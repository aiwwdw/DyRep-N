import numpy as np
import torch
from datetime import datetime, timedelta
from torch.nn import Linear, ModuleList, Parameter


class DyRepNode(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, random_state, first_date, end_datetime, num_neg_samples= 5, num_time_samples = 10,
                 device='cpu', all_comms=False, train_td_max=None):
        super(DyRepNode, self).__init__()

        self.batch_update = True
        self.all_comms = all_comms
        self.include_link_features = False
    
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.first_date = first_date
        self.end_datetime = end_datetime
        self.num_neg_samples = num_neg_samples
        self.device = device
        self.num_time_samples = num_time_samples
        self.train_td_max = train_td_max

        self.w_t = Parameter(0.5*torch.ones(1))
        self.alpha = Parameter(0.5*torch.ones(1))
        self.psi = Parameter(0.5*torch.ones(1))
        self.omega = Linear(in_features=hidden_dim, out_features=1)
        
        self.W_h = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_event_to_neigh = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_rec_event = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_rec_neigh = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_t = Linear(4,hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        모델의 모든 Linear 레이어의 파라미터를 초기화
        각 Linear 레이어의 초기화 메서드를 호출하여 파라미터를 재설정
        """
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def reset_state(self, node_embeddings_initial, A_initial, node_degree_initial, time_bar):
        """
        모델의 상태를 초기화
        main에서 모델 만들고 실행됨
        Args:
        node_embeddings_initial(=self.z) : one hot vector encoding
        A_initial(=self.A) : adjacency matrix (fixed)
        node_degree_initial(=self.node_degree_global) : can be computed by A
        time_bar : 각 node에 대한 initial time bar (=train_set.FIRST_DATE.timestamp()) (shape: (train_set.N_nodes, 1))
        """
        z = np.pad(node_embeddings_initial, ((0, 0), (0, self.hidden_dim - node_embeddings_initial.shape[1])),'constant')
        z = torch.from_numpy(z).float().to(self.device)
        A = torch.from_numpy(A_initial).float().to(self.device) ## *** (num_node, num_node)로 설정해야함 
        if len(A.shape) == 2:
            A = A.unsqueeze(2)
        self.register_buffer('z', z)
        self.register_buffer('A', A)
        self.node_degree_global = node_degree_initial ## ***train에는 (n_assoc_type,num_node로) -> (num_node)로 설정해야함
        self.time_bar = time_bar

        ## S를 1/deg(u)로 initalize
        self.initialize_S_from_A()

        assert torch.sum(torch.isnan(A)) == 0, (torch.sum(torch.isnan(A)), A)

        self.Lambda_dict = torch.zeros(5000, device=self.device)
        self.time_keys = []

    def initialize_S_from_A(self):
        S = self.A.new_zeros((self.num_nodes, self.num_nodes))
        
        D = torch.sum(self.A[:,:], dim=1)
        for v in torch.nonzero(D, as_tuple=False):
            u = torch.nonzero(self.A[v,:].squeeze(), as_tuple=False)
            S[v,u] = 1. / D[v]
        self.S = S
        # Check that values in each row of S add up to 1
        S = self.S[:, :]
        assert torch.sum(S[self.A[:, :] == 0]) < 1e-5, torch.sum(S[self.A[:, :] == 0])

    def forward(self, data):
        u, time_delta, time_bar, time_cur,significance,magnitudo = data[:6]
        
        batch_size = len(u)
        u_all = u.data.cpu().numpy()
        
        # testing일때, A_pred, surv를 초기화 세팅
        if not self.training:
            lambda_all_list, surv_all_list, lambda_pred = None, None, None
            lambda_all_list = self.A.new_zeros((batch_size, self.num_nodes))
            surv_all_list = self.A.new_zeros((batch_size, self.num_nodes))

        # *** time의 shape 알고 수정 필요
        # *** time을 normalize할 바에, fixed encoding을 하는 방식을 어떨까?
        # [sw] normalize 하는 이유는 모르겠고, 24,12진법을 10진법으로 변환은 의미가 있을듯
        time_mean = torch.from_numpy(np.array([0, 0, 0, 0])).float().to(self.device).view(1, 1, 4)
        time_sd = torch.from_numpy(np.array([50, 7, 15, 15])).float().to(self.device).view(1, 1, 4)
        time_delta = (time_delta - time_mean) / time_sd

        # 기본 세팅
        lambda_list,  lambda_u_neg = [], []
        batch_embeddings_u, batch_embeddings_u_neg = [], []
        ts_diff_neg = []
        z_prev = self.z # 초기 세팅
        expected_time = []
        # 모든 training data에 대해서 for문 시행
       
        for it in range(batch_size):
            """
            input:
            u_it: event node와 neighbor (shape: (1 + degree(u)))
            time_delta_it : delta_u + delta_N(u)
            time_bar_it: 
            time_cur_it: 
            significance_it: 
            magnitudo_it: 
            """
            # event edge 하나씩
            u_it, time_delta_it, time_bar_it, time_cur_it,significance_it,magnitudo_it = u_all[it], time_delta[it], time_bar[it], time_cur[it],significance[it],magnitudo[it] 
            u_event = u_it[0]
            u_neighborhood = u_it[1:]
            time_delta_event = time_delta_it[0]
            time_delta_neighborhood = time_delta_it[1:]

            ## 1. lambda 구하기
            # batch_update면 다 batch 끝나고 기록된 v,u 임베딩에 대해서 계산
            # 아니면 실시간 람다 계산 및 리스트 저장
            if self.batch_update:
                batch_embeddings_u.append(z_prev[u_event]) 
            else:
                lambda_u_it = self.compute_intensity_lambda(z_prev[u_it])
                lambda_list.append(lambda_u_it)

            ## 2. 노드별 embedding 계산
            z_new = self.update_node_embedding_without_attention(z_prev, u_event, u_neighborhood, time_delta_it)
            assert torch.sum(torch.isnan(z_new)) == 0, (torch.sum(torch.isnan(z_new)), z_new, it)
            

            ## 3. batch_update 업데이트가 아니면 매 순간 S,A 계산
            # batch_update면 뒤쪽에서 한번에 
            # *** 수정안함
            # if not self.batch_update:
            #     self.update_S(u_it, lambda_u_it)


            ## 4. survival probability를 위한 negative sampling 생성
            # u를 제외한 노드들에 대하여, num_neg_samples 만큼의 노드를 샘플링
            # 샘플링 된 노드에 대하여, time different 저장하기
            # 뒤에서 loss 계산에서 쓰일 예정
            batch_nodes = np.delete(np.arange(self.num_nodes), [u_event])
            batch_u_neg = self.random_state.choice(batch_nodes, size=self.num_neg_samples,
                                                    replace=len(batch_nodes) < self.num_neg_samples)
            batch_embeddings_u_neg.append(z_prev[batch_u_neg])
            last_t_neg = time_bar_it[batch_u_neg]
            ts_diff_neg.append(time_cur_it -last_t_neg)

            ## 5. 모든 node별 conditional density
            with torch.no_grad():
                # 모든 노드에 대한 hawkes lambda 계산하기
                time_diff_pred = time_cur_it - time_bar_it
                lambda_all_pred = self.compute_hawkes_lambda(z_prev, time_diff_pred)
                
                # survival probability 계산 후, 저장
                if not self.training:
                    lambda_all_list[it, :] = lambda_all_pred
                    assert torch.sum(torch.isnan(lambda_all_list[it])) == 0, (it, torch.sum(torch.isnan(lambda_all_list[it])))
                    s_u = self.compute_cond_density(u_it, time_bar_it)
                    surv_all_list[it,:] = s_u

                # *** 어떤 type의 시간을 사용할것인가?
                # Lambda_dict: 이벤트 일어나는 순서대로 람다 저장(최대 사이즈는 fix) -> survival probability 구할 때 이용
                time_key = time_cur_it
                # u,v에 연결할수는 있는 모든 노드 중, u,v를 제거해서 lambda를 계산, 
                idx = np.delete(np.arange(self.num_nodes), [u_event])
                
                # 크기를 넘어서면 예전것부터 없앰
                if len(self.time_keys) >= len(self.Lambda_dict):
                    time_keys = np.array(self.time_keys)
                    time_keys[:-1] = time_keys[1:]
                    self.time_keys = list(time_keys[:-1])
                    self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                    self.Lambda_dict[-1] = 0
                #발생하지 않은 노드들의 lambda를 더해서 lambda_dict에 저장
                self.Lambda_dict[len(self.time_keys)] = lambda_all_pred[idx].sum().detach()
                self.time_keys.append(time_key)

                # test for time prediction
                if not self.training:
                    t_cur_date = datetime.fromtimestamp(int(time_cur_it))
                    # Use the cur and most recent time
                    t_prev = datetime.fromtimestamp(int(time_bar_it[u_event]))
                    td = t_cur_date - t_prev
                    time_scale_hour = round((td.days*24 + td.seconds/3600),3)
                    surv_allsamples = z_new.new_zeros(self.num_time_samples)
                    factor_samples = 2*self.random_state.rand(self.num_time_samples)
                    sampled_time_scale = time_scale_hour*factor_samples

                    embeddings_u = z_new[u_event].expand(self.num_time_samples, -1)
                    all_td_c = torch.zeros(self.num_time_samples)

                    t_c_n = torch.tensor(list(map(lambda x: int((t_cur_date+timedelta(hours=x)).timestamp()),
                                                  np.cumsum(sampled_time_scale)))).to(self.device)
                    all_td_c = t_c_n - time_cur_it

                    all_u_neg_sample = self.random_state.choice(batch_nodes, size=self.num_neg_samples*self.num_time_samples,
                                        replace=len(batch_nodes) < self.num_neg_samples*self.num_time_samples)
                    surv_neg =  self.compute_hawkes_lambda(all_u_neg_sample, all_td_c)
                    surv_allsamples = surv_neg.view(-1,self.num_neg_samples).mean(dim=-1)
                    lambda_t_allsamples = self.compute_hawkes_lambda(embeddings_u, all_td_c)
                    f_samples = lambda_t_allsamples*torch.exp(-surv_allsamples)
                    expectation = torch.from_numpy(np.cumsum(sampled_time_scale))*f_samples
                    expectation = expectation.sum()
                    expected_time.append(expectation/self.num_time_samples)

            ## 6. Update the embedding z
            z_prev = z_new
        
        # training data에 대한 for문이 끝난후
        self.z = z_new

        # time prediction
        
        #### batch update for all events' intensity

        # batch_update면 위에서 했던거 for문 끝나고 한번에 시행
        # 아니면 이미 계산한거 합치기
        if self.batch_update:
            batch_embeddings_u = torch.stack(batch_embeddings_u, dim=0)
            last_t_u = time_delta[torch.arange(batch_size), [0]*batch_size]
            ts_diff = time_cur.view(-1)-last_t_u
            lambda_list = self.compute_hawkes_lambda(batch_embeddings_u, ts_diff)
            # *** attention은 제외함
            # for i,k in enumerate(event_types):
            #     u_it, v_it = u_all[i], v_all[i]
            #     self.update_A_S(u_it, v_it, k, lambda_uv[i].item())
            #     for j in [u_it, v_it]:
            #         self.node_degree_global[j] = torch.sum(self.A[j, :]>0).item()
        else: 
            lambda_list = torch.cat(lambda_list, dim=0)

        # 앞에 데이터들 형식 통일
        # neg 엣지들에 대해서도 람다 계산
        batch_embeddings_u_neg = torch.cat(batch_embeddings_u_neg, dim=0)
        neg_events_len = len(batch_embeddings_u_neg)
        lambda_u_neg = torch.zeros(neg_events_len, device=self.device)
        ts_diff_neg = torch.cat(ts_diff_neg)
        lambda_u_neg = self.compute_hawkes_lambda(batch_embeddings_u_neg, ts_diff_neg)
    
        # 리턴값: 이벤트 노드의 람다, neg 노드의 평균, A_pred, surv, 예상시간
        return lambda_list, lambda_u_neg / self.num_neg_samples,lambda_all_list, surv_all_list, expected_time
        
    def compute_hawkes_lambda(self, z_u, td):
        """
        주어진 node embedding과 시간 차이를 사용하여 Hawkes 프로세스를 통해 event 강도 (lambda)를 계산
        
        Args:
        z_u (torch.Tensor): Source node의 embedding (shape: [batch_size, hidden_dim])
        z_v (torch.Tensor): Target node의 embedding (shape: [batch_size, hidden_dim])
        et_uv (torch.Tensor): node u와 v 사이의 event 유형 (shape: [batch_size])
        td (torch.Tensor): 현재 event와 마지막 event 사이의 시간 차이 (shape: [batch_size])

        Returns:
        torch.Tensor: 주어진 node pair와 event 유형에 대한 계산된 lambda
        """

        # embedding이 올바른 shape을 가지도록 보장
        z_u = z_u.view(-1, self.hidden_dim)
        g = z_u.new_zeros(len(z_u))    # 강도 벡터를 초기화
        # 각 event 유형에 대해 해당 linear layer (omega)를 사용하여 강도 (g)를 계산
        g = self.omega(z_u).flatten()

        psi = self.psi
        alpha = self.alpha
        w_t = self.w_t
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g/(psi + 1e-7), -75, 75) # avoid overflow
        
        # Hawkes 프로세스 강도 (Lambda) 계산, 뒷부분이 hawkes를 의미
        Lambda = psi * torch.log(1 + torch.exp(g_psi)) + alpha*torch.exp(-w_t*(td/self.train_td_max))
        return Lambda

    def compute_intensity_lambda(self, z_u):
        """
        주어진 node embedding을 사용하여 event 강도 (lambda)를 계산
        """
        # embedding이 올바른 shape을 가지도록 보장
        z_u = z_u.view(-1, self.hidden_dim)
        g = z_u.new_zeros(len(z_u))  # 강도 벡터를 초기화
        # 각 event 유형에 대해 해당 linear layer (omega)를 사용하여 강도 (g)를 계산
        g = self.omega(z_u).flatten()

        psi = self.psi
        g_psi = torch.clamp(g/(psi + 1e-7), -75, 75) # avoid overflow

        # 강도 (Lambda) 계산
        Lambda = psi * torch.log(1 + torch.exp(g_psi))
        return Lambda
 
    def update_node_embedding_without_attention(self, prev_embedding, u_event, u_neighborhood, time_delta_it):
        """
        주어진 node embedding과 시간 차이를 사용하여 node embedding을 업데이트합니다.
        
        Args:
        prev_embedding (torch.Tensor): 이전 embedding (shape: [num_nodes, hidden_dim])
        u_event: event가 발생한 노드
        u_neighborhood: event가 발생 노드의 이웃
        time_delta_it (torch.Tensor): 시간 차이 (shape: [batch_size, 4])

        Returns:
        torch.Tensor: 업데이트된 node embedding
        """
        # *** attention 사용하지 않은 버전

        # 이전 embedding을 복제하여 새로운 embedding 생성
        z_new = prev_embedding.clone()
        
        #neighborhood node에 대한 업데이트
        # for u_neighborhood in u_neighborhoods:
        #     z_new[u_neighborhood] = torch.sigmoid(self.W_event_to_neigh(prev_embedding[u_event]) + \
        #                             self.W_rec_neigh(prev_embedding[u_neighborhood]) + \
        #                             self.W_t(time_delta_it[u_neighborhood]))
        
        z_new[u_neighborhood] = torch.sigmoid(self.W_event_to_neigh(prev_embedding[u_event]) + \
                                  self.W_rec_neigh(prev_embedding[u_neighborhood]) + \
                                  self.W_t(time_delta_it[u_neighborhood].view(len(u_neighborhood),4)))
        
        #event node에 대한 update 
        z_new[u_event] = torch.sigmoid(self.W_rec_event(prev_embedding[u_event]) + \
                                  self.W_t(time_delta_it[u_event]))
        return z_new

    
    def compute_cond_density(self, u, time_bar):
        N = self.num_nodes
        surv = self.Lambda_dict.new_zeros((1, N))

        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0) / len(self.Lambda_dict)
        
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]
        indices = []
        lambda_indices = []
        t_bar_u = time_bar[u].item()
        if t_bar_u < time_keys_min:
            start_ind_min = 0
            #노드의 이벤트가 dictionary 저장 전에 일어났따. 
        elif t_bar_u > time_keys_max:
            # 이벤트가 이 노드들에서는 발생하지 않았다.
            return surv
        else:
            # 일어난 노드부터 index시작
            start_ind_min = self.time_keys.index(int(t_bar_u))
        
        for i in range(N):
            t_bar = time_bar[i]
            if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
            elif t_bar > time_keys_max:
                continue  # it means t_bar is current event, so there is no history for this pair of nodes
            else:
                # t_bar is somewhere in after time_keys_min
                start_ind = self.time_keys.index(t_bar, start_ind_min)
                lambda_indices.append(start_ind)
        surv = Lambda_sum[lambda_indices].view(-1,N)

        return surv


    # def update_S(self, u_it, v_it, lambda_uv_t):
    #     """
    #     주어진 event 정보를 사용하여 adjacency matrix (A)와 influence matrix (S)를 업데이트
        
    #     Args:
    #     u_it (int): Source node의 인덱스.
    #     v_it (int): Target node의 인덱스.
    #     et_it (int): event 유형.
    #     lambda_uv_t (float): node u와 v 사이의 event 강도 (lambda).
    #     """
    #     # 모든 이벤트를 association = communication으로 보는 경우, 
    #     if self.all_comms:
    #         self.A[u_it, v_it, 0] = self.A[v_it, u_it, 0] = 1
    #     else:
    #         # event 유형이 0일 경우, 해당하는 타입의 adjacency matrix (A)를 업데이트
    #         self.A[u_it, v_it] = self.A[v_it, u_it] = 1
    #     A = self.A
    #     indices = torch.arange(self.num_nodes, device=self.device)

    #     # attention matrix update -> algorithm 1과 동일
    #     if (A[u_it, v_it]!=0):
    #         for j,i in [(u_it,v_it), (v_it, u_it)]:
    #             y = self.S[j, :]
    #             # TODO: check if this work (not use the node degree when compute embedding)
    #             degree_j = torch.sum(A[j,:] > 0).item()
    #             b = 0 if degree_j==0 else 1/(float(degree_j) + 1e-7)
    #             if A[j,i]==1:
    #                 y[i] = b + lambda_uv_t
    #             elif A[j,i]==1:
    #                 degree_j_bar = self.node_degree_global[j]
    #                 b_prime = 0 if degree_j_bar==0 else 1./(float(degree_j_bar) + 1e-7)
    #                 x = b_prime - b
    #                 y[i] = b + lambda_uv_t
    #                 w_idx = (y!=0) & (indices != int(i))
    #                 # w_idx[int(i)] = False
    #                 y[w_idx] = y[w_idx]-x
    #             y /= (torch.sum(y)+ 1e-7)
    #             self.S[j,:] = y