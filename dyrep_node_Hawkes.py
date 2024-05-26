import numpy as np
import torch
from datetime import datetime, timedelta
from torch.nn import Linear, ModuleList, Parameter
import bisect

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
        self.W_t = Linear(1,hidden_dim)

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
        A = torch.from_numpy(A_initial).float().to(self.device) ## ** (num_node, num_node)로 설정해야함 
        self.register_buffer('z', z)
        self.register_buffer('A', A)
        self.node_degree_global = node_degree_initial ## **train에는 (n_assoc_type,num_node로) -> (num_node)로 설정해야함
        self.time_bar = time_bar

        ## S를 1/deg(u)로 initalize
        self.initialize_S_from_A()

        assert torch.sum(torch.isnan(A)) == 0, (torch.sum(torch.isnan(A)), A)

        self.Lambda_dict = torch.zeros(500, device=self.device) #parameter 5000 원래
        self.time_keys = [] 

    def initialize_S_from_A(self):
        S = self.A.new_zeros((self.num_nodes, self.num_nodes))
        D = torch.sum(self.A, dim=1)
        for v in torch.nonzero(D, as_tuple=False):
            u = torch.nonzero(self.A[v, :].squeeze(), as_tuple=False)
            
            S[v, u] = 1. / D[v][0]
        self.S = S
        # Check that values in each row of S add up to 1
        for idx, row in enumerate(S):
            assert torch.isclose(torch.sum(row), torch.tensor(1.0, device=self.device, dtype=torch.float32), atol=1e-4)
        
        # A의 0인 부분에 대해 S의 값이 작은지 확인
        assert torch.sum(S[self.A == 0]) < 1e-5, torch.sum(S[self.A == 0])

    def forward(self, data):
        u, time_delta, time_bar, time_cur, significance, magnitudo = data[:6]

        # 기본 세팅
        batch_size = len(u)
        u_all = u.data.cpu().numpy()
        lambda_all_list, surv_all_list, lambda_pred = None, None, None
        lambda_list,  lambda_u_neg = [], []
        batch_embeddings_u, batch_embeddings_u_neg = [], []
        ts_diff_neg = []
        ts_diff = []
        z_prev = self.z
        expected_time = []

        # testing일때, A_pred, surv를 초기화 세팅
        if not self.training:
            lambda_all_list = self.A.new_zeros((batch_size, self.num_nodes))
            surv_all_list = self.A.new_zeros((batch_size, self.num_nodes))
        
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
            u_event, time_delta_event, time_bar_it, time_cur_it, significance_it, magnitudo_it = u_all[it], time_delta[it], time_bar[it], time_cur[it],significance[it],magnitudo[it] 
            u_event = int(u_event)

            # impact_nodes 계산
            u_neigh = torch.nonzero(self.A[u_event, :] == 1, as_tuple=True)[0]
            u_event_tensor = torch.tensor([u_event], dtype=torch.int64)
            impact_nodes = torch.cat((u_event_tensor, u_neigh)) # u_event랑 연결된애들 모음

            # time_delta_it 계산 - for update_node_embedding 함수
            time_delta_it = np.zeros((int(impact_nodes.shape[0]), 1)) # 같은 사이즈로 생성
            for i, k in enumerate(impact_nodes):
                time_delta_it[i] = time_cur_it - time_bar_it[i] # 현재 시간과 time_bar 차이 **** k아님?
            time_delta_it = torch.Tensor(time_delta_it) # impact_nodes의 t_curr - t_bar
            # event_node의 delta 계속 저장
            ts_diff.append(time_cur_it - time_bar_it[impact_nodes[0]]) # (1,) 저장
            
            # ts_diff_neg 계산 size:(num_neg_samples,1) - survivabsal probability를 위한 negative sampling 생성
            batch_nodes = np.delete(np.arange(self.num_nodes), [u_event])
            batch_u_neg = self.random_state.choice(batch_nodes, size=self.num_neg_samples, replace=len(batch_nodes) < self.num_neg_samples)
            last_t_neg = time_bar_it[batch_u_neg] # (num_neg_samples,1)
            ts_diff_neg.append(time_cur_it -last_t_neg) # neg의 t_curr - t_bar - (num_neg_samples,1)

            # z_new(node_num,1) - u_event랑 u_neigh에 대한 z만 수정
            z_new = self.update_node_embedding_without_attention(z_prev, u_event, u_neigh, time_delta_it)
            assert torch.sum(torch.isnan(z_new)) == 0, (torch.sum(torch.isnan(z_new)), z_new, it)
            
            # 각 노드의 embedding 값 저장
            batch_embeddings_u.append(z_prev[u_event]) #값 한개
            batch_embeddings_u_neg.append(z_prev[batch_u_neg]) # (num_neg_samples) list append

            ## 모든 node별 conditional density
            with torch.no_grad():
                # 모든 노드에 대한 hawkes lambda 계산하기
                time_diff_pred = time_cur_it - time_bar_it # node 별 
                lambda_all_pred = self.compute_hawkes_lambda(z_prev, time_diff_pred)
                
                # test - survival probability 계산 후, 저장
                if not self.training:
                    lambda_all_list[it, :] = lambda_all_pred.squeeze(1)
                    assert torch.sum(torch.isnan(lambda_all_list[it])) == 0, (it, torch.sum(torch.isnan(lambda_all_list[it])))
                    s_u = self.compute_cond_density(u_event, time_bar_it)
                    surv_all_list[it,:] = s_u

                # Lambda_dict: (survival probability 구하기용) 시간 순서대로 event에 대한 람다 저장
                if len(self.time_keys) >= len(self.Lambda_dict): # 안중요 - 크기를 넘어서면 예전것부터 없앰
                    time_keys = np.array(self.time_keys)
                    time_keys[:-1] = time_keys[1:]
                    self.time_keys = list(time_keys[:-1])
                    self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                    self.Lambda_dict[-1] = 0
                
                # 발생하지 않은 노드들의 lambda를 더해서 lambda_dict에 저장 - Lambda_dict[1]: 1번째 batch에 대한 lambda_all_pred
                idx = np.delete(np.arange(self.num_nodes), [u_event])
                self.Lambda_dict[len(self.time_keys)] = lambda_all_pred[idx].sum().detach()
                self.time_keys.append(time_cur_it)

                self.time_bar[u_event] = time_cur_it

                # test - for time prediction - 여기 아직 안봄
                if not self.training:
                    t_cur_date = time_cur_it
                    time_scale_hour = t_cur_date - time_bar_it[u_event] # 계산 시간 차이
                    
                    surv_allsamples = z_new.new_zeros(self.num_time_samples)
                    factor_samples = 2 * self.random_state.rand(self.num_time_samples) # ** 왜 2곱함?
                    sampled_time_scale = time_scale_hour * factor_samples

                    embeddings_u = z_new[int(u_event)].expand(self.num_time_samples, -1)
                    all_td_c = torch.zeros(self.num_time_samples)
                    
                    
                    t_c_n = torch.tensor(list(np.cumsum(sampled_time_scale))).to(self.device)
                    all_td_c = t_c_n
                    print(all_td_c.size())

                    all_u_neg_sample = self.random_state.choice(batch_nodes, size=self.num_neg_samples*self.num_time_samples,
                                        replace=len(batch_nodes) < self.num_neg_samples*self.num_time_samples)
                    embeddings_u_neg = z_new[all_u_neg_sample]
                    print("error part")
                    print(time_scale_hour.shape) # batch1 수
                    print(factor_samples.shape) # num_time_samples5 수
                    print(sampled_time_scale.shape) # 5
                    print(t_c_n.shape) # self.num_neg_samples*self.num_time_samples 25

                    surv_neg =  self.compute_hawkes_lambda(embeddings_u_neg, all_td_c)

                    surv_allsamples = surv_neg.view(-1,self.num_neg_samples).mean(dim=-1)
                    lambda_t_allsamples = self.compute_hawkes_lambda(embeddings_u, all_td_c)
                    print(surv_allsamples.size())
                    f_samples = lambda_t_allsamples*torch.exp(-surv_allsamples)
                    expectation = torch.cumsum(sampled_time_scale, dim=0)*f_samples
                    expectation = expectation.sum()
                    expected_time.append(expectation/self.num_time_samples)

            ## Update embedding
            z_prev = z_new
        
        # training 끝나면 z 반영
        self.z = z_new
        
        # event에 따른 lambda_list를 새로 계산 - batch_embeddings_u는 이전 없데이트에 대해
        # time_delta를 쓰는거에 대해 생각해봐야할듯 ****
        # batch_embeddings_u = torch.stack(batch_embeddings_u, dim=0)
        # last_t_u = time_delta[torch.arange(batch_size), [0]*batch_size]
        # ts_diff = time_cur.view(-1)-last_t_u
        # lambda_list = self.compute_hawkes_lambda(batch_embeddings_u, ts_diff)
        
        batch_embeddings_u = torch.stack(batch_embeddings_u, dim=0)
        last_t_u = time_delta[torch.arange(batch_size), [0]*batch_size]
        ts_diff = (time_cur.view(-1)-last_t_u).unsqueeze(1)
        lambda_list = self.compute_hawkes_lambda(batch_embeddings_u, ts_diff)
        


        batch_embeddings_u_neg = torch.cat(batch_embeddings_u_neg, dim=0)
        lambda_u_neg = torch.zeros(len(batch_embeddings_u_neg), device=self.device)
        ts_diff_neg = torch.cat(ts_diff_neg)
        lambda_u_neg = self.compute_hawkes_lambda(batch_embeddings_u_neg, ts_diff_neg)

        # attention은 제외함
        # for i,k in enumerate(event_types):
        #     u_it, v_it = u_all[i], v_all[i]
        #     self.update_A_S(u_it, v_it, k, lambda_uv[i].item())
        #     for j in [u_it, v_it]:
        #         self.node_degree_global[j] = torch.sum(self.A[j, :]>0).item()
        

        # 리턴값: 이벤트 노드의 람다, neg 노드의 평균, A_pred, surv, 예상시간
        return lambda_list, lambda_u_neg / self.num_neg_samples,lambda_all_list, surv_all_list, expected_time
        
    def compute_hawkes_lambda(self, z_u, td):
        """
        주어진 node embedding과 시간 차이를 사용하여 Hawkes 프로세스를 통해 event 강도 (lambda)를 계산
        
        Args:
        z_u (torch.Tensor): Source node의 embedding (shape: [batch_size, hidden_dim])
        td (torch.Tensor): 현재 event와 마지막 event 사이의 시간 차이 (shape: [batch_size])

        Returns:
        torch.Tensor: 주어진 node에 의해 계산된 lambda ()
        """

        # embedding이 올바른 shape을 가지도록 보장
        z_u = z_u.reshape(-1, self.hidden_dim)
        g = z_u.new_zeros(len(z_u))    # 강도 벡터를 초기화
        # 각 event 유형에 대해 해당 linear layer (omega)를 사용하여 강도 (g)를 계산
        g = self.omega(z_u).flatten().unsqueeze(1)

        psi = self.psi
        alpha = self.alpha
        w_t = self.w_t
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g/(psi + 1e-7), -75, 75) # avoid overflow (batch_size,1)
        
        # Hawkes 프로세스 강도 (Lambda) 계산, 뒷부분이 hawkes를 의미
        # Lambda = psi * torch.log(1 + torch.exp(g_psi))
        time_effect = alpha*torch.exp(-w_t*(td/self.train_td_max))
        
        Lambda = psi * torch.log(1 + torch.exp(g_psi)) + time_effect
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
        
        # 이전 embedding을 복제하여 새로운 embedding 생성
        z_new = prev_embedding.clone()
        
        #neighborhood node에 대한 업데이트
        # for u_neighborhood in u_neighborhoods:
        #     z_new[u_neighborhood] = torch.sigmoid(self.W_event_to_neigh(prev_embedding[u_event]) + \
        #                             self.W_rec_neigh(prev_embedding[u_neighborhood]) + \
        #                             self.W_t(time_delta_it[u_neighborhood]))
        z_new[u_neighborhood] = torch.sigmoid(self.W_event_to_neigh(prev_embedding[int(u_event)]) + \
                                  self.W_rec_neigh(prev_embedding[u_neighborhood]) + \
                                  self.W_t(time_delta_it[1:]))
        
        #event node에 대한 update 
        z_new[u_event] = torch.sigmoid(self.W_rec_event(prev_embedding[u_event]) + \
                                  self.W_t(time_delta_it[0]))
        return z_new

    def compute_cond_density(self, u, time_bar):
        # print(u)
        N = self.num_nodes
        surv = self.Lambda_dict.new_zeros((1, N))
        #단순 코너케이스
        if not self.time_keys:
            
            return surv
        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0) / len(self.Lambda_dict)
        
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]
        indices = []
        lambda_indices = []
        t_bar_u = self.time_bar[u].item()
        if t_bar_u < time_keys_min:
            start_ind_min = 0
            #노드의 이벤트가 dictionary 저장 전에 일어났따. 
        elif t_bar_u > time_keys_max:
            # 이벤트가 이 노드들에서는 발생하지 않았다.
            return surv
        else:
            # 일어난 노드부터 index시작
            # self.time_keys[-10:].index(t_bar_u)
            start_ind_min = self.time_keys.index(t_bar_u)
        
        for i in range(N):
            t_bar = self.time_bar[i].item()
            if i == u:
                start_ind = start_ind_min
            if t_bar < time_keys_min:
                start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
            elif t_bar > time_keys_max:
                start_ind = -1
                 # it means t_bar is current event, so there is no history for this pair of nodes
            else:
                # t_bar is somewhere in after time_keys_min
                start_ind = self.time_keys.index(t_bar)
            lambda_indices.append(start_ind)
        surv = Lambda_sum[lambda_indices].view(-1,N)
        return surv
        

        if t_bar_u < time_keys_min:
            start_ind_min = 0
            #노드의 이벤트가 dictionary 저장 전에 일어났따. 
        elif t_bar_u > time_keys_max:
            # 이벤트가 이 노드들에서는 발생하지 않았다.
            return surv
        else:
            # 일어난 노드부터 index시작
            # print(self.time_keys[0:10])
            # print(self.time_keys[-10:])
            # print(int(t_bar_u))
            index = bisect.bisect_left(self.time_keys, int(t_bar_u))
            start_ind_min = index
            # start_ind_min = self.time_keys.index(int(t_bar_u))
            # print(start_ind_min)
            # print(self.time_keys[start_ind_min-1])
            # print(self.time_keys[start_ind_min])
            # print(self.time_keys[start_ind_min+1])
            

        for i in range(N):
            t_bar = time_bar[i].item()
            if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
            elif t_bar > time_keys_max:
                continue  # it means t_bar is current event, so there is no history for this pair of nodes
            else:
                # t_bar is somewhere in after time_keys_min
                # start_ind = self.time_keys.index(t_bar, start_ind_min)
                index = bisect.bisect_left(self.time_keys, int(t_bar))
                start_ind = index
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