import numpy as np
import torch
from datetime import datetime, timedelta
from torch.nn import Linear, ModuleList, Parameter


class DyRepNode(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, random_state, first_date, end_datetime, num_neg_samples= 5, num_time_samples = 10,
                 device='cpu', all_comms=False, train_td_max=None):
        super(DyRepNode, self).__init__()

        self.batch_update = True
        self.hawkes = True
        self.bipartite = False
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

        # TODO: TB we bring bias term to the linear layer by using Linear (set bias=False to exempt or directly use parameter)
        # if not self.include_link_features:
        #     self.omega = ModuleList([Linear(in_features=2*hidden_dim, out_features=1),
        #                              Linear(in_features=2*hidden_dim, out_features=1)])
        # else:
        #     self.omega = ModuleList([Linear(in_features=2*hidden_dim+172, out_features=1),
        #                              Linear(in_features=2*hidden_dim+172, out_features=1)])

        self.omega = Linear(in_features=hidden_dim, out_features=1)
        

        self.W_h = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_evnet_to_neigh = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_rec_event = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_rec_neigh = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_t = Linear(4,hidden_dim) # [days, hours, minutes, seconds]

        self.reset_parameters()

    def reset_parameters(self):
        """
        모델의 모든 Linear 레이어의 파라미터를 초기화
        각 Linear 레이어의 초기화 메서드를 호출하여 파라미터를 재설정
        """
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def reset_state(self, node_embeddings_initial, A_initial, node_degree_initial, time_bar, resetS=False):
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
            A_pred, surv, lambda_pred = None, None, None
            A_pred = self.A.new_zeros((batch_size, self.num_nodes, self.num_nodes))
            surv = self.A.new_zeros((batch_size, self.num_nodes, self.num_nodes))

        # *** time의 shape 알고 수정 필요
        # *** time을 normalize할 바에, fixed encoding을 하는 방식을 어떨까?
        time_mean = torch.from_numpy(np.array([0, 0, 0, 0])).float().to(self.device).view(1, 1, 4)
        time_sd = torch.from_numpy(np.array([50, 7, 15, 15])).float().to(self.device).view(1, 1, 4)
        time_diff = (time_diff - time_mean) / time_sd

        # 기본 세팅
        lambda_list,  lambda_u_neg = [], []
        batch_embeddings_u, batch_embeddings_u_neg = [], [], [], []
        ts_diff_neg = []
        z_all = [] # *** 굳이 직전 임베딩 들고올건데 list까지 써야함?
        expected_time = []
        # 모든 training data에 대해서 for문 시행
       
        for it in range(batch_size):
            """
            input:
            u_it: event node와 neighbor (shape: (1 + degree(u)))
            time_delta_it : delta_u + delta_N(u)
            """
             # event edge 하나씩
            u_it, time_delta_it, time_bar_it, time_cur_it,significance_it,magnitudo_it = u_all[it], time_delta[it], time_bar[it], time_cur[it],significance[it],magnitudo[it] 
            u_event = u_it[0]
            u_neighborhood = u_it[1:]
            time_delta_event = time_delta_it[0]
            time_delta_neighborhood = time_delta_it[1:]

            #z는 1차원 배열, z_all은 2차원 배열
            z_prev = self.z if it == 0 else z_all[it - 1] 

            ## 1. lambda 구하기
            # batch_update면 다 batch 끝나고 기록된 v,u 임베딩에 대해서 계산
            # 아니면 실시간 람다 계산 및 리스트 저장
            if self.batch_update:
                batch_embeddings_u.append(z_prev[u_event]) 
            else:
                lambda_u_it = self.compute_intensity_lambda(z_prev[u_it])
                lambda_list.append(lambda_u_it)


            ## 2. 노드별 embedding 계산
            z_new = self.update_node_embedding(z_prev, u_it, time_delta_it)
            assert torch.sum(torch.isnan(z_new)) == 0, (torch.sum(torch.isnan(z_new)), z_new, it)
            

            ## batch_update 업데이트가 아니면 매 순간 S,A 계산
            # batch_update면 뒤쪽에서 한번에 
            if not self.batch_update:
                self.update_A_S(u_it, lambda_u_it)
                for j in [u_it, v_it]:
                    self.node_degree_global[j] = torch.sum(self.A[j, :]>0).item()


            ## to compute survival probability in loss 계산
            # u v를 제외한 노드들에 대하여, num_neg_samples 만큼의 노드를 샘플링
            # u와 샘플링된 노드를 엣지로, v와 샘플링된 노드를 엣지로 저장
            # u,v에 대해 엣지들의 t_bar 구하고, 종합 t_bar 구함
            batch_nodes = np.delete(np.arange(self.num_nodes), [u_it, v_it])
            # sample해야되는 총 개수가 batch_node보다 많으면 중복 허용인데 무의미한 코드
            batch_uv_neg = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                                                    replace=len(batch_nodes) < 2*self.num_neg_samples)
            batch_u_neg, batch_v_neg = batch_uv_neg[self.num_neg_samples:], batch_uv_neg[:self.num_neg_samples]
            # 뒤쪽에서 쓸듯 엣지 쌍 리스트로 저장
            batch_embeddings_u_neg.append(torch.cat((z_prev[u_it].expand(self.num_neg_samples, -1),
                                                     z_prev[batch_u_neg]), dim=0))
            batch_embeddings_v_neg.append(torch.cat([z_prev[batch_v_neg],
                                                     z_prev[v_it].expand(self.num_neg_samples, -1)], dim=0))
            # 샘플 노드들에 대해 각 t_bar에 대한 값 구하기 t_bar: (num_node, 1)
            last_t_u_neg = t_bar[it, np.concatenate([[u_it] * self.num_neg_samples, batch_u_neg]), 0]
            last_t_v_neg = t_bar[it, np.concatenate([batch_v_neg, [v_it] * self.num_neg_samples]), 0]
            # 각 엣지에 해당하는 노드 두개의 t_bar을 비교하여 negative samples edge의 t bar를 계산
            last_t_uv_neg = torch.cat([last_t_u_neg.view(-1,1), last_t_v_neg.view(-1,1)], dim=1).max(-1)[0].to(self.device)
            ts_diff_neg.append(t[it] - last_t_uv_neg)


            ## 5. 모든 pair별 conditional density
            with torch.no_grad():
                # (2 * self.num_nodes, hidden_dim)
                z_uv_it = torch.cat((z_prev[u_it].detach().unsqueeze(0).expand(self.num_nodes,-1),
                           z_prev[v_it].detach().unsqueeze(0).expand(self.num_nodes, -1)), dim=0)
                
                # hawkes라고 가정하고 쓸데없는 코드 지움
                # u,v에 연결되는 엣지의 람다만 구하기
                # two type of events: assoc + comm
                last_t_pred = torch.cat([
                    t_bar[it, [u_it, v_it], 0].unsqueeze(1).repeat(1, self.num_nodes).view(-1,1),
                    t_bar[it, :, 0].repeat(2).view(-1,1)], dim=1).max(-1)[0]
                ts_diff_pred = t[it].repeat(2*self.num_nodes) - last_t_pred
                lambda_uv_pred = self.compute_hawkes_lambda(z_uv_it, z_prev.detach().repeat(2,1),
                                                            et_it.repeat(len(z_uv_it)), ts_diff_pred).detach()
                
                # testing - 예측 A_pred 게산, Surv 계산
                if not self.training:
                    A_pred[it, u_it, :] = lambda_uv_pred[:self.num_nodes]
                    A_pred[it, v_it, :] = lambda_uv_pred[self.num_nodes:]
                    assert torch.sum(torch.isnan(A_pred[it])) == 0, (it, torch.sum(torch.isnan(A_pred[it])))
                    s_u_v = self.compute_cond_density(u_it, v_it, t_bar[it])
                    surv[it, [u_it, v_it], :] = s_u_v
                
                time_key = int(t[it])
                # u,v에 연결할수는 있는 모든 노드 중, u,v를 제거해서 lambda를 계산, 
                idx = np.delete(np.arange(self.num_nodes), [u_it, v_it])
                idx = np.concatenate((idx, idx+self.num_nodes))
                
                # Lambda_dict는 리스트 & 비효율적인 코드
                # 크기를 넘어서면 예전것부터 없앰
                # Lambda_dict: event 순서에 따
                if len(self.time_keys) >= len(self.Lambda_dict):
                    time_keys = np.array(self.time_keys)
                    time_keys[:-1] = time_keys[1:]
                    self.time_keys = list(time_keys[:-1])
                    self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                    self.Lambda_dict[-1] = 0
                #구해진 idx로 lambda를 더해서 lambda_dict에 저장
                self.Lambda_dict[len(self.time_keys)] = lambda_uv_pred[idx].sum().detach()
                self.time_keys.append(time_key)
                # test for time prediction
                if not self.training:
                    t_cur_date = datetime.fromtimestamp(int(t[it]))
                    # Use the cur and most recent time
                    t_prev = datetime.fromtimestamp(int(max(t_bar[it][u_it], t_bar[it][v_it])))
                    td = t_cur_date - t_prev
                    time_scale_hour = round((td.days*24 + td.seconds/3600),3)
                    surv_allsamples = z_new.new_zeros(self.num_time_samples)
                    factor_samples = 2*self.random_state.rand(self.num_time_samples)
                    sampled_time_scale = time_scale_hour*factor_samples

                    embeddings_u = z_new[u_it].expand(self.num_time_samples, -1)
                    embeddings_v = z_new[v_it].expand(self.num_time_samples, -1)
                    all_td_c = torch.zeros(self.num_time_samples)

                    t_c_n = torch.tensor(list(map(lambda x: int((t_cur_date+timedelta(hours=x)).timestamp()),
                                                  np.cumsum(sampled_time_scale)))).to(self.device)
                    all_td_c = t_c_n - t[it]

                    all_uv_neg_sample = self.random_state.choice(
                        batch_nodes,
                        size=self.num_neg_samples*2*self.num_time_samples,
                        replace=len(batch_nodes) < self.num_neg_samples*2*self.num_time_samples)
                    u_neg_sample = all_uv_neg_sample[:self.num_neg_samples*self.num_time_samples]
                    v_neg_sample = all_uv_neg_sample[self.num_neg_samples*self.num_time_samples:]

                    embeddings_u_neg = torch.cat((
                        z_new[u_it].view(1, -1).expand(self.num_neg_samples*self.num_time_samples, -1),
                        z_new[u_neg_sample]), dim=0).to(self.device)
                    embeddings_v_neg = torch.cat((
                        z_new[v_neg_sample],
                        z_new[v_it].view(1, -1).expand(self.num_neg_samples*self.num_time_samples, -1)), dim=0).to(self.device)
                    all_td_c_expand = all_td_c.unsqueeze(1).repeat(1,self.num_neg_samples).view(-1)
                    surv_0 = self.compute_hawkes_lambda(embeddings_u_neg, embeddings_v_neg,
                                                        torch.zeros(len(embeddings_u_neg)),
                                                        torch.cat([all_td_c_expand, all_td_c_expand]))
                    surv_1 = self.compute_hawkes_lambda(embeddings_u_neg, embeddings_v_neg,
                                                        torch.ones(len(embeddings_u_neg)),
                                                        torch.cat([all_td_c_expand, all_td_c_expand]))
                    surv_01 = (surv_0 + surv_1).view(-1,self.num_neg_samples).mean(dim=-1)
                    surv_allsamples = surv_01[:self.num_time_samples]+surv_01[self.num_time_samples:]

                    # for n in range(1, self.num_time_samples+1):
                    #     t_c_n = int((t_cur_date + timedelta(hours=sum(sampled_time_scale[:n]))).timestamp())
                    #     td_c = t_c_n - t[it]
                    #     all_td_c[n - 1] = td_c
                    #
                    #     batch_uv_neg_sample = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                    #                                             replace=len(batch_nodes) < 2 * self.num_neg_samples)
                    #     u_neg_sample = batch_uv_neg_sample[self.num_neg_samples:]
                    #     v_neg_sample = batch_uv_neg_sample[:self.num_neg_samples]
                    #     embeddings_u_neg = torch.cat((z_new[u_it].view(1,-1).expand(self.num_neg_samples,-1),
                    #                                     z_new[u_neg_sample]),dim=0)
                    #     embeddings_v_neg = torch.cat([z_new[v_neg_sample],
                    #                                   z_new[v_it].view(1,-1).expand(self.num_neg_samples,-1)],dim=0)
                    #
                    #     surv_0 = self.compute_hawkes_lambda(embeddings_u_neg, embeddings_v_neg,
                    #                                         torch.zeros(len(embeddings_u_neg)), td_c)
                    #     surv_1 = self.compute_hawkes_lambda(embeddings_u_neg, embeddings_v_neg,
                    #                                         torch.ones(len(embeddings_u_neg)), td_c)
                    #
                    #     surv_allsamples[n-1] = (torch.sum(surv_0) + torch.sum(surv_1)) / self.num_neg_samples

                    lambda_t_allsamples = self.compute_hawkes_lambda(embeddings_u, embeddings_v,
                                                                     torch.zeros(self.num_time_samples)+et_it,
                                                                     all_td_c)
                    f_samples = lambda_t_allsamples*torch.exp(-surv_allsamples)
                    expectation = torch.from_numpy(np.cumsum(sampled_time_scale))*f_samples
                    expectation = expectation.sum()
                    # 여기가 시간 예측 부분 !!!!
                    expected_time.append(expectation/self.num_time_samples)

            ## 6. Update the embedding z
            z_all.append(z_new)
        
        # training data에 대한 for문이 끝난후
        self.z = z_new

        # time prediction
        
        #### batch update for all events' intensity

        # batch_update면 위에서 했던거 for문 끝나고 한번에 시행
        # 아니면 이미 계산한거 합치기
        if self.batch_update:
            batch_embeddings_u = torch.stack(batch_embeddings_u, dim=0)
            batch_embeddings_v = torch.stack(batch_embeddings_v, dim=0)
            # hawkes라고 하고 쓸데없는거 지움
            last_t_u = t_bar[torch.arange(batch_size), u_all, [0]*batch_size]
            last_t_v = t_bar[torch.arange(batch_size), v_all, [0]*batch_size]
            last_t_uv = torch.cat([last_t_u.view(-1,1), last_t_v.view(-1,1)], dim=1).max(-1)[0]
            ts_diff = t.view(-1)-last_t_uv
            lambda_uv = self.compute_hawkes_lambda(batch_embeddings_u, batch_embeddings_v, event_types, ts_diff)
            for i,k in enumerate(event_types):
                u_it, v_it = u_all[i], v_all[i]
                self.update_A_S(u_it, v_it, k, lambda_uv[i].item())
                for j in [u_it, v_it]:
                    self.node_degree_global[j] = torch.sum(self.A[j, :]>0).item()
        else: 
            lambda_uv = torch.cat(lambda_uv, dim=0)

        # 앞에 데이터들 형식 통일
        # neg 엣지들에 대해서도 람다 계산
        batch_embeddings_u_neg = torch.cat(batch_embeddings_u_neg, dim=0)
        batch_embeddings_v_neg = torch.cat(batch_embeddings_v_neg, dim=0)
        neg_events_len = len(batch_embeddings_u_neg)
        lambda_uv_neg = torch.zeros(neg_events_len * 2, device=self.device)
        # hawkes라고 하고 쓸데없는거 지움
        ts_diff_neg = torch.cat(ts_diff_neg)
        lambda_uv_neg[:neg_events_len] = self.compute_hawkes_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg,
                                                                        torch.zeros(neg_events_len), ts_diff_neg)
        lambda_uv_neg[neg_events_len:] = self.compute_hawkes_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg,
                                                                        torch.ones(neg_events_len), ts_diff_neg)
    
        # 리턴값: 엣지의 람다, neg엣지의 평균, A_pred, surv, 예상시간
        return lambda_uv, lambda_uv_neg / self.num_neg_samples, A_pred, surv, expected_time
        
    def compute_hawkes_lambda(self, z_u, z_v, et_uv, td):
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
        
        Args:
        z_u (torch.Tensor): Source node의 embedding (shape: [batch_size, hidden_dim])
        z_v (torch.Tensor): Target node의 embedding (shape: [batch_size, hidden_dim])
        et_uv (torch.Tensor): node u와 v 사이의 event 유형 (shape: [batch_size])

        Returns:
        torch.Tensor: 주어진 node pair와 event 유형에 대한 계산된 lambda
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

    def update_node_embedding(self, prev_embedding, u_event,u_neighborhood, time_delta_it)
        """
        주어진 node embedding과 시간 차이를 사용하여 node embedding을 업데이트합니다.
        
        Args:
        prev_embedding (torch.Tensor): 이전 embedding (shape: [num_nodes, hidden_dim])
        u_event: event가 발생한 노드
        u_neighborhood: event가 발생 노드의 이웃
        time_delta_it (torch.Tensor): 시간 차이 (shape: [batch_size, 4]) , [days, hours, minutes, seconds]

        Returns:
        torch.Tensor: 업데이트된 node embedding
        """
        # *** attention 사용하지 않은 버전

        # 이전 embedding을 복제하여 새로운 embedding 생성
        z_new = prev_embedding.clone()


        # u와 v를 번갈아 가며 embedding 업데이트
        for cnt, (v,u) in enumerate([(node_u,  node_v), (node_v, node_u)]):
                u_nb = self.A[u, :] > 0 # u의 이웃 노드 선택
                num_u_nb = torch.sum(u_nb).item() # 이웃 노드의 수 계산
                if num_u_nb > 0:
                    # 이웃 노드의 embedding을 W_h를 통해 변환
                    h_i_bar = self.W_h(prev_embedding[u_nb]).view(num_u_nb, self.hidden_dim)
                    q_ui = torch.exp(self.S[u, u_nb])   # attention 계산
                    q_ui = q_ui / (torch.sum(q_ui) + 1e-7)  # attention 정규화 
                    h_u_struct[cnt, :] = torch.max(torch.sigmoid(q_ui.view(-1,1)*h_i_bar), dim=0)[0] #h_u_struct 계산
        #node u와 v의 embedding update 논문 수식 바탕
        z_new[[node_u, node_v]] = torch.sigmoid(self.W_struct(h_u_struct.view(2, self.hidden_dim)) + \
                                  self.W_rec(prev_embedding[[node_u, node_v]]) + \
                                  self.W_t(td).view(2, self.hidden_dim))
        return z_new
    
    def update_node_embedding_without_attention(self, prev_embedding, u_event,u_neighborhood, time_delta_it)
        """
        주어진 node embedding과 시간 차이를 사용하여 node embedding을 업데이트합니다.
        
        Args:
        prev_embedding (torch.Tensor): 이전 embedding (shape: [num_nodes, hidden_dim])
        u_event: event가 발생한 노드
        u_neighborhood: event가 발생 노드의 이웃
        time_delta_it (torch.Tensor): 시간 차이 (shape: [batch_size, 4]) , [days, hours, minutes, seconds]

        Returns:
        torch.Tensor: 업데이트된 node embedding
        """
        # *** attention 사용하지 않은 버전

        # 이전 embedding을 복제하여 새로운 embedding 생성
        z_new = prev_embedding.clone()
        
        #neighborhood node에 대한 업데이트
        z_new[u_event] = torch.sigmoid(self.W_evnet_to_neigh(prev_embedding([u_neighborhood])) + \
                                  self.W_rec(prev_embedding[[node_u, node_v]]) + \
                                  self.W_t(td).view(2, self.hidden_dim))
        
        #event node에 대한 update 
        z_new[u_event] = torch.sigmoid(self.W_struct(h_u_struct.view(2, self.hidden_dim)) + \
                                  self.W_rec(prev_embedding[u_event]) + \
                                  self.W_t(td).view(2, self.hidden_dim))
        return z_new


    def update_A_S(self, u_it, v_it, lambda_uv_t):
        """
        주어진 event 정보를 사용하여 adjacency matrix (A)와 influence matrix (S)를 업데이트
        
        Args:
        u_it (int): Source node의 인덱스.
        v_it (int): Target node의 인덱스.
        et_it (int): event 유형.
        lambda_uv_t (float): node u와 v 사이의 event 강도 (lambda).
        """
        # 모든 이벤트를 association = communication으로 보는 경우, 
        if self.all_comms:
            self.A[u_it, v_it, 0] = self.A[v_it, u_it, 0] = 1
        else:
            # event 유형이 0일 경우, 해당하는 타입의 adjacency matrix (A)를 업데이트
            self.A[u_it, v_it] = self.A[v_it, u_it] = 1
        A = self.A
        indices = torch.arange(self.num_nodes, device=self.device)

        # attention matrix update -> algorithm 1과 동일
        if (A[u_it, v_it]!=0):
            for j,i in [(u_it,v_it), (v_it, u_it)]:
                y = self.S[j, :]
                # TODO: check if this work (not use the node degree when compute embedding)
                degree_j = torch.sum(A[j,:] > 0).item()
                b = 0 if degree_j==0 else 1/(float(degree_j) + 1e-7)
                if A[j,i]==1:
                    y[i] = b + lambda_uv_t
                elif A[j,i]==1:
                    degree_j_bar = self.node_degree_global[j]
                    b_prime = 0 if degree_j_bar==0 else 1./(float(degree_j_bar) + 1e-7)
                    x = b_prime - b
                    y[i] = b + lambda_uv_t
                    w_idx = (y!=0) & (indices != int(i))
                    # w_idx[int(i)] = False
                    y[w_idx] = y[w_idx]-x
                y /= (torch.sum(y)+ 1e-7)
                self.S[j,:] = y
    
    def compute_cond_density(self, u, v, time_bar):
        N = self.num_nodes
        s_uv = self.Lambda_dict.new_zeros((2, N))

        # Lambda_sum는 expected return 같은 느낌인거 같은데
        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0) / len(self.Lambda_dict)
        
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]
        indices = []
        l_indices = []
        t_bar_min = torch.min(time_bar[[u, v]]).item()
        if t_bar_min < time_keys_min:
            start_ind_min = 0
            #노드의 이벤트가 dictionary 저장 전에 일어났따. 
        elif t_bar_min > time_keys_max:
            # 이벤트가 이 노드들에서는 발생하지 않았다.
            return s_uv
        else:
            start_ind_min = self.time_keys.index(int(t_bar_min))
            # 그 한참전에 일어난 노드부터 index시작
        
        # 각 노드와 다른 모든 노드 간의 가장 최근 이벤트 시간
        max_pairs = torch.max(torch.cat((time_bar[[u, v]].view(1, 2).expand(N, -1).t().contiguous().view(2 * N, 1),
                                         time_bar.repeat(2, 1)), dim=1), dim=1)[0].view(2, N).long().data.cpu().numpy()  # 2,N

        # compute cond density for all pairs of u and some i, then of v and some i

        for c, j in enumerate([u, v]):  # range(i + 1, N):
            for i in range(N):
                if i == j:
                    continue
                # most recent timestamp of either u or v
                t_bar = max_pairs[c, i]

                if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
                elif t_bar > time_keys_max:
                    continue  # it means t_bar is current event, so there is no history for this pair of nodes
                else:
                    # t_bar is somewhere in after time_keys_min
                    start_ind = self.time_keys.index(t_bar, start_ind_min)

                indices.append((c, i))
                l_indices.append(start_ind)

        indices = np.array(indices)
        l_indices = np.array(l_indices)
        s_uv[indices[:, 0], indices[:, 1]] = Lambda_sum[l_indices]

        return s_uv
