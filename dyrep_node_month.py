import numpy as np
import torch
from datetime import datetime, timedelta
from torch.nn import Linear, ModuleList, Parameter
import bisect
import torch.nn.functional as F

class DyRepNode_month(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, random_state, first_date, end_datetime, num_neg_samples= 20, num_time_samples = 5,
                 device='cpu', all_comms=False, train_td_max=None):
        super(DyRepNode_month, self).__init__()

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
        
        self.prob_1 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.prob_2 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.prob_3 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.prob_4 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.prob_5 = Linear(in_features=hidden_dim, out_features=1)

        self.time_1 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.time_2 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.time_3 = Linear(in_features=hidden_dim, out_features=1)

        self.mae = torch.nn.L1Loss()
        self.cross = torch.nn.CrossEntropyLoss()
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
        A = torch.from_numpy(A_initial).float().to(self.device) ## (num_node, num_node)로 설정해야함 
        self.register_buffer('z', z)
        self.register_buffer('A', A)
        self.node_degree_global = node_degree_initial ## train에는 (n_assoc_type,num_node로) -> (num_node)로 설정해야함
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

    def forward(self, data, change_month_prev, changing_month):
        u, time_delta, time_bar, time_cur, significance, magnitudo = data[:6]
        # Count the number of parameters that were updated
        u = u[change_month_prev:changing_month]
        time_delta = time_delta[change_month_prev:changing_month]
        time_bar = time_bar[change_month_prev:changing_month]
        time_cur = time_cur[change_month_prev:changing_month]
        significance = significance[change_month_prev:changing_month]
        magnitudo = magnitudo[change_month_prev:changing_month]


        # 기본 세팅
        batch_size = len(u)
        u_all = u.data.cpu().numpy()
        ts_diff = []
        z_prev = self.z
        prob_list = torch.zeros((batch_size, self.num_nodes))
        time_list = torch.zeros((batch_size))

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
                time_delta_it[i] = time_cur_it - time_bar_it[k] # 현재 시간과 time_bar 차이
            
            time_delta_it = torch.Tensor(time_delta_it) # impact_nodes의 t_curr - t_bar
            # event_node의 delta 계속 저장
            ts_diff.append(time_cur_it - time_bar_it[impact_nodes[0]]) # (1,) 저장

            prob = self.compute_prob(z_prev)
            time = self.compute_time(z_prev[u_event])

            
            # z_new(node_num,1) - u_event랑 u_neigh에 대한 z만 수정
            z_new = self.update_node_embedding_without_attention(z_prev, u_event, u_neigh, time_delta_it)
            assert torch.sum(torch.isnan(z_new)) == 0, (torch.sum(torch.isnan(z_new)), z_new, it)
            prob_list[it] = prob.squeeze(1)
            time_list[it] = time
            
        return prob_list, time_list
        

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
    
    def compute_prob(self, x):
        prob = np.zeros((self.num_nodes))
        x = self.prob_1(x)
        x = F.leaky_relu(x) 
        x = self.prob_2(x)
        x = F.leaky_relu(x) 
        x = self.prob_3(x)
        x = F.leaky_relu(x)
        x = self.prob_4(x)
        x = F.leaky_relu(x)
        x = self.prob_5(x)
        x = F.sigmoid(x) 
        
        return x
    
    def compute_time(self, x):
        
        x = self.time_1(x)
        x = F.leaky_relu(x)
        x = self.time_2(x)
        x = F.leaky_relu(x)
        x = self.time_3(x)
        return x
