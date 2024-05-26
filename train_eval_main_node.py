import platform

import numpy as np
import sys
import os
import time
import copy
import pickle
import torch.nn as nn
from datetime import datetime

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import matplotlib.pyplot as plt

from torch import autograd

from earthquake_data_loader import EarthquakeDataset

from utils import *
from dyrep import DyRep
from dyrepHawkes import DyRepHawkes
from dyrepHawkes_re import DyRepHawkesRe
from dyrep_node_Hawkes import DyRepNode
from tqdm import tqdm
from collections import defaultdict

def get_return_time(data_set):
    reoccur_dict = {}
    for sources,timestamps_date, _, _ in data_set.all_events:
        if sources not in reoccur_dict:
            reoccur_dict[sources] = [timestamps_date]
        elif timestamps_date != reoccur_dict[sources][-1]:
            reoccur_dict[sources].append(timestamps_date)
    reoccur_time_hr = np.zeros(len(data_set.all_events))

    for idx, (sources,timestamps_date,significance,magnitudo) in enumerate(data_set.all_events):
        val = reoccur_dict[sources]
        if val.index(timestamps_date) < len(val)-1:
            reoccur_time = val[val.index(timestamps_date) +1] - timestamps_date
        else:  
            reoccur_time = data_set.END_DATE - timestamps_date
        reoccur_time_hr[idx] = reoccur_time
    
    return reoccur_time_hr

def mae_error(u, time_cur, expected_time, reoccur_dict, end_date):

    u, time_cur = u.data.cpu().numpy(), time_cur.data.cpu().numpy()
    batch_predict_time = []
    N = len(u)
    ae = 0
    for idx in range(N):
        val = reoccur_dict[u[idx]]
        td_pred_hour = expected_time[idx]
        if len(val) == 1 or time_cur[idx]==val[-1]:
            next_ts = end_date
        else:
            next_ts = val[val.index(time_cur[idx])+1]

        true_td = next_ts - time_cur[idx]
        td_true_hour = true_td
        # td_true_hour = round((true_td.days*24 + true_td.seconds/3600), 3)
        ae += abs(td_pred_hour-td_true_hour)
        batch_predict_time.append((td_pred_hour, td_true_hour))
    return ae, batch_predict_time

def MAE(expected_time_hour, batch_ts_true, t_cur):
    t_cur = t_cur.data.cpu().numpy()
    valid_idx = np.where(batch_ts_true != 0)
    t_cur_dt = np.array(list(t_cur[valid_idx]))
    batch_dt_true = np.array(list(batch_ts_true[valid_idx]))
    batch_time_true = batch_dt_true - t_cur_dt
    batch_time_hour_true = np.array(list(batch_time_true))
    expected_time_hour = np.array(expected_time_hour)[valid_idx]
    batch_ae = sum(abs(expected_time_hour-batch_time_hour_true))
    batch_res = list(zip(expected_time_hour, batch_time_hour_true))
    return batch_ae, batch_res
    
def test_all(model, return_time_hr, device,batch_size):
    model.eval()
    time_bar = copy.deepcopy(model.time_bar)
    loss = 0
    total_ae= 0
    aps, aucs = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            if batch_idx==0:
                model.time_bar = time_bar
            
            data[0] = data[0].float().to(device)
            data[1] = data[1].float().to(device)
            data[2] = data[2].double().to(device)
            data[3] = data[3].double()# no need of GPU
            batch_size = len(data[0])
            
            # 리턴값: 이벤트 노드의 람다, neg 노드의 평균, A_pred, surv, 예상시간
            lambda_event, average_neg, A_pred, Survival_term, pred_time = model(data) # data는 6*batch_size

            cond = A_pred * torch.exp(-Survival_term) # cond (100(batch_size)*100(node 수))
            
            loss += (-torch.sum(torch.log(lambda_event) + 1e-10) + torch.sum(average_neg).item())
        
            u = np.asarray(data[:6][0]).astype(int) # (batch_size) - event별 활성화 노드
            neg_u_all = np.delete(np.arange(train_set.N_nodes), u)
            neg_u = torch.tensor(rnd.choice(neg_u_all, size=batch_size, replace=len(neg_u_all) < batch_size), device=device)
            
            # for y_pred 계산 및 y_true 계산 - 다음 event일때, node별 지진이 날 확률
            pos_prob = cond[np.arange(batch_size), u] #(batch_size,1)
            neg_prob = cond[np.arange(batch_size), neg_u] #(batch_size,1) 샘플 하나만 뽑기에 문제가 있긴함
            y_pred = torch.cat([pos_prob, neg_prob], dim=0).cpu()
            y_true = torch.cat([torch.ones(pos_prob.size(0)), torch.zeros(neg_prob.size(0))], dim=0)
            
            ap = average_precision_score(y_true, y_pred)
            aps.append(ap)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            
            # total_ae 구하기
            return_time_real = return_time_hr[batch_idx*batch_size:((batch_idx+1)*batch_size)] #(batch_size,)
            return_time_pred = torch.stack(pred_time).cpu().numpy() # size = (batch_size,
            assert len(return_time_real) == len(return_time_pred)
            mae = abs(return_time_pred - return_time_real).sum()
            total_ae += mae
            
    return total_ae / len(test_set.all_events), loss / len(test_set.all_events), \
           float(torch.tensor(aps).nanmean()), float(torch.tensor(aucs).nanmean())


if __name__ == '__main__':

    ## 기본적인 입력 parameter 세팅
    
    
    parser = argparse.ArgumentParser(description='DyRep Model Training Parameters')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=100, help='test_batch size')
    parser.add_argument('--sample_num', type=int, default=5, help='sample_num')
    parser.add_argument('--neg_sample_num', type=int, default=20, help='neg_sample_num')
    
    
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden layer dimension in DyRep')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda or mps')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_step', type=str, default='20', help='number of epochs after which to reduce lr')
    parser.add_argument('--all_comms', type=bool, default=False, help='assume all of the links in Jodie as communication or not')
    parser.add_argument('--include_link_feat', type=bool, default=False, help='include link features or not')
    args = parser.parse_args()
    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    
    # argument 외 변수 설정
    total_losses = []
    total_losses_lambda, total_losses_surv = [], []
    test_MAR, test_HITS10, test_loss = [], [], []
    all_test_mae, all_test_loss = [], []
    all_test_ap, all_test_auc = [], []
    first_batch = []

    #데이터 불러오기
    train_set = EarthquakeDataset("train")
    test_set = EarthquakeDataset("test")
    A_initial = train_set.get_Adjacency() # csv 파일 통해서 구하기
    
    #환경 세팅
    np.random.seed(args.seed)
    rnd = np.random.RandomState(args.seed)
    #cPyTorch의 cuDNN 백엔드 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    initial_embeddings = np.random.randn(train_set.N_nodes, args.hidden_dim) # (100, hidden_dim)
    
    # 계산이 필요한 환경 세팅
    time_bar_initial = np.zeros((train_set.N_nodes, 1)) + train_set.FIRST_DATE # FIRST_DATE 형식의 (100,1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    
    test_reoccur_time_hr = get_return_time(test_set) #event별 t_bar 구하기 (event수, 1)
    tain_reoccur_time_hr = get_return_time(train_set)
    train_td_max = tain_reoccur_time_hr.max()

    # 모델 설정
    model = DyRepNode(num_nodes=train_set.N_nodes,
                  hidden_dim=args.hidden_dim,
                  random_state= rnd,
                  first_date=train_set.FIRST_DATE,
                  end_datetime=test_set.END_DATE,
                  num_neg_samples=args.neg_sample_num, # ****
                  num_time_samples=args.sample_num,
                  device=args.device,
                  all_comms=args.all_comms,
                  train_td_max=train_td_max
                  ).to(args.device)
    
    # learning parameter 설정
    params_main = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_main, lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    # Training Part
    for epoch in range(1, args.epochs + 1):
        # 기타 변수 세팅
        start = time.time()
        total_loss = 0
        total_loss_lambda, total_loss_surv = 0, 0
        node_degree_initial = []
        node_degree_initial.append(np.sum(A_initial, axis=0))
        
        time_bar = copy.deepcopy(time_bar_initial)
        train_loader.dataset.time_bar = time_bar
        test_loader.dataset.time_bar = time_bar

        # 모델 세팅
        model.train() # train 환경으로 model 설정
        model.reset_state(node_embeddings_initial=initial_embeddings,
                          A_initial=A_initial,
                          node_degree_initial=node_degree_initial,
                          time_bar=time_bar)

    
        # Batch_size 만큼의 event 한번에 계산
        for batch_idx, data_batch in enumerate(tqdm(train_loader)): #tqdm이 막대그래프 표시하는 역할
            
            optimizer.zero_grad()

            data_batch[0] = data_batch[0].float().to(args.device)
            data_batch[1] = data_batch[1].float().to(args.device)
            data_batch[2] = data_batch[2].double().to(args.device)
            data_batch[3] = data_batch[3].double()# no need of GPU
            
            # 단순히 log(pos)- neg 형태의 기울기 반영
            output = model(data_batch) # 모델 forward 돌리고 역전파 생성
            losses = [-torch.sum(torch.log(output[0]) + 1e-10), 10 * torch.sum(output[1])]
            loss = torch.sum(torch.stack(losses))/args.batch_size
            
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 100)
            optimizer.step()
            model.psi.data = torch.clamp(model.psi.data, 1e-1, 1e+3)
            
            time_iter = time.time() - start
            model.z = model.z.detach()
            model.S = model.S.detach()
            
            if batch_idx == 0:
                first_batch.append(loss)

            total_loss += loss*args.batch_size
            total_loss_lambda += losses[0]
            total_loss_surv += losses[1]
            scheduler.step()


        # loss 리스트에 추가
        total_loss = float(total_loss)/len(train_set.all_events)
        total_loss_lambda = float(total_loss_lambda)/len(train_set.all_events)
        total_loss_surv = float(total_loss_surv)/len(train_set.all_events)
        total_losses.append(total_loss)
        total_losses_lambda.append(total_loss_lambda)
        total_losses_surv.append(total_loss_surv)
        
        
        # test_all: for 문으로 똑같이 한번씩 돌리는 코드 들어있음
        test_mae, test_loss, test_ap, test_auc = test_all(model, test_reoccur_time_hr, args.device ,args.batch_size)
        
        #리스트에 정보 추가
        all_test_mae.append(test_mae)
        all_test_loss.append(test_loss)
        all_test_ap.append(test_ap)
        all_test_auc.append(test_auc)

        print("epoch {}/{}".format(epoch+1, args.epochs + 1))
        print("Train: loss {:.5f}, loss_lambda {:.5f}, loss_surv {:.5f}, time per batch {:.5f}".format(
            total_loss, total_loss_lambda, total_loss_surv, time_iter/float(batch_idx+1)))

        print('Test: loss={:.5f}, time prediction MAE {:.5f}, ap {:.5f}, auc{:.5f}\n\n'.format(
            test_loss, test_mae, test_ap, test_auc))
    
    # visualization codes

    # 무시해도 됨 - numpy 관련 에러 수정(지우지는 마삼)
    detached_tensors = [tensor.detach() for tensor in first_batch]
    detached_arrays = [tensor.numpy() for tensor in detached_tensors]
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses), 'k', label='total loss')
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses_lambda), 'r', label='loss events')
    plt.plot(np.arange(1, args.epochs + 1), np.array(total_losses_surv), 'b', label='loss nonevents')
    plt.legend()
    plt.title("DyRep, training loss")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, args.epochs + 1), np.array(detached_arrays), 'r')
    plt.title("DyRep, loss for the first batch for each epoch")
    fig.savefig('dyrepHawkes_social_train.png')

    fig = plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_loss), 'k', label='total loss')
    plt.title("DyRep, test loss")
    plt.subplot(1, 4, 2)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_ap), 'r')
    plt.title("DyRep, test ap")
    plt.subplot(1, 4, 3)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_mae), 'r')
    plt.title("DyRep, test mae")
    
    plt.subplot(1, 4, 4)
    plt.plot(np.arange(1, args.epochs + 1), np.array(all_test_auc), 'r')
    plt.title("DyRep, test auc")


    fig.savefig('dyrepHawkes_social_test.png')

    # 두 날짜 사이의 차이 계산
    # delta = datetime.fromtimestamp((test_set.END_DATE-test_set.FIRST_DATE)/1000) - datetime(1970, 1, 1)
    
    # 모델 정보
    # print(model)
    # print('number of training parameters: %d' %
    #       np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    # dt = datetime.now()
    # print('start time:', dt)
    # experiment_ID = '%s_%06d' % (platform.node(), dt.microsecond)
    # print('experiment_ID: ', experiment_ID)