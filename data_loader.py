import numpy as np
import datetime
import torch
import torch.utils
from datetime import datetime, timezone
from torch.utils.data import Dataset


class EventsDataset(torch.utils.data.Dataset):
    '''
    Base class for event datasets
    '''
    def __init__(self, TZ=None):
        self.TZ = TZ  # timezone.utc

    def get_Adjacency(self):
        return None

    def __len__(self):
        return self.n_events

    def __getitem__(self, index):

        tpl = self.all_events[index]
        
        # sources,timestamps_date,significance,magnitudo
        if self.link_feat:
            u, v, rel, time_cur, link_feature = tpl
        else:
            u, time_cur,significance,magnitudo = tpl
        # self.start_time = min(self.start_time, min(time_cur))
        # Compute time delta in seconds (t_p - \bar{t}_p_j) that will be fed to W_t
        time_delta = np.zeros((int(1+sum(self.A_initial[int(u)])), 4))  # two nodes x 4 values
        # most recent previous time for all nodes
        time_bar = self.time_bar.copy()
        # assert u != v, (tpl, rel)
        impact_nodes = [u]
        for i, k in enumerate(self.A_initial[int(u)]):
            # print(str(i)+" "+str(k))
            if k == 1:
                impact_nodes.append(i)
                
        for i, k in enumerate(impact_nodes):
            # t = datetime.fromtimestamp(int(self.time_bar[int(k)]), tz=self.TZ)
            t = self.time_bar[int(k)]
            # if t.toordinal() >= self.FIRST_DATE.toordinal():  # assume no events before FIRST_DATE
            if t >= self.FIRST_DATE:  # assume no events before FIRST_DATE
                td = time_cur - t
                time_delta[i] = np.array([td],np.float64)
            else:
                raise ValueError('unexpected result', t, self.FIRST_DATE)
            
            self.time_bar[int(k)] = time_cur  # last time stamp for nodes u and v
            
            # self.time_bar[int(k)] = time_cur.timestamp()  # last time stamp for nodes u and v
            
        # sanity checks
        # assert np.float64(time_cur.timestamp()) == time_cur.timestamp(), (
        # np.float64(time_cur.timestamp()), time_cur.timestamp())
        time_cur = np.float64(time_cur)
        time_bar = np.array(list(time_bar))
        time_bar = time_bar.astype(np.float64)
        time_cur = torch.from_numpy(np.array([time_cur])).double()
        assert time_bar.max() <= time_cur, (time_bar.max(), time_cur)
        
        # if self.link_feat:
            # return u, v, time_delta_uv, k, time_bar, time_cur, link_feature
        # else:
        #return impact_nodes, time_delta, time_bar, time_cur,significance,magnitudo
        return impact_nodes[0], time_delta[0], time_bar, time_cur,significance,magnitudo