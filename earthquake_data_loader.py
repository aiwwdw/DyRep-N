import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from data_loader import EventsDataset


class EarthquakeDataset(EventsDataset):

    def __init__(self, split, dataset_name, data_dir=None, link_feat=False):
        super(EarthquakeDataset, self).__init__()

        self.rnd = np.random.RandomState(1111)
        self.dataset_name = dataset_name

        self.link_feat = link_feat

        graph_df = pd.read_csv('./result.csv')
        graph_df = graph_df.sort_values('time')
        test_time = np.quantile(graph_df.ts, 0.90)
        sources = graph_df.cluster.values
        significance = graph_df.significance.values
        magnitudo = graph_df.magnitudo.values

        # if not all_comms:
        #     visited = set()
        #     for idx, (source, des) in enumerate(zip(sources, destinations)):
        #         if (source,des) not in visited:
        #             event_type[idx]=0
        #             visited.add((source,des))

        timestamps = graph_df.time.values
        # milisecond 맞는지 확인할것
        timestamps_date = np.array(list(map(lambda x: datetime.fromtimestamp(int(x / 1000), tz=None), timestamps)))

        train_mask = timestamps<=test_time
        test_mask = timestamps>test_time

        all_events = list(zip(sources,timestamps_date,significance,magnitudo ))

        if split == 'train':
            self.all_events = np.array(all_events)[train_mask].tolist()
        elif split == 'test':
            self.all_events = np.array(all_events)[test_mask].tolist()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime.fromtimestamp(0)
        self.END_DATE = timestamps_date[-1]

        self.N_nodes = max(sources.max())

        self.n_events = len(self.all_events)

        self.A_initial = self.cluster_to_adj()

        # random_source = self.rnd.choice(np.unique(sources), size=500, replace=False)
        # random_des =self.rnd.choice(np.unique(destinations), size=500, replace=False)
        #
        # for i, j  in zip(random_source, random_des):
        #     self.A_initial[i,j] = 1
        #     self.A_initial[j,i] = 1

        print('\nA_initial', np.sum(self.A_initial))


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial

    def cluster_to_adj(self, cluster_path="cluster_assign.csv", adj_path="adj.csv"):
        df = pd.read_csv(cluster_path)
        n_cluster = len(df['cluster'])
        adj = np.zeros((n_cluster, n_cluster))
        plate_element = list()
        for plate in df.columns.values[1:]:
            plate_element.append(list())
            for cluster_index, in_plate in enumerate(df[plate]):
                if bool(in_plate):
                    plate_element[-1].append(cluster_index)
        for plate_cluster in plate_element:
            for i in range(len(plate_cluster)):
                for j in range(i+1, len(plate_cluster)):
                    adj[plate_cluster[i]][plate_cluster[j]] = 1
                    adj[plate_cluster[j]][plate_cluster[i]] = 1
        header = [str(i) for i in range(n_cluster)]
        pd.DataFrame(adj).to_csv(adj_path,  header = header)
        return adj
