import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from data_loader import EventsDataset
import networkx as nx
# import matplotlib.pyplot as plt
import csv
from collections import Counter

class EarthquakeDataset(EventsDataset):

    def __init__(self, split, dataset_name=None, data_dir=None, link_feat=False):
        super(EarthquakeDataset, self).__init__()

        self.rnd = np.random.RandomState(1111)
        # self.dataset_name = dataset_name
        self.link_feat = link_feat

        graph_df = pd.read_csv('result.csv')
        graph_df = graph_df.sort_values('time')
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
        timestamps_date = np.array(list(map(lambda x: datetime.fromtimestamp(int(x/1000), tz=None), timestamps)))
        
        
        test_time = np.quantile(graph_df.time, 0.9)
        train_mask = timestamps<=test_time
        test_mask = timestamps>test_time
        

        # mini data set 생성기
        # mini_time = np.quantile(graph_df.time, 0.082)
        # mini_test_time = np.quantile(graph_df.time, 0.08)
        # train_mask = (timestamps<=mini_test_time) & (timestamps<=mini_time)
        # test_mask = (timestamps>mini_test_time) & (timestamps<=mini_time)


        # all_events = list(zip(sources,timestamps_date,significance,magnitudo ))
        all_events = list(zip(sources,timestamps,significance,magnitudo ))

        if split == 'train':
            self.all_events = np.array(all_events)[train_mask].tolist()
        elif split == 'test':
            self.all_events = np.array(all_events)[test_mask].tolist()
        else:
            raise ValueError('invalid split', split)

        # self.FIRST_DATE = datetime.fromtimestamp(0)
        self.FIRST_DATE = timestamps[0]
        self.END_DATE = timestamps[-1]
        series = set(sources)
        self.N_nodes = len(series)

        self.time_bar = [self.FIRST_DATE for i in range(self.N_nodes)]
        
        self.n_events = len(self.all_events)

        self.A_initial = self.cluster_to_adj()

    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial

    def cluster_to_adj(self, cluster_path="cluster_assign.csv", adj_path="adj.csv"):
        clusters = pd.read_csv(cluster_path,header=None).to_numpy()
        n_cluster = len(clusters)
        # print(n_cluster)
        adj = np.zeros((n_cluster, n_cluster))
        plate_element = list()
        for plate in range(len(clusters[0])):
            plate_element.append(list())
            for cluster_index, in_plate in enumerate(clusters.T[plate]):
                if bool(in_plate):
                    plate_element[-1].append(cluster_index)
        for plate_cluster in plate_element:
            for i in range(len(plate_cluster)):
                for j in range(i+1, len(plate_cluster)):
                    adj[plate_cluster[i]][plate_cluster[j]] = 1
                    adj[plate_cluster[j]][plate_cluster[i]] = 1
        header = [str(i) for i in range(n_cluster)]
        pd.DataFrame(adj).to_csv(adj_path,  header = header)
        # G = nx.Graph()
        # print(adj)
        # for i in range(len(adj)):
        #     for j in range(len(adj[i])):
        #         if adj[i][j] == 1:
        #             G.add_edge(i+1, j+1)
        # nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=16, font_color='black')
        # plt.show()  
        return adj
