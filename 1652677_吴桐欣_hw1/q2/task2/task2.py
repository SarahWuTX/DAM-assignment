import copy
import csv
import time

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from DensityPeaks import DensityPeaks
from task2_prepare import get_dataset


class AlgorithmTry:
    def __init__(self, dataset, origin):
        self.dimension = len(dataset[0])
        self.dataset = dataset
        self.origin = origin
        self.labels = []
        self.center_index = []
        self.cluster_count = 0
        self.assess = {}
        self.sc = 0
        self.ch = 0

    def init(self):
        self.labels = []
        self.center_index = []
        self.cluster_count = 0
        self.assess = {}
        self.sc = 0
        self.ch = 0

    def print_result_assess(self, algorithm='', **kwargs):
        try:
            self.assess['silhouette_score'] = metrics.silhouette_score(self.origin, self.labels)
            self.assess['calinski_harabaz_score'] = metrics.calinski_harabaz_score(self.origin, self.labels)
            self.sc = self.assess['silhouette_score']
            self.ch = self.assess['calinski_harabaz_score']
        except ValueError:
            print('Only one label!')

        print(algorithm.upper())
        for k, v in kwargs.items():
            print(k, ':', v)
        print('clusters :', self.cluster_count)
        print('*'.ljust(50, '-') + '*')
        for k, v in self.assess.items():
            print('|', format(k, '<22'), '|', format(v, '<22'), '|')
        print('*'.ljust(50, '-') + '*\n')

    def print_result(self):
        # number of cluster & number of points in each cluster
        clusters = []
        for label in np.unique(self.labels):
            cluster = {
                'label': label,
                'size': 0,
                'dimension': [0]*self.dimension,
                'feature': []
            }
            clusters.append(cluster)
        for cluster in clusters:
            for i, data in enumerate(self.dataset):
                if self.labels[i] == cluster['label']:
                    cluster['size'] += 1
                    for d in range(len(data)):
                        cluster['dimension'][d] += data[d]
            for d, num in enumerate(cluster['dimension']):
                cluster['dimension'][d] = num/cluster['size']
                cluster['feature'].append((d, cluster['dimension'][d]))
            cluster['feature'].sort(key=lambda x: x[1], reverse=True)
        for index, cluster in enumerate(clusters):
            print('cluster:', index, 'size:', cluster['size'])
            print(cluster['feature'])
        print()

    def density_peak(self, dc):
        task = DensityPeaks(dc, self.dimension, copy.deepcopy(self.dataset))
        task.set_peak_count(3)
        task.run()
        self.cluster_count = task.peak_count
        self.labels = task.labels

    def k_means(self, n_clusters):
        task = KMeans(n_clusters)
        task.fit(self.dataset)
        self.cluster_count = n_clusters
        self.labels = task.labels_

    def dbscan(self, eps, min_samples):
        task = DBSCAN(eps, min_samples)
        task.fit(self.dataset)
        self.labels = task.labels_
        self.cluster_count = len(np.unique(self.labels))-1

    def em(self, n_components):
        task = GaussianMixture(n_components, random_state=0)
        task.fit(self.dataset)
        self.labels = task.predict(self.dataset)
        self.cluster_count = np.unique(self.labels)

    def spectral(self, n_clusters):
        task = SpectralClustering(3, affinity='nearest_neighbors', n_neighbors=n_clusters)
        task.fit_predict(self.dataset)
        self.labels = task.labels_
        self.cluster_count = np.unique(task.labels_)

    def fclusterdata(self, t):
        self.labels = fclusterdata(self.dataset, t, criterion='maxclust')
        self.cluster_count = np.unique(self.labels)


def set_args():
    dataset = get_dataset('./data/user_data_26.json').get('data')
    origin = get_dataset('./data/user_data.json').get('data')
    task2 = AlgorithmTry(dataset, origin)
    # task2.print_k_distance_graph()
    sc = []
    ch = []
    x = []
    time1 = time.time()
    ''' density peak '''
    # for i in range(1, 10):
    # for i in range(1, 2):
    #     task2.density_peak(0.1*i)
    #     task2.print_result()
    #     task2.print_result_assess('density peak', dc=0.1*i)
    #     x.append(0.1*i)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)
    ''' k means '''
    # for i in range(2, 9):
    # for i in range(3, 4):
    #     task2.k_means(i)
    #     task2.print_result()
    #     task2.print_result_assess('k means', k=i)
    #     x.append(i)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)
    ''' db scan '''
    # for i in range(20):
    # for i in range(10, 11):
    #     task2.dbscan(3+i*0.1, 38)
    #     task2.print_result()
    #     task2.print_result_assess('db scan', eps=3+i*0.1, min_samples=38)
    #     x.append(3+i*0.1)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)
    ''' em '''
    # for i in range(2, 9):
    # for i in range(3, 4):
    #     task2.em(i)
    #     task2.print_result()
    #     task2.print_result_assess('em', n=i)
    #     x.append(i)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)

    ''' spectral '''
    # for i in range(10):
    # # for i in range(3, 4):
    #     task2.spectral(i+25)
    #     task2.print_result()
    #     task2.print_result_assess('spectral', n=i+25)
    #     x.append(i+25)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)

    ''' hierarchy '''
    # for i in range(2, 10):
    # for i in range(3, 4):
    #     task2.fclusterdata(i)
    #     task2.print_result()
    #     task2.print_result_assess('hierarchy', n=i)
    #     x.append(i)
    #     sc.append(task2.sc)
    #     ch.append(task2.ch)

    print('TIME : {0}'.format(time.time() - time1))

    plt.subplot(121)
    plt.plot(x, sc, label='sc', color='blue')
    plt.legend(loc='upper right')
    plt.xlabel('n')

    plt.subplot(122)
    plt.plot(x, ch, label='ch', color='green')
    plt.legend(loc='upper right')
    plt.xlabel('n')

    plt.show()


def save_csv(filepath):
    dataset = get_dataset('./data/user_data_26.json').get('data')
    origin = get_dataset('./data/user_data.json').get('data')
    user = []
    for key in get_dataset('./data/user_business_test.json').keys():
        user.append(key)
    columns = [user]

    task2 = AlgorithmTry(dataset, origin)
    for i in range(6):
        task2.init()

        if i == 0:
            task2.density_peak(1)
        elif i == 1:
            task2.k_means(3)
        elif i == 2:
            task2.dbscan(4, 38)
        elif i == 3:
            task2.fclusterdata(3)
        elif i == 4:
            task2.spectral(32)
        elif i == 5:
            task2.em(3)

        columns.append(task2.labels)

    rows = np.array(columns).T.tolist()
    with open(filepath, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(['uid', 'sci2014_label', 'kmeans_label', 'dbscan_label', 'hierarchical_label', 'spectral_label', 'em_label'])
        for i in range(len(user)):
            writer.writerow(rows[i])


if __name__ == '__main__':
    # set_args()
    save_csv('task2.csv')

