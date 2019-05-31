import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import csv
from sklearn.metrics import pairwise_distances


class DensityPeaks:
    def __init__(self, dc, dimension, dataset=None, peak_count=0):
        # 0-x, 1-y, 2-p, 3-delta, 4-gamma, 5-neighbour, 6-label
        self.dc = dc
        self.dataset = []
        self.distance_matrix = []
        self.dimension = dimension
        self.data_size = 0
        if dataset is not None:
            self.set_dataset(dataset, dimension)
        self.peaks = []
        self.peak_index = []
        self.peak_count = peak_count
        self.__gamma__ = []
        self.__n__ = []
        self.labels = []

    def set_dataset(self, dataset, dimension):
        self.data_size = len(dataset)
        self.dimension = dimension
        self.dataset = []
        for i in range(self.data_size):
            self.dataset.append(dataset[i][:self.dimension])
        self.distance_matrix = pairwise_distances(dataset)

    def set_peak_count(self, peak_count):
        self.peak_count = peak_count

    def set_peaks(self, peaks):
        self.peaks = peaks

    def set_dc(self, dc):
        self.dc = dc

    def run(self):
        self.__execute_algorithm__()

    def distance_of(self, point1, point2):
        return 0

    def __execute_algorithm__(self):
        print('calculate p')
        total = 0
        for i in range(self.data_size):
            p = 0
            for j in range(self.data_size):
                # distance = self.distance_of(self.dataset[i], self.dataset[j])
                distance = self.distance_matrix[i][j]
                if distance < self.dc:
                    p += 1
            self.dataset[i].append(p - 1)
            total += p-1
        print('dc: {0}, average neighbours: {1}'.format(self.dc, total/self.data_size))
        origin = copy.deepcopy(self.dataset)
        for i in range(self.data_size):
            origin[i].append(i)

        # sort
        self.dataset.sort(key=lambda x: x[self.dimension], reverse=True)
        origin.sort(key=lambda x: x[self.dimension], reverse=True)
        self.distance_matrix = pairwise_distances(np.delete(self.dataset, -1, axis=1))

        print('calculate delta、gamma、neighbour')
        for i, point_i in enumerate(self.dataset):
            if i == 0:
                continue
            # distance = self.distance_of(self.dataset[0], self.dataset[i])
            delta = -1
            neighbour = 0
            for j, point_j in enumerate(self.dataset):
                if j < i:
                    # distance = self.distance_of(point_i, point_j)
                    distance = self.distance_matrix[i][j]
                    if delta == -1 or distance < delta:
                        delta = distance
                        neighbour = j
                else:
                    break
            point_i.append(delta)
            point_i.append(point_i[self.dimension] * delta)
            point_i.append(neighbour)
        self.dataset[0].append(max(self.distance_matrix[0]))
        self.dataset[0].append(self.dataset[0][self.dimension] * self.dataset[0][self.dimension+1])
        self.dataset[0].append(0)

        # clustering
        self.__find_peaks__()
        self.__clustering__()

        # label
        self.labels = [0] * self.data_size
        for i in range(self.data_size):
            self.labels[origin[i][-1]] = self.dataset[i][self.dimension + 4]

    def __find_peaks__(self):
        print('__find_peaks__')
        gamma = np.array(self.dataset).T[self.dimension+2].tolist()
        gamma.sort(reverse=True)
        for i in range(self.data_size):
            self.__n__.append(i)
            self.__gamma__.append(gamma[i])

        if self.peak_count == 0:
            decision_array = np.diff(np.diff(gamma)).tolist()
            max_diff = max(decision_array)
            first = 0
            if max_diff == decision_array[0]:
                decision_array.pop(0)
                max_diff = max(decision_array)
                first = 1
            self.peak_count = decision_array.index(max_diff) + 1 + first

        gamma_limit = gamma[self.peak_count-1]

        if len(self.peaks) == 0:
            i = 0
            for point in self.dataset:
                if point[self.dimension+2] >= gamma_limit:
                    point.append(i)
                    self.peaks.append(point)
                    i += 1
                if len(self.peaks) >= self.peak_count:
                    break

    def __clustering__(self):
        print('__clustering__')
        for i in range(self.data_size):
            if len(self.dataset[i]) >= self.dimension+5:
                self.peak_index.append(i)
                continue
            neighbour = self.dataset[i][self.dimension+3]
            if len(self.dataset[neighbour]) >= self.dimension+5:
                self.dataset[i].append(self.dataset[neighbour][self.dimension+4])
            else:
                self.dataset[i].append(-1)

    def print_peak_decision_graph(self):
        plt.title = 'find peaks (current peaks count:{0})'.format(self.peak_count)
        plt.scatter(self.__n__, self.__gamma__, self.__gamma__, self.__gamma__, alpha=0.5)
        plt.title = 'decision graph'
        plt.show()

    def print_decision_graph(self):
        ds = np.array(self.dataset)
        # plt.scatter(ds[:, self.dimension], ds[:, self.dimension+1], ds[:, self.dimension+2], ds[:, self.dimension+2], alpha=0.5)
        plt.scatter(ds[:, self.dimension], ds[:, self.dimension + 1], c=ds[:, self.dimension + 2], alpha=0.5)
        plt.ylabel('delta')
        plt.xlabel('p  (dc={0})'.format(self.dc))
        plt.title = 'decision graph'
        plt.show()

    '''
        Can only print data with 2 dimensions.
    '''
    def print_point_distribution(self):
        ds = np.array(self.dataset)
        plt.scatter(ds[:, 0], ds[:, 1], 10, alpha=0.5)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title = 'point distribution'
        plt.show()

    def print_cluster_result(self):
        peaks = np.asarray(self.peaks)
        ds = np.asarray(self.dataset)
        plt.scatter(ds[:, 0], ds[:, 1], 10, ds[:, 6], cmap=cm.cmapname)
        plt.scatter(peaks[:, 0], peaks[:, 1], 20, c='red', alpha=0.5)
        plt.title = 'cluster result'
        plt.show()

    def print(self, out='screen'):
        plt.figure(figsize=(10, 10))
        ds = np.array(self.dataset)

        plt.subplot(221)
        plt.scatter(ds[:, 0], ds[:, 1], 10)
        plt.ylabel('y')
        plt.xlabel('x')

        plt.subplot(222)
        plt.scatter(ds[:, 2], ds[:, 3], ds[:, 4], ds[:, 4], alpha=0.5)
        plt.ylabel('delta')
        plt.xlabel('p  (dc={0})'.format(self.dc))

        plt.subplot(223)
        plt.scatter(self.__n__, self.__gamma__, self.__gamma__, self.__gamma__, alpha=0.5)
        plt.xlabel('peaks count:{0}'.format(self.peak_count))

        plt.subplot(224)
        peaks = np.asarray(self.peaks)
        plt.scatter(ds[:, 0], ds[:, 1], 10, ds[:, 6], cmap=cm.cmapname)
        plt.scatter(peaks[:, 0], peaks[:, 1], 20, c='red', alpha=0.5)

        if out == 'screen':
            plt.show()
        if out == 'file':
            plt.savefig('fig_{0}.png'.format(self.dc))

    def save_csv(self, ouput_filepath='result.csv'):
        with open(ouput_filepath, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'label'])
            for point in self.dataset:
                row = []
                for i in range(self.dimension):
                    row.append(point[i])
                row.append(point[self.dimension+4])
                writer.writerow(row)



