import numpy as np
from sklearn import metrics

from DensityPeaks import DensityPeaks


def get_dataset(input_filepath):
    dataset = []
    with open(input_filepath, 'r') as f:
        for line in f.readlines():
            data = []
            for x in line.split(','):
                data.append(float(x))
            dataset.append(data)
    return dataset


def main():
    dataset = get_dataset('./data/Aggregation.txt')
    for i in range(2):
        task1 = DensityPeaks(1.79+0.01*i, 2, dataset.copy())
        task1.run()
        labels = np.array(task1.dataset).T[task1.dimension + 4]
        int_labels = []
        for label in labels:
            int_labels.append(int(label))
        try:
            ds = np.array(task1.dataset)
            sc = metrics.silhouette_score(ds[:, [0, 1]], labels)
            ch = metrics.calinski_harabaz_score(ds[:, [0, 1]], labels)
            print('clusters:', task1.peak_count, '   sc:', sc, '   ch:', ch)
        except ValueError:
            pass
        task1.print(out='file')

    # task1.save_csv('task1.csv')


if __name__ == '__main__':
    main()
