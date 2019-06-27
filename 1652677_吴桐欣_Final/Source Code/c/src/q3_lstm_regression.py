import utm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow import keras

import utility


def group(dataset):
    groups = []
    g = []
    traj_id = dataset[0][0]
    for data in dataset:
        if traj_id != data[0]:
            traj_id = data[0]
            groups.append(g)
            g = []
        g.append(data[3:])
    return groups


def pre_processing_q3():
    train_set = utility.get_train_set()
    train_set_group = group(train_set)

    for train_set in train_set_group:
        feature_set = []
        label_set = []
        for data in train_set:
            feature_set.append(np.asarray(data[:-2]))
            label_set.append(np.asarray(data[-2:]))
        # feature_set = normalize(feature_set, axis=0, norm='max')
        # train_features, test_features, train_labels, test_labels = \
        yield train_test_split(feature_set, label_set, test_size=0.2)


def build_model_q3(input_shape):
    model = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(64, input_shape=input_shape, kernel_initializer='normal'),
        keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer='normal'),
        keras.layers.Dense(2, activation=tf.nn.relu, kernel_initializer='normal')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer='adadelta', metrics=['mae', 'acc'])

    return model


def main():
    predictions = []
    test_group = group(utility.get_test_set())
    model = build_model_q3((1, 44))
    p = []
    truth = []

    # dataset: [0]train_features, [1]test_features, [2]train_labels, [3]test_labels
    for i, dataset in enumerate(pre_processing_q3()):
        if i >= 69:
            break
        dataset[0] = np.asarray(dataset[0]).reshape(len(dataset[0]), 1, 44)
        dataset[2] = utility.utm_labels_to_gps(dataset[2])
        dataset[2] = utility.transform_gps_labels(dataset[2])
        dataset[2] = np.asarray(dataset[2]).reshape(len(dataset[2]), 2)
        dataset[1] = np.asarray(dataset[1]).reshape(len(dataset[1]), 1, 44)
        dataset[3] = utility.utm_labels_to_gps(dataset[3])
        dataset[3] = utility.transform_gps_labels(dataset[3])
        dataset[3] = np.asarray(dataset[3]).reshape(len(dataset[3]), 2)

        model.fit(dataset[0], dataset[2], epochs=50)
        test_pred = model.predict(dataset[1])
        print(test_pred)
        # gps = utility.utm_labels_to_gps(list(test_pred))
        p += list(test_pred)
        # print(dataset[3])
        truth += list(dataset[3])

        # test_group[i] = normalize(test_group[i], axis=0, norm='max')
        preds = model.predict(np.asarray(test_group[i]).reshape(len(test_group[i]), 1, 44))
        for pred in preds:
            predictions.append([pred[0], pred[1]])

    model.summary()
    p = utility.transform_gps_labels(p, True)
    truth = utility.transform_gps_labels(truth, True)
    print(p)
    print(truth)
    utility.get_metrics(p, truth)
    print(predictions)
    predictions = utility.transform_gps_labels(predictions, True)
    print(predictions)
    # utility.generate_pred_csv(3, predictions)
    # utility.visualize(3, predictions)


if __name__ == '__main__':
    main()
