import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from tensorflow import keras
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.optimizers import SGD

import utility

left_lat = 328770
left_lon = 3462224
right_lat = left_lat + 165 * 10  # > 330420
right_lon = left_lon + 127 * 10  # > 3463487
SHAPE = (5, 6, 1)

"""
utm.from_latlon(51.2, 7.5) 
>>> (395201.3103811303, 5673135.241182375, 32, 'U') UTM坐标
utm.to_latlon(340000, 5710000, 32, 'U')
>>> (51.51852098408468, 6.693872395145327) 经纬度
"""


def pre_processing_q2():
    train_set = utility.get_train_set()
    feature_set = []
    label_set = []
    for i in range(len(train_set)):
        feature_set.append(np.asarray(train_set[i][:-2]))
        label_set.append(np.asarray(train_set[i][-2:]))

    feature_set = process_features(feature_set)
    label_set = process_labels(label_set)
    return feature_set, label_set


def process_features(features):
    new_features = []
    for feature in features:
        new_feature = []
        respective = feature[5:]
        for i in range(0, 42, 7):
            new_feature.append(respective[i + 2:i + 7])
        new_feature = np.asarray(new_feature)
        new_feature = np.reshape(new_feature, (SHAPE[1], SHAPE[0])).T
        new_feature = np.reshape(new_feature, SHAPE)
        new_features.append(new_feature)
    return np.asarray(new_features)


def process_labels(labels):
    x_labels = []
    y_labels = []
    for label in labels:
        x, y = utility.utm_to_gird(longitude=label[0], latitude=label[1])
        x_labels.append(x)
        y_labels.append(y)
    return [np.asarray(x_labels), np.asarray(y_labels)]


def process_predictions(predictions):
    predictions = list(predictions)
    new_pred = []
    for pred in predictions:
        lon = int(round(pred[0]))
        lat = int(round(pred[1]))
        # print(lon, lat)
        location = utility.grid_to_gps(lon, lat)
        new_pred.append([location[0], location[1]])
    return new_pred


def group(features, split=False):
    groups = []
    g = []
    traj_id = features[0][0]
    for feature in features:
        if feature[0] != traj_id:
            traj_id = feature[0]
            groups.append(g)
            g = []
        g.append(feature[1:])

    if not split:
        return groups

    result = []
    for g in groups:
        features = []
        labels = []
        for data in g:
            features.append(data[2:])
            labels.append(data[:2])
        result.append(train_test_split(features, labels, test_size=0.2))
    return result


def build_cnn_model(input_shape):
    # layers
    input_layer = keras.layers.Input(shape=input_shape)
    # layer = keras.layers.BatchNormalization()(input_layer)
    layer = keras.layers.Conv2D(64, kernel_size=2, activation=tf.nn.tanh,
                                padding='same')(input_layer)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 1), padding='same')(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.1)(layer)
    layer = keras.layers.Conv2D(16, kernel_size=2, activation=tf.nn.relu,
                                padding='same')(layer)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 1), padding='same')(layer)
    layer = keras.layers.BatchNormalization()(layer)

    layer = keras.layers.Flatten()(layer)
    initializer = TruncatedNormal(mean=0.0, stddev=0.05, seed=3)
    layer = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer='uniform')(layer)
    layer = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer='uniform')(layer)

    lat = keras.layers.Dense(166, activation='softmax', name='lat')(layer)
    lon = keras.layers.Dense(128, activation='softmax', name='lon')(layer)


    optimizer = tf.keras.optimizers.RMSprop()
    loss = {
        'lat': 'sparse_categorical_crossentropy',
        'lon': 'sparse_categorical_crossentropy'
    }

    # build and compile
    model = keras.Model(inputs=input_layer, outputs=[lon, lat])
    model.compile(loss=loss, optimizer='adam',
                  metrics=['acc'])  # metrics=['mean_absolute_error', 'mean_squared_error']
    model.summary()
    return model


def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=input_shape),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.relu)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mae', 'acc'])

    model.summary()
    return model


def phase_1():
    train_features, train_labels = pre_processing_q2()
    cnn_model = build_cnn_model(SHAPE)
    cnn_model.fit(train_features, train_labels, epochs=120)
    lstm_train_features = cnn_model.predict(train_features)
    origin = utility.get_train_set()
    with open('out.csv', 'w+') as file:
        writer = csv.writer(file)
        for i, feature_lon in enumerate(lstm_train_features[0]):
            if int(origin[i][0]) >= 70:
                break
            row = [int(origin[i][0]), train_labels[0][i], train_labels[1][i]]
            row += list(feature_lon)
            row += list(lstm_train_features[1][i])
            writer.writerow(row)
    # test
    test_set = utility.get_test_set()
    test_features = process_features(test_set)
    test_features = cnn_model.predict(test_features)
    with open('test.csv', 'w+') as file:
        writer = csv.writer(file)
        for i, test_row in enumerate(test_set):
            row = [int(test_row[0])]
            row += list(test_features[0][i])
            row += list(test_features[1][i])
            writer.writerow(row)


def phase_2():
    train_set = utility.get_csv_rows('out.csv')
    test_set = utility.get_csv_rows('test.csv')
    groups = group(train_set, split=True)
    test_groups = group(test_set)
    dims = len(train_set[0]) - 3
    lstm_model = build_lstm_model((1, dims))
    test_preds = []
    test_truth = []

    predictions = []
    for i, g in enumerate(groups):
        # g: [0]train_features, [1]test_features, [2]train_labels, [3]test_labels
        g[0] = np.asarray(g[0]).reshape(len(g[0]), 1, dims)
        g[2] = np.asarray(g[2])
        g[1] = np.asarray(g[1]).reshape(len(g[1]), 1, dims)
        g[3] = np.asarray(g[3])

        lstm_model.fit(g[0], g[2], epochs=50)

        test_preds += list(lstm_model.predict(g[1]))
        test_truth += list(g[3])

        pred = lstm_model.predict(np.asarray(test_groups[i]).reshape(len(test_groups[i]), 1, dims))
        predictions += list(pred)

    test_preds = process_predictions(test_preds)
    for i, t in enumerate(test_truth):
        gps = utility.grid_to_gps(t[0], t[1])
        test_truth[i] = [gps[0], gps[1]]
    utility.get_metrics(test_preds, test_truth)
    predictions = process_predictions(predictions)
    # utility.generate_pred_csv(5, predictions)
    # utility.visualize(5, predictions)


def main():
    # phase_1()
    phase_2()


if __name__ == '__main__':
    main()
