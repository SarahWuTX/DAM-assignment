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
    train_features, test_features, train_labels, test_labels = \
        train_test_split(feature_set, label_set, test_size=0.1)

    train_features = process_features(train_features)
    test_features = process_features(test_features)
    train_labels = process_labels(train_labels)
    test_labels = process_labels(test_labels)
    return train_features, test_features, train_labels, test_labels


def process_features(features):
    new_features = []
    for feature in features:
        new_feature = []
        public = feature[:5]
        respective = feature[5:]
        for i in range(0, 42, 7):
            new_feature.append(respective[i + 2:i + 7])
            # new_feature.append(np.append(public, respective[i:i + 7]))
        new_feature = np.asarray(new_feature)
        # print(new_feature)
        new_feature = np.reshape(new_feature, (SHAPE[1], SHAPE[0])).T
        # new_feature = normalize(new_feature)
        # for i, f in enumerate(new_feature):
        #     new_feature[i] = normalize(list(f))

        # new_feature = normalize(new_feature, axis=0, norm='max')
        # print(new_feature)
        new_feature = np.reshape(new_feature, SHAPE)
        new_features.append(new_feature)
    return np.asarray(new_features)


def process_labels(labels, reverse=False):
    if not reverse:
        x_labels = []
        y_labels = []
        for label in labels:
            x, y = utility.utm_to_gird(longitude=label[0], latitude=label[1])
            x_labels.append(x)
            y_labels.append(y)
        return [np.asarray(x_labels), np.asarray(y_labels)]
    else:
        new_labels = []
        for i in range(len(labels[0])):
            lon, lat = utility.grid_to_gps(labels[0][i], labels[1][i])
            new_labels.append([lon, lat])
        return new_labels


def process_predictions(predictions):
    predictions = [list(predictions[0]), list(predictions[1])]
    new_pred = []
    for i in range(len(predictions[0])):
        pred_lon = list(predictions[0][i])
        pred_lat = list(predictions[1][i])
        lon = pred_lon.index(max(pred_lon))
        lat = pred_lat.index(max(pred_lat))
        # print(lon, lat)
        location = utility.grid_to_gps(lon, lat)
        new_pred.append([location[0], location[1]])
    return new_pred


def build_model_q2(input_shape):
    # layers
    input_layer = keras.layers.Input(shape=input_shape)
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

    # settings
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


def predict(model):
    test_set = utility.get_test_set()
    feature_set = process_features(test_set)
    predictions = model.predict(feature_set)
    predictions = process_predictions(predictions)
    utility.generate_pred_csv(2, predictions)
    utility.visualize(2, predictions)


def main():
    train_features, test_features, train_labels, test_labels = pre_processing_q2()
    model = build_model_q2(SHAPE)
    model.fit(train_features, train_labels, epochs=120)
    predictions = model.predict(test_features)
    predictions = process_predictions(predictions)
    test_labels = process_labels(test_labels, reverse=True)
    utility.get_metrics(predictions, test_labels)
    # predict(model)


if __name__ == '__main__':
    main()
