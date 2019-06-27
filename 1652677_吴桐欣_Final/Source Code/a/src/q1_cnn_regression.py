import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import utility


SHAPE = (5, 6, 1)


def pre_processing_q1():
    train_set = utility.get_train_set()
    # test_set = utility.trans_dataset(utility.get_test_set())
    feature_set = []
    label_set = []
    for i in range(len(train_set)):
        # feature = change_shape(train_set[i][:-2])
        feature_set.append(np.asarray(train_set[i][:-2]))
        label_set.append(np.asarray(train_set[i][-2:]))
    # feature_set = normalize(np.asarray(feature_set), axis=0, norm='max')
    # split
    train_features, test_features, train_labels, test_labels = \
        train_test_split(feature_set, label_set, test_size=0.2, random_state=0)
    for i, feature in enumerate(train_features):
        train_features[i] = process_feature(feature)
    for i, feature in enumerate(test_features):
        test_features[i] = process_feature(feature)
    train_labels = utility.utm_labels_to_gps(train_labels)
    train_labels = utility.transform_gps_labels(train_labels)
    test_labels = utility.utm_labels_to_gps(test_labels)
    test_labels = utility.transform_gps_labels(test_labels)
    return train_features, test_features, train_labels, test_labels


def process_feature(feature):
    new_feature = []
    public = feature[:5]
    respective = feature[5:]
    for i in range(0, 42, 7):
        new_feature.append(respective[i+2:i+7])
        # new_feature.append(np.append(public, respective[i:i + 7]))
    new_feature = np.asarray(new_feature).T
    # print(new_feature)
    # new_feature = normalize(new_feature, axis=0, norm='max')
    # print(new_feature)
    new_feature = np.reshape(new_feature, SHAPE)
    # print(new_feature)
    return new_feature


def build_model_q1(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.tanh, input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer='normal'),
        keras.layers.Dense(8, activation=tf.nn.relu, kernel_initializer='normal'),
        keras.layers.Dense(2, activation=tf.nn.relu, kernel_initializer='normal')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_absolute_error'])
    model.summary()
    return model


def predict(model):
    test_set = utility.get_test_set()
    feature_set = []
    for i, feature in enumerate(test_set):
        feature = process_feature(feature)
        feature_set.append(feature)

    predictions = model.predict(np.asarray(feature_set))
    predictions = utility.transform_gps_labels(predictions, reverse=True)
    utility.generate_pred_csv(1, predictions)
    utility.visualize(1, predictions)


def main():
    dataset = pre_processing_q1()
    model = build_model_q1(input_shape=SHAPE)
    model.fit(np.asarray(dataset[0]), np.asarray(dataset[2]), epochs=30)
    p = model.predict(np.asarray(dataset[1]))
    p = utility.transform_gps_labels(p, reverse=True)
    truth = utility.transform_gps_labels(dataset[3], reverse=True)
    utility.get_metrics(p, truth)
    predict(model)


if __name__ == '__main__':
    main()
