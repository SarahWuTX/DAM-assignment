import utm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow import keras

import utility

left_lat = 328770
left_lon = 3462224
right_lat = left_lat + 165 * 10  # > 330420
right_lon = left_lon + 127 * 10  # > 3463487
SHAPE = (7, 6, 1)


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


def pre_processing_q4():
    train_set = utility.get_train_set()
    train_set_group = group(train_set)

    for train_set in train_set_group:
        feature_set = []
        label_set = []
        for data in train_set:
            feature_set.append(np.asarray(data[:-2]))
            label_set.append(np.asarray(data[-2:]))
        feature_set = normalize(feature_set, axis=0, norm='max')
        # train_features, test_features, train_labels, test_labels = \
        yield train_test_split(feature_set, label_set, test_size=0.2, random_state=0)


def process_labels(labels):
    x_labels = []
    y_labels = []
    for label in labels:
        x, y = utility.utm_to_gird(longitude=label[0], latitude=label[1])
        x_labels.append(x)
        y_labels.append(y)
    return [np.asarray(x_labels), np.asarray(y_labels)]


def build_model_q4(input_shape):
    # layers
    input_layer = keras.layers.Input(shape=input_shape)
    layer = keras.layers.LSTM(256)(input_layer)
    layer = keras.layers.Dense(256, activation=tf.nn.relu)(layer)
    layer = keras.layers.Dense(256, activation=tf.nn.relu)(layer)
    layer = keras.layers.Dense(256, activation=tf.nn.relu)(layer)
    layer = keras.layers.Dense(256, activation=tf.nn.relu)(layer)

    # settings
    lat = keras.layers.Dense(166, activation=tf.nn.softmax, name='lat')(layer)
    lon = keras.layers.Dense(128, activation=tf.nn.softmax, name='lon')(layer)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    loss = {
        'lat': 'sparse_categorical_crossentropy',
        'lon': 'sparse_categorical_crossentropy'
    }

    # build and compile
    model = keras.Model(inputs=input_layer, outputs=[lon, lat])
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])  # metrics=['mean_absolute_error', 'mean_squared_error']

    model.summary()
    return model


def process_predictions(predictions):
    predictions = [list(predictions[0]), list(predictions[1])]
    new_pred = []
    for i in range(len(predictions[0])):
        pred_lon = list(predictions[0][i])
        pred_lat = list(predictions[1][i])
        lon = pred_lon.index(max(pred_lon))
        lat = pred_lat.index(max(pred_lat))
        location = utility.grid_to_gps(lon, lat)
        new_pred.append([location[0], location[1]])
    return new_pred


def main():
    predictions = []
    test_preds = []
    test_truth = []
    test_group = group(utility.get_test_set())
    model = build_model_q4((1, 44))

    # dataset: [0]train_features, [1]test_features, [2]train_labels, [3]test_labels
    for i, dataset in enumerate(pre_processing_q4()):
        if i >= 69:
            break
        dataset[0] = dataset[0].reshape(len(dataset[0]), 1, 44)
        dataset[2] = process_labels(dataset[2])
        # dataset[2] = np.asarray(dataset[2])#  .reshape(len(dataset[2][0]), 1)
        dataset[1] = dataset[1].reshape(len(dataset[1]), 1, 44)
        # dataset[3] = process_labels(dataset[3])
        # dataset[3] = np.asarray(dataset[3])#  .reshape(len(dataset[3][0]), 1)

        model.fit(dataset[0], dataset[2], epochs=100)
        model.evaluate(dataset[1], process_labels(dataset[3]))
        test_pred = model.predict(dataset[1])
        test_pred = process_predictions(test_pred)
        test_preds += test_pred
        test_truth += dataset[3]

        preds = model.predict(np.asarray(test_group[i]).reshape(len(test_group[i]), 1, 44))
        preds = process_predictions(preds)
        for pred in preds:
            if len(pred) < 2:
                print(pred)
            predictions.append([pred[0], pred[1]])

    test_truth = utility.utm_labels_to_gps(test_truth)
    utility.get_metrics(test_preds, test_truth)
    # utility.generate_pred_csv(4, predictions)
    # utility.visualize(4, predictions)


if __name__ == '__main__':
    main()
