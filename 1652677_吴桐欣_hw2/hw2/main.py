import csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_dataset():
    users = {}
    with open('./data/dianping_gender.csv', 'r') as file:
        for row in csv.reader(file):
            users[row[1]] = {'gender': row[2]}
    with open('./data/dianping_features/dianping_user_vec.csv', 'r') as file:
        for row in csv.reader(file):
            if users.get(row[1]):
                elements = row[2].strip('[] ')
                elements = ' '.join(elements.split()).split(' ')
                elements = list(map(float, elements))
                users.get(row[1])['vec'] = elements
    vec_set = []
    gender_set = []
    keys = list(users.keys())
    keys.sort()
    for key in keys:
        try:
            vec_set.append(users.get(key)['vec'])
            gender_set.append(users.get(key)['gender'])
        except KeyError:
            pass
    return keys, vec_set, gender_set


class ClassifierTry:
    def __init__(self, keys, features, labels):
        self.users = keys
        self.features = features
        self.labels = labels
        self.train_features = []
        self.train_labels = []
        self.validation_features = []
        self.validation_labels = []
        self.test_features = []
        self.test_labels = []
        self.split_dataset()

    def split_dataset(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels = \
            train_test_split(self.features, self.labels, test_size=0.2, random_state=0)
        self.validation_features, self.test_features, self.validation_labels, self.test_labels = \
            train_test_split(self.test_features, self.test_labels, test_size=0.5, random_state=0)

    def logistic_regression(self):
        weight = [None, 'balanced']
        solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
        max_f1 = 0
        chosen_weight = ''
        chosen_solver = ''
        for s in solver:
            for w in weight:
                clf = LogisticRegression(random_state=0, class_weight=w, solver=s)
                clf.fit(self.train_features, self.train_labels)
                print('|  |', w, '|', s, end=' ')
                current_f1 = self.validate(clf.predict(self.validation_features))['f1']
                if current_f1 > max_f1:
                    max_f1 = current_f1
                    chosen_weight = w
                    chosen_solver = s
        print('| test |', chosen_weight, '|', chosen_solver, end=' ')
        clf = LogisticRegression(random_state=0, class_weight=chosen_weight, solver=chosen_solver)
        clf.fit(self.train_features, self.train_labels)
        self.test(clf.predict(self.test_features))

        print('max f1:', max_f1, 'arguments', chosen_weight, chosen_solver)

    def svm(self):
        clf = svm.SVC(random_state=0, cache_size=1000, kernel='rbf', gamma='scale')
        clf.fit(self.train_features, self.train_labels)
        self.validate(clf.predict(self.validation_features))
        # max_f1 = 0
        # chosen_d = 0
        # for d in range(1, 12):
        #     clf = svm.SVC(random_state=0, cache_size=1000, kernel='poly', degree=d, gamma='scale')
        #     clf.fit(self.train_features, self.train_labels)
        #     print('|  | poly | degree={0}, gamma=\'auto\''.format(d), end=' ')
        #     current_f1 = self.validate(clf.predict(self.validation_features))['f1']
        #     if current_f1 > max_f1:
        #         max_f1 = current_f1
        #         chosen_d = d
        # print('| test | poly | degree={0}, gamma=\'auto\''.format(chosen_d), end=' ')
        # clf = svm.SVC(random_state=0, cache_size=1000, kernel='poly', degree=chosen_d, gamma='scale')
        # clf.fit(self.train_features, self.train_labels)
        self.test(clf.predict(self.test_features))

    def random_forest(self):
        max_f1 = 0
        chosen_arg = 0
        for i in range(12, 30):
            clf = RandomForestClassifier(random_state=0, oob_score=True, n_estimators=70, max_depth=19, max_features=i)
            clf.fit(self.train_features, self.train_labels)
            print('|  |  | max_features={0}'.format(i), end=' ')
            current_f1 = self.validate(clf.predict(self.validation_features))['f1']
            if current_f1 > max_f1:
                max_f1 = current_f1
                chosen_arg = i
        print('| test |  | max_features={0}'.format(chosen_arg), end=' ')
        clf = RandomForestClassifier(random_state=0, oob_score=True, n_estimators=70, max_depth=19, max_features=chosen_arg)
        clf.fit(self.train_features, self.train_labels)
        self.test(clf.predict(self.test_features))

    def neural_network(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(32, activation=tf.nn.softmax),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        model.fit(np.array(self.train_features), np.array(self.train_labels), epochs=8)
        predictions = model.predict(np.array(self.validation_features))
        labels = []
        for p in predictions:
            if p[0] > p[1]:
                labels.append('0')
            else:
                labels.append('1')
        self.validate(labels)

        predictions = model.predict(np.array(self.test_features))
        labels = []
        for p in predictions:
            if p[0] > p[1]:
                labels.append('0')
            else:
                labels.append('1')
        self.test(labels)

    def validate(self, labels):
        metrics = ClassifierTry.get_metrics(labels, self.validation_labels)
        # print(format(' validation ', '*^30'))
        # print(metrics)
        print('|', metrics['accuracy'], '|', metrics['precision'], '|', metrics['recall'], '|', metrics['f1'], '|')
        return metrics

    def test(self, labels):
        metrics = ClassifierTry.get_metrics(labels, self.test_labels)
        # print(format(' test ', '*^30'))
        # print(metrics)
        print('|', metrics['accuracy'], '|', metrics['precision'], '|', metrics['recall'], '|', metrics['f1'], '|')

    @staticmethod
    def get_metrics(predict, actual):
        if len(predict) != len(actual):
            raise ValueError('different amount between predict labels and actual labels')
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'true_1': 0,
            'true_0': 0,
            'false_1': 0,
            'false_0': 0,
            'total': 0
        }
        for i in range(len(predict)):
            if predict[i] == actual[i] == '0':
                metrics['true_0'] += 1
            elif predict[i] == actual[i] == '1':
                metrics['true_1'] += 1
            elif predict[i] == '1':
                metrics['false_1'] += 1
            else:
                metrics['false_0'] += 1
        metrics['total'] = metrics['true_0'] + metrics['true_1'] + metrics['false_1'] + metrics['false_0']
        metrics['accuracy'] = (metrics['true_0'] + metrics['true_1']) / metrics['total']
        metrics['precision'] = metrics['true_1'] / (metrics['true_1'] + metrics['false_1'])
        metrics['recall'] = metrics['true_1'] / (metrics['true_1'] + metrics['false_0'])
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

        metrics['accuracy'] = round(metrics['accuracy'], 4)
        metrics['precision'] = round(metrics['precision'], 4)
        metrics['recall'] = round(metrics['recall'], 4)
        metrics['f1'] = round(metrics['f1'], 4)
        return metrics

    def final(self):
        clf = LogisticRegression(random_state=0, solver='newton-cg')
        clf.fit(self.train_features, self.train_labels)
        self.generate_csv('LR.csv', clf.predict(self.features))

        clf = svm.SVC(random_state=0, cache_size=1000, kernel='rbf', gamma='scale')
        clf.fit(self.train_features, self.train_labels)
        self.generate_csv('SVM.csv', clf.predict(self.features))

        clf = RandomForestClassifier(random_state=0, oob_score=True, n_estimators=70, max_depth=19, max_features=14)
        clf.fit(self.train_features, self.train_labels)
        self.generate_csv('RF.csv', clf.predict(self.features))

        model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(32, activation=tf.nn.softmax),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        model.fit(np.array(self.train_features), np.array(self.train_labels), epochs=8)
        predictions = model.predict(np.array(self.features))
        predict_labels = []
        for p in predictions:
            if p[0] > p[1]:
                predict_labels.append('0')
            else:
                predict_labels.append('1')
        self.generate_csv('NN.csv', predict_labels)

    def generate_csv(self, filepath, predict_labels):
        with open(filepath, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(['user_id', 'predict_label', 'true_label'])
            for i in range(len(self.labels)):
                writer.writerow([self.users[i], predict_labels[i], self.labels[i]])


def main():
    keys, features, labels = get_dataset()
    task = ClassifierTry(keys, features, labels)
    # task.logistic_regression()
    # task.svm()
    # task.random_forest()
    # task.neural_network()
    task.final()


if __name__ == '__main__':
    main()
