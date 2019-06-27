from keras.layers import *
from keras.models import *
from mycode.dataset.lstm_trans import lstmClassificationTrans as lrt
from mycode.dataset.lstm_trans import cnnlstmTrans as clt
from sklearn.model_selection import *
from keras import optimizers
from mycode.trainvis import LossHistory
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# np.random.seed(0)  # - loss: 0.0648 - acc: 0.1329 - val_loss: 0.0648 - val_acc: 0.1522
# np.random.seed(1000)  # - loss: 0.0612 - acc: 0.1687 - val_loss: 0.0612 - val_acc: 0.1603
np.random.seed(100000)  # - loss: 0.0648 - acc: 0.2257 - val_loss: 0.0648 - val_acc: 0.2049  五维


history = LossHistory()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

def moving_window(x, y, sequence_length):
    # 处理成lstm输入格式
    sequence_x = []
    sequence_y = []
    for i in range(len(x) - sequence_length):
        sequence_x.append(x[i: i + sequence_length])  # [i, i+sequence_length-1] [0, 4]
        sequence_y.append(y[i + sequence_length])  # 5
    x = np.array(sequence_x)

    y = np.array(sequence_y)

    # 分割训练验证集
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[-1]))  # (2422, 10, 30) --> (2422, 10, 30)
    return x, y


def preprocess_data(x, y, sequence_length=10):
    """
    处理输入成lstm输入形式，划分训练，验证，测试集
    :param x: 一个完整的时序序列训练集数据, (一部手机)
    :param y: 一个完整的每时刻预测值序列, (一部手机)
    :param sequence_length: 用于对x进行训练长度分割，形成序列列表
    :return:
    """
    x, y = moving_window(x, y, sequence_length)
    train_x_disorder, tmp_x, train_y_disorder, tmp_y = train_test_split(x, y, train_size=0.8, random_state=33)
    return train_x_disorder, tmp_x


def deep_ae(data, valdata):
    inpE = Input((10, 32))  # here, you don't define the batch size
    outE = Dense(units=28, activation='tanh')(inpE)
    outE = LSTM(units=16, return_sequences=True)(outE)
    outE = BatchNormalization()(outE)
    outE = LSTM(units=8, return_sequences=True)(outE)
    outE = LSTM(units=4, return_sequences=True)(outE)
    outE = LSTM(units=2, return_sequences=True)(outE)

    encoder = Model(inpE, outE)

    inpD = Input((10, 2))
    outD = LSTM(4, return_sequences=True)(inpD)
    outD = LSTM(8, return_sequences=True)(outD)
    outD = BatchNormalization()(outD)
    outD = LSTM(16, return_sequences=True)(outD)
    outD = Dense(units=28, activation='tanh')(outD)
    outD = LSTM(32, return_sequences=True)(outD)

    decoder = Model(inpD, outD)
    autoencoder = Model(encoder.inputs, decoder(encoder(encoder.inputs)))

    autoencoder.compile(loss='mse',
                        optimizer=optimizers.SGD(lr=0.01, momentum=0.1),
                        metrics=['accuracy'])

    autoencoder.fit(data, data,
                    batch_size=20,
                    epochs=300,
                    validation_data=(valdata, valdata), callbacks=[reduce_lr, early_stopping, history])

    return encoder, decoder, autoencoder


def main():
    x, y = lrt.main()
    x, valx = preprocess_data(x[0], y[0])
    encoder, decoder, autoencoder = deep_ae(x, valx)
    encoder.save("../models/deep_encoder_p1.h5")
    decoder.save("../models/depp_decoder_p1.h5")
    autoencoder.save("../models/deep_autoencoder_p1.h5")


if __name__ == "__main__":
    main()
