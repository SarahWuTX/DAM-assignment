from mycode.dataset.autoencoder import autoencoderTrans as ect
from keras.layers import *
from keras.models import *
from keras import optimizers
from sklearn.model_selection import train_test_split
from mycode.trainvis import LossHistory
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

row_col = (131, 166)
curphone = 3

np.random.seed(10000000)

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
history = LossHistory()


#  - loss: 4.2204 - row_loss: 2.1363 - col_loss: 2.0841 - row_acc: 0.4292 - col_acc: 0.4229
#  - loss: 4.8570 - row_loss: 2.4307 - col_loss: 2.4263 - row_acc: 0.3356
#  - loss: 4.1609 - row_loss: 2.0934 - col_loss: 2.0675 - row_acc: 0.4150 - col_acc: 0.4273
# - loss: 4.2029 - row_loss: 2.1139 - col_loss: 2.0890 - row_acc: 0.4289
# - loss: 3.4669 - row_loss: 1.7421 - col_loss: 1.7248 - row_acc: 0.5154 - col_acc: 0.5146
# - loss: 3.8063 - row_loss: 1.9131 - col_loss: 1.8932 - row_acc: 0.4731 - col_acc: 0.4743
# - loss: 3.3450 - row_loss: 1.6645 - col_loss: 1.6804 - row_acc: 0.5395 - col_acc: 0.5296
def build_lstm_model():
    main_input = Input(shape=(10, 2))  # 降维了
    x = Dense(10, activation='relu')(main_input)
    x = BatchNormalization()(x)
    x = Dense(100, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(250, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = LSTM(100, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Reshape((1000, ))(x)
    x = Dense(200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    row = Dense(row_col[0], activation='softmax', name="row")(x)
    col = Dense(row_col[1], activation='softmax', name="col")(x)

    model = Model(inputs=main_input,
                  outputs=[row, col],
                  name="RCNet"
                  )

    losses = {
        "row": "sparse_categorical_crossentropy",
        "col": "sparse_categorical_crossentropy"
    }
    # optimizers.SGD(lr=0.001, momentum=0.5)
    model.compile(loss=losses, optimizer='adadelta', metrics=['accuracy'])
    model.summary()
    return model


def main():
    x, y = ect.main(curphone)
    print(x[:5])
    print(y[0])
    model = build_lstm_model()
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8,random_state=33)
    # 喂入数据训练
    model.fit(x_train,
              {"row": y_train[:, 0], "col": y_train[:, 1]},
              validation_data=(x_valid,
                               {"row": y_valid[:, 0], "col": y_valid[:, 1]}),
              batch_size=int(0.01 * len(x_train)), epochs=300, callbacks=[reduce_lr, early_stopping, history])

    model.save('../models/aelstm_class_phone' + str(curphone + 1) + '_2.h5')


if __name__ == "__main__":
    main()
