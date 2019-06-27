from mycode.dataset.autoencoder import autoencoderTrans as ect
from keras.layers import *
from keras.models import *
from keras import optimizers
from sklearn.model_selection import train_test_split
from mycode.trainvis import LossHistory
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

row_col = (131, 166)
curphone = 1

np.random.seed(10000000)

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
history = LossHistory()

# - loss: 7.4566 - row_loss: 3.7141 - col_loss: 3.7425 - row_acc: 0.1421 - col_acc: 0.1432
#  - loss: 7.2114 - row_loss: 3.5716 - col_loss: 3.6398 - row_acc: 0.1499 - col_acc: 0.1409 relu2tanh
# - loss: 6.8279 - row_loss: 3.4052 - col_loss: 3.4226 - row_acc: 0.1600 - col_acc: 0.1376  one more dense100
# - loss: 6.3000 - row_loss: 3.1504 - col_loss: 3.1496 - row_acc: 0.1957 - col_acc: 0.1745  one more dense256
# - loss: 6.5974 - row_loss: 3.2734 - col_loss: 3.3240 - row_acc: 0.2248 - col_acc: 0.2103
# - loss: 6.2313 - row_loss: 3.1273 - col_loss: 3.1040 - row_acc: 0.1991 - col_acc: 0.1991
# - loss: 5.8884 - row_loss: 2.9441 - col_loss: 2.9444 - row_acc: 0.2371 - col_acc: 0.2036 more cell in Dense after LSTM
#  - loss: 5.4823 - row_loss: 2.7406 - col_loss: 2.7418 - row_acc: 0.2640 - col_acc: 0.2562 more cell in LSTM
def build_lstm_model():
    main_input = Input(shape=(10, 2))  # 降维了
    x = Dense(10, activation='relu')(main_input)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = LSTM(200, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='tanh')(x)
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

    model.save('../models/aelstm_class_phone' + str(curphone + 1) + '.h5')


if __name__ == "__main__":
    main()
