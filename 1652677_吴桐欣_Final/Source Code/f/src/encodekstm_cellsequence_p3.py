from mycode.dataset.autoencoder import autoencoderTrans as ect
from keras.layers import *
from keras.models import *
from keras import optimizers
from sklearn.model_selection import train_test_split
from mycode.trainvis import LossHistory
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

row_col = (131, 166)
curphone = 2

np.random.seed(10000000)

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
history = LossHistory()


#  - loss: 3.6797 - row_loss: 1.8205 - col_loss: 1.8591 - row_acc: 0.4498 - col_acc: 0.4347
#  - loss: 3.0844 - row_loss: 1.5192 - col_loss: 1.5652 - row_acc: 0.5307 - col_acc: 0.5209
#  - loss: 2.4103 - row_loss: 1.2003 - col_loss: 1.2101 - row_acc: 0.6196 - col_acc: 0.6302
def build_lstm_model():
    main_input = Input(shape=(10, 2))  # 降维了
    x = Dense(10, activation='relu')(main_input)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = LSTM(500, return_sequences=False)(x)
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
