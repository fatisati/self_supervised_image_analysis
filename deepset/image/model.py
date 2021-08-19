from eval_ import *


def get_deepset_model(images, max_length):
    input_img = Input(shape=(max_length,))
    x = Embedding(images.shape[0], images.shape[1], mask_zero=True, trainable=False)(input_img)
    x = Dense(300, activation='tanh')(x)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    summer.get_layer(index=1).set_weights([images])
    return summer


def get_lstm_model(images, max_length):
    input_img = Input(shape=(max_length,))
    x = Embedding(images.shape[0], images.shape[1], mask_zero=True, trainable=False)(input_img)
    x = Dense(300, activation='tanh')(x)  # One can try relu as well which results in similar performance
    x = Dense(100, activation='tanh')(x)
    x = LSTM(50)(x)
    x = Dense(30, activation='tanh')(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3)
    summer.compile(optimizer=adam, loss='mae')  # One can try mse as well which results in similar performance
    summer.get_layer(index=1).set_weights([images])
    return summer


def get_gru_model(images, max_length):
    input_img = Input(shape=(max_length,))
    x = Embedding(images.shape[0], images.shape[1], mask_zero=True, trainable=False)(input_img)
    x = Dense(300, activation='tanh')(x)  # One can try relu as well which results in similar performance
    x = Dense(100, activation='tanh')(x)
    x = GRU(50)(x)
    x = Dense(30, activation='tanh')(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3)
    summer.compile(optimizer=adam, loss='mae')  # One can try mse as well which results in similar performance
    summer.get_layer(index=1).set_weights([images])
    return summer


def train_deepset(img, X, sum_X):
    model = get_deepset_model(img, max_train_length)
    # visualize
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    # train
    reduce_lr = get_lr()

    model.fit(X, sum_X, epochs=500, batch_size=128,
              shuffle=True, validation_split=0.0123456789,
              callbacks=[reduce_lr])
    # save weights
    deep_we = []
    for i in [2, 3, 4, 6]:
        w = model.get_layer(index=i).get_weights()
        deep_we.append(w)
    return deep_we


def train_lstm(img, X, sum_X):
    # model
    K.clear_session()
    model = get_lstm_model(img, max_train_length)

    # visualize
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # train
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=10, min_lr=0.000001)

    model.fit(X, sum_X, epochs=100, batch_size=128,  # Fewer iterations, because each iteration is much more costlier
              shuffle=True, validation_split=0.0123456789,
              callbacks=[reduce_lr])

    # save weights
    lstm_we = []
    for i in [2, 3, 4, 5, 6]:
        w = model.get_layer(index=i).get_weights()
        lstm_we.append(w)
    return lstm_we


def train_gru(img, X, sum_X):
    # model
    K.clear_session()
    model = get_gru_model(img, max_train_length)

    # visualize
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # train
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=10, min_lr=0.000001)

    model.fit(X, sum_X, epochs=100, batch_size=128,  # Fewer iterations, because each iteration is much more costlier
              shuffle=True, validation_split=0.0123456789,
              callbacks=[reduce_lr])

    # save weights
    gru_we = []
    for i in [2, 3, 4, 5, 6]:
        w = model.get_layer(index=i).get_weights()
        gru_we.append(w)
    return gru_we


if __name__ == '__main__':
    # model
    img, label = load_mnist()
    img, X, sum_X = gen_train_data(img, label)

    deep_we = train_deepset(img, X, sum_X)
    lstm_we = train_lstm(img, X, sum_X)
    gru_we = train_gru(img, X, sum_X)

    print(evaluate_models(deep_we, get_deepset_model))
    print(evaluate_models(lstm_we, get_lstm_model))
    print(evaluate_models(gru_we, get_deepset_model))
