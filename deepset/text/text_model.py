from text_params_ import *

models_layer_arr = {'deepset': [1, 2, 4], 'lstm': [1, 2, 3], 'gru': [1, 2, 3]}


def get_deepset_model(max_length):
    input_txt = Input(shape=(max_length,))
    x = Embedding(11, 100, mask_zero=True)(input_txt)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_txt, encoded)
    adam = Adam(lr=1e-4, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer


def get_lstm_model(max_length):
    input_txt = Input(shape=(max_length,))
    x = Embedding(11, 100, mask_zero=True)(input_txt)
    x = LSTM(50)(x)
    encoded = Dense(1)(x)
    summer = Model(input_txt, encoded)
    adam = Adam(lr=1e-4)
    summer.compile(optimizer=adam, loss='mae')
    return summer


def get_gru_model(max_length):
    input_txt = Input(shape=(max_length,))
    x = Embedding(11, 100, mask_zero=True)(input_txt)
    x = GRU(80)(x)
    encoded = Dense(1)(x)
    summer = Model(input_txt, encoded)
    adam = Adam(lr=1e-4)
    summer.compile(optimizer=adam, loss='mae')
    return summer


def train_model(X, sum_X, get_model_func, layer_arr=[]):
    # model
    model = get_model_func(max_train_length)

    # visualize
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # train
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=0, save_best_only=True)

    model.fit(X, sum_X, epochs=10, batch_size=128,
              shuffle=True, validation_split=0.0123456789,
              callbacks=[checkpointer])

    model = load_model('/tmp/weights.hdf5')

    # save weights
    weights = []
    for i in layer_arr:
        w = model.get_layer(index=i).get_weights()
        weights.append(w)

    return weights
