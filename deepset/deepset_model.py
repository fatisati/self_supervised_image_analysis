from keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation, Lambda
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import adam_v2


def get_deepset_model(embedding_size, outshape):
    input_img = Input(shape=(2,))
    x = Embedding(embedding_size, mask_zero=False, trainable=False)(input_img)
    x = Dense(300, activation='tanh')(x)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(outshape)(x)
    summer = Model(input_img, encoded)
    adam = adam_v2(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer

if __name__ == '__main__':
    model = get_deepset_model(1024, 7)
    model.summary()