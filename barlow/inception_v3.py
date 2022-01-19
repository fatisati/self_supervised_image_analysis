import barlow.conf as conf
from utils.model_utils import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import experimental
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


def load_inception():
    return load_model(conf.model_path + 'inception')


def get_network():
    inputs = Input(shape=(299, 299, 3))
    x = experimental.preprocessing.Rescaling(scale=1.0 / 255.0)(inputs)

    inception = load_inception()
    x = inception(x)
    out = BatchNormalization()(x)
    return Model(inputs, out)


if __name__ == '__main__':
    model = get_network()
