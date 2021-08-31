import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import SVG
import keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation, Lambda, Concatenate
from tensorflow.keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm, trange

from tensorflow.keras.optimizers import Adam


def get_deepset_image_model(images, max_length):
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


def get_deepset_model_embedded_input(embedding_cnt, embedding_shape):
    inp = Input(shape=(embedding_cnt, embedding_shape))
    x = Dense(300, activation='tanh')(inp)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(inp, encoded)
    adam = Adam(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    # summer.get_layer(index=1).set_weights([images])
    return summer


def get_deepset_hybrid(twins_model, swav_model):
    input_img = Input((None, None, 3))
    twins_emb = twins_model(input_img)
    swav_emb = swav_model(input_img)
    concat_emb = tf.stack([twins_emb, swav_emb], axis=1)
    x = Dense(300, activation='tanh')(concat_emb)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer


def get_sample_deepset_concat():
    input_img = Input(shape=(3, 256))
    twins_emb = Dense(10)(input_img)
    swav_emb = Dense(20)(input_img)
    concat_layer = Concatenate()([twins_emb, swav_emb])
    x = Dense(300, activation='tanh')(concat_layer)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer


