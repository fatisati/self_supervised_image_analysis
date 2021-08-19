import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import SVG
import keras.backend as K
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation, Lambda
from keras.models import Model, load_model
# from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm, trange

num_train_examples = 100
max_train_length = 10

num_test_examples = 10000
min_test_length = 5
max_test_length = 55
step_test_length = 5


def get_lr():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=20, min_lr=0.000001)
