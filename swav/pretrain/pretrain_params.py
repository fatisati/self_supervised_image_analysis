import swav.utils.multicrop_dataset as multicrop_dataset
import swav.utils.architecture as architecture

import tensorflow as tf
# import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os

from itertools import groupby
from tqdm import tqdm

tf.random.set_seed(666)
np.random.seed(666)

# tfds.disable_progress_bar()


# Experimental options
options = tf.data.Options()
options.experimental_optimization.noop_elimination = True
# options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_deterministic = False
options.experimental_threading.max_intra_op_parallelism = 1

# Configs
# BS = 32
# SIZE_CROPS = [224, 96]
NUM_CROPS = [2, 3]
MIN_SCALE = [0.14, 0.05]
MAX_SCALE = [1., 0.14]

lr_decayed_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.1,
        decay_steps=500,
        end_learning_rate=0.001,
        power=0.5)
opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn)