import swav.utils.multicrop_dataset as multicrop_dataset
import swav.utils.architecture as architecture

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from itertools import groupby


from tqdm import tqdm
# from imutils import paths

tf.random.set_seed(666)
np.random.seed(666)

AUTO = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = 64
# EPOCHS = 35

# Configs
# CROP_SIZE = 224
MIN_SCALE = 0.5
MAX_SCALE = 1.

# Experimental options
options = tf.data.Options()
options.experimental_optimization.noop_elimination = True
# options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_deterministic = False
options.experimental_threading.max_intra_op_parallelism = 1