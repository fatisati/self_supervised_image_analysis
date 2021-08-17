import multicrop_dataset
import architecture

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os

from itertools import groupby
from tqdm import tqdm

# Configs
BS = 32
SIZE_CROPS = [224, 96]
NUM_CROPS = [2, 3]
MIN_SCALE = [0.5, 0.14]
MAX_SCALE = [1., 0.5]

# Experimental options
options = tf.data.Options()
options.experimental_optimization.noop_elimination = True
options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_deterministic = False
options.experimental_threading.max_intra_op_parallelism = 1
AUTO = tf.data.experimental.AUTOTUNE






