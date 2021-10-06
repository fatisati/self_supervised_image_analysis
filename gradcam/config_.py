import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"

if __name__ == '__main__':

    # The local path to our target image
    img_path = keras.utils.get_file(
        "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    )

    display(Image(img_path))