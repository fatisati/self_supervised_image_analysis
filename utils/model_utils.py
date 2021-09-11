import tensorflow as tf
import os

def get_checkpoint_callback(checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                     save_weights_only=False, verbose=1)
    return cp_callback


def model_exist(params):
    if params.get_summary() in os.listdir(params.save_path):
        return True
    return False

