import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def load_model(path):
    return tf.keras.models.load_model(path)


def get_checkpoint_callback(checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,

                                                     save_weights_only=False, verbose=1)
    return cp_callback


def model_exist(path, name):
    if name in os.listdir(path):
        return True
    return False


def save_checkpoint(model, epoch_loss_list, path, name):
    try:
        model.encoder.save(path + name)
    except:
        model.save(path + name)
    epoch_loss = []
    for hist in epoch_loss_list:
        epoch_loss += list(hist.history['loss'])

    plt.plot(epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(f'{path}/figures/{name}.png')


def train_model(model, data, checkpoints, path, name, test_ds=None):
    history = []

    checkpoints = np.array(checkpoints)
    epoch_change = np.diff(checkpoints)

    epoch_change = np.insert(epoch_change, 0, checkpoints[0])
    current_epoch = 0

    for change in epoch_change:

        current_epoch += change
        if model_exist(path, f'{name}_e{current_epoch}'):
            print(f'model {name}_e{current_epoch} existed in {path}')
            model = load_model(path+name + f'_e{current_epoch}')
            continue

        hist = model.fit(data, epochs=change, validation_data=test_ds)
        history.append(hist)

        save_checkpoint(model, history, path, name + f'_e{current_epoch}')

        print(model.predict(data))
    return model, history
