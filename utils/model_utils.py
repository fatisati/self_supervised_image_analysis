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


def check_folder(path, folder):
    if not folder in os.listdir(path):
        os.mkdir(path + folder)


def save_checkpoint(model, epoch_loss_list, path, name, epoch):
    check_folder(path, name)
    try:
        model.encoder.save(path + name + '/' + f'e{epoch}')
    except:
        model.save(path + name + '/' + f'e{epoch}')

    epoch_loss = []
    for hist in epoch_loss_list:
        epoch_loss += list(hist.history['loss'])

    plt.plot(epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    check_folder(path + name + '/', 'figures')
    plt.savefig(f'{path + name}/figures/{name}.png')


def train_model(model, data, checkpoints, path, name, test_ds=None):
    history = []

    checkpoints = np.array(checkpoints)
    epoch_change = np.diff(checkpoints)

    epoch_change = np.insert(epoch_change, 0, checkpoints[0])
    current_epoch = 0

    for change in epoch_change:

        current_epoch += change
        print(f'epoch {current_epoch}')

        # if model_exist(path, f'{name}_e{current_epoch}'):
        #     print(f'model {name}_e{current_epoch} existed in {path}')
        #     model = load_model(path+name + f'_e{current_epoch}')
        #     model.compile(optimizer=tf.keras.optimizers.Adam())
        #     continue

        hist = model.fit(data, epochs=change, validation_data=test_ds)
        history.append(hist)

        save_checkpoint(model, history, path, name, current_epoch)

        try:

            print(model.encoder.predict(data))
        except:
            print(model.predict(data))
    return model, history
