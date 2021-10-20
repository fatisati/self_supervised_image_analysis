import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import CSVLogger


def load_model(path):
    return tf.keras.models.load_model(path)


def get_checkpoint_callback(checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,

                                                     save_weights_only=False, verbose=1)
    return cp_callback


def model_exist(path, name, epoch):
    if not name in os.listdir(path):
        return False
    if f'e{epoch}' in os.listdir(path + name):
        return True
    return False


def check_folder(path, folder):
    if not folder in os.listdir(path):
        os.mkdir(path + folder)


def get_save_path(path, name, epoch):
    return f'{path}{name}/e{epoch}'


def save_checkpoint(model, epoch_loss_list, path, name, epoch):
    save_path = get_save_path(path, name, epoch)
    try:
        model.encoder.save(save_path, save_format='tf')
    except:
        model.save(save_path, save_format='tf')

    epoch_loss = []
    for hist in epoch_loss_list:
        epoch_loss += list(hist.history['loss'])

    plt.plot(epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # check_folder(path + name + '/', 'figures')
    plt.savefig(f'{path + name}/{name}.png')


def find_latest_model(path, name):
    if name not in os.listdir(path):
        return -1, -1
    files = os.listdir(path + name)
    model_files = []
    for file in files:
        if ('.' not in file) and (file[0] == 'e') :
            model_files.append(file)
    if len(model_files) == 0:
        return -1, -1
    model_files = sorted(model_files)
    model = load_model(f'{path}{name}/{model_files[-1]}')
    print(f'best founded model {model_files[-1]}')
    return model, int(model_files[-1][1:])


def train_model(model, data, checkpoints, path, name, test_ds=None):
    history = []
    checkpoints = np.array(checkpoints)

    latest_model, latest_epoch = find_latest_model(path, name)
    if latest_model != -1:
        model = latest_model
        checkpoints = checkpoints[checkpoints > latest_epoch]
        checkpoints = np.insert(checkpoints, 0, latest_epoch)
    else:
        checkpoints = np.insert(checkpoints, 0, 0)

    epoch_change = np.diff(checkpoints)
    current_epoch = 0

    for change in epoch_change:

        current_epoch += change
        print(f'current-epoch {current_epoch}, epoch-change: {change}')

        check_folder(path, name)
        csv_logger = CSVLogger(path + name + '/log.csv', append=True, separator=',')

        hist = model.fit(data, epochs=change, validation_data=test_ds, callbacks=[csv_logger])
        history.append(hist)

        save_checkpoint(model, history, path, name, current_epoch)

        try:
            print(model.encoder.predict(data))
        except:
            print(model.predict(data))
    # except Exception as e:
    #     print(f'cant train model. exception {e}')
    return model, history


def get_metrics():
    METRICS = [
        # keras.metrics.TruePositives(name='tp'),
        # keras.metrics.FalsePositives(name='fp'),
        # keras.metrics.TrueNegatives(name='tn'),
        # keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    return METRICS
