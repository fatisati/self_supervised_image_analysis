import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import CSVLogger
import time

import barlow.inception_v3 as inception_v3
import segmentation.unet_model as unet_model
from barlow import resnet20


def print_time(st):
    print(f'done took {time.time() - st}')


def load_model(path):
    print(f'loading model in path: {path} ...')
    st = time.time()
    model = tf.keras.models.load_model(path)
    print_time(st)
    return model


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
        if ('.' not in file) and (file[0] == 'e'):
            model_files.append(int(file[1:]))
    if len(model_files) == 0:
        return -1, -1
    model_files = sorted(model_files)
    print(model_files)
    model = load_model(f'{path}{name}/e{model_files[-1]}')
    print(f'best founded model {model_files[-1]}')
    return model, int(model_files[-1])


def train_model(model, data, checkpoints, path, name,
                test_ds=None, load_latest_model=True, debug=False, checkpoint_function=None, compile_function=None):
    history = []
    checkpoints = np.array(checkpoints)

    latest_model, latest_epoch = find_latest_model(path, name)

    if (latest_model != -1) and load_latest_model:
        model = latest_model
        checkpoints = checkpoints[checkpoints > latest_epoch]
        checkpoints = np.insert(checkpoints, 0, latest_epoch)
    else:
        checkpoints = np.insert(checkpoints, 0, 0)

    epoch_change = np.diff(checkpoints)
    current_epoch = checkpoints[0]

    for change in epoch_change:

        print(f'current-epoch {current_epoch}-{current_epoch + change}, epoch-change: {change}')
        current_epoch += change

        check_folder(path, name)
        csv_logger = CSVLogger(path + name + '/log.csv', append=True, separator=',')

        if debug:
            print('backbone out before train')
            try:
                print(model.encoder.predict(data))
            except:
                print(model.predict(data))

        if compile_function:
            compile_function(model)
        hist = model.fit(data, epochs=change, validation_data=test_ds, callbacks=[csv_logger])
        history.append(hist)

        save_checkpoint(model, history, path, name, current_epoch)
        if checkpoint_function:
            checkpoint_function(model, f'{path}/{name}/e{current_epoch}')
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


def load_custom_object_model(weighted_loss):
    custom_objects = {"weighted_loss": weighted_loss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model('../../models/twins/finetune/ct64_bs128_loss_weighted/e100')
    print('model loaded')
    return model


def get_backbone(backbone_name, crop_to = None, project_dim = None):
    backbone = None
    if backbone_name == 'resnet':
        backbone = resnet20.get_network(crop_to, hidden_dim=project_dim, use_pred=False,
                                        return_before_head=False)
    elif backbone_name == 'inception':
        backbone = inception_v3.get_network()

    elif backbone_name == 'unet':
        backbone = unet_model.get_unet_backbone((crop_to, crop_to))
    return backbone
