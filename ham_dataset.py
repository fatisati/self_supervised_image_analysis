import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tensorflow as tf
# from skimage import io
from numpy import asarray
from PIL import Image
from copy import deepcopy

import zipfile
from io import BytesIO
from PIL import Image

# from skimage.transform import resize
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
import keras.backend as K

from sklearn.utils import shuffle

AUTO = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE


def balance_data(x, y):
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(x, y)
    return X_res, y_res


def class_weights(y):
    y = y.astype(np.int)
    class_counts = np.bincount(y)
    sum_ = sum(class_counts)
    weights = sum_ / class_counts
    return weights / weights.sum()


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = class_weights(y_true[:, i])
    # weights = weights / weights.sum()
    print(weights[:, 0], weights[:, 1])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        loss = (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred)
        return loss / weights.sum()  # K.mean(loss, xis=-1)

    return weighted_loss


# model.compile(optimizer=Adam(), loss=get_weighted_loss(class_weights))

def label_report(labels):
    for i in range(labels.shape[1]):
        print(sum(labels[:, i]), end=',')


class MyDataset:

    def __init__(self, data_path='data/', image_folder='ISIC2018_Task1-2_Training_Input_resized/',
                 label_filename='labels.xlsx', image_col='img_id',
                 data_size=-1, balanced=False):
        self.image_col = image_col

        self.data_size = data_size
        self.data_path = data_path
        self.image_folder = image_folder
        self.balanced = balanced

        self.label_df = self.read_label_df(data_path + label_filename)

        if data_size != -1:
            self.label_df = self.label_df[:data_size]

        file_names = self.label_df[image_col]
        labels = self.label_df.drop([image_col, 'is_train'], axis=1).values

        class_weights = calculating_class_weights(labels)
        self.weighted_loss = get_weighted_loss(class_weights)

        train_idx = self.label_df['is_train'] == 1

        train_size = len(self.label_df[train_idx])
        test_size = len(file_names) - train_size
        print(f'train size: {train_size}, test size: {test_size}')

        self.train_names = file_names[train_idx]
        self.test_names = file_names[~train_idx]

        self.train_labels = labels[train_idx]
        self.test_labels = labels[~train_idx]

        print('train label report')
        label_report(self.train_labels)
        print('test label report')
        label_report(self.test_labels)

        self.supervised_train_size = int(0.1*len(self.train_names))
        self.train_names_ds_sample = tf.data.Dataset.from_tensor_slices\
            (self.train_names[:self.supervised_train_size])
        self.test_names_ds = tf.data.Dataset.from_tensor_slices(self.test_names)

        self.train_labels_ds_sample = tf.data.Dataset.from_tensor_slices\
            (self.train_labels[:self.supervised_train_size])
        self.test_labels_ds = tf.data.Dataset.from_tensor_slices(self.test_labels)

        self.train_zip_sample = tf.data.Dataset.zip((self.train_names_ds_sample, self.train_labels_ds_sample))
        self.test_zip = tf.data.Dataset.zip((self.test_names_ds, self.test_labels_ds))

    def get_image_path(self):
        return self.data_path + self.image_folder

    @staticmethod
    def read_label_df(df_name):
        if df_name[-3:] == 'csv':
            return pd.read_csv(df_name, index_col=0)
        return pd.read_excel(df_name, index_col=0)

    def get_x_train_test_ds(self):
        return self.train_names_ds_sample.map(self.read_tf_image), \
               self.test_names_ds.map(self.read_tf_image)

    def get_supervised_ds(self):
        print(f'train size: {len(self.train_zip_sample)}')
        supervised_train_ds = (self.train_zip_sample
                               .map(self.process_path, num_parallel_calls=AUTOTUNE)
                               )

        supervised_test_ds = (self.test_zip
                              .map(self.process_path, num_parallel_calls=AUTOTUNE)
                              )
        return supervised_train_ds, supervised_test_ds

    def get_batched_ds(self, batch_size):
        train_ds, test_ds = self.get_supervised_ds()
        return train_ds.batch(batch_size), test_ds.batch(batch_size)

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.per_image_standardization(img)
        # img = tf.divide(img, 255)
        return img

    def read_tf_image(self, file_path):
        path = self.get_image_path() + file_path

        img = tf.io.read_file(path)
        img = self.decode_img(img)
        return img

    def process_path(self, file_path, label):
        img = self.read_tf_image(file_path)
        return img, label


def fix_validation_data(df, train_size, save_path):
    train_idxs = [0] * len(df)
    train_idxs[:train_size] = [1] * train_size
    train_idxs = shuffle(train_idxs)
    df['is_train'] = train_idxs
    df.to_csv(save_path)


if __name__ == '__main__':
    ds = MyDataset(data_path='../data/ISIC/ham10000/', image_folder='resized256/',
                   label_filename='disease_labels.csv', image_col='image')
    # x_train, x_test = ds.get_x_train_test_ds()
    #
    # sample_img = list(x_train.take(1))[0]
    # plt.imshow(sample_img)
    # plt.show()
