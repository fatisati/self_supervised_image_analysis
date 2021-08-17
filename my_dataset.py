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

AUTO = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE


def balance_data(x, y):
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(x, y)
    return X_res, y_res


class MyDataset:

    def __init__(self, data_path='data/', image_folder='ISIC2018_Task1-2_Training_Input_resized/',
                 label_filename='labels.xlsx', image_col='img_id',
                 data_size=-1, balanced=False):
        self.image_col = image_col

        self.data_size = data_size
        self.data_path = data_path
        self.image_folder = image_folder

        self.label_df = self.read_label_df(data_path + label_filename)

        file_names = self.label_df[image_col]
        labels = self.label_df.drop([image_col], axis=1).values

        if balanced:
            print('balancing data...')
            file_names, labels = balance_data(np.array(file_names).reshape(-1, 1), np.array(labels))
            file_names = file_names.flatten()


        if data_size != -1:
            file_names = file_names[:data_size]
            labels = labels[:data_size]

        test_size = int(0.1 * 0.8 * len(file_names))
        train_size = len(file_names) - test_size
        print('train size: ', train_size, ' test size:', len(file_names) - train_size)

        self.train_names = file_names[:train_size]
        self.test_names = file_names[train_size:]

        self.train_labels = labels[:train_size]
        self.test_labels = labels[train_size:]

        self.train_names_ds = tf.data.Dataset.from_tensor_slices(self.train_names)
        self.test_names_ds = tf.data.Dataset.from_tensor_slices(self.test_names)

        self.train_labels_ds = tf.data.Dataset.from_tensor_slices(self.train_labels)
        self.test_labels_ds = tf.data.Dataset.from_tensor_slices(self.test_labels)

        self.train_zip = tf.data.Dataset.zip((self.train_names_ds, self.train_labels_ds))
        self.test_zip = tf.data.Dataset.zip((self.test_names_ds, self.test_labels_ds))

    def get_image_path(self):
        return self.data_path + self.image_folder

    @staticmethod
    def read_label_df(df_name):
        if df_name[-3:] == 'csv':
            return pd.read_csv(df_name, index_col=0)
        return pd.read_excel(df_name, index_col=0)

    def get_x_train_test_ds(self):
        return self.train_names_ds.map(self.read_tf_image), \
               self.test_names_ds.map(self.read_tf_image)

    def get_supervised_ds(self):
        supervised_train_ds = (self.train_zip
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
        return img

    def read_tf_image(self, file_path):
        path = self.get_image_path() + file_path

        img = tf.io.read_file(path)
        img = self.decode_img(img)
        return img

    def process_path(self, file_path, label):
        img = self.read_tf_image(file_path)
        return img, label


if __name__ == '__main__':
    ds = MyDataset(data_path='../data/ISIC/ham10000/', image_folder='resized256/',
                   label_filename='disease_labels.csv', image_col='image', balanced=True)
    print(np.array(ds.train_labels.shape[-1]))