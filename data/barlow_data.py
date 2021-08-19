import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
# from skimage import io
from numpy import asarray
from PIL import Image
from copy import deepcopy

import zipfile
from io import BytesIO
from PIL import Image

import cv2

# from skimage.transform import resize

AUTO = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE


def download_flower_ds(as_supervised=False):
    # Gather Flowers dataset
    train_ds, validation_ds = tfds.load(
        "tf_flowers",
        split=["train[:85%]", "train[85%:]"],
        as_supervised=as_supervised
    )
    return train_ds, validation_ds


class MyDataset:

    def __init__(self, data_path='data/', image_folder='ISIC2018_Task1-2_Training_Input_resized/',
                 label_filename='labels.xlsx', image_col='img_id',
                 data_size=-1):

        self.image_col = image_col

        self.data_size = data_size
        self.image_path = data_path + image_folder
        print(self.image_path)
        # drive_path = 'drive/MyDrive/'
        # mask_path = 'drive/MyDrive/data/ISIC2018_Task2_Training_GroundTruth_v3/'
        # sample_image_path = image_path #'data/sample100/input/'

        self.label_df = self.read_label_df(data_path + label_filename)
        file_names = self.get_image_names()

        np.random.shuffle(file_names)

        labels, file_names = self.get_label_list(file_names)

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

    @staticmethod
    def read_label_df(df_name):
        if df_name[-3:] == 'csv':
            return pd.read_csv(df_name, index_col=0)
        return pd.read_excel(df_name, index_col=0)

    def get_x_train_test_ds(self):
        return self.train_names_ds.map(self.read_tf_image), self.test_names_ds.map(self.read_tf_image)

    def get_x_train_test(self):
        x_train = self.get_images_nparray(self.train_names)
        x_test = self.get_images_nparray(self.test_names)
        return x_train, x_test

    # def resize_image(self, img):
    #     return img.resize((self.image_size, self.image_size))

    def get_images_nparray(self, img_names):
        print('reading images...')
        cnt = 0
        res_arr = []

        avg_height = 0
        avg_width = 0

        for name in img_names:
            # img = Image.open(self.image_path + name)
            img = cv2.imread(self.image_path + name)
            norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = asarray(norm_image)
            res_arr.append(img)

        return np.array(res_arr)

    def get_supervised_ds(self):
        supervised_train_ds = (self.train_zip
                               .map(self.process_path, num_parallel_calls=AUTOTUNE)
                               #  .batch(BATCH_SIZE)
                               #  .prefetch(AUTO)
                               )

        supervised_test_ds = (self.test_zip
                              .map(self.process_path, num_parallel_calls=AUTOTUNE)
                              #  .batch(BATCH_SIZE)
                              #  .prefetch(AUTO)
                              )
        return supervised_train_ds, supervised_test_ds

    def get_batched_ds(self, batch_size):
        train_ds, test_ds = self.get_supervised_ds()
        return train_ds.batch(batch_size), test_ds.batch(batch_size)

    def get_dict_ds(self):
        train_ds = (self.train_zip
                    .map(self.dict_ds_func, num_parallel_calls=AUTOTUNE)
                    #  .batch(BATCH_SIZE)
                    #  .prefetch(AUTO)
                    )
        test_ds = (self.test_zip
                   .map(self.dict_ds_func, num_parallel_calls=AUTOTUNE)
                   #  .batch(BATCH_SIZE)
                   #  .prefetch(AUTO)
                   )

        return train_ds, test_ds

    def get_image_names(self):
        list_ds = []
        for file in os.listdir(self.image_path):
            if file[-3:] == 'jpg':
                list_ds.append(file)

            if self.data_size == -1:
                continue
            if len(list_ds) == self.data_size:
                break
        return list_ds

    def get_label(self, file_path):
        try:
            res = self.label_df[self.label_df[self.image_col] == file_path].drop([self.image_col], axis=1).iloc[0].values
        except:
            arr_size = 5  # len(self.label_df.drop(['img_id'], axis=1).iloc[0].values)
            res = np.array([-1] * arr_size)
        return res

    def get_label_list(self, img_names):
        res = []
        new_img_names = []
        for i in range(len(img_names)):
            y = self.get_label(img_names[i])

            if y[0] == -1:
                continue
            else:
                res.append(y)
                new_img_names.append(img_names[i])
        return res, new_img_names

    # from skimage import io
    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.per_image_standardization(img)
        return img

    def read_tf_image(self, file_path):
        img = tf.io.read_file(self.image_path + file_path)
        img = self.decode_img(img)
        # img = tf.image.resize(img, size=[self.image_size, self.image_size])
        return img

    def process_path(self, file_path, label):
        # file_path = bytes.decode(path)
        # label = get_label(file_path)
        img = self.read_tf_image(file_path)
        return img, label

    def dict_ds_func(self, file_path, label):
        # file_path = bytes.decode(path)
        # label = get_label(file_path)
        img = self.read_tf_image(file_path)
        return {'image': img, 'label': label}


if __name__ == '__main__':
    ds = MyDataset(data_path='../../data/ISIC/', image_folder='ISIC_task1_resized_input/')