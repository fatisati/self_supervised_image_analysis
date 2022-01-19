import random

import pandas as pd
from utils import tf_utils
import tensorflow as tf
from utils.data_utils import get_train_test_idx

from data_codes.razi.razi_utils import *
import os

AUTO = tf.data.AUTOTUNE


def augment_sample(samples, pid):
    row = samples.iloc[pid]
    img_names = row['img_names']
    r = random.randint(0, len(img_names) - 1)
    return img_names[r]


class RaziDataset:

    def __init__(self, data_folder, img_size, img_folder=None):
        self.data_folder = data_folder
        if img_folder:
            self.img_folder = img_folder
        else:
            self.img_folder = self.data_folder + 'imgs/'
        self.img_size = img_size
        self.all_labels = None
        self.valid_names = list(os.listdir(self.img_folder))

    def remove_invalid_samples(self, img_names: []):
        valid_imgs = []
        for name in img_names:
            if name in self.valid_names:
                valid_imgs.append(name)
        return valid_imgs

    def get_pretrain_samples(self):
        samples = pd.read_excel(self.data_folder + 'all_samples.xlsx')
        samples['img_names'] = get_samples_valid_img_names(samples, list(os.listdir(self.img_folder)))
        samples['img_cnt'] = [len(names) for names in samples['img_names']]
        samples = samples[samples['img_cnt'] > 0]
        train = samples[samples['is_train'] == 1]
        test = samples[samples['is_train'] == 0]
        print(f'train-size: {len(train)}, test-size: {len(test)}')
        return train, test

    def process_ssl_names(self, ssl_samples, bs):
        return tf_utils.tf_ds_from_arr(ssl_samples).map(self.load_img).batch(bs).prefetch(AUTO)

    def prepare_ssl_ds(self, samples, bs):
        ssl_one = [augment_sample(samples, i) for i in range(len(samples))]
        ssl_two = [augment_sample(samples, i) for i in range(len(samples))]

        ssl_ds_one = self.process_ssl_names(ssl_one, bs)
        ssl_ds_two = self.process_ssl_names(ssl_two, bs)
        ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
        return ssl_ds

    def load_img(self, name):
        img = tf_utils.read_tf_image(self.img_folder + name)
        return tf.image.resize(img, (self.img_size, self.img_size))

    def make_zip_ds(self, df):
        names = [get_img_name(url) for url in df['img_url']]
        names_ds = tf_utils.tf_ds_from_arr(names)
        labels = [get_one_hot(label, self.all_labels) for label in df['label']]
        labels_ds = tf_utils.tf_ds_from_arr(labels)
        return tf.data.Dataset.zip((names_ds.map(self.load_img), labels_ds))

    def filter_valid_samples(self, samples):
        samples['img_name'] = [get_img_name(url) for url in samples['img_url']]
        return samples[samples['img_name'].isin(self.valid_names)]

    def get_supervised_ds(self, train_ratio):
        print('generating razi supervised ds...')
        samples = pd.read_excel(self.data_folder + 'supervised_samples.xlsx')
        print(f'all sample size: {len(samples)}')

        samples = self.filter_valid_samples(samples)
        print(f'valid sample size: {len(samples)}')

        train = samples[samples['is_train'] == 1]
        train_sample_idx = get_train_test_idx(len(train), 1 - train_ratio)
        train_sample_idx = [idx == 1 for idx in train_sample_idx]
        train = train[train_sample_idx]

        test = samples[samples['is_train'] == 0]

        self.all_labels = list(set(samples['label']))
        train_ds = self.make_zip_ds(train)
        test_ds = self.make_zip_ds(test)
        print('done')
        return train_ds, test_ds
