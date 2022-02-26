import random

import pandas as pd
from utils import tf_utils
import tensorflow as tf
from utils.data_utils import get_train_test_idx

from data_codes.razi.razi_utils import *
import os
from IRV2.data_utils import *

from barlow import augmentation_utils

AUTO = tf.data.AUTOTUNE


def get_random_instance(samples, pid):
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

        print('listing all valid img names...')
        self.valid_names = list(os.listdir(self.img_folder))
        self.valid_names = [name.lower() for name in self.valid_names]
        print(f'{len(self.valid_names)} valid images founded.')
        # the order is the same as order in irv2 datagen
        self.ham_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        print('done')

    def remove_invalid_samples(self, img_names: []):
        valid_imgs = []
        for name in img_names:
            if name in self.valid_names:
                valid_imgs.append(name)
        return valid_imgs

    def get_pretrain_samples(self):
        samples = pd.read_excel(self.data_folder + 'all_samples.xlsx')
        samples['img_names'] = get_samples_valid_img_names(samples, self.valid_names)
        samples['img_cnt'] = [len(names) for names in samples['img_names']]
        samples = samples[samples['img_cnt'] > 0]
        return samples

    def process_ssl_path(self, ssl_samples, bs):
        return tf_utils.tf_ds_from_arr(ssl_samples).map(self.process_path).batch(bs).prefetch(AUTO)

    def prepare_ssl_ds(self, samples, bs):
        ssl_one = [self.img_folder + get_random_instance(samples, i) for i in range(len(samples))]
        ssl_two = [self.img_folder + get_random_instance(samples, i) for i in range(len(samples))]

        ssl_ds_one = self.process_ssl_path(ssl_one, bs)
        ssl_ds_two = self.process_ssl_path(ssl_two, bs)
        ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
        return ssl_ds

    def process_path(self, path):
        img = tf_utils.read_tf_image(path)
        return self.augment_img(img)

    def augment_img(self, img):
        aug_func = augmentation_utils.get_tf_augment(self.img_size)
        return aug_func(img)

    def prepare_imgs_labels(self, df, prepare_label=None):
        if not prepare_label:
            prepare_label = lambda label: get_one_hot(label, self.all_labels)
        all_path = [self.img_folder + get_img_name(url) for url in df['img_url']]
        path_ds = tf_utils.tf_ds_from_arr(all_path)
        labels = [prepare_label(label) for label in df['label']]
        labels_ds = tf_utils.tf_ds_from_arr(labels)
        return path_ds, labels_ds

    def make_zip_ds(self, df):
        path_ds, labels_ds = self.prepare_imgs_labels(df)
        return tf.data.Dataset.zip((path_ds.map(self.process_path), labels_ds))

    def filter_valid_samples(self, samples):
        samples['img_name'] = [get_img_name(url) for url in samples['img_url']]
        return samples[samples['img_name'].isin(self.valid_names)]

    def generating_valid_samples(self, group, label_set=None):
        print('generating razi supervised ds...')
        samples = pd.read_excel(self.data_folder + 'supervised_samples.xlsx')
        samples = samples[samples['group'] == group]
        if label_set:
            samples = samples[samples['label'].isin(label_set)]
        print(f'all sample size: {len(samples)}')

        samples = self.filter_valid_samples(samples)
        print(f'valid sample size: {len(samples)}')
        return samples

    def prepare_supervised_data(self, train_ratio, group):

        samples = self.generating_valid_samples(group)
        train_sample_idx = get_train_test_idx(len(samples), 1 - train_ratio)
        train_sample_idx = [idx == 1 for idx in train_sample_idx]
        train = samples[train_sample_idx]

        test_idx = [not (x) for x in train_sample_idx]
        test = samples[test_idx]

        self.all_labels = list(set(samples['label']))
        return train, test

    def prepare_ham_labels(self, label):
        return get_one_hot(label, self.ham_labels)

    def get_ham_format_x_y(self):
        samples = self.generating_valid_samples('tumor', self.ham_labels)
        path_ds, labels_ds = self.prepare_imgs_labels(samples, self.prepare_ham_labels)
        return path_ds.map(self.process_path), labels_ds

    def get_supervised_ds(self, train_ratio, group):

        train, test = self.prepare_supervised_data(train_ratio, group)
        train_ds = self.make_zip_ds(train)
        test_ds = self.make_zip_ds(test)

        print('done')
        return train_ds, test_ds
