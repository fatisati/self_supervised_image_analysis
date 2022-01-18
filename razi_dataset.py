import pandas as pd
import ast
from utils import tf_utils
import tensorflow as tf
from utils.data_utils import get_train_test_idx

import os

def get_img_name(url):
    slash_idx = url.find('/')
    return url[slash_idx + 1:]


def get_one_hot(label, all_labels: []):
    label_idx = all_labels.index(label)
    one_hot = [0] * len(all_labels)
    one_hot[label_idx] = 1
    return one_hot


class RaziDataset:

    def __init__(self, data_folder, img_size):
        self.data_folder = data_folder
        self.img_folder = data_folder + 'imgs/'
        self.img_size = img_size
        self.all_labels = None

    def for_pretrain(self):
        self.samples = pd.read_excel(self.data_folder + 'all_samples.csv')
        all_urls = [ast.literal_eval(urls) for urls in self.samples['img_urls']]
        self.img_names = [[get_img_name(url) for url in urls] for urls in all_urls]
        self.labels = list(self.samples['label'])

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
        valid_names = list(os.listdir(self.img_folder))
        samples['img_name'] = [get_img_name(url) for url in samples['img_url']]
        return samples[samples['img_name'].isin(valid_names)]

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
