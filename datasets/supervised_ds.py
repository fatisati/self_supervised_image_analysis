import numpy as np

from utils.tf_utils import read_tf_image
import tensorflow as tf
from zip_ import *

AUTOTUNE = tf.data.AUTOTUNE


class SupervisedDs:
    def __init__(self, image_folder, image_names, labels):
        self.image_folder = image_folder
        self.image_names = image_names
        self.label_set = list(set(labels))
        print(self.label_set)
        self.labels = [self.one_hot(label) for label in labels]

    def one_hot(self, label):
        one_hot = [0] * len(self.label_set)
        label_idx = self.label_set.index(label)
        one_hot[label_idx] = 1
        return one_hot

    def process_sample(self, name, label, aug_func):
        img = read_tf_image(self.image_folder + name, 512)
        img = aug_func(img)
        return img, label

    def get_ds(self, aug_func, batch_size):
        # .map(lambda img: read_tf_image(img, 512))
        img_ds = tf.data.Dataset.from_tensor_slices(self.image_names)
        # img_ds = img_ds.map(aug_func)
        labels_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        ds = tf.data.Dataset.zip((img_ds, labels_ds)).map(lambda img, label: self.process_sample(img, label, aug_func))

        return ds.shuffle(1024).batch(batch_size)


class ZipDs(SupervisedDs):
    def __init__(self, zip_folder, zip_subfolder, image_names, labels):
        self.zip_folder = zip_folder
        self.zip_subfolder = zip_subfolder
        super(ZipDs, self).__init__('', image_names, labels)

    def process_sample(self, name, label, aug_func):
        read_img = lambda url: read_tf_image(url, 512)
        name_str = name.numpy()
        print(type(name_str))
        img = load_img_from_zip(self.zip_folder, self.zip_subfolder, './',
                                name_str, read_img)
        img = aug_func(img)
        return img, label
