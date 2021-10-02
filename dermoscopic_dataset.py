import os

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

import random


def split_train_test(input_img_paths, target_img_paths):
    val_samples = 1000
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths


def get_img_path(data_path, image_folder, mask_folder, train_size):
    targets = []
    for file in os.listdir(data_path + mask_folder):
        if 'streaks' in file:
            targets.append(data_path + mask_folder + file)

    def append_image_folder(img): return data_path + image_folder + img

    input_imgs = [append_image_folder(img) for img in os.listdir(data_path + image_folder)]
    return sorted(input_imgs)[:train_size], sorted(targets)[:train_size], \
           sorted(input_imgs)[train_size:], sorted(targets)[train_size:]


class DermoscopicImage(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y
