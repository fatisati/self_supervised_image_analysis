import os

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

import random


def split_train_test(input_img_path, val_ratio = 0.2):
    input_img_files = list(os.listdir(input_img_path))
    val_samples = int(val_ratio * len(input_img_files))
    random.Random(1337).shuffle(input_img_files)
    train_input_img_paths = [input_img_path + img_name for img_name in input_img_files[:-val_samples]]
    val_input_img_paths = [input_img_path + img_name for img_name in input_img_files[-val_samples:]]
    return train_input_img_paths, val_input_img_paths


def get_img_path(data_path, image_folder, train_size):
    def append_image_folder(img): return data_path + image_folder + img

    input_imgs = [append_image_folder(img) for img in os.listdir(data_path + image_folder)]

    return sorted(input_imgs)[:train_size], sorted(input_imgs)[train_size:]


class DermoscopicImage(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths,
                 target_img_folder):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_folder = target_img_folder
        self.target_class_names = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
        self.num_classes = len(self.target_class_names)

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        # batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,) , dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            y[j] = self.load_mask(path.split('/')[-1])
        # for j, path in enumerate(batch_target_img_paths):
        #     img = load_img(path, target_size=self.img_size, color_mode="grayscale")
        #     y[j] = np.expand_dims(img, 2)
        #     # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        #     # y[j] -= 1

        return x, y

    def load_mask(self, img_name):
        img_name = img_name[:-4]
        y = np.zeros(self.img_size + (self.num_classes,))
        for idx in range(len(self.target_class_names)):
            class_ = self.target_class_names[idx]
            mask_path = f'{self.target_img_folder}{img_name}_attribute_{class_}.png'
            mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            mask = np.array(mask)
            mask = np.array(mask) // 255
            y[:, :, idx]  = mask
        return y


if __name__ == '__main__':
    train_input_img_paths, val_input_img_paths = split_train_test('../../data/ISIC/dermoscopic/resized255/')
    ds = DermoscopicImage(10, (160, 160), [], '../data/ISIC/dermoscopic/ISIC2018_Task2_Training_GroundTruth_v3/')

    mask = ds.load_mask('ISIC_0011345.jpg')


