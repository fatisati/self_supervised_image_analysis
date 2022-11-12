import os
import zipfile
import random
import zip_
from sklearn.model_selection import train_test_split


def sample_data(zip_path, dest_path, sample_cnt):
    zip_file = zipfile.ZipFile(zip_path)
    all_imgs = [img for img in zip_file.namelist() if img.endswith('.jpg')]
    samples = random.sample(all_imgs, sample_cnt)
    zip_.copy_img_from_zip(zip_file, dest_path, samples)


def sample_test_data(image_names, labels):
    X_train, X_test, y_train, y_test = train_test_split(image_names, labels, test_size=0.33)
    return X_train, X_test, y_train, y_test
