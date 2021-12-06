from ham_dataset import MyDataset
from barlow.data_utils import prepare_data_loader
from barlow.augmentation_utils import custom_augment
from google.colab.patches import cv2_imshow
import numpy as np

import cv2


def get_ham_ds():
    ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/')
    return ds


def get_barlow_dataset(sample_size):
    ds = get_ham_ds()
    return [ds.read_tf_image(path) for path in ds.train_names[:sample_size]]


def show_augmented_img(img, crop_to=256, augment_cnt=5):
    augmented_imgs = [custom_augment(img, crop_to, '') for i in range(augment_cnt - 1)]
    augmented_imgs.append(img)
    augmented_imgs = [get_showable_img(pic) for pic in augmented_imgs]
    all_imgs = concat_all(augmented_imgs)
    cv2_imshow(all_imgs)


def get_showable_img(img):
    return cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)


def concat_all(imgs):
    if len(imgs) < 2:
        return imgs[0]
    all_imgs = np.concatenate((imgs[0], imgs[1]), axis=1)
    for i in range(2, len(imgs)):
        all_imgs = np.concatenate((all_imgs, imgs[i]), axis=1)
    return all_imgs
