import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_img(path, size):
    img = cv2.imread(path)
    return cv2.resize(img, (size, size))

def plot_img_mask(img_path, mask_path, size=255):
    img = load_img(img_path, size)
    mask = load_img(mask_path, size)

    cv2.imshow(img+0.5*mask)
    cv2.imshow(img)
    plt.show()

if __name__ == '__main__':
    data_path = '../../data/ISIC/dermoscopic/'
    input_path = data_path + 'resized255/'
    mask_path = data_path + 'ISIC2018_Task2_Training_GroundTruth_v3/'
    sample_name = list(os.listdir(input_path))[0]
    sample_mask = list(os.listdir(mask_path))[4]
    print(sample_name, sample_mask)
    plot_img_mask(input_path + sample_name, mask_path + sample_mask)
