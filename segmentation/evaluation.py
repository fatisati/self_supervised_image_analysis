import os
import random

import numpy as np
import cv2
from copy import deepcopy


class SegmentationVisualization:
    def __init__(self, size):
        self.pattern_names = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
        self.data_path = '../data/ISIC/dermoscopic/'
        self.input_path = self.data_path + 'resized255/'
        self.mask_path = self.data_path + 'ISIC2018_Task2_Training_GroundTruth_v3/'
        self.model_path = '../models/segmentation/'
        self.size = size
        self.sample_names = ['ISIC_0011356.jpg', 'ISIC_0012099.jpg', 'ISIC_0012092.jpg',
                             'ISIC_0012221.jpg', 'ISIC_0012313.jpg', 'ISIC_0013026.jpg', 'ISIC_0013031.jpg',
                             'ISIC_0013048.jpg']

    def cv2_load_img(self, path):
        img = cv2.imread(path)
        return cv2.resize(img, (self.size, self.size))

    @staticmethod
    def get_masked_img(img, mask):
        masked = cv2.addWeighted(img, 1, mask, 0.2, 0)

        both = np.concatenate((masked, mask), axis=0)

        return both

    def get_all_masks_img(self, img, masks):
        masked_imgs = [self.get_masked_img(img, mask) for mask in masks]
        final_img = np.concatenate((masked_imgs[0], masked_imgs[1]), axis=1)
        for i in range(2, len(masked_imgs)):
            final_img = np.concatenate((final_img, masked_imgs[i]), axis=1)

        return final_img

    @staticmethod
    def show_img(img):
        cv2.imshow('', img)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        # closing all open windows
        cv2.destroyAllWindows()

    def get_sample_image(self, n=5):
        all_imgs = os.listdir(self.input_path)
        samples = random.choices(all_imgs, k=n)
        return samples

    def get_mask_names(self, name):
        dot_idx = name.find('.')
        img_id = name[:dot_idx]
        return [f'{img_id}_attribute_{pattern}.png' for pattern in self.pattern_names]

    @staticmethod
    def save_img(name, img):
        cv2.imwrite(name, img)

    @staticmethod
    def convert_to_rgb(mask):
        dim = deepcopy(mask)
        rgb = np.stack((mask, dim, dim), axis=2)
        return rgb

    def process_predicted_mask(self, mask):
        mask = mask.round()
        mask = mask * 255
        mask = np.asarray(mask, np.uint8)
        return self.convert_to_rgb(mask)

    def load_img_and_masks(self, img_names):
        sample_mask_names = [self.get_mask_names(sample_name) for sample_name in img_names]
        sample_imgs = [self.cv2_load_img(self.input_path + sample_name) for sample_name in img_names]
        sample_masks = [[self.cv2_load_img(self.mask_path + mask_name) for mask_name in sample_mask] for
                        sample_mask in
                        sample_mask_names]
        return np.array(sample_imgs), np.array(sample_masks)

    def get_sample_prediction(self, model, sample_img_names):
        sample_imgs, sample_masks = self.load_img_and_masks(sample_img_names)
        print(f'sample-imgs shape: {sample_imgs.shape}')
        predicted_masks = model.predict(sample_imgs)
        print(f'prection done. out shape {predicted_masks.shape}')
        return sample_imgs, sample_masks, predicted_masks

    def visualize(self, sample_imgs, sample_masks, predicted_masks):
        corrected_masks = [[self.process_predicted_mask(out_mask[:, :, idx]) for idx in range(len(self.pattern_names))]
                           for
                           out_mask in predicted_masks]
        predicted_masked_imgs = [self.get_all_masks_img(img, masks) for
                                 img, masks in zip(sample_imgs, corrected_masks)]

        original_masked_images = [self.get_all_masks_img(img, masks) for
                                  img, masks in zip(sample_imgs, sample_masks)]

        final_imgs = [np.concatenate((original, predicted), axis=0) for original, predicted in
                      zip(original_masked_images, predicted_masked_imgs)]
        return final_imgs

    def visualize_samples(self, model, save_img_path):
        sample_imgs, sample_masks, predicted_masks = self.get_sample_prediction(model, self.sample_names)
        final_imgs = self.visualize(sample_imgs, sample_masks, predicted_masks)
        idx = 0
        for img in final_imgs:
            self.save_img(f'{save_img_path}/sample{idx}.png', img)
            idx += 1
