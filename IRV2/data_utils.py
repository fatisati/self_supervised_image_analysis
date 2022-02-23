import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import numpy as np


def get_preprocessing_datagen():
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
    return datagen


def duplicates(x, unique_df):
    unique = set(unique_df['lesion_id'])
    if x in unique:
        return 'no'
    else:
        return 'duplicates'


class AugmentHam:
    def __init__(self, ham_folder, target_folder, from_test_folder, train_sample_ratio):
        self.ham_folder = ham_folder
        self.train_dir = os.path.join(target_folder, 'train_dir/')
        self.test_dir = os.path.join(target_folder, 'test_dir/')
        self.target_folder = target_folder
        data_pd = pd.read_csv(ham_folder + 'HAM10000_metadata.csv')
        self.targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        if from_test_folder:
            test_img_ids = set([])
            for subdir in os.listdir(self.test_dir):
                for img in os.listdir(self.test_dir + '/' + subdir):
                    dot_idx = img.find('.')
                    test_img_ids.add(img[:dot_idx])
            # split train and test in a way that no duplicate in test data
            data_pd['train_test_split'] = data_pd['image_id'].apply(
                lambda x: self.identify_trainOrtest(x, test_img_ids))
            train_df = data_pd[data_pd['train_test_split'] == 'train']
            self.test_list = list(test_img_ids)

        else:

            train, test_df = self.generate_random_train_test(data_pd)
            data_pd['train_test_split'] = data_pd['image_id'].apply(
                lambda x: self.identify_trainOrtest(x, set(test_df['image_id'])))
            train_df = data_pd[data_pd['train_test_split'] == 'train']
            print(f'train size before sample: {len(train_df)}')

            train_size = int(len(train_df) * train_sample_ratio)
            print(f'train sample size: {train_size}')

            _, train_df = train_test_split(train_df, test_size=train_sample_ratio, stratify=train_df['dx'])
            self.test_list = list(test_df['image_id'])

        self.train_list = list(train_df['image_id'])
        print(f'train size: {len(self.train_list)}, test size: {len(self.test_list)}')
        data_pd.set_index('image_id', inplace=True)
        self.data_pd = data_pd

    def generate_random_train_test(self, data_pd):
        samples = self.get_unique_samples(data_pd)
        train, test_df = train_test_split(samples, test_size=0.2, stratify=samples['dx'])
        return train, test_df

    def copy_train_test_from_ham(self):
        os.mkdir(self.target_folder)
        self.generate_train_test_folders()
        self.copy_imgs_to_train_test_folders()

    def copy_imgs_to_train_test_folders(self):
        self.copy_imgs_to_label_subdir(self.train_list, self.ham_folder, self.train_dir)
        self.copy_imgs_to_label_subdir(self.test_list, self.ham_folder, self.test_dir)

    def copy_imgs_to_label_subdir(self, img_names, src_dir, target_dir):
        print(f'copy imgs from {src_dir} to {target_dir}...')
        for image in img_names:
            file_name = image + '.jpg'
            label = self.data_pd.loc[image, 'dx']

            # path of source image
            source = os.path.join(src_dir, file_name)

            # copying the image from the source to target file
            target = os.path.join(target_dir, label, file_name)

            shutil.copyfile(source, target)
        print('done')

    def generate_train_test_folders(self):
        os.mkdir(self.train_dir)
        os.mkdir(self.test_dir)

        for i in self.targetnames:
            directory1 = self.train_dir + '/' + i
            directory2 = self.test_dir + '/' + i
            os.mkdir(directory1)
            os.mkdir(directory2)

    def identify_trainOrtest(self, x, test_img_ids):
        if str(x) in test_img_ids:
            return 'test'
        else:
            return 'train'

    def get_unique_samples(self, data_pd):
        df_count = data_pd.groupby('lesion_id').count()

        unique_df = df_count[df_count['dx'] == 1]
        unique_df.reset_index(inplace=True)

        data_pd['is_duplicate'] = data_pd['lesion_id'].apply(lambda x: duplicates(x, unique_df))
        unique_samples = data_pd[data_pd['is_duplicate'] == 'no']
        return unique_samples

    def copy_imgs(self, src, dst):
        print(f'copy all imgs from {src} to {dst}')
        img_list = os.listdir(src)

        # Copy images from the class train dir to the img_dir
        for file_name in img_list:
            # path of source image in training directory
            source = os.path.join(src, file_name)

            # creating a target directory to send images
            target = os.path.join(dst, file_name)

            # copying the image from the source to target file
            shutil.copyfile(source, target)

    def augment_all_classes(self):
        for img_class in self.targetnames:
            self.augment_class(img_class)

    def augment_class(self, img_class):
        print(f'augmenting class {img_class}')
        # creating temporary directories
        # creating a base directory
        aug_dir = '/content/tmp_aug_dir/'
        os.mkdir(aug_dir)
        # creating a subdirectory inside the base directory for images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)
        self.copy_imgs(self.train_dir + img_class, img_dir)

        source_path = aug_dir
        save_path = self.train_dir + img_class

        batch_size = 50
        aug_datagen = self.get_aug_datagen(source_path, save_path, batch_size)
        # Generate the augmented images
        aug_images = 8000

        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((aug_images - num_files) / batch_size))

        # creating 8000 augmented images per class
        for i in range(0, num_batches):
            images, labels = next(aug_datagen)

        # delete temporary directory
        shutil.rmtree(aug_dir)
        print('done')

    def get_aug_datagen(self, src, dst, batch_size):
        # Creating Image Data Generator to augment images
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(

            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'

        )

        aug_datagen = datagen.flow_from_directory(src, save_to_dir=dst, save_format='jpg',
                                                  target_size=(299, 299), batch_size=batch_size)
        return aug_datagen


if __name__ == '__main__':
    ham_folder = ''
    target_folder = ''
    AugmentHam(ham_folder, target_folder).augment_all_classes()
