import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import tf_utils


def identify_train_test(img, test_imgs):
    if img in test_imgs:
        return 'test'
    return 'train'


def split_data(df, label_col, id_col, train_ratio):
    train, test = train_test_split(df, train_size=train_ratio, stratify=df[label_col])

    df['split'] = [identify_train_test(img, list(test[id_col])) for img in df[id_col]]
    return df


def split_razi(data_folder):
    razi_folder = data_folder + 'razi/'
    razi_df = pd.read_excel(razi_folder + 'supervised_samples.xlsx', index_col=0)
    tumor_df = razi_df[razi_df['group'] == 'tumor']
    tumor_df = split_data(tumor_df, 'label', 'img_name', 0.8)
    tumor_df.to_csv(razi_folder + 'tumor-stratify-split.csv')


def make_razi_split_from_tumor(data_folder):
    razi_folder = data_folder + 'razi/'
    tumor_df = pd.read_excel(razi_folder + 'tumor-stratify-split.xlsx')
    razi_df = pd.read_excel(razi_folder + 'supervised_samples.xlsx', index_col=0)
    razi_df['split'] = ['train'] * len(razi_df)
    razi_df.loc[razi_df['group'] == 'tumor', 'split'] = list(tumor_df['split'])
    razi_df.to_csv(razi_folder + 'razi-stratify-split.csv')


def split_isic(data_folder):
    isic_folder = data_folder + 'ISIC/2020/'
    df = pd.read_csv(isic_folder + 'train-labels.csv')
    df = split_data(df, 'benign_malignant', 'image_name', 0.8)
    df.to_csv(isic_folder + 'stratify-split.csv')


class SslData:
    def __init__(self, img_folder, split_df, img_name_col, split='train'):
        self.train = split_df[split_df['split'] == split]
        self.img_urls = [img_folder + name for name in self.train[img_name_col]]
        print(f'train size: {len(self.train)}, test size: {len(split_df) - len(self.train)}')


class SslDataset:
    def __init__(self, all_ssl_data: [SslData]):
        self.all_urls = set([])
        for ds in all_ssl_data:
            self.all_urls = self.all_urls.union(set(ds.img_urls))

        print('number of images in ssl-dataset: ', len(self.all_urls))

    def get_ds(self, batch_size):
        urls_ds = tf.data.Dataset.from_tensor_slices(list(self.all_urls))
        return urls_ds.batch(batch_size).map(tf_utils.read_tf_image)


if __name__ == '__main__':
    data_folder = '../data/'
    make_razi_split_from_tumor(data_folder)
