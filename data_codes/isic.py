import cv2
import numpy as np
import os

import zipfile
from io import BytesIO
from PIL import Image
import imghdr
import pandas as pd

import matplotlib.pyplot as plt

HEIGHT = 255
WIDTH = 255


def read_img(img_path):
    return Image.open(img_path)


def is_image(filename):
    if filename[-3:] == 'jpg':
        return True
    return False


def get_name(zip_file_name):
    return zip_file_name.split('/')[-1]


def read_zip(src_path, dst_path):
    imgzip = open(src_path, 'rb')
    z = zipfile.ZipFile(imgzip)

    cnt = 0
    avg_height = 0
    avg_width = 0

    for file in z.namelist():
        if is_image(file):
            data = z.read(file)

            dataEnc = BytesIO(data)
            img = Image.open(dataEnc)
            height, width = img.size
            avg_height += height
            avg_width += width

            img = img.resize((255, 255))

            img.save(dst_path + get_name(file))
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt, avg_height, avg_width)
    print(avg_height / cnt, avg_width / cnt)


class Labels:
    def __init__(self, data_path):
        dis_labels_path = data_path + 'ISIC2018_Task3_Training_GroundTruth.csv'
        derm_features_path = data_path + 'labels.csv'
        self.derm_features_df = pd.read_csv(derm_features_path)
        self.dis_label_df = pd.read_csv(dis_labels_path)

    def get_disease_label(self, isic_id):
        return self.find_row(self.dis_label_df, 'image', isic_id)

    def get_derm_features(self, isic_id):
        return self.find_row(self.derm_features_df, 'image', isic_id)

    @staticmethod
    def find_row(df, base_col, val):
        return df[df[base_col] == val].drop([base_col], axis=1).values


data_path = '../../data/ISIC/'
res_path = './results/'


def derm_features_report():
    df = pd.read_excel(data_path + 'derm_features.xlsx', index_col=0).drop(['img_id'], axis=1)
    derm_features_stats = df.sum(axis=0)
    plt.bar(x=df.columns, height=list(derm_features_stats))
    plt.title('number of class samples in ISIC dermoscopic features data')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.show()

    pd.DataFrame(derm_features_stats).to_excel(res_path + 'derm_features_stats.xlsx')


def disease_df_report():
    df = pd.read_csv(data_path + 'disease_labels.csv').drop(['image'], axis=1)
    stats = df.sum(axis=0)
    plt.bar(x=df.columns, height=list(stats))
    stats = pd.DataFrame(stats)
    stats.to_excel(res_path + 'disease_stats.xlsx')
    plt.title('number of class samples in ISIC 10000 disease data')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.show()


def preprocess_disease_labels_df(filepath):
    df = pd.read_csv(filepath)

    image_names = []
    for i in range(len(df)):
        image_names.append(df.iloc[i]['image'] + '.jpg')
    df['image'] = image_names
    df.to_csv(filepath)


if __name__ == '__main__':
    data_path = '../../data/ISIC/'
    preprocess_disease_labels_df(data_path + 'ham10000/disease_labels.csv')
    # res_path = './results/'
    # # disease_df_report()
    # derm_features_report()
