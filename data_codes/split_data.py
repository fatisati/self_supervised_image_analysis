import os

from sklearn.model_selection import train_test_split


def identify_train_test(img, test_imgs):
    if img in test_imgs:
        return 'test'
    return 'train'


def split_data(df, label_col, id_col, train_ratio):
    print('splitting data...')
    train, test = train_test_split(df, train_size=train_ratio, stratify=df[label_col])

    df['split'] = [identify_train_test(img, list(test[id_col])) for img in df[id_col]]
    return df


import pandas as pd

if __name__ == '__main__':
    # isic_folder = '../../data/ISIC/ham10000/'
    # df = pd.read_csv(isic_folder + 'HAM10000_metadata.csv')
    # splited = split_data(df, 'dx', 'image_id', 0.8)
    # splited.to_csv(isic_folder + 'HAM10000_metadata.csv')

    data_folder = '../../data/ISIC/ham10000/'
    data = pd.read_csv(data_folder + 'HAM10000_metadata.csv')
    # train = tumor[tumor['split'] == 'train']
    splited_train = split_data(data, 'dx', 'image_id', 0.8)
    splited_train['image_id'] = [img + '.jpg' for img in splited_train['image_id']]
    splited_train.to_csv(data_folder + 'stratify-split.csv')
