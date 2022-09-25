import pandas as pd
import shutil
import os


def prepare_isic2020(img_folder, split_path, isic_path):
    os.mkdir(isic_path)
    os.mkdir(isic_path + 'train')
    os.mkdir(isic_path + 'test')

    split = pd.read_csv(split_path)

    for row in split.iterrows():
        src_path = img_folder + row['image_name']
        dest_path = f'{isic_path}/{row["split"]}/{row["image_name"]}'
        shutil.move(src_path, dest_path)
