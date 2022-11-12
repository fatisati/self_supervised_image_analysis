import os
import pandas as pd

from datasets.supervised_ds import SupervisedDs


def isic2020(image_folder, csv_path):
    label_col = 'target'
    image_col = 'image_name'
    image_names = list(os.listdir(image_folder))
    label_df = pd.read_csv(csv_path)
    labels = [label_df[label_df[image_col] == img][label_col] for img in image_names]
    return SupervisedDs(image_folder, image_names, labels)
