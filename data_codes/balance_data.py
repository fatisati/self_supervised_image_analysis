import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
from data_codes.split_data import split_data


def balance_data(train_df, x_col, y_col, Sampler):
    x_train, y_train = np.array(train_df[x_col]), np.array(train_df[y_col])
    sampler = Sampler(random_state=42)

    X_res, y_res = sampler.fit_resample(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    print(f"Training target statistics: {Counter(y_res)}")

    balanced_df = pd.DataFrame({x_col: list(X_res[:, 0]), y_col: list(y_res)})
    return balanced_df


if __name__ == '__main__':
    isic = '../../data/ISIC/2020/'
    df = pd.read_csv(isic + 'stratify-split.csv')
    # tumor = df[df['group'] == 'tumor']
    train = df[df['split'] == 'train']

    train_splited = split_data(train, 'benign_malignant', 'image_name', 0.1)
    train_splited = train_splited[train_splited['split'] == 'train']
    print('10 percent train size: ', len(train_splited))

    balanced_df = balance_data(train_splited, 'image_name', 'benign_malignant', RandomOverSampler)
    balanced_df.to_csv(isic + 'ten-percent-train-oversampled.csv')
