import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd


def oversample_data(train_df, x_col, y_col):
    x_train, y_train = np.array(train_df[x_col]), np.array(train_df[y_col])
    over_sampler = RandomOverSampler(random_state=42)

    X_res, y_res = over_sampler.fit_resample(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    print(f"Training target statistics: {Counter(y_res)}")

    balanced_df = pd.DataFrame({x_col: list(X_res[:, 0]), y_col: list(y_res)})
    return balanced_df


if __name__ == '__main__':
    razi_folder = '../../data/razi/'
    df = pd.read_csv(razi_folder + 'razi-stratify-split.csv')
    tumor = df[df['group'] == 'tumor']
    train = tumor[tumor['split'] == 'train']
    balanced_df = oversample_data(train, 'img_name', 'label')
    balanced_df.to_csv(razi_folder + 'tumor-balanced.csv')
