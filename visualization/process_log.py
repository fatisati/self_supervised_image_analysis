import os

import pandas as pd


def clean_log(path, name):
    log_df = pd.read_csv(path + name + '.csv')
    print(name, len(log_df))
    log_df['epoch_diff'] = log_df['epoch'].diff()
    log_df = log_df[log_df['epoch_diff'] != 0]
    print(len(log_df))
    print('------')
    log_df.to_csv(path + name + '.csv')


import matplotlib.pyplot as plt

if __name__ == '__main__':

    log_path = '../../results/model_logs/isic2020/'
    for file in os.listdir(log_path):
        clean_log(log_path, file[:-4])
        df = pd.read_csv(log_path + file)
        plt.plot(df['val_prc'], label=file)
    plt.legend()
    plt.show()
