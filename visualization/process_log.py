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


if __name__ == '__main__':
    for ct in [128]:
        path = f'../../models/twins/finetune/train0.1_test0.9_logs/'
        # clean_log(path+'tumor/', 'ham-pretrain-train0.8')
        for model in os.listdir(path):
            # for model in os.listdir(path + group):
            try:
                clean_log(path, model[:-4])
            except Exception as e:
                print(model, e)
