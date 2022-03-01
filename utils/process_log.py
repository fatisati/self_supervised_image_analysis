import os

import pandas as pd


def clean_log(path, name):
    log_df = pd.read_csv(path + name + '.csv')
    log_df['epoch_diff'] = log_df['epoch'].diff()
    log_df = log_df[log_df['epoch_diff'] != 0]
    log_df.to_csv(path + name + '.csv')


if __name__ == '__main__':
    for ct in [128]:
        path = f'../../models/twins/finetune/razi/'
        clean_log(path+'tumor/', 'ham-pretrain-train0.8')
        # for group in os.listdir(path):
        #     for model in os.listdir(path + group):
        #         try:
        #             clean_log(path+'/'+group+'/'+model+'/' , 'log')
        #         except Exception as e:
        #             print(model, e)
