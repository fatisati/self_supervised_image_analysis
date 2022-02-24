import pandas as pd


def clean_log(path, name):
    log_df = pd.read_csv(path + name + '.csv')
    log_df['epoch_diff'] = log_df['epoch'].diff()
    log_df = log_df[log_df['epoch_diff'] != 0]
    log_df.to_csv(path + name + '.csv')


if __name__ == '__main__':
    for ct in [128]:
        path = f'../../models/twins/finetune/dropout0.2_ct128_bs{64}_aug_tf/'

        name = 'log'
        clean_log(path, name)
