import os

import pandas as pd

if __name__ == '__main__':
    log_folder = '../../results/model_logs/ham/'
    for file in os.listdir(log_folder):
        if file[-3:] != 'csv':
            continue
        df = pd.read_csv(log_folder + file)

        best_row = df['val_categorical_accuracy'].argmax()
        print(file, best_row, len(df),f'best acc: {df.iloc[best_row]["val_categorical_accuracy"]}',
              f'prc: {df.iloc[best_row]["val_prc"]}')

