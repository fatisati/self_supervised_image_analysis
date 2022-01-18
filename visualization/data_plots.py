import matplotlib.pyplot as plt
import pandas as pd
from visualization.vis_utils import VisUtils

def prepare_bar_data(label_df):
    label_sum = {}
    for col in label_df.columns:
        label_sum[col] = label_df[col].sum()
    return label_sum


class Ham10000:
    def __init__(self):
        self.data_path = '../../data/ISIC/ham10000/'
        labels_path = self.data_path + 'disease_labels.csv'
        self.labels = pd.read_csv(labels_path, index_col= 0).drop(['image', 'is_train'], axis=1)
        self.vis_utils = VisUtils('../../results/data-plots/')

    def plot_disease_dist(self):
        bar_data = prepare_bar_data(self.labels)
        plt.bar(list(bar_data.keys()), list(bar_data.values()))
        plt.title('Disease sample count in HAM10000 dataset')
        plt.xlabel('Disease')
        plt.ylabel('Number of samples')
        self.vis_utils.save_and_show('ham10000_dist_bar')

if __name__ == '__main__':
    ham = Ham10000()
    ham.plot_disease_dist()
