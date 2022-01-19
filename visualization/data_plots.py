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
        self.labels = pd.read_csv(labels_path, index_col=0).drop(['image', 'is_train'], axis=1)
        self.vis_utils = VisUtils('../../results/data-plots/')

    def plot_disease_dist(self):
        bar_data = prepare_bar_data(self.labels)
        plt.bar(list(bar_data.keys()), list(bar_data.values()))
        plt.title('Disease sample count in HAM10000 dataset')
        plt.xlabel('Disease')
        plt.ylabel('Number of samples')
        self.vis_utils.save_and_show('ham10000_dist_bar')


class Razi:
    def __init__(self):
        self.data_folder = '../../data/razi/'
        self.all_ids = pd.read_excel(self.data_folder + 'all_ids_report.xlsx')
        self.vis_utils = VisUtils('../../results/data-plots/')

    def all_ids_bar(self):
        all_ids_filtered = self.all_ids[self.all_ids['count'] > 90]
        plt.bar(list(all_ids_filtered['label']), list(all_ids_filtered['count']))
        plt.xlabel('Disease')
        plt.ylabel('Number of patients')
        plt.title('Disease sample count in razi dataset')
        self.vis_utils.save_and_show('razi_disease_id_bar')

    def all_ids_hist(self):
        all_ids_filtered = self.all_ids[self.all_ids['count'] > 10]
        plt.hist(list(all_ids_filtered['count']))
        self.vis_utils.save_and_show('razi_ids_hist')
if __name__ == '__main__':
    razi = Razi()
    razi.all_ids_bar()
