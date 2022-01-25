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
        self.all_ids = pd.read_excel(self.data_folder + 'reports/all_ids_report.xlsx')
        self.vis_utils = VisUtils('../../results/data-plots/razi/')

        self.all_samples = pd.read_excel(self.data_folder + 'all_samples.xlsx')
        self.label_set = pd.read_excel(self.data_folder + 'label_set.xlsx')
        self.groups = list(self.label_set.columns)

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

    def all_groups_bar(self):
        group_cnts = []
        for group in self.groups:
            group_samples = self.all_samples[self.all_samples['group'] == group]
            print(group, len(group_samples))
            group_cnts.append(len(group_samples))
        plt.bar(self.groups, group_cnts)
        plt.ylabel('Sample count')
        self.vis_utils.save_and_show('group_bar')

    def count_labels(self, df, label):
        return len(df[df['label'] == label])

    def all_groups_samples_bar(self):
        for group in self.groups:
            group_samples = self.all_samples[self.all_samples['group'] == group]
            group_labels = list(set(group_samples['label']))
            group_label_cnt = [self.count_labels(group_samples, label) for label in group_labels]

            fig_width = 0
            for l in group_labels:
                fig_width += len(l) * 0.25
                print(l, len(l), len(l) * 0.5)
            print(f'--------{fig_width}-------')
            plt.figure(figsize=(fig_width, 5))
            plt.bar(group_labels, group_label_cnt)
            plt.ylabel('Number of samples')
            plt.title(f'Labels distribution in {group} group')
            self.vis_utils.save_and_show(f'{group}')

    def general_report(self):

        report = {'sample_cnt': len(self.all_samples),
                  'valid_img_cnt': self.all_samples['valid_cnt'].sum()}
        pd.DataFrame([report]).to_excel(self.data_folder + '/reports/general_report.xlsx')


if __name__ == '__main__':
    razi = Razi()
    razi.general_report()
    # razi.all_groups_samples_bar()
