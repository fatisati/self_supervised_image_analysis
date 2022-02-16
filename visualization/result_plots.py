import os

import matplotlib.pyplot as plt
import pandas as pd
from visualization.vis_utils import VisUtils


# link for 0 f1 if precision and recall both 0:
# https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec#:~:text=In%20each%20case%20where%20TP,predict%20any%20correct%20positive%20result.
def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def make_folder_if_not(path, folder):
    if folder not in os.listdir(path):
        os.mkdir(path + '/' + folder)
    else:
        print('folder existed.')


def calc_log_f1_score(log):
    precisions = log['val_precision']
    recalls = log['val_recall']
    f1 = [f1_score(precision, recall) for precision, recall in zip(precisions, recalls)]
    return f1


class BarlowResVis:
    def __init__(self, model_name1, model_name2, label1, label2, experiment):
        self.res_folder = '../../results/barlow-results/'
        self.model_folder = '../../models/twins/finetune/'

        self.model_path1 = self.model_folder + model_name1
        self.model_path2 = self.model_folder + model_name2
        self.model_log1 = pd.read_csv(self.model_path1 + 'log.csv')
        self.model_log2 = pd.read_csv(self.model_path2 + 'log.csv')

        self.label1 = label1
        self.label2 = label2

        self.experiment = experiment

        vis_utils = VisUtils(self.res_folder)
        self.save_and_show = vis_utils.save_and_show

    def plot_f1_score(self):
        model1_f1 = calc_log_f1_score(self.model_log1)
        model2_f1 = calc_log_f1_score(self.model_log2)
        plt.plot(model1_f1, label=self.label1)
        plt.plot(model2_f1, label=self.label2)
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.title(self.experiment)
        make_folder_if_not(self.res_folder, self.experiment)
        self.save_and_show(f'{self.experiment}/f1_score')

    def precision_recall(self):
        plt.plot(self.model_log1['val_recall'], self.model_log1['val_precision'], 'x', label=self.label1)
        plt.plot(self.model_log2['val_recall'], self.model_log2['val_precision'], '.', label=self.label2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title(self.experiment)
        make_folder_if_not(self.res_folder, self.experiment)
        self.save_and_show(f'{self.experiment}/precision_recall')

    def plot_metric(self, metric):
        plt.plot(self.model_log1[metric], label=self.label1)
        plt.plot(self.model_log2[metric], label=self.label2)
        plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.title(self.experiment)
        plt.legend()
        make_folder_if_not(self.res_folder, self.experiment)
        self.save_and_show(f'{self.experiment}/{metric}')


def compare_to_best_model(model_name, best_model_label, model_label, experiment):
    best_model = 'dropout0.2_ct128_bs64_aug_tf/'
    vis = BarlowResVis(best_model, model_name, best_model_label, model_label, experiment)
    vis.plot_f1_score()
    vis.precision_recall()
    vis.plot_metric('val_accuracy')
    vis.plot_metric('val_auc')
    vis.plot_metric('val_prc')


def batchnorm_effect(ct):
    bs = 64
    model1 = f'batchnorm/batchnorm_ct{ct}_bs{bs}_aug_tf/'
    model2 = f'batchnorm/no-batchnorm_ct{ct}_bs{bs}_aug_tf/'
    vis = BarlowResVis(model1, model2, 'batchnorm', 'no-batchnorm', f'batchnorm_ct{ct}')
    vis.plot_f1_score()
    vis.precision_recall()


def dropout_effect():
    best_model = f'batchnorm/batchnorm_ct{128}_bs{64}_aug_tf/'
    dropout_model = 'dropout0.2_ct128_bs64_aug_tf/'
    vis = BarlowResVis(best_model, dropout_model, 'no-dropout', 'dropout p=0.2', f'dropout_p0.2')
    vis.plot_f1_score()
    vis.precision_recall()
    vis.plot_metric('val_accuracy')
    vis.plot_metric('val_auc')
    vis.plot_metric('val_prc')


def aug_effect():
    orig_aug = 'ct128_bs64_aug_original/'
    best_model = f'batchnorm/batchnorm_ct{128}_bs{64}_aug_tf/'
    vis = BarlowResVis(best_model, orig_aug, 'our augmentation', 'original augmentation', f'augmentation')

    compare_to_best_model(orig_aug, 'our augmentation', 'original augmentation', 'augmentation')


def pretrain_effect():
    no_pretrain = 'no-pretrain_ct128_loss_normal/'
    compare_to_best_model(no_pretrain, 'pretrain', 'no-pretrain', 'no-pretrain')


def weighted_loss():
    weighted = 'weighted-loss_batchnorm_ct128_bs64_aug_tf/'
    compare_to_best_model(weighted, 'binary crossentropy', 'weighted loss', 'weighted_loss')


# def img_size_effect():
#     model_names = []
#     for img_size in [32, 64, 128]:
#         model_names.append(f'batchnorm/batchnorm_ct{img_size}_bs{64}_aug_tf/')
#     metrics = []

if __name__ == '__main__':
    weighted_loss()
