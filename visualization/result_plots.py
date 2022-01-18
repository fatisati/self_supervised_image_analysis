import matplotlib.pyplot as plt
import pandas as pd
from visualization.vis_utils import VisUtils


# link for 0 f1 if precision and recall both 0:
# https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec#:~:text=In%20each%20case%20where%20TP,predict%20any%20correct%20positive%20result.
def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def calc_log_f1_score(log):
    precisions = log['val_precision']
    recalls = log['val_recall']
    f1 = [f1_score(precision, recall) for precision, recall in zip(precisions, recalls)]
    return f1


class BarlowResVis:
    def __init__(self):
        self.res_folder = '../../results/'
        self.model_folder = '../../models/twins/finetune/'

        self.best_model_path = self.model_folder + '10percent-tf-aug_ct128 _bs128_pretrain_e20/'
        self.no_pretrain_path = self.model_folder + '10percent-no-pretrain_ct128 _bs128_loss_normal/'
        self.best_model_log = pd.read_csv(self.best_model_path + 'log.csv')
        self.no_pretrain_log = pd.read_csv(self.no_pretrain_path + 'log.csv')

        vis_utils = VisUtils(self.res_folder + 'barlow-results/')
        self.save_and_show = vis_utils.save_and_show

    def plot_f1_score(self):
        best_model_f1 = calc_log_f1_score(self.best_model_log)
        no_pretrain_f1 = calc_log_f1_score(self.no_pretrain_log)
        plt.plot(best_model_f1, label='pretrained')
        plt.plot(no_pretrain_f1, label='no-pretrain')
        plt.xlabel('epochs')
        plt.ylabel('f1 score')
        plt.legend()
        self.save_and_show('f1_score')

    def train_val_loss(self):
        plt.plot(self.best_model_log['loss'], label='train')
        plt.plot(self.best_model_log['val_loss'], label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        self.save_and_show('train_val_loss')

    def train_val_prec_recall_epoch(self):
        plt.plot(self.best_model_log['precision'], label='train precision')
        plt.plot(self.best_model_log['recall'], label='train recall')
        plt.plot(self.best_model_log['val_precision'], label='validation precision')
        plt.plot(self.best_model_log['val_recall'], label='validation recall')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('precision_recall_epoch')

    def train_val_precision_recall(self):
        plt.plot(self.best_model_log['recall'], self.best_model_log['precision'], label='train')
        plt.plot(self.best_model_log['val_recall'], self.best_model_log['val_precision'], label='validation')
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.legend()
        self.save_and_show('precision_recall')

    def no_pretrain_precision(self):
        plt.plot(self.no_pretrain_log['val_precision'], label='no-pretrain precision')
        plt.plot(self.best_model_log['val_precision'], label='pretrained precision')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('no_pretrain_precision')

    def no_pretrain_recall(self):
        plt.plot(self.no_pretrain_log['val_recall'], label='no-pretrain recall')
        plt.plot(self.best_model_log['val_recall'], label='pretrained recall')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('no_pretrain_recall')

    def no_pretrain_precision_recall(self):
        plt.plot(self.no_pretrain_log['val_recall'], self.no_pretrain_log['val_precision'], label='no-pretrain')
        plt.plot(self.best_model_log['val_recall'], self.best_model_log['val_precision'], label='pretrained')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        self.save_and_show('no_pretrain_precision_recall')


if __name__ == '__main__':
    vis = BarlowResVis()
    vis.plot_f1_score()
