import matplotlib.pyplot as plt
import pandas as pd


class BarlowResVis:
    def __init__(self):
        self.res_folder = '../../results/'
        self.res = pd.read_csv(self.res_folder + 'tables/best-model.csv')
        self.save_path = self.res_folder + 'barlow-results/'
        self.no_pretrain = pd.read_csv(self.res_folder + 'tables/no-pretrain.csv')

    def save_with_format(self, name, frm):
        plt.savefig(self.save_path + f'{frm}/{name}.{frm}')

    def save_and_show(self, name):
        self.save_with_format(name, 'svg')
        self.save_with_format(name, 'png')
        plt.show()

    def train_val_loss(self):
        plt.plot(self.res['loss'], label='train')
        plt.plot(self.res['val_loss'], label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        self.save_and_show('train_val_loss')

    def train_val_prec_recall_epoch(self):
        plt.plot(self.res['precision'], label='train precision')
        plt.plot(self.res['recall'], label='train recall')
        plt.plot(self.res['val_precision'], label='validation precision')
        plt.plot(self.res['val_recall'], label='validation recall')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('precision_recall_epoch')

    def train_val_precision_recall(self):
        plt.plot(self.res['recall'], self.res['precision'], label='train')
        plt.plot(self.res['val_recall'], self.res['val_precision'], label='validation')
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.legend()
        self.save_and_show('precision_recall')

    def no_pretrain_precision(self):
        plt.plot(self.no_pretrain['val_precision'], label = 'no-pretrain precision')
        plt.plot(self.res['val_precision'], label = 'pretrained precision')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('no_pretrain_precision')

    def no_pretrain_recall(self):
        plt.plot(self.no_pretrain['val_recall'], label = 'no-pretrain recall')
        plt.plot(self.res['val_recall'], label = 'pretrained recall')
        plt.xlabel('epoch')
        plt.legend()
        self.save_and_show('no_pretrain_recall')

    def no_pretrain_precision_recall(self):
        plt.plot(self.no_pretrain['val_recall'], self.no_pretrain['val_precision'], label = 'no-pretrain')
        plt.plot(self.res['val_recall'], self.res['val_precision'], label='pretrained')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        self.save_and_show('no_pretrain_precision_recall')
if __name__ == '__main__':
    vis = BarlowResVis()
    vis.no_pretrain_precision_recall()
