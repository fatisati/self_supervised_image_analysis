import matplotlib.pyplot as plt
import pandas as pd


class BarlowResVis:
    def __init__(self):
        self.res = pd.read_csv('../../results/tables/sample-log.csv')

    def all(self):
        vis_cols = ['val_precision', 'val_recall', 'val_auc']
        for col in vis_cols:
            plt.plot(self.res[col], label=col)
        plt.legend()
        plt.show()

    def precision_recall(self):
        plt.plot(self.res['val_precision'], self.res['val_recall'])
        plt.xlabel('val precision')
        plt.ylabel('val recall')
        plt.show()


if __name__ == '__main__':
    vis = BarlowResVis()
    vis.precision_recall()
