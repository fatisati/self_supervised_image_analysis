import os

import matplotlib.pyplot as plt
import pandas as pd
from visualization.vis_utils import VisUtils, CompareModels


class BarlowResVis:
    def __init__(self):
        self.res_folder = '../../results/barlow-results/'
        self.model_folder = '../../models/twins/finetune/'
        self.best_model = f'batchnorm/batchnorm_ct{128}_bs{64}_aug_tf/'
        self.no_bacthnorm_model = f'batchnorm/no-batchnorm_ct128_bs{64}_aug_tf/'
        self.no_pretrain_name = 'no-pretrain'
        self.aug_original = f'batchnorm_ct128_bs{64}_aug_original/'

    def batchnorm_effect(self, ct):
        bs = 64
        model1 = f'batchnorm/batchnorm_ct{ct}_bs{bs}_aug_tf/'
        model2 = f'batchnorm/no-batchnorm_ct{ct}_bs{bs}_aug_tf/'
        models = [model1, model2]
        labels = ['batchnorm', 'no-batchorm']

        cm = CompareModels(self.res_folder, self.model_folder, f'batchnorm-effect{ct}')
        cm.compare_all_metrics(models, labels)

    def no_pretrain_effect(self):
        no_pretrain_name = 'no-pretrain'
        cm = CompareModels(self.res_folder, self.model_folder, f'no-pretrain')
        labels = ['no-pretrain', 'pretrained']
        cm.compare_all_metrics([no_pretrain_name, self.best_model], labels)

    def compare_all(self):
        models = [self.best_model, self.no_bacthnorm_model,
                  self.no_pretrain_name, self.aug_original]
        labels = ['best', 'no-batchnorm', 'no-pretrain', 'augmentation-original']
        cm = CompareModels(self.res_folder, self.model_folder, f'compare-all')
        cm.compare_all_metrics(models, labels)


if __name__ == '__main__':
    bv = BarlowResVis()
    bv.compare_all()
