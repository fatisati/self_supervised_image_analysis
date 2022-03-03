import os

import matplotlib.pyplot as plt
import pandas as pd
from visualization.vis_utils import VisUtils, CompareModels


class BarlowResVis:
    def __init__(self):
        self.res_folder = '../../results/barlow-results/'
        #twins/finetune/
        self.model_folder = '../../models/twins/finetune/train0.1_test0.9_logs/'

        self.best_model = 'dropout0.2_ct128_bs64_aug_tf'
        self.batchnorm = 'batchnorm_ct128_bs64_aug_tf'
        self.no_bacthnorm_model = 'no-batchnorm_ct128_bs64_aug_tf'
        self.no_pretrain_name = 'no-pretrain_dropout0.2_ct128'
        self.aug_original = 'batchnorm_ct128_bs64_aug_original'
        self.drop_out = 'dropout0.2_ct128_bs64_aug_tf'
        self.irv2 = 'irv2'

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
        models = [self.drop_out, self.batchnorm, self.no_bacthnorm_model,
                  self.no_pretrain_name, self.aug_original, self.irv2]

        labels = ['dropout (p=0.2), batch-normalization', 'no-dropout, batch-normalization'
                  'no-dropout, no-batch-normalization',
                  'no-pretrain',
                  'no-dropout, batch-normalization, original-augmentation', 'IRV2']
        cm = CompareModels(self.res_folder, self.model_folder, f'Compare all models')
        cm.compare_all_metrics(models, labels)


if __name__ == '__main__':
    bv = BarlowResVis()
    bv.compare_all()
