from visualization.vis_utils import VisUtils, CompareModels


class RaziVis:
    def __init__(self, save_path, log_folder):
        self.log_folder = log_folder
        self.res_folder = save_path

        self.ham_pretrain = 'ham-pretrain'
        self.no_pretrain = 'no-pretrain'

        self.groups = ['hair', 'nail', 'tumor']

        self.cm = CompareModels(save_path, log_folder, 'Razi')

    def compare(self, group):
        models = [self.ham_pretrain + f'-{group}', self.no_pretrain + f'-{group}']
        labels = ['ham-pretrain', 'no-pretrain']
        self.cm.experiment = group
        self.cm.compare_all_metrics(models, labels)


if __name__ == '__main__':
    rv = RaziVis('../../results/razi/', '../../models/twins/finetune/razi-logs/')
    groups = ['tumor', 'nail', 'hair']
    for group in groups:
        rv.compare(group)
