from visualization.vis_utils import VisUtils, CompareModels


class RaziVis:
    def __init__(self, save_path, finetune_folder):
        self.finetune_folder = finetune_folder
        self.res_folder = save_path

    def compare(self, group):
        model_folder = self.finetune_folder + group + '/'
        pretrain_types = ['razi-pretrain', 'no-pretrain']
        model_names = [ptype + '-bs128' for ptype in pretrain_types]
        cp = CompareModels(self.res_folder, model_folder, group)
        cp.compare_all_metrics(model_names, model_names)


if __name__ == '__main__':
    rv = RaziVis('../../results/razi/', '../../models/twins/finetune/razi/')
    groups = ['tumor']
    for group in groups:
        rv.compare(group)
