import matplotlib.pyplot as plt

from self_supervised_model import *
from swav.pretrain import pretrain_utils, pretrain_model

from swav.fine_tune.fine_tune import *

from swav.config_ import *
from my_dataset import MyDataset


class PretrainParams:
    def __init__(self, epochs, size_crops, batch_size, model_path):
        self.epochs = epochs
        self.size_crops = size_crops
        self.batch_size = batch_size
        self.model_path = model_path

    def get_model_path(self):
        return self.model_path + 'fbackbone_' + self.get_summary() + '.h5', \
               self.model_path + 'proj_' + self.get_summary() + '.h5'

    def get_summary(self):
        return 'e{0}_crops{1}_b{2}'.format(self.epochs, self.size_crops, self.batch_size)


class FineTuneParams:
    def __init__(self, warmup_epochs, fine_tune_epochs, model_path, bs, crop_to):
        self.warmup_epochs, self.fine_tune_epochs = warmup_epochs, fine_tune_epochs
        self.model_path = model_path
        self.batch_size = bs
        self.crop_to = crop_to

    def get_summary(self):
        return 'we{0}_e{1}_imgsize{2}'.format(self.warmup_epochs, self.fine_tune_epochs, self.crop_to)


class SwAVModel(SelfSupervisedModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.feature_name = 'features_'
        self.prototypes_name = 'prototypes_'
        self.model_path = model_path

    def get_models_path(self, epochs, end):
        end = str(epochs) + '_' + end + '.h5'
        return self.model_path + self.feature_name + end, \
               self.model_path + self.prototypes_name + end

    def pretrain(self, train_ds, params: PretrainParams):
        print('preparing data...')
        trainloaders_zipped = pretrain_utils.prepare_data(train_ds, params.size_crops, params.batch_size)
        print('done.')
        epoch_wise_loss, models = pretrain_model.fit_swav(trainloaders_zipped, params.epochs)
        fpath, ppath = params.get_model_path()  # self.get_models_path(params.epochs, pretrain_end_name)

        pretrain_model.save_models(models, fpath, ppath)
        plt.plot(epoch_wise_loss)
        plt.savefig(params.model_path + 'figures/pretrain_' + params.get_summary())
        # pretrain_model.visualize_training(epoch_wise_loss)
        return models

    def fine_tune(self, train_ds, test_ds, pretrain_models, outshape, params: FineTuneParams):
        # warmup_epochs, fine_tune_epochs = epochs
        training_ds = prepare_fine_tune_data(train_ds, params.batch_size, params.crop_to)
        testing_ds = test_ds.batch(params.batch_size)

        feature_backbone, prot = pretrain_models
        ft = FineTune(feature_backbone)
        warmup_model, history = ft.warm_up(training_ds, params.crop_to, outshape, params.warmup_epochs)
        fine_tuned_model, history = ft.fine_tune_model(training_ds, testing_ds, params.crop_to, params.fine_tune_epochs,
                                                       warmup_model)

        plt.plot(history.history['loss'])
        plt.title('fine-tune loss')
        plt.savefig(params.model_path + 'figures/' + params.get_summary() + '.png')

        fine_tuned_model.save(params.model_path + 'fine_tuned_' + params.get_summary())
        # augmented_training_ds = prepare_fine_tune_augment_data(train_ds)
        # ft.warm_up(augmented_training_ds, testing_ds)
        # ft.fine_tune_model(augmented_training_ds, testing_ds)

        return fine_tuned_model, warmup_model

    def evaluate(self, model, test_ds):
        super().evaluate()


def run_pretrain(ds: MyDataset, model_path, params):
    train_ds, test_ds = ds.get_x_train_test_ds()
    swav = SwAVModel(model_path)
    # epochs, size_crops, batch_size = 2, [32, 64], 32
    # params = PretrainParams(epochs, size_crops, batch_size)
    pretrain_models = swav.pretrain(train_ds, params)
    return pretrain_models


def run_fine_tune(ds: MyDataset, pretrain_params: PretrainParams, fine_tune_params):
    swav = SwAVModel(pretrain_params.model_path)

    train_ds, test_ds = ds.get_supervised_ds()
    fpath, ppath = pretrain_params.get_model_path()
    pretrain_models = pretrain_model.load_models(fpath, ppath)
    outshape = ds.train_labels.shape[-1]
    fine_tuned_model, warmup_model = swav.fine_tune(train_ds, test_ds, pretrain_models, outshape, fine_tune_params)
    return fine_tuned_model, warmup_model


if __name__ == '__main__':
    train_ds, test_ds = ''
    outshape = 7
    swav = SwAVModel('models/swav/')
    pretrain_models = swav.pretrain(train_ds)
    fine_tuned_model, warmup_model = swav.fine_tune(train_ds, pretrain_models, outshape)
    swav.evaluate(fine_tuned_model, test_ds)
