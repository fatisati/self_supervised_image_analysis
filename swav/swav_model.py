from self_supervised_model import *
from swav.pretrain import pretrain_utils, pretrain_model

from swav.fine_tune.fine_tune import *

from swav.config_ import *


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

    def pretrain(self, train_ds, epochs):
        print('preparing data...')
        trainloaders_zipped = pretrain_utils.prepare_data(train_ds)
        print('done.')
        epoch_wise_loss, models = pretrain_model.fit_swav(trainloaders_zipped, epochs)
        fpath, ppath = self.get_models_path(epochs, pretrain_end_name)

        pretrain_model.save_models(models, fpath, ppath)
        pretrain_model.visualize_training(epoch_wise_loss)
        return models

    def fine_tune(self, train_ds, pretrain_models, outshape, epochs):

        warmup_epochs, fine_tune_epochs = epochs
        training_ds = prepare_fine_tune_data(train_ds)
        feature_backbone, prot = pretrain_models
        ft = FineTune(feature_backbone)
        warmup_model, history = ft.warm_up(training_ds, outshape, warmup_epochs)
        fine_tuned_model, history = ft.fine_tune_model(training_ds, fine_tune_epochs)
        fine_tuned_model.save(self.model_path + 'fine_tuned_'+str(fine_tune_epochs)+'_epochs.h5')
        # augmented_training_ds = prepare_fine_tune_augment_data(train_ds)
        # ft.warm_up(augmented_training_ds, testing_ds)
        # ft.fine_tune_model(augmented_training_ds, testing_ds)

        return fine_tuned_model, warmup_model

    def evaluate(self, model, test_ds):
        super().evaluate()

if __name__ == '__main__':
    train_ds, test_ds = ''
    outshape = 7
    swav = SwAVModel('models/swav/')
    pretrain_models = swav.pretrain(train_ds)
    fine_tuned_model, warmup_model = swav.fine_tune(train_ds, pretrain_models, outshape)
    swav.evaluate(fine_tuned_model, test_ds)
