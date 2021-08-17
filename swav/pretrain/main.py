import swav.pretrain.pretrain_utils as pretrain_utils
from swav.pretrain.pretrain_model import *

train_ds, val_ds = pretrain_utils.get_flower_ds()
trainloaders_zipped = pretrain_utils.prepare_data(train_ds)

epochs = 10
epoch_wise_loss, models = fit_swav(trainloaders_zipped, epochs)
visualize_training(epoch_wise_loss)
save_models(models)
