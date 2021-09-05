from barlow_pretrain.barlow_model import *

from ham_dataset import *

ds = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
               image_col='image', image_folder='resized256/', balanced=True, data_size=15)

outshape = 7
train_ds, test_ds = ds.get_supervised_ds()
resnet_enc = resnet20.get_network(input_shape=(CROP_TO, CROP_TO, 3), hidden_dim=PROJECT_DIM, use_pred=False,
                                  return_before_head=False)
epochs = 2
STEPS_PER_EPOCH = len(train_ds) // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * epochs
WARMUP_EPOCHS = int(epochs * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
cosine_lr = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.3, decay_steps=epochs * STEPS_PER_EPOCH
)
linear_model = fine_tune(train_ds, test_ds, resnet_enc, outshape, cosine_lr,
                         'binary_crossentropy', epochs)
linear_model.save('../../models/twins/finetune_test')
