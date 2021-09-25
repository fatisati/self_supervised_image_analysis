import tensorflow as tf
from ham_dataset import *
from barlow.data_utils import *
from barlow.barlow_finetune import *
path = '../models/twins/'
barlow_encoder = tf.keras.models.load_model(path + 'adama_pretrain_projdim2048_bs16_ct128_e50')
print(barlow_encoder.summary())

ds = MyDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/', data_size=100, balanced=True)

x, _ = ds.get_x_train_test_ds()
x = x.map(lambda x: tf.image.resize(x, (128, 128))).batch(32)
model = get_linear_model(barlow_encoder, 128, 7)

print(model.predict(x))
# train, test = ds.get_supervised_ds()
# train, test = prepare_supervised_data_loader(train, test, 32, 128)
# print(backbone.predict(train))