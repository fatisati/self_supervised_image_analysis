import swav.utils.architecture as swav_architecture
import barlow.resnet20 as barlow_architecture
from embeddings.config import *

from my_dataset import *


def load_swav_pretrain():
    feature_backbone = swav_architecture.get_resnet_backbone()
    feature_backbone.load_weights(swav_weights)

    print('swav loaded.')
    return feature_backbone


def load_twins_pretrain(img_size):
    encoder = barlow_architecture.get_network(input_shape=(img_size, img_size, 3),
                                              hidden_dim=twins_proj_dim, use_pred=False,
                                              return_before_head=False)
    encoder.load_weights(twins_weights)
    print('twins loaded')
    return encoder


class Embedding:
    def __init__(self, img_size):
        self.swav_backbone = load_swav_pretrain()
        self.twins_backbone = load_twins_pretrain(img_size)

    def get_embeddings(self, x):
        twins_emb = self.twins_backbone.predict(x)
        swav_emb = self.swav_backbone.predict(x)
        return twins_emb, swav_emb

    def get_embedding_ds(self, ds, batch_size):
        batched_ds = ds.batch(batch_size)
        return batched_ds.map(lambda x: (self.twins_backbone(x), self.swav_backbone(x)))


if __name__ == '__main__':
    embeddings = Embedding(256)

    ds = MyDataset(data_path='../../data/ISIC/ham10000/', image_folder='resized256/',
                   label_filename='disease_labels.csv', image_col='image', data_size=15)

    x_train, x_test = ds.get_x_train_test_ds()
    new_ds = embeddings.get_embedding_ds(x_train, 32)
