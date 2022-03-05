import swav.utils.architecture as swav_architecture
import barlow_pretrain.resnet20 as barlow_architecture
from embeddings_.config import *

from datasets.ham_dataset import *

from deepset.deepset_model import *


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
        twins_emb = self.twins_backbone(x)
        swav_emb = self.swav_backbone(x)

        return twins_emb, swav_emb

    def get_embedding_ds(self, ds):
        batched_ds = ds.batch(32)
        return batched_ds.map(lambda x, y: ((self.twins_backbone(x), self.swav_backbone(x)), y))


def deepset_hybrid(ds):
    twins_model = load_twins_pretrain(256)
    swav_model = load_swav_pretrain()
    model = get_deepset_hybrid(twins_model, swav_model)

    train_ds, test_ds = ds.get_supervised_ds()

    model.fit(train_ds.batch(32))


def deepset_embedded_input(ds):
    emb = Embedding(256)
    train_ds, test_ds = ds.get_supervised_ds()
    emb_ds = emb.get_embedding_ds(train_ds)
    print(emb_ds)

    model = get_deepset_model_embedded_input(2, 2048)
    model.fit(emb_ds)


if __name__ == '__main__':

    # embeddings = Embedding(256)

    ds = HAMDataset(data_path='../../data/ISIC/ham10000/', image_folder='resized256/',
                    label_filename='disease_labels.csv', image_col='image', data_size=15)

    deepset_hybrid(ds)
    # # deepset_embedded_input(ds)
