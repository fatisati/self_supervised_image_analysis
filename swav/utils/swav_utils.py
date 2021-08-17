# import tensorflow_datasets as tfds
import swav.utils.architecture as architecture

# def download_flower_ds(supervised=False, split=["train[:85%]", "train[85%:]"]):
#     # Gather Flowers dataset
#     ds = tfds.load(
#         "tf_flowers",
#         split=split,
#         as_supervised=supervised
#     )
#     return ds


def load_feature_backbone(feature_backbone_weights):
    feature_backbone = architecture.get_resnet_backbone()
    # load trained weights
    feature_backbone.load_weights(feature_backbone_weights)
    return feature_backbone