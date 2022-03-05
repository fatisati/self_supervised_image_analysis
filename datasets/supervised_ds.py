from utils.tf_utils import read_tf_image
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class SupervisedDs:
    def __init__(self, image_folder, image_names, labels):
        self.image_folder = image_folder
        self.image_urls = [image_folder + name for name in image_names]
        self.labels = labels

    def get_ds(self, aug_func, batch_size):
        img_ds = tf.data.Dataset.from_tensor_slices(self.image_urls).map(lambda img: read_tf_image(img, 512))
        img_ds = img_ds.map(aug_func)
        labels_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        zip_ds = tf.data.Dataset.zip(img_ds, labels_ds)

        return zip_ds.batch(batch_size)
