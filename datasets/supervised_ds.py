from utils.tf_utils import read_tf_image
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class SupervisedDs:
    def __init__(self, image_folder, image_names, labels):
        self.image_folder = image_folder
        self.image_urls = [image_folder + name for name in image_names]
        self.labels = labels

    @staticmethod
    def process_sample(url, label):
        return read_tf_image(url, 512), label

    @staticmethod
    def augment_sample(img, label, augment_function):
        return augment_function(img), label

    def get_ds(self, aug_func, batch_size):
        urls_ds = tf.data.Dataset.from_tensor_slices(self.image_urls)
        labels_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        zip_ds = tf.data.Dataset.zip((urls_ds, labels_ds))

        aug = lambda img, label: self.augment_sample(img, label, aug_func)

        return (zip_ds.map(self.process_sample, num_parallel_calls=AUTOTUNE)).map(aug).batch(batch_size)
