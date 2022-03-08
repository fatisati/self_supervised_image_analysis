import tensorflow as tf
import keras.backend as K


# https://stackoverflow.com/questions/59305514/tensorflow-how-to-use-tf-keras-metrics-in-multiclass-classification

class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

        self.cat_true_positives.assign_add(true_poss)

    def result(self):
        return self.cat_true_positives


def get_multi_class_metrics(num_class, batch_size):
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        CategoricalTruePositives(num_class, batch_size),
    ]
    return METRICS


from sklearn.metrics import classification_report

def weighted_recall(y_true, y_pred):
    y_pred = y_pred.numpy().round()
    print(y_pred, y_true.numpy())
    res = classification_report(y_true.numpy(), y_pred, output_dict=True)
    return res['weighted avg']['recall']


def weighted_precision(y_true, y_pred): pass


def macro_precision(): pass


def macro_recall(): pass

def get_metircs():
    metrics = ['accuracy', f1_score, precision_func, recall_func]
    return metrics
