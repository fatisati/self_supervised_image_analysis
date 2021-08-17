import sklearn.metrics as skm


def print_scores(y_true, y_pred):
    cm = skm.multilabel_confusion_matrix(y_true, y_pred)
    print(cm)
    print(skm.classification_report(y_true, y_pred))

