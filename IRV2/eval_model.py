from tensorflow.keras import models
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def eval_model(model, test_batches, test_size, batch_size):
    predictions = model.predict(test_batches, steps=test_size / batch_size, verbose=0)
    # geting predictions on test dataset
    y_pred = np.argmax(predictions, axis=1)
    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    # getting the true labels per image
    y_true = test_batches.classes
    # getting the predicted labels per image
    y_prob = predictions
    from tensorflow.keras.utils import to_categorical
    y_test = to_categorical(y_true)

    # Creating classification report
    report = classification_report(y_true, y_pred, target_names=targetnames)

    print("\nClassification Report:")
    print(report)

    print('---weighted result---')
    print("Precision: " + str(precision_score(y_true, y_pred, average='weighted')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='weighted')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("weighted Roc score: " + str(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')))
    print('-----------------------')

    print('---macro---')
    print("Precision: " + str(precision_score(y_true, y_pred, average='macro')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='macro')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("Macro Roc score: " + str(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')))
    print('-----------------------')

    print('---micro---')
    print("Precision: " + str(precision_score(y_true, y_pred, average='micro')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='micro')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    tpr = {}
    fpr = {}
    roc_auc = {}
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("Micro Roc score: " + str(roc_auc["micro"]))
    print('------------------------')

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(7):
        r = roc_auc_score(y_test[:, i], y_prob[:, i])
        print("The ROC AUC score of " + targetnames[i] + " is: " + str(r))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = dict()
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
