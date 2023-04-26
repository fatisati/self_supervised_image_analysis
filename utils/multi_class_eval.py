from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from datasets.ham_dataset import HAMDataset


def eval_model(model, x, y):
    y_pred = model.predict(x)
    return eval_res(y, y_pred)


def eval_res(y, y_pred):
    report = classification_report(y, y_pred, output_dict=True)
    macro = report['macro avg']
    return macro['precision'], macro['recall'], macro['f1-score']


if __name__ == '__main__':
    model = ''
    ds = HAMDataset(data_path='../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                    image_col='image', image_folder='resized256/')
    x_train, x_test = ds.get_x_train_test_ds()
    y_test = ds.test_labels

    x_test = x_test.batch(64)
    eval_model(model, x_test, y_test)
