from sklearn.metrics import classification_report

# y_true = [0, 1, 2, 2, 2]
# y_pred = [0, 0, 2, 2, 1]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))

y_pred = [[1, 0], [0, 1], [0, 1]]
y_true = [[1, 0], [1, 0], [0,1]]


res = classification_report(y_true, y_pred, target_names=['class1', 'class2'], output_dict=True)
print(res['weighted avg']['precision'])
