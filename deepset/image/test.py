from data import *

img, label = load_mnist()
img, X, sum_X = gen_train_data(img, label)
print(X[0], sum_X[0])