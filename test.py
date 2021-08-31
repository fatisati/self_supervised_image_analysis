import numpy as np

def class_weights(y):
    class_counts = np.bincount(y)
    sum_ = sum(class_counts)
    weights = sum_ / class_counts
    return weights, 1/ weights

arr = np.array([1,1,1,0])
print(class_weights(arr))