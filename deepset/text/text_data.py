from text_params_ import *

def gen_text_train_data():
    X = np.zeros((num_train_examples, max_train_length))
    sum_X = np.zeros((num_train_examples))
    for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
        n = np.random.randint(1, max_train_length)
        for j in range(1, n + 1):
            X[i, -j] = np.random.randint(1, 10)
        sum_X[i] = np.sum(X[i])
    return X, sum_X


def gen_text_test_data(num_examples, length):
    Y = np.zeros((num_examples, length))
    sum_Y = np.zeros((num_examples))
    for i in range(num_examples):
        for j in range(1, length + 1):
            Y[i, -j] = np.random.randint(1, 10)
        sum_Y[i] = np.sum(Y[i])
    return Y, sum_Y
