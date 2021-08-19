# from params_ import *
from params_ import *
import pickle as pkl


def load_mnist(pack=0):
    # with open('../data/parts/mnist8m_{0}_features.bin'.format(pack), 'r') as f:
    #     v = np.fromfile(f, dtype='int32', count=1)
    #     D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
    #     N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
    #     arr = np.fromfile(f).astype('float32')
    # img = np.reshape(arr, (N, D))
    #
    # with open('../data/parts/mnist8m_{0}_labels.bin'.format(pack), 'r') as f:
    #     v = np.fromfile(f, dtype='int32', count=1)
    #     D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
    #     N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
    #     arr = np.fromfile(f).astype('float32')
    # label = np.reshape(arr, (N, D))
    label = pkl.load(open('../data/parts/mnist8m_{0}_features.pkl'.format(pack), 'rb'))
    img = pkl.load(open('../data/parts/mnist8m_{0}_labels.pkl'.format(pack), 'rb'))
    return img, label


def gen_train_data(img, label):
    # Random shuffle of data
    print(num_train_examples)
    rng_state = np.random.get_state()
    np.random.shuffle(img)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

    X = np.zeros((num_train_examples, max_train_length))
    sum_X = np.zeros((num_train_examples))
    label_poiner = 0
    for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
        sample_length = np.random.randint(1, max_train_length)
        for j in range(1, sample_length + 1):
            while label[label_poiner] == 0.:
                label_poiner += 1
            X[i, -j] = label_poiner
            sum_X[i] += label[label_poiner]
            label_poiner += 1

    img = img[:label_poiner]
    return img, X, sum_X


def gen_test_data(num_examples, length):
    img, label = load_mnist(np.random.randint(1, 8))
    Y = np.zeros((num_examples, length))
    sum_Y = np.zeros((num_examples))
    m = 0
    for i in range(num_examples):
        for j in range(1, length + 1):
            while label[m] == 0.:
                m += 1
            Y[i, -j] = m
            sum_Y[i] += label[m]
            m += 1
    return img[:m], Y, sum_Y


def check_generated_data(index, X, sum_X, label):
    print('sample length: ', len(X[index]))
    sample_sum = 0
    start_idx = 0
    while X[index][start_idx] == 0:
        start_idx += 1
    for i in range(start_idx, len(X[index])):
        print('i, sample[i], label[sample[i]]: ', i, X[index][i], label[int(X[index][i])])
        sample_sum += label[int(X[index][i])]
    print('sample sum, sample sum_X', sample_sum, sum_X[index])

# def prepare_deepset_data(embeddings, labels)

if __name__ == '__main__':
    img, label = load_mnist()
    print(img.shape)
    img, X, sum_X = gen_train_data(img, label)
    print(img.shape, X.shape)
    # check_generated_data(0, X, sum_X, label)
    # check_generated_data(10, X, sum_X, label)
