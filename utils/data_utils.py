import random


def get_train_test_idx(size, test_ratio):
    test_size = int(size * test_ratio)
    # print(f'test size: {test_size}')
    is_train = [1] * size
    for i in range(test_size):
        is_train[i] = 0
    random.shuffle(is_train)
    # print(f'train size: {sum(is_train)}')
    return is_train


if __name__ == '__main__':
    print(get_train_test_idx(5, 0.2))
