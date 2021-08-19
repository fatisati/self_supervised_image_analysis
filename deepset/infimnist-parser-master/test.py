from infimnist_parser import convert

from infimnist_parser import convert

data = convert('../data/test10k-labels', '../data/test10k-patterns',
               save=True)


def split_digits(digits):
    split_idx = int(len(digits) / 2)
    return digits[:split_idx], digits[split_idx:]


def split_part(part1, label_part1):
    part2 = []
    label_part2 = []
    for i in range(len(part1)):
        splited = split_digits(part1[i])
        splited_labels = split_digits(label_part1[i])

        for arr in splited:
            part2.append(arr)
        for label_arr in splited_labels:
            label_part2.append(label_arr)
    return part2, label_part2


part1 = split_digits(data[0])
label_part1 = split_digits(data[1])

part2, label_part2 = split_part(part1, label_part1)
part3, label_part3 = split_part(part2, label_part2)

import pickle as pkl

for i in range(len(part3)):
    f = open('mnist8m_' + str(i) + '_features.pkl', 'wb')
    pkl.dump(part3[i], f)

    f = open('mnist8m_' + str(i) + '_labels.pkl', 'wb')
    pkl.dump(label_part3[i], f)
