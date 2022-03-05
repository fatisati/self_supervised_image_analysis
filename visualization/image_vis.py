import random

from datasets.razi_dataset import RaziDataset

if __name__ == '__main__':
    ds = RaziDataset('../../data/', 256)
    samples = ds.prepare_multi_instance_samples()
    samples = samples[samples['img_cnt'] > 1]
    for group in ['tumor']:
        group_samples = samples[samples['group'] == group]
        group_labels = set(samples['label'])
        random_labels = random.sample(group_labels, 2)
        for label in random_labels:
            label_samples = group_samples[group_samples['label'] == label]
            random_idx = random.randint(0, len(label_samples))
            ds.plot_sample(label_samples.iloc[random_idx])
