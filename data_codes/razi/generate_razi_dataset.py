import ast
import enum
import os
import time
import shutil

from data_codes.razi.razi_db import RaziDb
import data_codes.razi.razi_cols as razi_cols
import pandas as pd

from utils import data_utils
from data_codes.razi.razi_utils import *

razi_folder = '../../../data/razi/'


class RaziSample:
    def __init__(self, img_urls, study, pid, disease_dic, img_types, date, valid_labels):
        self.img_urls = img_urls
        self.disease_dic = disease_dic
        self.valid_labels = valid_labels

        self.disease_id = self.get_disease_id(study)
        self.label = self.get_disease_label(self.disease_id)
        self.pid = self.get_id(pid)

        self.img_types = img_types
        self.date = date

    def get_disease_id(self, study):
        if not study:
            return -1
        disease_id = self.disease_dic[self.disease_dic['disease'] == study.lower()]['disease ID']
        if len(disease_id) == 0:
            print(f'{study} not found in dic')
            return -1
        disease_id = disease_id.iloc[0]
        if type(disease_id) == type(1):
            print(f'{study}: int id -> {disease_id}')
            return disease_id
        return disease_id.lower()

    def get_disease_label(self, disease_id):
        if disease_id in self.valid_labels:
            return disease_id
        return 'others'

    def get_id(self, pid):
        pid = pid.replace(' ', '')
        if len(pid) == 0:
            return -1
        return pid

    def get_dict(self):
        return {'pid': self.pid, 'label': self.label, 'disease_id': self.disease_id, 'img_urls': self.img_urls,
                'date': self.date, 'img_types': self.img_types}


class RaziDataset:
    def __init__(self, save_path):
        self.db = RaziDb()
        self.save_path = save_path

    def generate_samples_from_db(self):
        samples = []
        all_disease_ids = []
        all_labels = []
        df = self.db.read_table()
        disease_dict = pd.read_excel('../../data/razi/disease_ids.xlsx').fillna(-1)
        disease_dict['disease'] = [l.lower() for l in disease_dict['disease']]
        valid_labels = pd.read_excel('../../data/razi/all_ids_report.xlsx')
        valid_labels = valid_labels[valid_labels['count'] > 90]['label']
        valid_labels = [label.lower() for label in valid_labels]

        for i, g in df.groupby(razi_cols.pid):
            img_types = list(g[razi_cols.img_type])

            g = g[g[razi_cols.img_type] == razi_cols.micro_type]

            dates = list(g[razi_cols.shooting_date])
            if len(g) == 0:
                continue
            img_urls = [path + name for path, name in zip(g[razi_cols.img_path], g[razi_cols.img_name])]
            first_row = g.iloc[0]
            pid, study = first_row[razi_cols.pid], first_row[razi_cols.study]
            sample = RaziSample(img_urls, study, pid, disease_dict, img_types, dates, valid_labels)

            if sample.pid != -1 and sample.disease_id != -1:
                all_disease_ids.append(sample.disease_id)
                all_labels.append(sample.label)
                samples.append(sample)
        return samples, all_disease_ids, all_labels

    def get_label_report(self, all_labels, name):
        labels = set(all_labels)
        print(len(all_labels), len(labels))
        labels_report = [{'label': label, 'count': all_labels.count(label)} for label in labels]
        pd.DataFrame(labels_report).to_excel(f'{self.save_path}/{name}.xlsx')

    def save_samples(self, samples):
        all_samples = [sample.get_dict() for sample in samples]
        pd.DataFrame(all_samples).to_excel(f'{self.save_path}/all_samples.xlsx')


def print_time(st):
    print(f'done took {time.time() - st}')


def generate_supervised_samples():
    samples = pd.read_excel(razi_folder + 'all_samples.xlsx')
    supervised_samples = []
    for _, row in samples.iterrows():
        if row['label'] in ['unk', 'other']:
            continue
        urls = ast.literal_eval(row['img_urls'])
        for url in urls:
            supervised_samples.append({'label': row['label'], 'img_url': url, 'group': row['group']})
    print(f'supervised_samples cnt: {len(supervised_samples)}')
    supervised_samples = pd.DataFrame(supervised_samples)
    # is_train = data_utils.get_train_test_idx(len(supervised_samples), 0.2)
    # supervised_samples['is_train'] = is_train
    supervised_samples.to_excel(razi_folder + 'supervised_samples_grouped.xlsx')

def process_all_samples():
    samples = pd.read_excel(razi_folder + 'all_samples.xlsx')
    print(f'samples initial size: {len(samples)}')

    valid_names = os.listdir(razi_folder + 'imgs/')
    samples['img_names'] = get_samples_valid_img_names(samples, list(valid_names))
    samples['valid_cnt'] = [len(img_names) for img_names in samples['img_names']]
    samples = samples[samples['valid_cnt'] > 0]
    print(f'valid samples size: {len(samples)}')

    is_train = data_utils.get_train_test_idx(len(samples), 0.2)
    samples['is_train'] = is_train
    samples.to_excel(razi_folder + 'all_samples_processed.xlsx')


def move_imgs_to_label_folders(root_folder):
    os.mkdir(root_folder)
    samples = pd.read_excel(razi_folder + 'all_samples.xlsx')
    all_ids = list(set(samples['disease_id']))
    all_ids = [x.replace('/', '_') for x in all_ids]
    for id_ in all_ids:
        try:
            os.mkdir(root_folder + id_)
        except:
            print(id_)
    for _, row in samples.iterrows():
        names = ast.literal_eval(row['img_names'])
        sample_id = row['disease_id'].replace('/', '_')
        for name in names:
            try:
                shutil.copy(razi_folder + 'imgs/' + name, f'{root_folder}{sample_id}/{name}')
            except:
                print(sample_id)


def copy_remained_imgs():
    samples = pd.read_excel(razi_folder + 'all_samples.xlsx')
    labels = ['ev ', 'conderomatits ', 'trichotilomai ']
    print(set(samples['disease_id']).intersection(set(labels)))
    samples = samples[samples['disease_id'].isin(labels)]
    print(len(samples))
    for _, row in samples.iterrows():
        names = ast.literal_eval(row['img_names'])
        row_id = row['disease_id'].replace(' ', '')
        for img in names:
            try:
                shutil.copy(razi_folder + 'imgs/' + img, razi_folder + f'grouped_imgs/{row_id}/{img}')
            except Exception as e:
                print(e, row_id)


def set_sample_label_group(data_path):
    all_samples = pd.read_excel(data_path + 'all_samples.xlsx')
    label_set = pd.read_excel(data_path + 'label_set.xlsx')
    all_samples['group'] = [-1]*len(all_samples)
    for col in label_set.columns:
        group_idx = all_samples['label'].isin(label_set[col])
        print(group_idx.sum())
        all_samples.loc[group_idx, 'group'] = [col]*group_idx.sum()
    all_samples.to_excel(data_path + 'all_samples_grouped.xlsx')

if __name__ == '__main__':
    generate_supervised_samples()
    # set_sample_label_group('../../../data/razi/')
    # copy_remained_imgs()
    # ds = RaziDataset('../../data/razi')
    # print('reading samples from db...')
    # st = time.time()
    # samples, all_ids, all_labels = ds.generate_samples_from_db()
    # print_time(st)
    #
    # print('label report...')
    # st = time.time()
    # ds.get_label_report(all_labels, 'supervised_labels_report')
    # print_time(st)
    #
    # print('saving samples...')
    # ds.save_samples(samples)
