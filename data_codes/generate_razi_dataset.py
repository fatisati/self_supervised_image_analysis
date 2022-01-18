import ast
import enum
import time

from data_codes.razi.razi_db import RaziDb
import data_codes.razi.razi_cols as razi_cols
import pandas as pd

from utils import data_utils

razi_folder = '../../data/razi/'


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
        urls = ast.literal_eval(row['img_urls'])
        for url in urls:
            supervised_samples.append({'label': row['label'], 'img_url': url})
    print(f'supervised_samples cnt: {len(supervised_samples)}')
    supervised_samples = pd.DataFrame(supervised_samples)
    is_train = data_utils.get_train_test_idx(len(supervised_samples), 0.2)
    supervised_samples['is_train'] = is_train
    supervised_samples.to_csv(razi_folder + 'supervised_samples.csv')


if __name__ == '__main__':
    generate_supervised_samples()
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
