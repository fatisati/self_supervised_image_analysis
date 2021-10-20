import enum
from data_codes.razi.razi_db import RaziDb
import data_codes.razi.razi_cols as razi_cols


class RaziSample:
    def __init__(self, img_urls, study, pid):
        self.img_urls = img_urls
        self.label = self.get_label(study)
        self.pid = self.get_id(pid)

    def get_label(self, study):
        return study

    def get_id(self, pid):
        pid = pid.replace(' ', '')
        if len(pid) == 0:
            return -1


class RaziDataset:
    def __init__(self):
        self.db = RaziDb()
        df = self.db.read_table()
        self.samples = []
        for i, g in df.groupby(razi_cols.pid):
            g = g[g[razi_cols.img_type] == razi_cols.micro_type]
            if len(g) == 0:
                continue
            img_urls = [path + name for path, name in zip(g[razi_cols.img_path], g[razi_cols.img_name])]
            first_row = g.iloc[0]
            pid, study = first_row[razi_cols.pid], first_row[razi_cols.study]

            self.samples.append(RaziSample(img_urls, study, pid))

        print(self.samples[0].img_urls, self.samples[0].pid, self.samples[0].label)


if __name__ == '__main__':
    ds = RaziDataset()
