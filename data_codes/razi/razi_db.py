import pyodbc
import pandas as pd
import os
import shutil
import enum
import data_codes.razi.razi_cols as razi_cols


class RaziDb:
    def __init__(self):
        self.conn = pyodbc.connect('Driver={SQL Server};'
                                   'Server=FATEMEH;'
                                   'Database=FotoFinder.Universe;'
                                   'Trusted_Connection=yes;')

        self.cursor = self.conn.cursor()
        self.image_path = 'D:\\Fatemeh\\razi data\\Backup\\Images\\Images\\'
        self.res_path = '../results/'
        self.useful_cols = ['ImagePath', 'ImageName', 'ImageTypeText', 'PatientRecordNumber',
                            'MedReference', 'MedLevel', 'PatientStudy', 'ShootingDate']

    def cast_image_path(self, path):
        return self.image_path + path.split('\\')[-1] + '/'

    def read_table(self):
        query = f'select {", ".join(self.useful_cols)} ' \
                'from ImagesInfoTable inner join MasterTable ' \
                'on ImagesInfoTable.ObjectID = MasterTable.ObjectID ' \
                'where MasterTable.Deleted != 1' \
            # 'group by PatientRecordNumber'
        print(query)
        self.cursor.execute(query)
        results = list(self.cursor)
        df = []
        for path, name, type_text, pid, med_ref, med_level, study, shooting_date in results:
            df.append({razi_cols.img_path: self.cast_image_path(path),
                       razi_cols.img_name: name,
                       razi_cols.img_type: type_text,
                       razi_cols.pid: pid, 'med-ref': med_ref, 'med-level': med_level,
                       razi_cols.study: study, razi_cols.shooting_date: shooting_date})
        df = pd.DataFrame(df)

        return df

    def save_images_with_types(self, df):
        try:
            os.mkdir(self.res_path + 'image-types')
        except Exception as e:
            print(e)

        def image_type_name(row, index):
            return f'{row["med-ref"]}_{row["med-level"]}_{index}.jpg'

        cnt = 0
        for i, g in df.groupby(['pid']):
            cnt += 1
            if cnt > 10:
                break
            if len(str(g.iloc[0]['pid']).replace(' ', '')) < 1:
                continue
            dst_folder = self.res_path + 'image-types/' + str(g.iloc[1]['pid'])
            print(f'--------{dst_folder}----------')
            try:
                os.mkdir(dst_folder)
            except:
                print('dst existed')
            for index, row in g.iterrows():
                # print(row['image-path'] + row['image-name'], image_type_name(row, index))
                shutil.copy(row['image-path'] + row['image-name'], dst_folder + '/' + image_type_name(row, index))

    def check_image_types(self):
        df = self.read_table()
        self.save_images_with_types(df)


if __name__ == '__main__':
    db = RaziDb()
    db.check_image_types()
