import pyodbc
import pandas as pd


class RaziDbReport:
    def __init__(self):
        self.data_path = '../../data/razi/'
        self.conn = pyodbc.connect('Driver={SQL Server};'
                                   'Server=FATEMEH;'
                                   'Database=FotoFinder.Universe;'
                                   'Trusted_Connection=yes;')
        self.res_path = './results/'
        self.cursor = self.conn.cursor()

    def count_all_disease_patients_by_id(self):
        # [PatientRecordnumber], PatientStudy
        self.cursor.execute('SELECT PatientStudy, count(PatientStudy) from MasterTable group by PatientStudy')

        print(self.cursor)
        mapping_df = pd.read_excel(self.data_path + 'all_disease.xlsx', index_col=0).fillna('')
        df = []

        disease_dic = {}

        def add_dic(dic, k, v):
            if k not in dic:
                dic[k] = 0
            dic[k] += v

        all_dis = []
        for i in range(len(mapping_df)):
            try:
                all_dis.append(mapping_df.iloc[i]['disease'].lower())
            except:
                all_dis.append('-')
                print('no lower', mapping_df.iloc[i]['disease'])

        mapping_df['disease'] = all_dis

        for row in self.cursor:
            try:
                disease_id = mapping_df[mapping_df['disease'] == row[0].lower()]['disease ID'].values
                if len(disease_id) > 1:
                    print('founded more than once', len(disease_id), disease_id)
                disease_id = disease_id[0]
                add_dic(disease_dic, disease_id, int(row[1]))
            except:
                print('not founded', row[0])

        for dis in disease_dic:
            df.append({'disease id': dis, 'count': disease_dic[dis]})

        pd.DataFrame(df).to_excel(self.res_path + 'razi_processed_disease.xlsx')

    def patient_cnt(self):
        query = 'select count(PatientRecordnumber) from MasterTable'
        self.cursor.execute(query)
        for row in self.cursor:
            return row[0]

    def image_cnt(self):
        query = 'select count(*) from ImagesInfoTable'
        self.cursor.execute(query)
        for row in self.cursor:
            print(row)
            return row[0]

    def class_count(self):
        query = 'SELECT count(*) from MasterTable group by PatientStudy'
        self.cursor.execute(query)
        for row in self.cursor:
            print('class count', row)
            return row[0]

    def razi_db_report(self):
        df = [{'image count': self.image_cnt(), 'patient count': self.patient_cnt(),
               'class count (before process)': self.class_count()}]
        pd.DataFrame(df).to_excel(self.res_path + 'razi_db_report.xlsx')

