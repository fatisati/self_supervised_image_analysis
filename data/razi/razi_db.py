import pyodbc
import pandas as pd
import os
import shutil


class RaziDb:
    def __init__(self):
        self.conn = pyodbc.connect('Driver={SQL Server};'
                                   'Server=FATEMEH;'
                                   'Database=FotoFinder.Universe;'
                                   'Trusted_Connection=yes;')

        self.cursor = self.conn.cursor()
        self.image_path = 'D:\\Fatemeh\\razi data\\Backup\\Images\\Images\\'
        self.res_path = './results/'

    def get_image_list(self, label):
        query = 'SELECT ImagePath, ImageName, PatientRecordnumber ' \
                'FROM ImagesInfoTable ' \
                'INNER JOIN MasterTable ' \
                'ON ImagesInfoTable.ObjectID = MasterTable.ObjectID ' \
                "WHERE PatientStudy=" "'" + label + "'"
        print(query)
        self.cursor.execute(query)
        image_list = list(self.cursor)
        print(len(image_list))
        return image_list

    def cast_image_path(self, path):
        return self.image_path + path.split('\\')[-1] + '/'

    @staticmethod
    def make_folder(src, folder):
        if folder in os.listdir(src):
            return
        os.mkdir(src + folder)

    def generate_image_folder(self, image_list, folder_name):
        folder_path = self.res_path + folder_name
        os.mkdir(folder_path)
        folder_path += '/'
        print(folder_path)
        for img in image_list:
            src_path = self.cast_image_path(img[0]) + img[1]
            patient_id = img[2]
            self.make_folder(folder_path, patient_id)
            dst_path = folder_path + patient_id + '/' + img[1]
            shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    db = RaziDb()
    image_list = db.get_image_list('mm')
    db.generate_image_folder(image_list, 'mm')
