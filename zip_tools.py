import os
import zipfile

# generate a class and open zip_file in the constructor
def get_all_files_in_zip(zip_file):
    return list(zip_file.namelist())


def copy_img_from_zip(src_path, dest_path, img_name):
    zip_file = zipfile.ZipFile(src_path)
    zip_file.extract(img_name, dest_path)
def load_img_from_zip(zip_path, zip_subfolder, dest_folder,
                      img_name, load_func):
    zip_file = zipfile.ZipFile(zip_path)
    zip_file.extract(zip_subfolder + img_name, dest_folder + img_name)
    img = load_func(dest_folder + img_name)
    os.remove(dest_folder + img_name)
    return img
