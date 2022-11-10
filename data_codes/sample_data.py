import zipfile
import random
import zip_


def sample_data(zip_path, dest_path, sample_cnt):
    zip_file = zipfile.ZipFile(zip_path)
    all_imgs = [img for img in zip_file.namelist() if img.endswith('.jpeg')]
    samples = random.sample(all_imgs, sample_cnt)
    zip_.copy_img_from_zip(zip_file, dest_path, samples)
