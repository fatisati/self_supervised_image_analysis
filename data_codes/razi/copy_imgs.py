import time

from PIL import Image
import os
import pandas as pd

import ast


def copy_and_resize(src, dst, size):
    # Image.open() can also open other image types
    try:
        img = Image.open(src)
    except Exception as e:
        print(e)
        return
    # width, height = img.size
    # print(width, height)
    # WIDTH and HEIGHT are integers
    resized_img = img.resize((size, size))
    resized_img.save(dst)


def get_img_name(url):
    slash_idx = url.find('/')
    return url[slash_idx + 1:]


if __name__ == '__main__':
    samples = pd.read_excel('../../../data/razi/all_samples.xlsx')
    save_path = '../../../data/razi/imgs/'

    print(f'number of samples: {len(samples)}')
    st = time.time()
    # cnt = len(os.listdir(save_path))
    cnt = 0
    sample_cnt = 3600
    for urls in samples['img_urls'].iloc[3600:]:

        urls = ast.literal_eval(urls)

        for url in urls:
            name = get_img_name(url)
            if name in os.listdir(save_path):
                break
            copy_and_resize(url, f'{save_path}/{name}', 512)
        cnt += len(urls)
        if sample_cnt %10 == 0:
            print(f'sample {sample_cnt}: copied {cnt} images till now. took {time.time() - st}')
        sample_cnt += 1
