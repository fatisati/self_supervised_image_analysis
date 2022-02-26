import ast


def get_img_name(url):
    slash_idx = url.find('/')
    return url[slash_idx + 1:]


def get_one_hot(label, all_labels: []):
    label_idx = all_labels.index(label)
    one_hot = [0] * len(all_labels)
    one_hot[label_idx] = 1
    return one_hot


def get_valid_names(img_names, valid_names):
    return list(set(img_names).intersection(set(valid_names)))


def get_samples_valid_img_names(samples, all_valid_names):
    all_urls = [ast.literal_eval(urls) for urls in samples['img_urls']]
    img_names = [[get_img_name(url) for url in urls] for urls in all_urls]
    img_names = [get_valid_names(names, all_valid_names) for names in img_names]
    return img_names
