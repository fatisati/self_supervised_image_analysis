import conf
from utils.model_utils import load_model


def get_model():
    return load_model(conf.model_path + 'inception')


if __name__ == '__main__':
    model = get_model()
