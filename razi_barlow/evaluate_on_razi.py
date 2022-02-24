from razi_dataset import RaziDataset
from utils.model_utils import load_model
if __name__ == '__main__':
    ct, bs = 128, 64
    ds = RaziDataset('../../data/razi/', 128)
    x, y = ds.get_ham_format_x_y()
    x_batched = x.batch(64)

    model_folder = '../../models/twins/finetune/'
    model = load_model(model_folder + f'dropout0.2_ct128_bs64_aug_tf/e200')

    y_pred = model.predict(x_batched)
    print(y_pred)
