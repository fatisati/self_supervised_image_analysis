from tf_explain.core import GradCAM
from ham_dataset import MyDataset
from utils.model_utils import load_model
if __name__ == '__main__':
    ds_sample = MyDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                   image_col='image', image_folder='resized256/')
    train_ds, test_ds = ds_sample.get_supervised_ds_sample()

    model_path = '../../models/twins/finetune/ct64_bs128_loss_normal/e100'
    model = load_model(model_path)
    explainer = GradCAM()
    grid = explainer.explain(test_ds, model, class_index=0)

    explainer.save(grid, ".", "grad_cam.png")
