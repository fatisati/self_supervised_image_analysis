class SelfSupervisedModel:

    def __init__(self, model_path):
        self.model_path = model_path
        pass

    def pretrain(self, train_ds, epochs):
        pass

    def fine_tune(self, train_ds, pretrain_models, outshape, epochs):
        pass

    def prepare_pretrain_data(self):
        pass

    def prepare_fine_tune_data(self):
        pass

    def evaluate(self, model, test_ds):
        pass
