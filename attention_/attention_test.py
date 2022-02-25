from barlow.resnet20 import ResNet
from barlow.barlow_finetune import *
if __name__ == '__main__':
    encoder = ResNet(True, False, 0.2).get_network(32, n=2, hidden_dim=128, use_pred=False, return_before_head=False)
    model = get_linear_model(encoder, 32, 7, True, True)
    model.compile()
