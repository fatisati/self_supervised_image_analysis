from barlow.resnet20 import ResNet
from barlow.barlow_finetune import *
if __name__ == '__main__':
    resnet = ResNet(True, False, '').get_network(128, hidden_dim=2048, use_pred=False,
                                      return_before_head=False)

    encoder = get_resnet_encoder(resnet, False)
    encoder.summary()

    model = get_linear_model(encoder, 128, 7, False, True)
    model.summary()
    model.compile()
