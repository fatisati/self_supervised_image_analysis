from keras import activations
from vis.visualization import visualize_saliency
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np


def vis_saliency_lat_layer(model, layer_index, x, y):
    model.layers[layer_index].activation = activations.linear
    model = utils.apply_modifications(model)
    x = x.as_numpy_iterator()
    # Visualize
    for i in range(len(x)):
        input_image = x[i]
        # Get input
        input_class = np.argmax(y[i])
        # Matplotlib preparations
        fig, axes = plt.subplots(1, 2)
        # Generate visualization
        visualization = visualize_saliency(model, layer_index,
                                           filter_indices=input_class,
                                           seed_input=input_image)
        axes[0].imshow(input_image[..., 0])
        axes[0].set_title('Original image')
        axes[1].imshow(visualization)
        axes[1].set_title('Saliency map')
        fig.suptitle(f'MNIST target = {input_class}')
        plt.show()
