
import numpy as np
import warnings
import logging

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

    import torch


class CNN_pt:
    def __init__(self, name, model_path):
        """
        Class initializer.

        Args:
          name: dead parameter for compatibility
          model_path: path to model file of the pre-trained CNN model

        Raise:
          ValueError: if provided network name is not provided
        """
        # create the CNN network
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def to(self, other):
        return self.model.to(other)

    def extract(self, inputs):
        """
        Function that extracts intermediate CNN features for
        each input image.

        Args:
          image_tensor: numpy tensor of input images
          batch_sz: batch size

        Returns:
          features: extracted features from each input image
        """
        with torch.no_grad():
            return self.model(inputs)



