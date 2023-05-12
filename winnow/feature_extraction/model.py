from os.path import join, dirname
import torch

# Default pretrained model path
DEFAULT_MODEL = join(dirname(__file__), "model", "model.pt")


def default_model_path(directory=None, base_url=None):
    """Get default pretrained model path."""
    return DEFAULT_MODEL

