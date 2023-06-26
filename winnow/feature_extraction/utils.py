import numpy as np
import torch
from torchvision import transforms
import PIL.Image as Image


TRANSFORMS = [
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5,), (0.5,)),
]


def read_image(path, shape=(224, 224)):
    im = Image.open(path)
    im.draft("L", shape)
    return im


def load_video(video, frame_sampling) -> np.ndarray:
    """
          Function that loads a video and converts it to the desired size.

          Args:
            video: path to video
            frame_sampling: Dead parameter

          Returns:
            video_tensor: the tensor of the given video
    cfg['pretrained_model_local_path']
          Raise:
            Exception: if provided video can not be load
    """
    try:
        image = read_image(video)
        transform = transforms.Compose(TRANSFORMS)
        image = transform(image)
        return image
    except Exception as e:
        raise Exception("Can't load video {}\n{}".format(video, e))
