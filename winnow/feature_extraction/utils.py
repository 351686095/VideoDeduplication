import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image


class FloorCeil(object):
    def __call__(self, sample):
        sample_min = torch.min(sample)
        sample_max = torch.max(sample)
        threshold = (sample_min + sample_max) / 2
        sample[sample < threshold] = sample_min
        sample[sample > sample_min] = sample_max
        return sample


TRANSFORMS = [
    transforms.Resize((224, 224)),
    transforms.ConvertImageDtype(torch.float32),
    # FloorCeil(),
    transforms.Normalize((0.5,), (0.5,)),
]


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
        image = read_image(video).cpu()
        if image.shape[-3] == 3:
            transform = transforms.Compose([transforms.Grayscale()] + TRANSFORMS)
        elif image.shape[-3] == 1:
            transform = transforms.Compose(TRANSFORMS)
        else:
            raise Exception(f"Image loaded with invalid shape {image.shape}.")
        image = transform(image)
        return image
    except Exception as e:
        raise Exception("Can't load video {}\n{}".format(video, e))
