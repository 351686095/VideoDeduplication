import logging
import os
from itertools import repeat
from typing import Callable, Any, Collection, Tuple

import numpy as np
from tqdm import tqdm
import torch

from winnow.utils.multiproc import multiprocessing as mp
from .model_pt import CNN_pt
from .utils import load_video

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def process_videos(tasks: Collection) -> Collection:
    """Process a batch of images.

    Takes a list of tuple (video_path, video_id, model, frame_sampling, batch_size).
    Returns a list of tuple (video_id, frames_tensor, frame_features)
    """
    return [process_video(task) for task in tasks]


def process_video(task: Tuple[str, Any, int]) -> Tuple[Any, np.ndarray]:
    """Process a single video.

    Takes a tuple (video_path, video_id, model, frame_sampling, batch_size).
    Returns a tuple (video_id, frames_tensor, frame_features)
    """
    logger = logging.getLogger(f"{__name__}.process_video")
    video_path, video_id, frame_sampling = task
    logger.debug("Preparing frames for %s", video_path)
    frames_tensor = load_video(video_path, frame_sampling)
    logger.debug("Done preparing frames for %s", video_path)
    return (video_id, frames_tensor)


# Type hint for a function tha will be called when a particular
# file is processed: callback(path_or_id, frame_features)
OnExtractedCallback = Callable[[Any, np.ndarray], Any]


def feature_extraction_videos(
    model,
    video_paths: Collection[str],
    video_ids: Collection[Any],
    on_extracted: OnExtractedCallback,
    cores: int = 4,
    batch_sz: int = 8,
    frame_sampling: int = 1,
    logger: logging.Logger = logging.getLogger(f"{__name__}.feature_extraction_videos"),
):
    """
    Function that extracts the intermediate CNN features
    of each video in a provided video list.
    Args:
        model: CNN network
        video_paths: list of video file paths
        video_ids: list of video ids (must be of the same size as video paths).
        on_extracted: a callback receiving (file_path, frames_tensor, frames_features) to handle
            extracted file features which is invoked on each file processing finish
        cores: CPU cores for the parallel video loading
        batch_sz: batch size fed to the CNN network
        frame_sampling: Minimal distance (in sec.) between frames to be saved.
        logger: logger to be used.
    """
    file_count = len(video_paths)
    logger.info("Number of videos: %s", file_count)
    logger.info("CPU cores: %s", cores)
    logger.info("Batch size: %s", batch_sz)
    logger.info("Starting Feature Extraction Process")
    logger.info("GPU is available: %s", torch.cuda.is_available())

    tasks = list(
        zip(
            video_paths,
            video_ids,
            repeat(frame_sampling, file_count),
        )
    )
    batches = [tasks[i * batch_sz : (i + 1) * batch_sz] for i in range(len(tasks) // batch_sz + 1)]
    batches = [batch for batch in batches if len(batch) > 0]
    logger.debug(f"{len(batches)} batches")
    for batch in batches:
        logger.debug(len(batch))

    # Chunk tasks into as close as possible to 1,0000 chunks, but with chunksize bounded [1, 10] (inclusive)
    chunksize = max(min(len(tasks) // 10000, 10), 1)

    # Semaphore pattern used below from https://stackoverflow.com/questions/30448267/multiprocessing-pool-imap-unordered-with-fixed-queue-size-or-buffer
    semaphore = mp.Semaphore(cores * chunksize)

    progress_bar = iter(tqdm(range(file_count), mininterval=1.0, unit=" image"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:    
        with torch.no_grad():
            with mp.Pool(cores) as pool:
                for batch_outputs in pool.imap_unordered(
                    process_videos, semaphore_producer(semaphore, batches), chunksize=chunksize
                ):
                    logger.debug(batch_outputs)
                    if len(batch_outputs) == 0:
                        logger.warn("Unknown glyph failed to process, skipping.")
                        semaphore.release()
                        continue
                    ids, images = list(zip(*batch_outputs))
                    batch_data = torch.stack(images, dim=0)
                    logger.debug(f"Extracting features for {len(ids)} files.")
                    batch_features = model.extract(batch_data.to(device)).cpu().numpy()
                    logger.debug(batch_data.shape)
                    logger.debug(batch_features.shape)
                    logger.debug("Extracted, processing")
                    for image_id, image_features in zip(ids, [batch_features[i] for i in range(batch_features.shape[0])]):
                        logger.debug(image_id)
                        on_extracted(image_id, image_features)
                        next(progress_bar)
                    logger.debug("Processed.")
                    semaphore.release()
                    logger.debug("Released.")
    except Exception as e:
        logger.error(f"{e}")
        raise e


def semaphore_producer(semaphore: mp.Semaphore, tasks: Collection[tuple]):
    for task in tasks:
        # Reduce Semaphore by 1 or wait if 0
        semaphore.acquire()
        # Now deliver an item to the caller (pool)
        yield task


def load_featurizer(pretrained_local_path) -> CNN_pt:
    """Load pretrained model."""
    return CNN_pt("vgg", pretrained_local_path)
