import datetime
import logging
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from tqdm import tqdm

from winnow.pipeline.progress_monitor import BaseProgressMonitor, ProgressMonitor
from winnow.storage.base_repr_storage import BaseReprStorage
from winnow.storage.file_key import FileKey
from winnow.utils.scene_detection import *

logger = logging.getLogger(__name__)

def detect_scenes(frame_level_features, minimum_duration=0):
    if frame_level_features.shape[0] > minimum_duration:
        diffs = cosine_series(frame_level_features)
        scene_ident = ((diffs > np.quantile(diffs, upper_thresh)) & (diffs > min_dif))
        idxs = np.array(list(range(len(scene_ident))))[scene_ident]
        if 0 not in idxs:
            idxs = [0] + list(idxs)
        if (len(frame_level_features) - 1) not in idxs:
            idxs = list(idxs) + [len(frame_level_features) - 1]
        scenes = [(start, end) for start, end in zip(idxs[:-1], idxs[1:])]
    else:
        scenes = [(0, len(frame_level_features) - 1)]

    results = {}
    results["scene_duration_seconds"] = get_duration(scenes)
    results["scenes_timestamp"] = seconds_to_time(results["scene_duration_seconds"])
    results["num_scenes"] = len(scenes)
    results["avg_duration_seconds"] = np.mean(results["scene_duration_seconds"])
    results["video_duration_seconds"] = np.sum(results["scene_duration_seconds"])
    results["total_video_duration_timestamp"] = datetime.timedelta(seconds=results["video_duration_seconds"])

    scene_features = np.array([frame_level_features[start, end] for start, end in scenes])

    return results, scene_features
