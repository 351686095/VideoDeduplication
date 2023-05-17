import datetime
import logging
import numpy as np

from winnow.utils.scene_detection import get_duration, seconds_to_time, cosine_series

logger = logging.getLogger(__name__)


def detect_scenes(frame_level_features, minimum_duration=1, upper_thresh: float = 0.793878, min_dif: float = 0.04):
    if frame_level_features.shape[0] > minimum_duration:
        logging.debug("Extracting difference info")
        diffs = cosine_series(frame_level_features)
        logging.debug("Identifying scene markers")
        scene_ident = (diffs > np.quantile(diffs, upper_thresh)) & (diffs > min_dif)
        idxs = np.array(list(range(len(scene_ident))))[scene_ident]
        idxs = [0] + [(i + 1) for i in idxs]
        if (len(frame_level_features)) not in idxs:
            idxs = list(idxs) + [len(frame_level_features)]
        else:
            idxs[-1] = len(frame_level_features)
        scenes = [(start, end) for start, end in zip(idxs[:-1], idxs[1:])]
    else:
        scenes = [(0, len(frame_level_features))]

    logging.debug(f"scenes: {scenes}")
    results = {}
    logging.debug("Extracting scene durations")
    results["scene_duration_seconds"] = get_duration(scenes)
    logging.debug("Extracting scene timestamps")
    results["scenes_timestamp"] = seconds_to_time(results["scene_duration_seconds"])
    logging.debug("Extracting scene count")
    results["num_scenes"] = len(scenes)
    logging.debug("Extracting mean scene duration")
    results["avg_duration_seconds"] = np.mean(results["scene_duration_seconds"])
    logging.debug("Extracting video duration")
    results["video_duration_seconds"] = np.sum(results["scene_duration_seconds"])
    logging.debug("Extracting video timestamp")
    results["total_video_duration_timestamp"] = datetime.timedelta(seconds=int(results["video_duration_seconds"]))

    logging.debug("Extracting scene features")
    scene_features = [frame_level_features[start:end] for start, end in scenes]

    return results, scene_features
