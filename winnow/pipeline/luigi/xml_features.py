import logging
import multiprocessing
from typing import Collection

import luigi
import numpy as np
import tensorflow as tf

from winnow.feature_extraction import IntermediateCnnExtractor
from winnow.pipeline.luigi.targets import (
    PrefixFeatureTarget,
    PathListFileFeatureTarget,
    PathListFeatureTarget,
)
from winnow.pipeline.luigi.platform import PipelineTask
from winnow.pipeline.pipeline_context import PipelineContext
from winnow.pipeline.progress_monitor import ProgressMonitor, BaseProgressMonitor
from winnow.storage.file_key import FileKey
from winnow.feature_extraction.loading_utils import global_vector

from winnow.utils.scene_extraction import detect_scenes


class VideoProcessingTask(PipelineTask):
    """Extract __?__ for files with prefix."""

    prefix: str = luigi.Parameter(default=".")
    batch_size: int = luigi.Parameter(default=512)

    def output(self) -> PrefixFeatureTarget:
        return PrefixFeatureTarget(
            prefix=self.prefix,
            coll=self.pipeline.coll,
            reprs=self.pipeline.repr_storage.features,
        )

    def run(self):
        target = self.output()
        self.logger.info(
            "Starting feature processing for %s files with prefix '%s'",
            len(target.remaining_keys),
            self.prefix,
        )

        tf.compat.v1.reset_default_graph()

        extract_frame_level_features(
            file_keys=target.remaining_keys,
            pipeline=self.pipeline,
            progress=self.progress,
            logger=self.logger,
            batch_size=self.batch_size
        )


def process_features(file_key: FileKey, frames_features: np.ndarray):
    data = {}
    data["scene_data"] = None
    data["scene_features"] = None
    frames_features = np.reshape(frames_features, (1, -1))
    data["video_features"] = frames_features
    return data


def extract_frame_level_features(
    file_keys: Collection[FileKey],
    pipeline: PipelineContext,
    progress: BaseProgressMonitor = ProgressMonitor.NULL,
    logger: logging.Logger = logging.getLogger(__name__),
    batch_size: int = 256,
):
    """Extract frame-level features from dataset videos."""

    config = pipeline.config

    # Skip step if required results already exist
    if not file_keys:
        logger.info("All required data already exist. Skipping...")
        progress.complete()
        return

    progress.scale(total_work=len(file_keys), unit="files")

    def process_wrapper(file_key: FileKey, frames_features: np.ndarray):
        """Handle features extracted from a single video file."""
        data_dict = process_features(file_key, frames_features)
        pipeline.repr_storage.features.write(file_key, data_dict)
        progress.increase(1)

    logger.info("Initializing IntermediateCnnExtractor")
    extractor = IntermediateCnnExtractor(
        video_paths=[pipeline.coll.local_fs_path(key) for key in file_keys],
        video_ids=file_keys,
        on_extracted=process_wrapper,
        frame_sampling=config.proc.frame_sampling,
        model=pipeline.pretrained_model,
    )

    logger.info("Extracting features.")
    extractor.extract_features(batch_size=batch_size, cores=multiprocessing.cpu_count())
    logger.info("Done feature extraction.")
    progress.complete()
