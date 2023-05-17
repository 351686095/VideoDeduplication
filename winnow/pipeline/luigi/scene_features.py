import logging
from typing import Collection
import numpy as np
import pandas as pd
from glob import glob
import os

import luigi

from winnow.feature_extraction.loading_utils import global_vector
from winnow.pipeline.luigi.targets import (
    PrefixFeatureTarget,
)
from winnow.pipeline.luigi.scenes import ScenesReportTask
from winnow.pipeline.luigi.frame_features import (
    FrameFeaturesTask,
)
from winnow.pipeline.luigi.platform_winnow import PipelineTask
from winnow.pipeline.pipeline_context import PipelineContext
from winnow.pipeline.progress_monitor import ProgressMonitor, BaseProgressMonitor
from winnow.storage.file_key import FileKey


class SceneFeaturesTask(PipelineTask):
    """Extract scene-level features for files with prefix."""

    prefix: str = luigi.Parameter(default=".")

    def requires(self):
        yield FrameFeaturesTask(
            config=self.config,
            prefix=self.prefix,
        )
        yield ScenesReportTask(config=self.config, prefix=self.prefix)

    def output(self) -> PrefixFeatureTarget:
        return PrefixFeatureTarget(
            prefix=self.prefix,
            coll=self.pipeline.coll,
            reprs=self.pipeline.repr_storage.scene_level,
        )

    def run(self):
        target = self.output()
        self.logger.info(
            "Starting scene-level feature extraction for %s file with prefix '%s'",
            len(target.remaining_keys),
            self.prefix,
        )

        extract_scene_level_features(
            file_keys=target.remaining_keys,
            pipeline=self.pipeline,
            progress=self.progress,
            logger=self.logger,
        )


def extract_scene_level_features(
    file_keys: Collection[FileKey],
    pipeline: PipelineContext,
    progress: BaseProgressMonitor = ProgressMonitor.NULL,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Extract scene-level features from the dataset videos."""

    # Skip step if required results already exist
    if not file_keys:
        logger.info("All required scene-level features already exist. Skipping...")
        progress.complete()
        return

    # Convert frame features to global features
    progress.scale(len(file_keys))
    frame_to_global(file_keys, pipeline, progress, logger)
    logger.info("Done scene-level feature extraction.")
    progress.complete()


def get_scene_durations(pipeline: PipelineContext):
    config = pipeline.config
    scene_metadata_path = os.path.join(config.repr.directory, "scenes__*.csv")
    # Pick most recent scene metadata path
    viable_paths = glob(scene_metadata_path)
    if len(viable_paths) == 0:
        raise Exception("Error loading scene metadata, no files of pattern '%s' found" % scene_metadata_path)
    elif len(viable_paths) == 1:
        scene_metadata_path = viable_paths[0]
    else:
        scene_metadata_path = sorted(viable_paths)[-1]
    try:
        scene_metadata = pd.read_csv(scene_metadata_path)
        scene_durations = []
        for _, row in scene_metadata.iterrows():
            scene_durations += [[row["path"], row["scene_duration_seconds"]]]
        return scene_durations
    except Exception as e:
        raise Exception("Error loading scene metadata from file '%s' due to error '%s'" % (scene_metadata_path, str(e)))


def frame_to_global(
    file_keys: Collection[FileKey],
    pipeline: PipelineContext,
    progress=ProgressMonitor.NULL,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Calculate and save scene-level feature vectors based on frame-level representation."""
    scene_durations = get_scene_durations(pipeline)
    config = pipeline.config
    # Seconds per frame
    spf = config.proc.frame_sampling
    progress.scale(len(file_keys))
    for key in file_keys:
        try:
            frame_features = pipeline.repr_storage.frame_level.read(key)

            scenes_dur = None
            for i in range(len(scene_durations)):
                filename, durs = scene_durations[i]
                if filename in key.path:
                    scenes_dur = [int(d) for d in durs.strip(']""[').split(", ") if len(d) > 0]
                    scene_durations = scene_durations[:i] + scene_durations[i + 1 :]
                    break
            if scenes_dur is None:
                raise Exception("No scene metadata available for file '%s'" % key.path)

            if len(scenes_dur) == 0:
                scene_features = [frame_features]
            else:
                scene_features = []
                for dur in scenes_dur:
                    scene_frames = dur // spf
                    scene_features += [frame_features[:scene_frames]]
                    frame_features = frame_features[scene_frames:]

            scene_representations = np.concatenate([global_vector(sf) for sf in scene_features])

            pipeline.repr_storage.scene_level.write(key, scene_representations)
        except Exception as e:
            logger.exception("Error computing scene-level features for file: %s (%s)", key, str(e))
        finally:
            progress.increase(1)
    progress.complete()
