import logging
from typing import Collection

import luigi

from winnow.pipeline.luigi.platform_winnow import PipelineTask
from winnow.pipeline.luigi.targets import PrefixFeatureTarget
from winnow.pipeline.luigi.xml_features import VideoProcessingTask
from winnow.pipeline.pipeline_context import PipelineContext
from winnow.pipeline.progress_monitor import ProgressMonitor, BaseProgressMonitor
from winnow.storage.file_key import FileKey
from winnow.storage.repr_utils import bulk_read, bulk_write


class SignaturesTask(PipelineTask):
    """Extract fingerprints for files with prefix."""

    prefix: str = luigi.Parameter(default="video")
    feature_batch_size: int = luigi.Parameter(default=512)

    def requires(self):
        yield VideoProcessingTask(config=self.config, prefix=self.prefix, batch_size=self.feature_batch_size)

    def output(self) -> PrefixFeatureTarget:
        return PrefixFeatureTarget(
            prefix=self.prefix,
            coll=self.pipeline.coll,
            reprs=self.pipeline.repr_storage.signature,
        )

    def run(self):
        target = self.output()
        self.logger.info(
            "Starting fingerprint extraction for %s file with prefix '%s'",
            len(target.remaining_keys),
            self.prefix,
        )

        extract_signatures(
            file_keys=target.remaining_keys,
            pipeline=self.pipeline,
            progress=self.progress,
            logger=self.logger,
        )


def extract_signatures(
    file_keys: Collection[FileKey],
    pipeline: PipelineContext,
    progress: BaseProgressMonitor = ProgressMonitor.NULL,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Calculate and save signatures for the given files to repr-storage
    assuming the corresponding video-level features are already available.
    """

    # Skip step if required results already exist
    if not file_keys:
        logger.info("Representation storage contains all required signatures. Skipping...")
        progress.complete()
        return

    logger.info("Reading video-level features.")
    video_features = bulk_read(pipeline.repr_storage.features, select=file_keys)
    for key in video_features.keys():
        video_features[key] = video_features[key]["video_features"]
    logger.info("Loaded %s video-level features", len(video_features))

    logger.info("Saving fingerprints.")
    bulk_write(pipeline.repr_storage.signature, video_features)

    logger.info("Done signature extraction.")
    progress.complete()
