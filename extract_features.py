import logging.config
import os

import click
import luigi

from winnow.pipeline.luigi.scenes import ScenesTask
from winnow.pipeline.luigi.signatures import (
    SignaturesTask,
    ScenelessSignaturesTask,
    SignaturesByPathListFileTask,
)
from winnow.utils.config import resolve_config


@click.command()
@click.option("--config", "-cp", help="path to the project config file", default=os.environ.get("WINNOW_CONFIG"))
@click.option(
    "--frame-sampling",
    "-fs",
    help=(
        "Sets the sampling strategy (values from 1 to 10 - eg "
        "sample one frame every X seconds) - overrides frame "
        "sampling from the config file"
    ),
    default=None,
)
@click.option(
    "--save-frames",
    "-sf",
    help="Whether to save the frames sampled from the videos - overrides save_frames on the config file",
    default=None,
    is_flag=True,
)
@click.option(
    "--skip_scenes",
    "-ss",
    help="Whether to process scenes.",
    default=None,
    is_flag=True,
)
def main(config, frame_sampling, save_frames, skip_scenes):
    config = resolve_config(config_path=config, frame_sampling=frame_sampling, save_frames=save_frames)

    logging.config.fileConfig("./logging.conf")
    sig_task = ScenelessSignaturesTask if skip_scenes else SignaturesTask

    luigi.build(
        [
            sig_task(config=config),
        ],
        local_scheduler=True,
        workers=1,
    )


if __name__ == "__main__":
    main()
