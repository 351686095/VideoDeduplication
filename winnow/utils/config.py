"""The config module offers high-level operations with pipeline configuration."""

from winnow.config import Config
from winnow.config.path import resolve_config_path, ensure_config_exists


def resolve_config(
    config_path=None,
    frame_sampling=None,
    save_frames=None,
    override_templates=None,
    templates_dir=None,
    templates_distance=None,
    templates_distance_min=None,
):
    """Resolve config from command-line arguments."""
    config_path = resolve_config_path(config_path)
    ensure_config_exists(config_path)
    config = Config.read(config_path)

    # Read from command-line arguments
    if frame_sampling is not None:
        config.proc.frame_sampling = int(frame_sampling)
    if save_frames is not None:
        config.proc.save_frames = save_frames
    if override_templates is not None:
        config.templates.override = override_templates
    if templates_dir is not None:
        config.templates.source_path = templates_dir
    if templates_distance is not None:
        config.templates.distance = templates_distance
    if templates_distance_min is not None:
        config.templates.distance_min = templates_distance_min
    return config
