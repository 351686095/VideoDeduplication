import hashlib
import logging
import os
from os import PathLike
from typing import Union

from cached_property import cached_property

from template_support.file_storage import FileStorage, LocalFileStorage
from winnow.collection.file_collection import FileCollection
from winnow.collection.local_collection import LocalFileCollection
from winnow.config import Config
from winnow.config.config import HashMode
from winnow.storage.repr_storage import ReprStorage
from winnow.utils.files import FileHashFunc, hash_path, HashCache, hash_file
from winnow.utils.repr import repr_storage_factory

logger = logging.getLogger(__name__)


class ComponentNotAvailable(Exception):
    """Error indicating component is not available."""


class PipelineContext:
    """Pipeline components created and wired consistently according to the pipeline Config."""

    TEXT_SEARCH_INDEX_NAME = "text_search_annoy_index"
    TEXT_SEARCH_DATABASE_IDS_NAME = "text_search_database_ids"
    TEXT_SEARCH_N_FEATURES = 2048

    def __init__(self, config: Config):
        """Create pipeline context."""
        self._config = config

    @cached_property
    def config(self) -> Config:
        """Get pipeline config."""
        return self._config

    @cached_property
    def repr_storage(self) -> ReprStorage:
        """Get representation storage."""
        return ReprStorage(
            directory=self.config.repr.directory,
            storage_factory=repr_storage_factory(self.config.repr.storage_type),
        )

    @cached_property
    def pretrained_model(self):
        """Load default model."""
        from winnow.feature_extraction import default_model_path, load_featurizer

        model_path = default_model_path(self.config.proc.pretrained_model_local_path)
        logger.info("Loading pretrained model from: %s", model_path)
        return load_featurizer(model_path)

    @cached_property
    def file_storage(self) -> FileStorage:
        """Create file storage for template examples."""
        return LocalFileStorage(directory=self.config.file_storage.directory)

    @cached_property
    def calculate_hash(self) -> FileHashFunc:
        """Get file hashing function."""

        if self.config.sources.hash_mode == HashMode.PATH:
            return hash_path
        if self.config.sources.hash_mode == HashMode.PATH_MTIME:
            return lambda path: hash_path(path, mtime=True)
        if self.config.sources.hash_cache is None:
            return hash_file

        # Otherwise, cache file hashes
        data_folder = self.config.sources.root
        cache_folder = self.config.sources.hash_cache
        os.makedirs(cache_folder, exist_ok=True)
        cache = HashCache(map_path=HashCache.rebase_path(data_folder, cache_folder, suffix="sha256"))

        @cache.wrap
        def calculate_hash(path: Union[str, PathLike]) -> str:
            """Calculate file hash."""
            return hash_file(path, algorithm=hashlib.sha256)

        return calculate_hash

    @cached_property
    def coll(self) -> FileCollection:
        """Get collection."""

        return LocalFileCollection(
            root_path=self.config.sources.root,
            extensions=self.config.sources.extensions,
            calculate_hash=self.calculate_hash,
        )
