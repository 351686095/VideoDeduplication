"""This module offers functions to find duplicates among the nearest neighbors."""
import logging
from typing import Any, Sequence, Iterable, Optional

from dataclasses import dataclass

from .annoy_neighbors import AnnoyNNeighbors
from ..pipeline.luigi.utils import FileKeyDF
from ..pipeline.progress_monitor import BaseProgressMonitor, ProgressMonitor


@dataclass(frozen=True)
class FeatureVector:
    """
    FeatureVector is a single point in a dataset for neighbor detection.

    The `key` attribute may be anything that associates a feature-vector with the corresponding object in the real world
    (e.g. it could be an entity id in a database, or filepath of the corresponding file, or any other natural key).
    """

    key: Any
    features: Sequence[float]


@dataclass(frozen=True)
class DetectedMatch:
    """Match represents a pair of close feature-vectors."""

    needle_key: Any
    haystack_key: Any
    distance: float


class NeighborMatcher:
    """NeighborMatcher is a low-level feature-vector duplicate detector based on nearest neighbor evaluation.

    NeighborMatcher takes a (potentially larger) set of feature-vectors (a haystack) and builds a reusable index based
    on that vectors. Then it allows to search near-duplicates for any set of feature-vectors (needles) among the
    haystack vectors.

    The NeighborMatcher is a low-level meaning it knows nothing about the origin and the semantics of the
    feature-vectors it operates on.
    """

    logger = logging.getLogger(__name__)

    @staticmethod
    def load(
        index_path,
        keys_path,
        max_matches=20,
        metric="euclidean",
        progress: BaseProgressMonitor = ProgressMonitor.NULL,
    ) -> "NeighborMatcher":
        """Load NeighborMatcher from files."""
        progress.scale(1.0)

        NeighborMatcher.logger.info("Loading NeighborMatcher")
        model = AnnoyNNeighbors.load(index_path, n_neighbors=max_matches, metric=metric)
        progress.increase(0.1)
        NeighborMatcher.logger.info("Loaded annoy index from %s", index_path)

        NeighborMatcher.logger.info("Loading file keys.")
        file_keys_df = FileKeyDF.read_csv(keys_path)
        progress.increase(0.1)
        file_keys = FileKeyDF.to_file_keys(file_keys_df, progress.remaining())
        NeighborMatcher.logger.info("Loaded file %s keys", len(file_keys))

        matcher = NeighborMatcher(
            haystack=[],
            metric=metric,
            max_matches=max_matches,
        )
        matcher._model = model
        matcher._haystack_keys = file_keys
        return matcher

    def __init__(
        self,
        haystack: Iterable[FeatureVector],
        max_matches=20,
        metric="euclidean",
    ):
        self._haystack = tuple(haystack)
        self._max_matches = max_matches
        self._metric = metric
        self._model: Optional[AnnoyNNeighbors] = None
        self._haystack_keys = None

    @property
    def model(self) -> AnnoyNNeighbors:
        if self._model is None:
            neighbors = min(self._max_matches, len(self._haystack))
            nearest_neighbors = AnnoyNNeighbors(n_neighbors=neighbors, metric=self._metric)
            nearest_neighbors.fit([vector.features for vector in self._haystack])
            self._model = nearest_neighbors
        return self._model

    @property
    def haystack_keys(self) -> Sequence[Any]:
        """Get haystack keys."""
        if self._haystack_keys is None:
            self._haystack_keys = [vector.key for vector in self._haystack]
        return self._haystack_keys

    def find_matches(
        self,
        needles: Iterable[FeatureVector],
        max_distance: float = None,
    ) -> Sequence[DetectedMatch]:
        """Find close matches of needle-vectors among haystack-vectors set."""
        needles = tuple(needles)
        distances, indices = self.model.kneighbors([needle.features for needle in needles])

        if max_distance is not None:
            distances, indices = self._filter_results(distances, indices, threshold=max_distance)

        sorted_results = sorted(zip(needles, indices, distances), key=lambda entry: len(entry[1]), reverse=True)
        seen = set()
        results = []

        for needle, haystack_indices, distances in sorted_results:
            for haystack_index, distance in zip(haystack_indices, distances):
                # haystack_item = self.haystack_keys[haystack_index]
                haystack_key = self.haystack_keys[haystack_index]
                # Skip duplicates
                if (needle.key, haystack_key) in seen:
                    continue
                # Skip self-matches
                if needle.key == haystack_key:
                    continue
                results.append(DetectedMatch(needle_key=needle.key, haystack_key=haystack_key, distance=distance))
                seen.add((needle.key, haystack_key))
                seen.add((haystack_key, needle.key))

        return results

    def _filter_results(self, distances, indices, threshold):
        results_indices = []
        results_distances = []
        mask = distances < threshold
        for i, row in enumerate(mask):
            results_indices.append(indices[i, row])
            results_distances.append(distances[i, row])
        return results_distances, results_indices
