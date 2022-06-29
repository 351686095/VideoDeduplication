import logging
import os

import numpy as np

from .siamese_net import DNN

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))
similarity_model_pretrained = os.path.join(package_directory, "model")


class SimilarityModel:
    def __init__(self):
        self.model = None

    def predict(self, file_feature_dict):
        """
        Args:
            file_feature_dict: A dictionary mapping from original (path,hash)
            to video-level feature tensor.
        """
        # Get array of (path,hash) and array of corresponding feature
        # values in the same order
        keys, features = ([], [])
        for key, feats in file_feature_dict.items():
            for i in range(feats.shape[0]):
                keys += [key]
                features += [feats[i]]
        features = np.array(features)
        embeddings = self.predict_from_features(features)
        n_keys, n_feats = ([], [])
        while len(keys) > 0:
            assert len(keys) == len(embeddings)
            key = keys[0]
            inds = []
            for i in range(len(keys)):
                if keys[i] == key:
                    inds += [i]
            feats = np.array([embeddings[i] for i in inds])
            n_keys += [key]
            n_feats += [feats]
            keys = [keys[i] for i in range(len(keys)) if not (i in inds)]
            embeddings = [embeddings[i] for i in range(len(embeddings)) if not (i in inds)]

        return dict(zip(n_keys, n_feats))

    def predict_from_features(self, features):

        # Create model
        if self.model is None:
            logger.info("Creating similarity model for shape %s", features.shape)
            self.model = DNN(features.shape[1], None, similarity_model_pretrained, load_model=True, trainable=False)

        embeddings = self.model.embeddings(features)
        embeddings = np.nan_to_num(embeddings)
        return embeddings
