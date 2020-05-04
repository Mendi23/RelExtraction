import pickle
from collections import deque
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from sklearn import svm

import ass4.parsers as parsers
from ass4.features_funcs import Features
from ass4utils.hashing import MagicHash
from ass4utils.measuretime import measure

_exclude_features = """first_ent_child_lemma_pos
first_ent_child_word
second_ent_child_dependency
second_ent_parent_lemma_pos
second_ent_parent_dependency""".split('\n')


class RelClassifier:
    def __init__(self, excludeFeatures=None):
        self.excludeFeatures = excludeFeatures if excludeFeatures is not None else _exclude_features
        self.features_dict = MagicHash()
        self.clf = svm.LinearSVC()

    def _extractFeatures(self, args: parsers.ArgTuple):
        return Features(args, self.excludeFeatures).items()

    def _featuresToIndexes(self, features):
        for feature in features:
            index = self.features_dict.get(feature, None)
            if index is not None:
                yield index

    @measure
    def transform(self, argsIter: Iterable[parsers.ArgTuple]):
        features_indices = deque()
        features_indptr = deque()
        for arg in argsIter:
            features = self._featuresToIndexes(self._extractFeatures(arg))

            features_indptr.append(len(features_indices))
            features_indices += features

        features_indptr.append(len(features_indices))

        features_data = np.ones(len(features_indices))
        return sp.csr_matrix((features_data, features_indices, features_indptr),
            shape=(len(features_indptr) - 1, len(self.features_dict)))

    def train(self, dataIter: Iterable[parsers.DataType]):
        X = self.transform((x.args for x in dataIter))
        Y = [x.tag for x in dataIter]
        self.features_dict.freeze()
        self.clf.fit(X, Y)

    def predict(self, dataIter: Iterable[parsers.DataType]):
        X = self.transform((x.args for x in dataIter))
        return self.clf.predict(X)

    def save(self, filePath):
        pickle.dump(self, open(filePath, 'wb'))

    @classmethod
    def load(cls, filePath):
        ret = pickle.load(open(filePath, 'rb'))
        return ret
