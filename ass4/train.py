"""
python train.py data/Corpus.TRAIN.txt data/TRAIN.annotations
"""

import sys

from ass4.model import RelClassifier
from ass4.parsers import CorpusParser


def main(trainCorpusFile, trainAnnotationsFile):
    trainData = CorpusParser(trainCorpusFile, trainAnnotationsFile)

    model = RelClassifier()
    model.train(trainData.data)

    # print("Total features:", len(model.features_dict))
    model.save("model")


if __name__ == "__main__":
    main(*sys.argv[1:3])
