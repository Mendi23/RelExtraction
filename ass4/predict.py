"""
python predict.py data/Corpus.DEV.txt DEV.annotations.Pred data/Corpus.TRAIN.txt TRAIN.annotations.pred
"""

import sys

from ass4.model import RelClassifier
from ass4.parsers import CorpusParser, NO_CONNECTION


TARGET_TAG = "Live_In"

def main(devCorpusFile, outputFileName):

    devData = CorpusParser(devCorpusFile)

    model = RelClassifier.load("model")

    predicted = model.predict(devData.data)

    # print("Positive predictions", sum(p != NO_CONNECTION for p in predicted))

    devData.output_to_file(outputFileName, predicted, TARGET_TAG)


if __name__ == "__main__":
    assert len(sys.argv[1:]) % 2 == 0, "number of args must be even"
    for i in range(1, len(sys.argv), 2):
        main(*sys.argv[i:i+2])
