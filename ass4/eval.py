"""
python eval.py data/DEV.annotations DEV.annotations.Pred data/TRAIN.annotations TRAIN.annotations.Pred
"""

import sys
from collections import namedtuple, defaultdict
from sklearn import metrics as skmetrics

from ass4.predict import TARGET_TAG

Metric = namedtuple("Metric", "Precision Recall F1")

additional_data = ["_correct",
                   "_wrong",
                   "_missed"]

def parse_annotations_file(annotationFilePath):
    with open(annotationFilePath) as f:
        connections = defaultdict(set)
        for line in f:
            tokens = line.strip().split('\t')
            id, first_chunk, connection, second_chunk = tokens[:4]
            connections[connection].add(
                (id, first_chunk.rstrip('.'), second_chunk.rstrip('.')))
        return connections


def metrics(gold, predicted):
    metrics = {}
    relations = (r for r in predicted.keys() if r in gold)
    for relation in relations:
        correct = gold[relation].intersection(predicted[relation])
        wrong = predicted[relation] - correct
        missed = gold[relation] - correct
        precision = len(correct) / float(len(predicted[relation]))
        recall = len(correct) / float(len(gold[relation]))
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / float(precision + recall)
        metrics[relation] = Metric(precision, recall, f1)
        metrics[relation+"_correct"] = correct
        metrics[relation+"_wrong"] = wrong
        metrics[relation+"_missed"] = missed
    return metrics


def print_metrics(metrics):
    space = " "*4
    print(" "*15 + space.join(f"{t:^12}" for t in Metric._fields))
    for relation, metric in metrics.items():
        if TARGET_TAG is not None and relation != TARGET_TAG: continue
        if any(relation.endswith(c) for c in additional_data): continue
        details = [f"{v:.10f}" for v in metric]
        print(f"{relation:15}" + space.join(details))

        # global i
        # x = str(i)+"_"
        # for c in additional_data:
        #     open(x+relation+f"{c}.txt", "w", encoding="utf8").\
        #         write("\n".join(" + ".join(x) for x in metrics[relation+c]))


def main(goldFile, predFile):
    gold_annotations = parse_annotations_file(goldFile)
    predicted_annotations = parse_annotations_file(predFile)
    res = metrics(gold_annotations, predicted_annotations)

    print(f"evaluating {' '*4} {goldFile} -vs- {predFile}:")
    print("   -" * 14)
    print_metrics(res)
    print("-" * 70)
    print()


if __name__ == "__main__":
    assert len(sys.argv[1:]) % 2 == 0, "number of args must be even"
    for i in range(1, len(sys.argv), 2):
        main(*sys.argv[i:i + 2])
