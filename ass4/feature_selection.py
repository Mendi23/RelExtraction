from collections import namedtuple

from ass4.eval import parse_annotations_file
from ass4.model import RelClassifier #,debugSet
from ass4.parsers import CorpusParser

Metric = namedtuple("Metric", "Precision Recall F1")

#exclude_list = []
temp_max = Metric(0, 0, 0)

def run_optimize(feats):
    global temp_max
    with open("selection2.res", "w", encoding="utf8")as f:
        for i in range(len(feats)):
            model = RelClassifier(feats[:i]+feats[i+1:])
            f.write(f"---checking {feats[i]}---\n")
            model.train(trainData.data)
            f.write(f"Total features: {len(model.features_dict)}\n")
            predicted = model.predict(devData.data)
            devData.output_to_file("temp.out", predicted)
            predicted_annotations = parse_annotations_file("temp.out")

            correct = gold_annotations["Live_In"].intersection(predicted_annotations["Live_In"])
            precision = len(correct) / float(len(predicted_annotations["Live_In"]))
            recall = len(correct) / float(len(gold_annotations["Live_In"]))
            f1 = 0 if precision + recall == 0 else 2 * precision * recall / float(
                precision + recall)
            f.write("results:\n")
            f.write(f"precision: {precision}     recall: {recall}     F1: {f1}\n")
            f.write("change from global:\n")
            f.write(f"precision: {precision-temp_max.prprecision}     "
            f"recall: {recall-temp_max.recall}     F1: {f1-temp_max.f1}\n")
            cur = Metric(precision, recall, f1)
            if cur.F1 > temp_max.F1-0.0001:
                f.write(f"!   removing {feats[i]}   !\n")
                temp_max = cur
                f.flush()
                run_optimize(feats[:i]+feats[i+1:])
                return
            else:
                f.flush()


if __name__ == "__main__":
    trainData = CorpusParser("data/Corpus.TRAIN.txt", "data/TRAIN.annotations")
    devData = CorpusParser("data/Corpus.DEV.txt")
    # model = RelClassifier(exclude_list)
    # model.train(trainData.data)
    # with open("feats.txt", "w", encoding="utf8")as f:
    #     f.write("\n".join(debugSet))

    with open("feats.txt", encoding="utf8")as f:
        feats = [line.strip() for line in f]

    gold_annotations = parse_annotations_file("data/DEV.annotations")
    run_optimize(feats)
