import itertools
from collections import namedtuple, defaultdict

import spacy

from ass4utils.measuretime import measure

NO_CONNECTION = "_None_"
UNKNOWN_ENTITY = "UNKNOWN"

Annotation = namedtuple("Anno", "SRC LINK DEST")
DataType = namedtuple("D", "args tag")
ArgTuple = namedtuple("Arg", "ne1 ne2 data")
NlpWord_ = namedtuple("Word", "id word lemma pos tag parent dependency bio ner s")
NlpEntity_ = namedtuple("Ent", "text originalText entType root parent id start end children s")


class NlpWord(NlpWord_):
    @classmethod
    def create(cls, word):
        head_id = word.head.i + 1  # we want ids to be 1 based
        if word == word.head:  # and the ROOT to be 0.
            assert (word.dep_ == "ROOT"), word.dep_
            head_id = 0  # root

        return cls(
            word.i + 1,  # id
            word.text,  # word
            word.lemma_,  # lemma
            word.pos_,  # pos
            word.tag_,  # tag
            head_id,  # parent
            word.dep_,  # dependency
            word.ent_iob_,  # bio
            word.ent_type_,  # ner
            word,  # s
        )

    @classmethod
    def createEmpty(cls, id_, s):
        return cls(id_,
                   s,
                   s,
                   s,
                   s,
                   s,
                   s,
                   s,
                   s,
                   None
                   )


class NlpEntity(NlpEntity_):
    @classmethod
    def create(cls, sentId, entity, entType=None):
        root = entity.root
        if entType is None:
            entType = root.ent_type_

        head_id = root.head.i + 1  # we want ids to be 1 based
        if root == root.head:  # and the ROOT to be 0.
            assert (root.dep_ == "ROOT"), root.dep_
            head_id = 0  # root

        return cls(
            CorpusParser.cleanTxt(entity.text),  # text
            entity.text,  # originalText
            entType,  # entType
            root,  # root
            head_id,  # parent
            sentId,  # id
            entity.start + 1,  # start (1 based)
            entity.end,  # end (actual end is the next word but we are 1 based so end is inside the span)
            list(c.i + 1 for c in entity.subtree),  # children
            entity,  # s
        )


class CorpusParser:
    verbose = False
    nlp = spacy.load('en')

    def __init__(self, trainFile, trainAnnotFile=None):
        self.filters = []
        self.trainFile = trainFile

        self.corpus = self.readCorpus()
        self.annotations = self.readAnnotations(trainAnnotFile)

        self.data = self.createData()

        if self.verbose:
            print(f"Number of data items: {len(self.data)}")
            if len(self.annotations) > 0:
                print("\tNon unknown {}".format(
                    sum(t != NO_CONNECTION for _, t in self.data)))

    @staticmethod
    def cleanTxt(txt):
        return txt.rstrip('.')

    def readCorpus(self):
        sentences = {}
        for line in open(self.trainFile, 'r', encoding="utf8"):
            linePair = line.strip().split("\t")
            if not linePair: continue

            sentId, sent = linePair
            sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
            sentences[sentId] = sent
        return sentences

    def readAnnotations(self, filePath):
        if filePath is None:
            return {}

        annotations = defaultdict(list)
        for line in (l.strip() for l in open(filePath, 'r', encoding='utf8')):
            if not line: continue

            ID, SRC, LINK, DEST, COMMENT = line.split("\t")
            SRC = self.cleanTxt(SRC)
            DEST = self.cleanTxt(DEST)
            annotations[ID].append(Annotation(SRC, LINK, DEST))
        return annotations

    def _extractSentenceData(self, parsed):
        data = [NlpWord.createEmpty(0, '_S_')]
        for i, word in enumerate(parsed):
            data.append(NlpWord.create(word))

        data.append(NlpWord.createEmpty(len(data), '_E_'))

        return data

    def _extractEntities(self, sentId, parsed):
        entities_gen = (NlpEntity.create(sentId, entity) for entity in parsed.ents)
        entities = {entity.text: entity for entity in entities_gen}

        for chunk in parsed.noun_chunks:
            ent = NlpEntity.create(sentId, chunk, entType=UNKNOWN_ENTITY)
            if ent.text not in entities:
                entities[ent.text] = ent

        return entities.values()

    @measure
    def createData(self):
        return list(self._createData())

    def _createData(self):
        for sentId, sentence in self.corpus.items():
            parsed = CorpusParser.nlp(sentence)

            sentenceData = self._extractSentenceData(parsed)
            entities = self._extractEntities(sentId, parsed)

            for ne1, ne2 in itertools.product(entities, entities):
                if ne1.text != ne2.text:
                    isAdded = False
                    for annotation in (self.annotations.get(sentId) or []):
                        if ne1.text == annotation.SRC and ne2.text == annotation.DEST:
                            yield DataType(ArgTuple(ne1, ne2, sentenceData), annotation.LINK)
                            isAdded = True

                    if not isAdded:
                        yield DataType(ArgTuple(ne1, ne2, sentenceData), NO_CONNECTION)

    def extract_output_lines(self, predicted, targetTag):
        for i, prediction in enumerate(predicted):
            if prediction == NO_CONNECTION or \
                    (targetTag is not None and prediction != targetTag):
                continue
            ne1, ne2, _ = self.data[i].args
            yield "\t".join([ne1.id, ne1.text, prediction, ne2.text])

    def output_to_file(self, output_file_name, predicted, targetTag):
        with open(output_file_name, 'w') as output_file:
            for line in self.extract_output_lines(predicted, targetTag):
                output_file.write(line + '\n')
