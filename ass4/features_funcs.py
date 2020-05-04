from itertools import filterfalse

from ass4.parsers import ArgTuple

import inspect, os.path as path

currDirName = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
with open(path.join(currDirName, "../ass4utils/places.txt"), encoding="utf8") as f:
    _places = frozenset(line.strip() for line in f)


class Features:
    def __init__(self, args: ArgTuple, ex):
        self.exclude = ex
        self.data = args.data
        self.ne1, self.ne2 = args.ne1, args.ne2
        self.feats = self.data_features()
        for entity, prefix in zip((self.ne1, self.ne2), ("first_ent", "second_ent")):
            self.feats.extend(self.entity_features(entity, prefix))
            self.feats.extend(self.dependency_features(entity, prefix))

    def entity_features(self, ne, prefix):
        return [
            prefix + "_type_tag=" + ne.entType,
            prefix + "_text=" + ne.text,
            prefix + "_prev_lemma=" + self.data[ne.start - 1].lemma,
            prefix + "_follow_pos=" + self.data[ne.end + 1].pos,
            prefix + "_word_count=" + f"{ne.end-ne.start+1}",
            prefix + "_is_upper=" + ("Y" if ne.text.isupper() else "N"),
            prefix + "_is_capital=" + ("Y" if ne.text[0].isupper() else "N"),
            # prefix + "_is_place=" + ("Y" if ne.text.replace("-", " ").lower() in _places else "N"),
            # prefix + "_follow_lemma=" + self.data[ne.end + 1].lemma,
            # prefix + "_prev_word=" + self.data[ne.start - 1].word,
            # prefix + "_follow_word=" + self.data[ne.end + 1].word,
            # prefix + "_prev_pos=" + self.data[ne.start - 1].pos,
        ]

    def data_features(self):
        dep_path = self._get_dependency_path()
        retVal = [
            "dependency_distance=" + str(len(dep_path) - 1),
            "first_ent_is_ancestor=" + ("Y" if
                                        self.ne1.root.is_ancestor(self.ne2.root)
                                        else "N"),
            "second_ent_is place=" + ("Y" if self.ne2.text.replace("-", " ").lower() in _places
                                      else "N"),
            "second_ent_prev_pos=" + self.data[self.ne2.start - 1].pos,
            # "sentence_distance=" + self._aux_distance(),
        ]

        for i in range(len(dep_path) - 1):
            retVal.extend(self._retrieve_dependency_features("dependency_edge", dep_path[i + 1]))

        # for i in range(min(self.ne1.end, self.ne2.end) + 1, max(self.ne1.start, self.ne2.start)):
        #     retVal.append("bag_of_words_sentence=" + self.data[i].lemma)

        return retVal

    def _retrieve_dependency_features(self, prefix, word_index):
        word_token = self.data[word_index]
        return [
            prefix + "_word=" + word_token.word,
            prefix + "_dependency=" + word_token.dependency,
            prefix + "_lemma_pos=" + word_token.lemma + "|" + word_token.pos,
            # prefix + "_lemma_pos_dependency=" + word_token.lemma + "|" + word_token.pos + "|" +
            # word_token.dependency,
            # prefix + "_pos=" + word_token.pos,
            # prefix + "_pos_dependency=" + word_token.pos + "|" + word_token.dependency,
        ]

    # need cross with edge type and vertex attributes ?
    def dependency_features(self, ne, prefix):
        # retVal = [prefix + "_is_ancestor=" + ("Y" if
        #                                       ne.root.is_ancestor(self.ne1.root) or
        #                                       ne.root.is_ancestor(self.ne2.root)
        #                                       else "N"), ]
        retVal = []
        if ne.parent != 0:
            retVal.extend(self._retrieve_neighbors_features(prefix + "_parent", ne.parent))

        for child in ne.children:
            retVal.extend(self._retrieve_neighbors_features(prefix + "_child", child))

        return retVal

    def _get_dependency_path(self):
        w1, w2 = self.ne1.root, self.ne2.root

        first_path = [w1.i + 1]
        for ancestor in w1.ancestors:
            if ancestor.i == w2.i:
                return first_path
            if ancestor.is_ancestor(w2):
                break
            first_path.append(ancestor.i + 1)

        second_path = list()
        for ancestor in w2.ancestors:
            second_path.append(ancestor.i + 1)
            if ancestor.i == w1.i:
                second_path.reverse()
                return second_path
            if ancestor.is_ancestor(w1):
                second_path.reverse()
                return first_path + second_path

        return []

    def _retrieve_neighbors_features(self, prefix, word_index):
        word_token = self.data[word_index]
        return [
            prefix + "_word=" + word_token.word,
            prefix + "_dependency=" + word_token.dependency,
            prefix + "_lemma_pos_dependency=" + word_token.lemma + "|" + word_token.pos + "|" +
            word_token.dependency,
            prefix + "_lemma_pos=" + word_token.lemma + "|" + word_token.pos,
            # prefix + "_pos=" + word_token.pos,
            # prefix + "_pos_dependency=" + word_token.pos + "|" + word_token.dependency,
        ]

    def _aux_distance(self):
        return str(max(self.ne1.start, self.ne2.start) - min(self.ne1.end, self.ne2.end))

    def items(self):
        # return self.feats
        return list(filterfalse(lambda x: x.split("=")[0] in self.exclude, self.feats))
