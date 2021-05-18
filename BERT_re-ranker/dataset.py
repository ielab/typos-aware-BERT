from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from tools.read_files import read_collection, read_qrel, read_queries, read_top_results, read_qrel_lines
import random
import gc

from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution, WordSwapQWERTY
from textattack.augmentation import Augmenter
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation.min_word_length import MinWordLength

random.seed(313)
import copy


class FixWordSwapQWERTY(WordSwapQWERTY):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            if len(self._get_adjacent(word[i])) == 0:
                candidate_word = (
                    word[:i] + random.choice(list(self._keyboard_adjacency.keys())) + word[i + 1:]
                )
            else:
                candidate_word = (
                    word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1:]
                )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words


class MsmarcoPassageCrossEncoderTrainSet(Dataset):
    def __init__(self,
                 path_to_top_results,
                 path_to_qrel,
                 path_to_query,
                 path_to_collection,
                 tokenizer,
                 num_epochs=2,
                 num_neg_per_pos=4,
                 insert_typo=False,
                 typo_rate=0.5):

        qrel = read_qrel(path_to_qrel)  # qrel only contains relevant judge
        all_top_results = read_top_results(path_to_top_results, qrel.keys())
        self.tokenizer = tokenizer
        self.queries = read_queries(path_to_query)
        self.collection = read_collection(path_to_collection)
        self.typo_rate = typo_rate
        self.insert_typo = insert_typo

        if self.insert_typo:
            transformation = CompositeTransformation([
                WordSwapRandomCharacterDeletion(),
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterInsertion(),
                WordSwapRandomCharacterSubstitution(),
                FixWordSwapQWERTY(),
            ])
            constraints = [MinWordLength(3)]
            self.augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0)

        data = []

        for qid in tqdm(qrel.keys(), desc="Creating training set"):
            positives = qrel[qid].keys()
            top_results = copy.deepcopy(all_top_results[qid])
            negatives = list(set(top_results).difference(set(positives)))
            for pos_docid in positives:
                for _ in range(num_epochs):  # repeat num_epoch times for each sample
                    data.append((qid, pos_docid, 1))  # positive sample
                    # sample negatives for each pos pair
                    if len(negatives) >= num_neg_per_pos:
                        rand_negatives = random.sample(negatives, num_neg_per_pos)
                    else:
                        rand_negatives = list(map(str, torch.randint(len(self.collection), (num_neg_per_pos,)).numpy()))
                    for neg_docid in rand_negatives:
                        data.append((qid, neg_docid, 0))
        self.np_data = np.array(copy.deepcopy(data), dtype=np.string_)  # try to solve memory leaking problem here
        del all_top_results, data
        gc.collect()

    def __getitem__(self, index):
        qid, docid, label = self.np_data[index]
        query = self.queries[int(qid)]
        if self.insert_typo and random.random() < self.typo_rate:
            query = self.augmenter.augment(query)[0]

        passage = self.collection[int(docid)]
        label = torch.LongTensor([int(label)])

        # inputs = self.tokenizer(query,
        #                         passage,
        #                         return_tensors='pt',
        #                         padding='max_length',
        #                         max_length=512,  # 64 + 256
        #                         truncation=True
        #                         )
        #
        # for key in inputs.keys():
        #     inputs[key] = inputs[key].squeeze()
        #
        # return inputs, label
        return query, passage, label

    def __len__(self):
        return len(self.np_data)

