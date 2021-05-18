from tools.read_files import read_query_lines
from tools.write_files import write_query_file
from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution
from textattack.augmentation import Augmenter
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation.min_word_length import MinWordLength
from dataset import FixWordSwapQWERTY
from tqdm import tqdm


def main():
    query_path = '../data/queries/passage/queries.dev.small.tsv'
    out_path = '../data/queries/passage/queries.dev.small.SwapAdjacent'
    # query_path = '../data/queries/passage/DL2020-queries.tsv'
    # out_path = '../data/queries/passage/DL2020-queries.typo'
    query_lines = read_query_lines(query_path)
    transformation = CompositeTransformation([
        # WordSwapRandomCharacterDeletion(),
        # WordSwapNeighboringCharacterSwap(),
        # WordSwapRandomCharacterInsertion(),
        # WordSwapRandomCharacterSubstitution(),
        FixWordSwapQWERTY(),
    ])
    constraints = [MinWordLength(3)]
    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0)
    qids = []
    typo_queires = []
    for qid, query in tqdm(query_lines, desc="Making typo queries"):
        typo_query = augmenter.augment(query)[0]
        qids.append(qid)
        typo_queires.append(typo_query)
    write_query_file(qids, typo_queires, out_path)


if __name__ == '__main__':
    main()