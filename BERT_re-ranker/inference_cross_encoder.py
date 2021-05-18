from model import BertReranker
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
from tools.read_files import read_qrel, read_run
from tools.write_files import write_run_file
import argparse
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="Loading collection...."):
            docid, text = line.strip().split("\t")
            collection[docid] = text
    return collection


def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="Loading query...."):
            qid, text = line.strip().split("\t")
            query[qid] = text
    return query


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_name = args.model_name
    typo_type = args.typo_type

    # model_type = "nghuyong/ernie-2.0-large-en"
    model_type = "bert-large-uncased"
    ckpt_path = f"ckpts/CE/BertReranker/bert-large-uncased/{model_name}.ckpt"
    collection_path = "data/collection/passage/collection.tsv"

    # year = "2019"
    # dev_query_path = f"data/queries/passage/DL{year}-queries.typo.tsv"
    # res_path = f"data/runs/run.trec2019-bm25.res"
    # qrel_path = f"data/qrels/passage/{year}qrels-pass.txt"
    # output_path = f"data/runs/CE/DL{year}/{model_name}_rerank_DE_1000_typo"
    # msmarco = False
    # trec = True

    dev_query_path = f"data/queries/passage/queries.dev.small.{typo_type}.tsv"
    res_path = f"data/runs/RepBERT/repbert-ckpt210000-typo.dev.small.{typo_type}.top1k.tsv"
    qrel_path = "data/qrels/passage/qrels.dev.small.tsv"
    output_path = f"data/runs/CE/msmarco/{model_name}_rerank_RepBERT_{typo_type}"
    run_type = 'msmarco'
    msmarco = True
    trec = False

    do_typo = False
    top_results = False

    batch_size = 128
    rerank_cut = 1000

    tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=".cache")

    print("Loading model...")
    model = BertReranker.load_from_checkpoint(checkpoint_path=ckpt_path).eval().to(DEVICE)
    # encoder = AutoModel.from_pretrained("ckpts/ernie_large/adam/encoder").eval().to(DEVICE)
    # project = torch.load("ckpts/ernie_large/adam/encoder/project").to(DEVICE)
    # project.eval()

    collection = load_collection(collection_path)
    queries = load_queries(dev_query_path)
    run = read_run(res_path, run_type)
    qrel = read_qrel(qrel_path)

    for qid in tqdm(qrel.keys(), desc="Ranking queries...."):
        query = queries[qid]
        # if do_typo:
        #     query = insert_typo(query)
        # split batch of documents in top 1000
        docids = run[qid]
        num_docs = min(rerank_cut, len(docids))  # rerank top k
        numIter = num_docs // batch_size + 1

        total_scores = []
        for i in range(numIter):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > num_docs:
                end = num_docs
                if start == end:
                    continue

            batch_passages = []
            for docid in docids[start:end]:
                batch_passages.append(collection[docid])

            inputs = tokenizer([query] * len(batch_passages), batch_passages,
                               return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)

            with torch.no_grad():
                # hidden_states = encoder(**inputs).last_hidden_state[:, 0, :]
                # outputs = project(hidden_states)
                # scores = outputs[:, 0]
                scores = model.get_scores(inputs)
                total_scores.append(scores)

        total_scores = torch.cat(total_scores).cpu().numpy()

        # rerank documents
        zipped_lists = zip(total_scores, docids)
        sorted_pairs = np.array(sorted(zipped_lists, reverse=True))
        scores = sorted_pairs[:, 0]
        docids = sorted_pairs[:, 1]

        # write run file
        write_run_file([qid], [scores], [docids], output_path, msmarco=msmarco, trec=trec, top_results=top_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--typo_type", choices=["original", "RandInsert", "RandDelete", "RandSub", "SwapNeighbor", "SwapAdjacent"], required=True)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()

    main(args)
