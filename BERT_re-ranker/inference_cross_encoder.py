from model import BertReranker, LCEreranker
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from tools.read_files import read_qrel, read_run
from tools.write_files import write_run_file
from dataset import insert_typo
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

    model_type = "bert-large-uncased"
    ckpt_path = args.ckpt_path
    collection_path = "data/collection/passage/collection.tsv"

    dev_query_path = args.query_path
    res_path = args.run_path
    output_path = f"data/runs/bert_reranker"
    run_type = 'msmarco'
    msmarco = True
    trec = False

    top_results = False

    batch_size = 128
    rerank_cut = 1000

    tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=".cache")

    print("Loading model...")
    model = BertReranker.load_from_checkpoint(checkpoint_path=ckpt_path).eval().to(DEVICE)

    collection = load_collection(collection_path)
    queries = load_queries(dev_query_path)
    run = read_run(res_path, run_type)


    for qid in tqdm(queries.keys(), desc="Ranking queries...."):
        query = queries[qid]
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
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True, help='path to query file')
    parser.add_argument("--run_path", type=str, required=True, help='path to msmarco run file that you want to rerank')

    args = parser.parse_args()

    main(args)
