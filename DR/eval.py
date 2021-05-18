import os
import re
import subprocess

eval_script = "./ms_marco_eval.py"
qrels = "./data/msmarco-passage/qrels.dev.small.tsv"
run_type = "./data/retrieve/repbert-ckpt210000"

run_file_path_pairs = [(f"{run_type}.dev.small.original.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.original.tsv"),
                       (f"{run_type}.dev.small.RandInsert.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.RandInsert.tsv"),
                       (f"{run_type}.dev.small.RandDelete.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.RandDelete.tsv"),
                       (f"{run_type}.dev.small.RandSub.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.RandSub.tsv"),
                       (f"{run_type}.dev.small.SwapNeighbor.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.SwapNeighbor.tsv"),
                       (f"{run_type}.dev.small.SwapAdjacent.top1k.tsv", f"./data/retrieve/run.msmarco.bm25.dev.small.SwapAdjacent.tsv")]

run = ["original", "RandInsert", "RandDelete", "RandSub", "SwapNeighbor", "SwapAdjacent"]
all_mrr1 = 0
all_mrr2 = 0
all_recall1 = 0
all_recall2 = 0
original_mrr1 = 0
original_mrr2 = 0
original_recall1 = 0
original_recall2 = 0
for i, run_file_path in enumerate(run_file_path_pairs):
    run_file_path1, run_file_path2 = run_file_path
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path1, run_file_path2])
    mrr_match = re.findall('MRR @10: ([\d.]+)', result.decode('utf-8'))
    recall_match = re.findall('Recall @1000: ([\d.]+)', result.decode('utf-8'))
    mrr_p_match = re.findall('MRR P value: ([0-9]*\.?[0-9]*([Ee][+-]?[0-9]+)?)', result.decode('utf-8'))
    recall_p_match = re.findall('recall value: ([0-9]*\.?[0-9]*([Ee][+-]?[0-9]+)?)', result.decode('utf-8'))

    if i == 0:
        original_mrr1 = float(mrr_match[0])
        original_mrr2 = float(mrr_match[1])
        original_recall1 = float(recall_match[0])
        original_recall2 = float(recall_match[1])
    else:
        all_mrr1 += float(mrr_match[0])
        all_mrr2 += float(mrr_match[1])
        all_recall1 += float(recall_match[0])
        all_recall2 += float(recall_match[1])

    print(f"{run[i]}:")
    print(f"MRR@10: {mrr_match[0]}({1-(float(mrr_match[0])/original_mrr1)}), {mrr_match[1]}({1-(float(mrr_match[1])/original_mrr2)}), p: {mrr_p_match}")
    print(f"Recall@1000: {recall_match[0]}({1-(float(recall_match[0])/original_recall1)}), {recall_match[1]}({1-(float(recall_match[1])/original_recall2)}), p: {recall_p_match}")
    print()

print(f"Average MRR@10: {all_mrr1/5}({1-((all_mrr1/5)/original_mrr1)}), {all_mrr2/5}({1-((all_mrr2/5)/original_mrr2)})")
print(f"Average Recall@1000: {all_recall1/5}({1-((all_recall1/5)/original_recall1)}), {all_recall2/5}({1-((all_recall2/5)/original_recall2)})")