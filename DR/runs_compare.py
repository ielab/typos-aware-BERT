import json
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

def read_run(run_path, run_type='msmarco'):
    run = {}
    with open(run_path, 'r') as f:
        for line in f:
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split("\t")
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split(" ")
            else:
                raise TypeError("run_type needs to be 'trec' or 'msmarco'.")
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


with open("./data/retrieve/repbert-ckpt210000.dev.small.original.top1k.tsv.json") as f:
    standard = json.load(f)

with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.original.top1k.tsv.json") as f:
    typo_aware = json.load(f)

gains = []
for qid in standard.keys():
    gains.append((qid, standard[qid] - typo_aware[qid], typo_aware[qid], standard[qid]))
gains = sorted(gains, key=lambda x: x[1])




with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.RandInsert.top1k.tsv.json") as f:
    RandInsert_typo = json.load(f)

with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.RandDelete.top1k.tsv.json") as f:
    RandDelete_typo = json.load(f)

with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.RandSub.top1k.tsv.json") as f:
    RandSub_typo = json.load(f)

with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.SwapNeighbor.top1k.tsv.json") as f:
    SwapNeighbor_typo = json.load(f)

with open("./data/retrieve/repbert-ckpt210000-typo.dev.small.SwapAdjacent.top1k.tsv.json") as f:
    SwapAdjacent_typo = json.load(f)



with open("./data/retrieve/repbert-ckpt210000.dev.small.RandInsert.top1k.tsv.json") as f:
    RandInsert = json.load(f)

with open("./data/retrieve/repbert-ckpt210000.dev.small.RandDelete.top1k.tsv.json") as f:
    RandDelete = json.load(f)

with open("./data/retrieve/repbert-ckpt210000.dev.small.RandSub.top1k.tsv.json") as f:
    RandSub = json.load(f)

with open("./data/retrieve/repbert-ckpt210000.dev.small.SwapNeighbor.top1k.tsv.json") as f:
    SwapNeighbor = json.load(f)

with open("./data/retrieve/repbert-ckpt210000.dev.small.SwapAdjacent.top1k.tsv.json") as f:
    SwapAdjacent = json.load(f)

#
with open("./data/retrieve/run.msmarco.bm25.dev.small.RandInsert.tsv.json") as f:
    bm25_RandInsert = json.load(f)

with open("./data/retrieve/run.msmarco.bm25.dev.small.RandDelete.tsv.json") as f:
    bm25_RandDelete = json.load(f)

with open("./data/retrieve/run.msmarco.bm25.dev.small.RandSub.tsv.json") as f:
    bm25_RandSub = json.load(f)

with open("./data/retrieve/run.msmarco.bm25.dev.small.SwapNeighbor.tsv.json") as f:
    bm25_SwapNeighbor = json.load(f)

with open("./data/retrieve/run.msmarco.bm25.dev.small.SwapAdjacent.tsv.json") as f:
    bm25_SwapAdjacent = json.load(f)

ave_typo_rank = {}
ave_typo_rank_typo = {}
ave_typo_bm25_rank = {}
gain_loss = []
gain_loss_typo = []
gain_loss_typo_stand_on_typo =[]
gain_loss_typo_stand_on_original = []

l1 = []
l2 = []
for qid in standard.keys():
    ave_typo_rank[qid] = (RandInsert[qid] + RandDelete[qid] + RandSub[qid] + SwapNeighbor[qid] + SwapAdjacent[qid])/5
    ave_typo_rank_typo[qid] = (RandInsert_typo[qid] + RandDelete_typo[qid] + RandSub_typo[qid] + SwapNeighbor_typo[qid] + SwapAdjacent_typo[qid]) / 5

    if qid not in bm25_RandInsert.keys() or qid not in bm25_RandDelete.keys() or qid not in bm25_RandSub.keys() or qid not in bm25_SwapNeighbor.keys() or qid not in bm25_SwapAdjacent.keys():
        ave_typo_bm25_rank[qid] = 0
    else:
        ave_typo_bm25_rank[qid] = (bm25_RandInsert[qid] + bm25_RandDelete[qid] + bm25_RandSub[qid] + bm25_SwapNeighbor[qid] + bm25_SwapAdjacent[qid])/5
    gain_loss.append(standard[qid] - ave_typo_rank[qid])
    gain_loss_typo.append(typo_aware[qid] - ave_typo_rank_typo[qid])


    gain_loss_typo_stand_on_typo.append(ave_typo_rank[qid] - ave_typo_rank_typo[qid])
    gain_loss_typo_stand_on_original.append(standard[qid] - typo_aware[qid])

    l1.append(ave_typo_rank[qid])
    l2.append(ave_typo_bm25_rank[qid])

_, p = stats.ttest_ind(l1, l2)
print(len(l1))
print(np.sum(l1)/6980, np.sum(l2)/6980, p)

gain_loss = sorted(gain_loss)
gain_loss_typo = sorted(gain_loss_typo)
gain_loss_typo_stand_on_typo = sorted(gain_loss_typo_stand_on_typo, reverse=True)
gain_loss_typo_stand_on_original = sorted(gain_loss_typo_stand_on_original, reverse=True)
# print(gain_loss_typo)
fig = plt.figure(figsize =(10, 5))
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['right'].set_color('none')

ax.spines['top'].set_position('center')
ax.spines['bottom'].set_color('none')
ax.plot(range(len(gain_loss)), gain_loss,linestyle="--", label='Standard training')
ax.plot(range(len(gain_loss_typo)), gain_loss_typo, label='Typos-aware training')
ax.set_xlim([0, 6980])
x = [1000, 2000, 3000, 4000, 5000, 6000, 6980]
# ax.("Rank Gain", fontsize=22)
# ax.xticks(fontsize=18)
# ax.yticks(y, fontsize=18)
ax.set_ylim([-1000,1000])
ax.set_ylabel("$loss = original(q_i) - typo(q_i) $", fontsize=22)
plt.xticks(x, fontsize=18)
plt.yticks(fontsize=18)

ax.legend(loc='upper left', fontsize=18)
# plt.savefig("gain_loss.png", bbox_inches='tight')