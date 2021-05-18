"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task. Intenral hard coded eval files version. DO NOT PUBLISH!
Command line:
python msmarco_eval_ranking.py <path_to_candidate_file>
Creation Date : 06/12/2018
Last Modified : 4/09/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import sys
import statistics
from scipy import stats
import numpy as np
import json

from collections import Counter

MaxMRRRank = 10


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[2]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            pid = int(l[1])
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_passages:
                pass
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            if rank > 1000:
                continue
            qid_to_ranked_candidate_passages[qid][rank - 1] = pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages


def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """

    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, path_to_candidate):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    recall = 0
    qids_with_relevant_passages = 0
    ranking = []
    all_MRR = []
    all_recall = []
    rank_dic = {}
    recall_dic = {}
    mrr_dic = {}
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            rank_dic[qid] = 1000
            mrr_dic[qid] = 0
            ranking.append(0)
            all_MRR.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, 1000):
                if candidate_pid[i] in target_pid:
                    if i < MaxMRRRank:
                        MRR += 1 / (i + 1)
                        all_MRR.pop()
                        all_MRR.append(1 / (i + 1))
                    ranking.pop()
                    ranking.append(i + 1)

                    rank_dic[qid] = i + 1
                    mrr_dic[qid] = 1 / (i + 1)
                    break
            num_retreved_rel = 0
            for i in range(0, 1000):
                if candidate_pid[i] in target_pid:
                    num_retreved_rel += 1
            recall += num_retreved_rel / len(target_pid)
            all_recall.append(num_retreved_rel / len(target_pid))
            recall_dic[qid] = num_retreved_rel / len(target_pid)
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_ranked_candidate_passages)
    recall = recall / len(qids_to_ranked_candidate_passages)
    avg_ranking = 0
    num_valid = 0
    for r in ranking:
        if r == 0:
            pass
            # avg_ranking += 1001

        else:
            num_valid += 1
            avg_ranking += r

    all_scores['MRR @10'] = MRR
    all_scores['Recall @1000'] = recall
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)

    # avg_ranking = avg_ranking / num_valid
    # all_scores['Avg Ranking'] = avg_ranking
    # import json
    # json = json.dumps(mrr_dic)
    # f = open(f"{path_to_candidate}.json", "w")
    # f.write(json)
    return all_scores, all_MRR, all_recall


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID
            Where the values are separated by tabs and ranked in order of relevance
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """

    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, path_to_candidate)


def main():
    """Command line:
    python msmarco_eval_ranking.py <path to reference> <path_to_candidate_file>
    """
    path_to_candidate1 = sys.argv[2]
    path_to_reference = sys.argv[1]
    if len(sys.argv) == 3:
        metrics, _, _ = compute_metrics_from_files(path_to_reference, path_to_candidate1)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')
    else:
        path_to_candidate2 = sys.argv[3]
        metrics1, all_MRR1, all_recall1 = compute_metrics_from_files(path_to_reference, path_to_candidate1)
        print('#####################')
        print(path_to_candidate1)
        for metric in sorted(metrics1):
            print('{}: {}'.format(metric, metrics1[metric]))
        print('#####################')
        print()

        metrics2, all_MRR2, all_recall2 = compute_metrics_from_files(path_to_reference, path_to_candidate2)
        print('#####################')
        print(path_to_candidate2)
        for metric in sorted(metrics2):
            print('{}: {}'.format(metric, metrics2[metric]))
        print('#####################')
        print()

        print('#####################')
        _, p = stats.ttest_ind(all_MRR1, all_MRR2)
        print("MRR P value:", p)
        _, p = stats.ttest_ind(all_recall1, all_recall2)
        print("recall value:", p)
        print('#####################')



if __name__ == '__main__':
    main()