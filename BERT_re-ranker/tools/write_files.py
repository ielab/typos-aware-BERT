import os


def write_run_file(qids, scores, docids, output_path, msmarco=True, trec=True, top_results=False):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    num_quries = len(qids)
    for j in range(num_quries):
        qid = qids[j]
        num_docs = len(docids[j])
        ms_lines = []
        trec_lines = []
        top_lines = str(qid)
        for k in range(num_docs):
            score = scores[j][k]
            docid = docids[j][k]
            ms_lines.append(str(qid) + "\t" + str(docid) + "\t" + str(k + 1) + "\n")
            trec_lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(k + 1) + " " + str(score) + " " + "ielab" + "\n")
            top_lines += " " + str(docid)
        top_lines += "\n"

        if msmarco:
            with open(output_path+'.tsv', "a+") as f:
                f.writelines(ms_lines)
        if trec:
            with open(output_path+'.res', "a+") as f:
                f.writelines(trec_lines)
        if top_results:
            with open(output_path+'_top_results.txt', "a+") as f:
                f.writelines([top_lines])


def write_query_file(qids, queries, output_path):
    query_lines = []
    for i in range(len(qids)):
        query_lines.append(str(qids[i]) + "\t" + queries[i] + "\n")
    with open(output_path + '.tsv', "a+") as f:
        f.writelines(query_lines)

