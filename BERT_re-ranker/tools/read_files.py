from tqdm import tqdm


def read_qrel(path_to_qrel) -> dict:
    """
    return a dictionary that maps qid, docid pair to its relevance label.
    """
    qrel = {}
    with open(path_to_qrel, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading qrel"):
        if path_to_qrel.strip().split(".")[-1] == 'txt':
            qid, _, docid, rel = line.strip().split(" ")
        elif path_to_qrel.strip().split(".")[-1] == 'tsv':
            qid, _, docid, rel = line.strip().split("\t")
        if qid in qrel.keys():
            pass
        else:
            qrel[qid] = {}

        qrel[qid][docid] = int(rel)
    return qrel


def read_qrel_lines(path_to_qrel) -> dict:
    """
    return a dictionary that maps qid, docid pair to its relevance label.
    """
    qrel = []
    with open(path_to_qrel, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading qrel."):
        qid, _, docid, rel = line.strip().split("\t")
        qrel.append((qid, docid))

    return qrel


def read_queries(path_to_query):
    query_dict = {}
    with open(path_to_query, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading query"):
        qid, query = line.strip().split("\t")
        query_dict[int(qid)] = query
    return query_dict


def read_query_lines(path_to_query):
    query_lines = []
    with open(path_to_query, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading query"):
        qid, query = line.strip().split("\t")
        query_lines.append((qid, query))
    return query_lines


def read_top_results(path_to_top_results, valid_qids=None) -> dict:
    """
    return a dictionary that maps qid to its list of top docids.
    """
    top_results = {}
    with open(path_to_top_results, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading top results."):
        l = line.strip().split(" ")
        qid = l[0]  # the first element is the qid, rests are docids.
        if valid_qids is not None and qid not in valid_qids:
            continue
        docids = l[1:]
        top_results[qid] = docids

    return top_results


def read_collection(path_to_collection) -> dict:
    """
    return a dictionary that maps qid to its text.
    """
    passage_dict = {}
    with open(path_to_collection, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading collection"):
        docid, passage = line.strip().split("\t")
        passage_dict[int(docid)] = passage
    return passage_dict


def read_run(run_path, run_type='trec'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="Loading run file...."):
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