import json
import os
import random
from datasets import load_dataset
from .config import N_SAMPLES


class SentinelDatasetManager:
    def __init__(self, cache_dir="data/cache", use_cache=True, verbose=False):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.verbose = verbose

    def _cache_paths(self, target_docs):
        cache_key = f"finmteb_{target_docs}"
        return {
            "corpus": os.path.join(self.cache_dir, f"{cache_key}_corpus.json"),
            "queries": os.path.join(self.cache_dir, f"{cache_key}_queries.json"),
            "qrels": os.path.join(self.cache_dir, f"{cache_key}_qrels.json"),
        }

    def _load_cache(self, paths):
        with open(paths["corpus"], "r") as f:
            corpus = json.load(f)
        with open(paths["queries"], "r") as f:
            queries = json.load(f)
        with open(paths["qrels"], "r") as f:
            qrels = json.load(f)
        return corpus, queries, qrels

    def _save_cache(self, paths, corpus, queries, qrels):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(paths["corpus"], "w") as f:
            json.dump(corpus, f)
        with open(paths["queries"], "w") as f:
            json.dump(queries, f)
        with open(paths["qrels"], "w") as f:
            json.dump(qrels, f)

    def load_smart_subset(self, target_docs, loading_method="cached"):
        paths = self._cache_paths(target_docs)
        if (
            self.use_cache
            and loading_method == "cached"
            and all(os.path.exists(path) for path in paths.values())
        ):
            if self.verbose:
                print("   ✓ Loading dataset from cache")
            return self._load_cache(paths)

        if self.verbose:
            print("   -> Loading FiQA corpus, queries, and qrels from HuggingFace")

        corpus_ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
        queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
        qrels_ds = load_dataset("mteb/fiqa", "default", split="test")

        qrels_map = {}
        doc_ids_needed = []
        seen_docs = set()
        for row in qrels_ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            qrels_map.setdefault(qid, {})
            qrels_map[qid][did] = 1
            if did not in seen_docs:
                doc_ids_needed.append(did)
                seen_docs.add(did)
            if target_docs and len(doc_ids_needed) >= target_docs:
                break

        selected_doc_ids = set(doc_ids_needed)

        corpus = {}
        for row in corpus_ds:
            row_id = str(row["_id"])
            if row_id in selected_doc_ids:
                corpus[row_id] = {"title": row["title"], "text": row["text"]}
                if len(corpus) >= len(selected_doc_ids):
                    break

        queries = {}
        for row in queries_ds:
            row_id = str(row["_id"])
            if row_id in qrels_map:
                queries[row_id] = row["text"]

        if self.use_cache:
            self._save_cache(paths, corpus, queries, qrels_map)

        return corpus, queries, qrels_map

def load_financial_corpus(use_full_data=False):
    """
    Loads FinMTEB (FiQA) but ensures 'Answerable' queries.
    It guarantees that the relevant documents for the chosen queries 
    are actually included in the N_SAMPLES corpus.
    """
    print(f"--- Loading FinMTEB (FiQA) Smart-Subset (Full Data: {use_full_data}) ---")
    
    # 1. Load Everything (Metadata only, fast)
    corpus_ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
    qrels_ds = load_dataset("mteb/fiqa", "default", split="test")

    # 2. Build Qrels Map {query_id: [doc_id, doc_id]}
    qrels_dict = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        docid = str(row["corpus-id"])
        if qid not in qrels_dict:
            qrels_dict[qid] = []
        qrels_dict[qid].append(docid)

    # 3. Smart Subsetting
    if use_full_data:
        # Use everything
        selected_doc_ids = set(str(row["_id"]) for row in corpus_ds)
        selected_query_ids = list(qrels_dict.keys())
    else:
        # --- THE FIX ---
        # 1. Pick 100 queries that actually have answers
        valid_qids = [q for q in list(qrels_dict.keys()) if len(qrels_dict[q]) > 0]
        selected_query_ids = valid_qids[:100] 
        
        # 2. Collect all document IDs that are answers to these 100 queries
        # (This guarantees Recall > 0 is possible)
        must_have_doc_ids = set()
        for qid in selected_query_ids:
            for doc_id in qrels_dict[qid]:
                must_have_doc_ids.add(doc_id)
        
        print(f"   -> Identified {len(must_have_doc_ids)} 'Gold' documents required for these queries.")

        # 3. Fill the rest of N_SAMPLES with random documents (Distractors)
        # to simulate the "Haystack"
        remaining_slots = N_SAMPLES - len(must_have_doc_ids)
        if remaining_slots > 0:
            # Get all available doc IDs
            all_doc_ids = [str(row["_id"]) for row in corpus_ds]
            # Remove ones we already picked
            pool = list(set(all_doc_ids) - must_have_doc_ids)
            # Pick random distractors
            distractors = random.sample(pool, min(len(pool), remaining_slots))
            selected_doc_ids = must_have_doc_ids.union(set(distractors))
        else:
            selected_doc_ids = must_have_doc_ids

    # 4. Filter the Datasets based on our Smart Selection
    # (This takes a moment but ensures data consistency)
    
    print("   -> Filtering Corpus...")
    corpus_dict = {}
    # We iterate once to find our selected IDs
    # Note: For speed on large data, we use a set check
    for row in corpus_ds:
        row_id = str(row["_id"])
        if row_id in selected_doc_ids:
            corpus_dict[row_id] = f"{row['title']} {row['text']}"
            if len(corpus_dict) >= len(selected_doc_ids):
                break # Stop once we have all we need

    print("   -> Filtering Queries...")
    queries_dict = {}
    for row in queries_ds:
        row_id = str(row["_id"])
        if row_id in selected_query_ids:
            queries_dict[row_id] = row["text"]

    print(f"Loaded {len(corpus_dict)} docs, {len(queries_dict)} queries.")
    return corpus_dict, queries_dict, qrels_dict
