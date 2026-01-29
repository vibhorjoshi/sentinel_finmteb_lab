#!/usr/bin/env python3
"""
IEEE TMLCN Large-Scale Benchmark: 100k Query Evaluation
Optimized for both CPU and GPU environments.
"""

import json
import numpy as np
import time
import gc
from tqdm import tqdm
from src.dataset import load_financial_corpus
from src.embedder import FinancialEmbedder
from src.engine import SentinelEngine
from src.metrics import compute_fidelity_with_qrels, calculate_network_load
from src.config import RESULTS_PATH, GT_COLLECTION, BQ_COLLECTION, N_SAMPLES
from qdrant_client import models
import torch

def _load_finmteb_with_fallback():
    print("\n[Phase 1] Loading FinMTEB Financial Corpus with Smart Alignment...")
    start_load = time.time()
    try:
        corpus, queries, qrels = load_financial_corpus(use_full_data=True)
        load_mode = "full"
    except Exception as exc:
        print(f"   [Warn] Full dataset load failed: {exc}")
        print("   [Warn] Falling back to subset loading for reliability.")
        corpus, queries, qrels = load_financial_corpus(use_full_data=False)
        load_mode = "subset"

    if not corpus or not queries:
        raise RuntimeError(
            "FinMTEB dataset failed to load with any records. "
            "Verify dataset availability or connectivity."
        )

    corpus_ids = set(corpus.keys())
    query_ids = set(queries.keys())
    filtered_qrels = {}
    missing_doc_ids = 0
    missing_query_ids = 0

    for qid, doc_ids in qrels.items():
        if qid not in query_ids:
            missing_query_ids += 1
            continue
        filtered_docs = [doc_id for doc_id in doc_ids if doc_id in corpus_ids]
        if len(filtered_docs) != len(doc_ids):
            missing_doc_ids += len(doc_ids) - len(filtered_docs)
        if filtered_docs:
            filtered_qrels[qid] = filtered_docs

    print(f"   ✓ Loaded mode: {load_mode}")
    print(f"   ✓ Loaded {len(corpus):,} documents")
    print(f"   ✓ Loaded {len(queries):,} queries")
    print(f"   ✓ Loaded {len(filtered_qrels):,} qrels (ground truth)")
    if missing_query_ids or missing_doc_ids:
        print(
            "   [Info] Filtered qrels to align with loaded corpus/queries "
            f"(dropped {missing_query_ids:,} query ids, {missing_doc_ids:,} doc links)."
        )
    print(f"   [Time] {time.time() - start_load:.1f}s")

    return corpus, queries, filtered_qrels


def run_large_scale_experiment():
    print("=" * 70)
    print("   IEEE TMLCN: LARGE-SCALE BENCHMARK (100k Queries)")
    print("=" * 70)
    print(f"[Config] N_SAMPLES={N_SAMPLES:,} queries")
    print(f"[Config] Device: {'CUDA/GPU' if torch.cuda.is_available() else 'CPU (Multi-process)'}")
    start_load = time.time()
    
    # ========== Phase 1: Data Loading ==========
    corpus, queries, qrels = _load_finmteb_with_fallback()

    # Convert to lists
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())
    
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    
    # ========== Phase 2: Embedding ==========
    print("\n[Phase 2] Vectorizing Corpus & Queries...")
    embedder = FinancialEmbedder()
    
    start_embed = time.time()
    
    # Encode documents in batches to manage memory
    print(f"   Vectorizing {len(doc_texts):,} documents...")
    doc_vectors = embedder.encode(doc_texts, batch_size=64)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"   Vectorizing {len(query_texts):,} queries...")
    query_vectors = embedder.encode(query_texts, batch_size=64)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    embed_time = time.time() - start_embed
    print(f"   [Time] {embed_time:.1f}s ({len(doc_texts) / embed_time:.0f} docs/sec)")
    
    # ========== Phase 3: Qdrant Ingestion ==========
    print("\n[Phase 3] Ingesting into Qdrant (Float32 & Binary Quantization)...")
    engine = SentinelEngine()
    engine.init_collections()
    
    start_ingest = time.time()
    engine.ingest(doc_vectors, doc_texts, ids=doc_ids)
    ingest_time = time.time() - start_ingest
    print(f"   ✓ Ingestion complete in {ingest_time:.1f}s")
    
    # ========== Phase 4: Network Impact Analysis ==========
    print("\n[Phase 4] Network Impact Analysis (IEEE TMLCN)...")
    cloud_bw, edge_bw = calculate_network_load(len(query_texts))
    reduction_factor = cloud_bw / edge_bw if edge_bw > 0 else 1.0
    print(f"   Cloud Backhaul:      {cloud_bw:.2f} Gbps")
    print(f"   Sovereign (Edge):    {edge_bw:.4f} Gbps")
    print(f"   Efficiency Gain:     {reduction_factor:.1f}x Reduction")
    
    # ========== Phase 5: Large-Scale Retrieval Benchmark ==========
    print(f"\n[Phase 5] Benchmarking Retrieval Fidelity ({len(query_vectors):,} queries)...")
    results = {
        "metadata": {
            "n_queries": len(query_texts),
            "n_documents": len(doc_texts),
            "network_stats": {
                "cloud_gbps": cloud_bw,
                "edge_gbps": edge_bw,
                "reduction_factor": reduction_factor
            }
        },
        "fidelity_stats": {}
    }
    
    oversampling_factors = [1.0, 2.0, 4.0]
    
    for factor in oversampling_factors:
        print(f"\n   --- Oversampling Factor: {factor}x ---")
        recalls = []
        hits = []
        mrr_scores = []
        start_time = time.time()
        
        for i, q_vec in enumerate(tqdm(query_vectors, desc=f"Factor {factor}x", leave=False)):
            current_qid = query_ids[i]
            
            # Query Qdrant with oversampling
            bq_response = engine.client.query_points(
                collection_name=BQ_COLLECTION,
                query=q_vec,
                limit=int(10 * factor),
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=factor
                    )
                )
            )
            
            # Extract top-10 results
            retrieved_ids = [str(hit.id) for hit in bq_response.points][:10]
            
            # Compute fidelity metrics
            recall, is_hit = compute_fidelity_with_qrels(qrels, current_qid, retrieved_ids)
            recalls.append(recall)
            hits.append(1 if is_hit else 0)
            
            # MRR: Mean Reciprocal Rank
            if is_hit and retrieved_ids:
                mrr_scores.append(1.0 / (retrieved_ids.index([x for x in retrieved_ids if qrels.get(current_qid) and x in qrels.get(current_qid, [])][0]) + 1) if any(x in qrels.get(current_qid, []) for x in retrieved_ids) else 0.0)
            else:
                mrr_scores.append(0.0)
            
            # Periodic cleanup
            if (i + 1) % 10000 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        elapsed = time.time() - start_time
        
        # Aggregation
        avg_recall = np.mean(recalls) if recalls else 0.0
        mrr_score = np.mean(hits) if hits else 0.0
        avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        
        print(f"   ✓ Recall@10:       {avg_recall:.4f}")
        print(f"   ✓ MRR@10:          {mrr_score:.4f}")
        print(f"   ✓ Avg MRR Score:   {avg_mrr:.4f}")
        print(f"   ✓ Latency:         {(elapsed / len(query_vectors) * 1000):.2f} ms/query")
        print(f"   ✓ Total Time:      {elapsed:.1f}s ({len(query_vectors) / elapsed:.0f} queries/sec)")
        
        results["fidelity_stats"][f"Oversample_{factor}"] = {
            "recall_at_10": float(avg_recall),
            "mrr_at_10": float(mrr_score),
            "avg_mrr_score": float(avg_mrr),
            "latency_avg_ms": float((elapsed / len(query_vectors) * 1000)),
            "throughput_queries_per_sec": float(len(query_vectors) / elapsed),
            "total_time_sec": float(elapsed),
            "bandwidth_reduction": float(reduction_factor)
        }
    
    # ========== Phase 6: Save Results ==========
    print("\n[Phase 6] Saving IEEE TMLCN Results...")
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    result_file = f"{RESULTS_PATH}/ieee_tmlcn_100k_benchmark.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"   ✓ Results saved to {result_file}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("   BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total Execution Time: {time.time() - start_load:.1f}s")
    print(f"Queries Evaluated: {len(query_vectors):,}")
    print(f"Best Recall@10: {max([v['recall_at_10'] for v in results['fidelity_stats'].values()]):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_large_scale_experiment()
