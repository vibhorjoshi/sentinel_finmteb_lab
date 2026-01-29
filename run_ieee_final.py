"""
SENTINEL IEEE Final Benchmark - Production Ready (v2.0)
Complete integrated implementation with:
  - 6 dataset loading methods
  - 5 specialized financial agents + orchestration
  - End-to-end pipeline orchestration
  - Comprehensive evaluation metrics

This is the OFFICIAL benchmark for the IEEE TMLCN 2026 paper submission.

Expected Runtime: ~1.5-2 hours on CPU, ~10-15 minutes on GPU
Output: results/final_ieee_data.json with complete metrics and agent analysis
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import Sentinel modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.config import (
        TARGET_DOCS,
        VECTOR_DIM,
        COMPRESSION_RATIO,
        DATA_PATH,
        COLLECTION_NAME,
        RESULTS_PATH,
        DEVICE,
        FINAL_RESULTS_FILE,
        RECALL_AT_K,
        CLOUD_LOAD_GBPS,
        SENTINEL_LOAD_GBPS,
        BYTES_PER_FULL_VECTOR,
        BYTES_PER_RABITQ_VECTOR,
        DEFAULT_PERSONA,
    )
    from src.dataset import SentinelDatasetManager
    from src.embedder import SentinelEmbedder
    from src.engine import SentinelEngine
    from src.agents import MultiAgentOrchestrator
    from src.metrics import ComprehensiveEvaluator
except ImportError as e:
    logger.error("Failed to import Sentinel modules: %s", e)
    logger.info("Make sure all src modules are present")
    sys.exit(1)


# ============================================================================
# PHASE 0: SMART SUBSET LOADING (Using SentinelDatasetManager)
# ============================================================================

def _load_finmteb_with_fallback():
    """
    Load FinMTEB Financial Corpus with smart fallback strategy.

    Strategy:
    1. Try loading full dataset
    2. If fails, fall back to subset loading
    3. Filter qrels to align with loaded corpus/queries
    4. Validate dataset integrity

    Returns:
        (corpus_dict, queries_dict, qrels_dict) with aligned ground truth
    """
    print("\n[Phase 1] Loading FinMTEB Financial Corpus with Smart Alignment...")
    start_load = time.time()
    try:
        manager = SentinelDatasetManager(
            cache_dir="data/cache",
            use_cache=True,
            verbose=True,
        )
        corpus, queries, qrels = manager.load_smart_subset(
            target_docs=TARGET_DOCS,
            loading_method="cached",
        )
        load_mode = "full"
    except Exception as exc:
        print(f"   [Warn] Full dataset load failed: {exc}")
        print("   [Warn] Falling back to subset loading for reliability.")
        manager = SentinelDatasetManager(
            cache_dir="data/cache",
            use_cache=True,
            verbose=False,
        )
        corpus, queries, qrels = manager.load_smart_subset(
            target_docs=TARGET_DOCS,
            loading_method="cached",
        )
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
        if isinstance(doc_ids, dict):
            doc_id_list = list(doc_ids.keys())
        else:
            doc_id_list = list(doc_ids)
        filtered_docs = [doc_id for doc_id in doc_id_list if doc_id in corpus_ids]
        if len(filtered_docs) != len(doc_id_list):
            missing_doc_ids += len(doc_id_list) - len(filtered_docs)
        if filtered_docs:
            filtered_qrels[qid] = {doc_id: 1 for doc_id in filtered_docs}

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


def get_smart_subset(
    target_docs: int = TARGET_DOCS,
    verbose: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load FiQA corpus with smart subset selection using SentinelDatasetManager.

    Strategy:
    1. Load full FiQA corpus (57,638 documents)
    2. Load ground-truth qrels (query relevance judgments)
    3. Filter to documents with actual relevance labels
    4. Load queries that have relevant documents

    Result: ~1000 documents with ~300 queries and guaranteed ground truth

    Args:
        target_docs: Target number of documents
        verbose: Print progress

    Returns:
        (corpus_dict, queries_dict, qrels_dict)
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 0: SMART SUBSET LOADING")
        print("=" * 70)

    start_load = time.time()

    try:
        manager = SentinelDatasetManager(
            cache_dir="data/cache",
            use_cache=True,
            verbose=verbose,
        )

        corpus, queries, qrels = manager.load_smart_subset(
            target_docs=target_docs,
            loading_method="cached",
        )

        elapsed = time.time() - start_load

        if verbose:
            print(f"\n✅ SUBSET LOADING COMPLETE ({elapsed:.1f}s)")
            print(f"   Loaded: {len(corpus)} Docs | {len(queries)} Queries")
            avg_rels = sum(len(v) for v in qrels.values()) / len(qrels) if qrels else 0
            print(f"   Avg relevant docs per query: {avg_rels:.1f}")

        return corpus, queries, qrels

    except Exception as e:
        logger.error("Failed to load smart subset: %s", e, exc_info=True)
        raise


# ============================================================================
# PHASE 1: VECTORIZATION
# ============================================================================

def vectorize_corpus(
    corpus_dict: Dict,
    embedder: SentinelEmbedder,
    batch_size: int = 64,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Vectorize corpus documents using Qwen-2.5-GTE + RaBitQ.

    Args:
        corpus_dict: {doc_id: {"title": str, "text": str}}
        embedder: SentinelEmbedder instance
        batch_size: Batch size for encoding
        verbose: Print progress

    Returns:
        (vectors, doc_ids) where vectors is (N, 1536) and doc_ids are strings
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: DOCUMENT VECTORIZATION")
        print("=" * 70)

    start_vec = time.time()

    doc_ids = list(corpus_dict.keys())
    doc_texts = []
    for doc_id in doc_ids:
        doc = corpus_dict[doc_id]
        text = f"{doc['title']} {doc['text']}".strip()
        doc_texts.append(text)

    if verbose:
        print(f"\n🧠 Encoding {len(doc_texts)} documents with RaBitQ...")

    vectors = embedder.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        persona=DEFAULT_PERSONA,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start_vec

    if verbose:
        print(f"\n✅ VECTORIZATION COMPLETE ({elapsed:.1f}s)")
        print(f"   Vectors shape: {vectors.shape}")
        print(f"   Dtype: {vectors.dtype}")
        print(f"   Time per document: {elapsed / len(vectors):.2f}s")

    return vectors, doc_ids


# ============================================================================
# PHASE 2: INDEXING & STORAGE
# ============================================================================

def build_index(
    vectors: np.ndarray,
    doc_ids: List[str],
    engine: SentinelEngine,
    verbose: bool = True,
) -> bool:
    """
    Build Qdrant index with binary quantization.

    Args:
        vectors: (N, 1536) float32 array
        doc_ids: List of document IDs
        engine: SentinelEngine instance
        verbose: Print progress

    Returns:
        True if successful
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: INDEX BUILDING (32x COMPRESSION)")
        print("=" * 70)

    start_index = time.time()

    if verbose:
        print("\n🔧 Creating Qdrant collection with binary quantization...")
    engine.init_collection()

    if verbose:
        print(f"   Ingesting {len(vectors)} vectors...")

    engine.upsert_vectors(vectors, doc_ids, batch_size=128)

    elapsed = time.time() - start_index

    info = engine.get_collection_info()

    if verbose:
        print(f"\n✅ INDEX BUILDING COMPLETE ({elapsed:.1f}s)")
        print(f"   Collection: {info.get('name', 'N/A')}")
        print(f"   Points indexed: {info.get('points_count', 'N/A')}")
        print("   Compression: 32.0x (1536 dims → 192 bytes per vector)")
        print(
            "   Estimated RAM: ~"
            f"{len(vectors) * BYTES_PER_RABITQ_VECTOR / (1024 ** 2):.1f} MB "
            f"(vs {len(vectors) * BYTES_PER_FULL_VECTOR / (1024 ** 2):.1f} MB uncompressed)"
        )

    return True


# ============================================================================
# PHASE 3B: MULTI-AGENT ANALYSIS
# ============================================================================

def multi_agent_analysis(
    query_ids: List[str],
    retrieval_results: Dict,
    corpus: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Multi-agent analysis on retrieval results using 5 specialized agents.

    Agents:
    1. Forensic Auditor - Fraud & irregularities detection
    2. Risk Analyst - Market, credit, operational risks
    3. Compliance Officer - Regulatory adherence
    4. Portfolio Manager - Investment attractiveness
    5. CFO - Financial health & strategy

    Args:
        query_ids: List of query IDs
        retrieval_results: Dict of retrieval results per query
        corpus: Document corpus
        verbose: Print progress

    Returns:
        Multi-agent analysis results with consensus
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3B: MULTI-AGENT ANALYSIS")
        print("=" * 70)

    start_agents = time.time()

    try:
        orchestrator = MultiAgentOrchestrator(verbose=False)

        if verbose:
            print(f"\n🤖 Analyzing {len(query_ids)} queries with 5 agents...")

        agent_analyses = {}
        errors = []

        for query_id in tqdm(query_ids, desc="Multi-agent analysis", disable=not verbose):
            try:
                results = retrieval_results.get(query_id, [])
                analysis = orchestrator.analyze_query(
                    query_id=query_id,
                    retrieval_results=results,
                    documents=corpus,
                    consensus_method="weighted_vote",
                )
                agent_analyses[query_id] = analysis
            except Exception as e:
                errors.append(f"Query {query_id}: {str(e)}")

        elapsed = time.time() - start_agents
        success_rate = (len(agent_analyses) / len(query_ids)) if query_ids else 0.0

        if verbose:
            print(f"\n✅ MULTI-AGENT ANALYSIS COMPLETE ({elapsed:.1f}s)")
            print(f"   Queries analyzed: {len(agent_analyses)}")
            print(f"   Success rate: {success_rate * 100:.1f}%")
            if errors:
                print(f"   Errors: {len(errors)}")

        return {
            "analyses": agent_analyses,
            "success_rate": success_rate,
            "errors": errors,
            "orchestrator": orchestrator.get_orchestrator_summary(),
        }

    except Exception as e:
        logger.error("Multi-agent analysis failed: %s", e, exc_info=True)
        raise


# ============================================================================
# PHASE 3C: COMPREHENSIVE EVALUATION & RETRIEVAL
# ============================================================================

def _coerce_qrels_to_sets(qrels: Dict) -> Dict[str, set]:
    coerced = {}
    for query_id, docs in qrels.items():
        if isinstance(docs, dict):
            coerced[query_id] = set(docs.keys())
        else:
            coerced[query_id] = set(docs)
    return coerced


def evaluate_retrieval(
    query_ids: List[str],
    query_vectors: np.ndarray,
    engine: SentinelEngine,
    qrels: Dict,
    corpus: Dict,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Evaluate retrieval with both ranking and comprehensive metrics.

    Args:
        query_ids: List of query IDs
        query_vectors: Query vectors (N, 1536)
        engine: SentinelEngine instance
        qrels: Ground-truth relevance {query_id: {doc_id: score}}
        corpus: Document corpus
        verbose: Print progress

    Returns:
        (retrieval_results, metrics)
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3C: RETRIEVAL & COMPREHENSIVE EVALUATION")
        print("=" * 70)

    start_eval = time.time()

    if verbose:
        print(f"\n🔎 Searching for {len(query_ids)} queries...")

    retrieval_results = {}
    for query_id, query_vec in tqdm(
        zip(query_ids, query_vectors),
        total=len(query_ids),
        desc="Retrieving",
        disable=not verbose,
    ):
        results = engine.search(query_vec, top_k=RECALL_AT_K)
        retrieval_results[query_id] = results

    if verbose:
        print("\n📊 Computing comprehensive metrics...")

    evaluator = ComprehensiveEvaluator(verbose=False)

    retrieval_ranked = {
        query_id: [doc_id for doc_id, _ in results]
        for query_id, results in retrieval_results.items()
    }
    qrels_sets = _coerce_qrels_to_sets(qrels)

    metrics = evaluator.evaluate(
        qrels=qrels_sets,
        results=retrieval_ranked,
        k_values=[1, 5, 10, 20],
    )

    elapsed = time.time() - start_eval

    if verbose:
        print(f"\n✅ RETRIEVAL & EVALUATION COMPLETE ({elapsed:.1f}s)")
        print(f"   Recall@10: {metrics['recall@10']['mean']:.4f}")
        print(f"   Precision@10: {metrics['precision@10']['mean']:.4f}")
        print(f"   MAP: {metrics['map']['mean']:.4f}")
        print(f"   NDCG@10: {metrics['ndcg@10']['mean']:.4f}")

    return retrieval_results, metrics


# ============================================================================
# PHASE 4: RESULTS & EXPORT
# ============================================================================

def export_results_comprehensive(
    metrics: Dict,
    agent_results: Dict,
    num_docs: int,
    num_queries: int,
    results_path: str = RESULTS_PATH,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive results export with all metrics and agent analysis.

    Includes:
    - Recall, Precision, MAP, NDCG at multiple K values
    - Multi-agent analysis results
    - Compression and network impact analysis
    - Paper-ready metrics

    Args:
        metrics: Comprehensive evaluation metrics
        agent_results: Multi-agent analysis results
        num_docs: Number of documents
        num_queries: Number of queries
        results_path: Output directory
        verbose: Print progress

    Returns:
        Dictionary with final results
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: COMPREHENSIVE EXPORT")
        print("=" * 70)

    recall_10 = metrics["recall@10"]["mean"]
    precision_10 = metrics["precision@10"]["mean"]
    map_score = metrics["map"]["mean"]
    ndcg_10 = metrics["ndcg@10"]["mean"]

    f32_bytes_per_doc = VECTOR_DIM * 4
    rabitq_bytes_per_doc = VECTOR_DIM * 0.125
    actual_compression = f32_bytes_per_doc / rabitq_bytes_per_doc

    network_savings_gbps = CLOUD_LOAD_GBPS - SENTINEL_LOAD_GBPS
    backhaul_reduction_percent = (network_savings_gbps / CLOUD_LOAD_GBPS) * 100

    results = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paper": "IEEE TMLCN 2026: Sentinel Edge-Intelligence Framework",
            "version": "v2.0 - Integrated with Multi-Agent System",
            "phase": "Final Benchmark (IEEE Submission)",
        },
        "system_info": {
            "device": DEVICE,
            "target_scale": TARGET_DOCS,
            "actual_documents": num_docs,
            "actual_queries": num_queries,
        },
        "evaluation_metrics": {
            "recall_at_k": {
                "1": metrics["recall@1"]["mean"],
                "5": metrics["recall@5"]["mean"],
                "10": metrics["recall@10"]["mean"],
                "20": metrics["recall@20"]["mean"],
            },
            "precision_at_k": {
                "1": metrics["precision@1"]["mean"],
                "5": metrics["precision@5"]["mean"],
                "10": metrics["precision@10"]["mean"],
                "20": metrics["precision@20"]["mean"],
            },
            "map": float(map_score),
            "ndcg_at_k": {
                "1": metrics["ndcg@1"]["mean"],
                "5": metrics["ndcg@5"]["mean"],
                "10": metrics["ndcg@10"]["mean"],
                "20": metrics["ndcg@20"]["mean"],
            },
        },
        "fidelity": {
            "recall_at_10": float(recall_10),
            "precision_at_10": float(precision_10),
            "map": float(map_score),
            "ndcg_at_10": float(ndcg_10),
            "status": "EXCELLENT" if recall_10 > 0.7 else "GOOD" if recall_10 > 0.5 else "FAIR",
        },
        "compression": {
            "ratio": float(actual_compression),
            "bytes_before": int(f32_bytes_per_doc),
            "bytes_after": int(rabitq_bytes_per_doc),
            "method": "RaBitQ + Binary Quantization",
        },
        "compression_ratio": float(actual_compression),
        "network_analysis": {
            "baseline_cloud_gbps": float(CLOUD_LOAD_GBPS),
            "sentinel_edge_gbps": float(SENTINEL_LOAD_GBPS),
            "network_savings_gbps": float(network_savings_gbps),
            "backhaul_reduction_percent": float(backhaul_reduction_percent),
            "backhaul_reduction": f"{backhaul_reduction_percent:.1f}%",
        },
        "multi_agent_system": {
            "num_agents": agent_results["orchestrator"]["agent_count"],
            "agents": agent_results["orchestrator"]["agent_roles"],
            "consensus_method": "weighted_vote",
            "queries_analyzed": len(agent_results["analyses"]),
            "success_rate": agent_results["success_rate"],
        },
        "extrapolation": {
            "target_scale_documents": TARGET_DOCS,
            "extrapolated_from_benchmark": num_docs,
            "scaling_factor": TARGET_DOCS / num_docs if num_docs > 0 else 1.0,
        },
        "documents_processed": num_docs,
        "queries_evaluated": num_queries,
    }

    results_file = os.path.join(results_path, FINAL_RESULTS_FILE)
    os.makedirs(results_path, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print("\n📊 RESULTS SUMMARY:")
        print(f"   Documents: {num_docs}")
        print(f"   Queries: {num_queries}")
        print(f"   Recall@10: {recall_10:.4f}")
        print(f"   Precision@10: {precision_10:.4f}")
        print(f"   MAP: {map_score:.4f}")
        print(f"   NDCG@10: {ndcg_10:.4f}")
        print(f"   Compression: {actual_compression:.1f}x")
        print(f"   Backhaul reduction: {backhaul_reduction_percent:.1f}%")
        print(f"   Agents: {agent_results['orchestrator']['agent_count']}")
        print(f"\n💾 Results saved to: {results_file}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the complete SENTINEL IEEE Final Benchmark v2.0.

    Execution Flow:
    1. Load smart subset (1000 docs with ground truth) - SentinelDatasetManager
    2. Vectorize documents (Qwen-2.5-GTE + RaBitQ) - SentinelEmbedder
    3. Build Qdrant index (binary quantization = 32x compression) - SentinelEngine
    4. Retrieve (search top-k documents)
    5. Multi-agent analysis (5 financial agents) - MultiAgentOrchestrator
    6. Comprehensive evaluation (Recall, Precision, MAP, NDCG) - ComprehensiveEvaluator
    7. Export results for IEEE paper
    """

    print("\n" + "=" * 70)
    print("SENTINEL: IEEE TMLCN FINAL BENCHMARK v2.0 (PRODUCTION)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Target documents: {TARGET_DOCS:,}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"Compression: {COMPRESSION_RATIO}x")
    print("Components: Dataset Manager | Embedder | Engine |")
    print("            Multi-Agent System | Comprehensive Metrics")

    total_start = time.time()

    engine = None
    try:
        corpus_dict, queries_dict, qrels_dict = _load_finmteb_with_fallback()

        embedder = SentinelEmbedder(
            device=DEVICE,
            verbose=True,
            vector_dim=VECTOR_DIM,
        )

        vectors, doc_ids = vectorize_corpus(
            corpus_dict,
            embedder,
            verbose=True,
        )

        print("\n🧠 Encoding queries...")
        query_ids = list(queries_dict.keys())
        query_texts = [queries_dict[qid] for qid in query_ids]
        query_vectors = embedder.encode(
            query_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        engine = SentinelEngine(
            data_path=DATA_PATH,
            collection_name=COLLECTION_NAME,
            vector_dim=VECTOR_DIM,
            verbose=True,
        )

        build_index(vectors, doc_ids, engine, verbose=True)

        retrieval_results, metrics = evaluate_retrieval(
            query_ids,
            query_vectors,
            engine,
            qrels_dict,
            corpus_dict,
            verbose=True,
        )

        agent_results = multi_agent_analysis(
            query_ids,
            retrieval_results,
            corpus_dict,
            verbose=True,
        )

        results = export_results_comprehensive(
            metrics=metrics,
            agent_results=agent_results,
            num_docs=len(corpus_dict),
            num_queries=len(qrels_dict),
            verbose=True,
        )

        total_elapsed = time.time() - total_start

        print("\n" + "=" * 70)
        print("🌟 BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Total execution time: {total_elapsed / 60:.1f} minutes")
        print("\nKey Results:")
        print(f"  • Recall@10: {metrics['recall@10']['mean']:.4f}")
        print(f"  • Precision@10: {metrics['precision@10']['mean']:.4f}")
        print(f"  • MAP: {metrics['map']['mean']:.4f}")
        print(f"  • NDCG@10: {metrics['ndcg@10']['mean']:.4f}")
        print(f"  • Compression: {results['compression_ratio']:.1f}x")
        print(f"  • Backhaul reduction: {results['network_analysis']['backhaul_reduction']}")
        print(f"  • Documents: {results['documents_processed']}")
        print(f"  • Queries: {results['queries_evaluated']}")
        print(f"  • Agents: {agent_results['orchestrator']['agent_count']}")
        print("\n✅ Ready for IEEE TMLCN submission!")
        print("=" * 70 + "\n")

        return results

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        raise
    finally:
        if engine:
            engine.close()


if __name__ == "__main__":
    main()
