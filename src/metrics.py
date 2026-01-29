import numpy as np

class ComprehensiveEvaluator:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _precision_at_k(self, relevant, retrieved, k):
        if k == 0:
            return 0.0
        return len(relevant.intersection(retrieved[:k])) / k

    def _recall_at_k(self, relevant, retrieved, k):
        if not relevant:
            return 0.0
        return len(relevant.intersection(retrieved[:k])) / len(relevant)

    def _average_precision(self, relevant, retrieved):
        if not relevant:
            return 0.0
        hits = 0
        precision_sum = 0.0
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / idx
        return precision_sum / len(relevant)

    def _ndcg_at_k(self, relevant, retrieved, k):
        if not relevant:
            return 0.0
        dcg = 0.0
        for idx, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(idx + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, qrels, results, k_values):
        metrics = {
            "map": {"mean": 0.0},
        }
        for k in k_values:
            metrics[f"recall@{k}"] = {"mean": 0.0}
            metrics[f"precision@{k}"] = {"mean": 0.0}
            metrics[f"ndcg@{k}"] = {"mean": 0.0}

        query_ids = [qid for qid in results.keys() if qid in qrels]
        if not query_ids:
            return metrics

        map_scores = []
        for qid in query_ids:
            relevant = qrels[qid]
            retrieved = results.get(qid, [])

            for k in k_values:
                metrics[f"recall@{k}"]["mean"] += self._recall_at_k(relevant, retrieved, k)
                metrics[f"precision@{k}"]["mean"] += self._precision_at_k(relevant, retrieved, k)
                metrics[f"ndcg@{k}"]["mean"] += self._ndcg_at_k(relevant, retrieved, k)

            map_scores.append(self._average_precision(relevant, retrieved))

        n_queries = len(query_ids)
        for k in k_values:
            metrics[f"recall@{k}"]["mean"] /= n_queries
            metrics[f"precision@{k}"]["mean"] /= n_queries
            metrics[f"ndcg@{k}"]["mean"] /= n_queries

        metrics["map"]["mean"] = float(np.mean(map_scores)) if map_scores else 0.0
        return metrics


class RecallCalculator:
    @staticmethod
    def recall_at_k(relevant, retrieved, k):
        if not relevant:
            return 0.0
        return len(set(retrieved[:k]).intersection(relevant)) / len(relevant)


def calculate_network_impact(n_queries, k=10, vec_dim=1024):
    """
    Calculates the Backhaul Traffic Reduction for IEEE TMLCN.
    
    Scenario:
    - Cloud Mode: Edge sends Query Vector (4KB) -> Cloud sends 10 Result Vectors (40KB).
    - Sentinel Mode: Edge does local Retrieval -> Sends only Final Text Verdict (0.5KB).
    """
    # 1. Physics Constants
    FLOAT32_BYTES = 4
    BINARY_BYTES = 0.125  # 1 bit
    METADATA_BYTES = 500  # Avg size of a JSON text verdict
    
    # 2. Cloud Architecture (Vector Offloading)
    # Traffic = Uplink (Query) + Downlink (Top-K Vectors + Metadata)
    cloud_uplink = n_queries * (vec_dim * FLOAT32_BYTES)
    cloud_downlink = n_queries * k * (vec_dim * FLOAT32_BYTES + METADATA_BYTES)
    total_cloud_bytes = cloud_uplink + cloud_downlink
    
    # 3. Sentinel Edge Architecture (Semantic Offloading)
    # Traffic = Uplink (Text Query) + Downlink (Text Answer) - No Vectors transmitted!
    # We assume purely text-based intent communication
    edge_uplink = n_queries * 100  # Approx 100 bytes for text query
    edge_downlink = n_queries * METADATA_BYTES  # Final text answer
    total_edge_bytes = edge_uplink + edge_downlink
    
    # 4. Conversion to Gbps (assuming 1-second burst for 10k users)
    cloud_gbps = (total_cloud_bytes * 8) / (1024**3)
    edge_gbps = (total_edge_bytes * 8) / (1024**3)
    
    reduction_factor = cloud_gbps / edge_gbps if edge_gbps > 0 else 0
    
    return {
        "cloud_gbps": cloud_gbps,
        "edge_gbps": edge_gbps,
        "reduction_factor": reduction_factor,
        "bandwidth_saved_percent": (1 - (edge_gbps / cloud_gbps)) * 100
    }

def calculate_network_load(n_queries, k=10, vec_dim=1536):
    FLOAT32_BYTES = 4
    TEXT_BYTES = 500
    cloud_bytes = n_queries * (vec_dim * FLOAT32_BYTES + k * vec_dim * FLOAT32_BYTES)
    edge_bytes = n_queries * TEXT_BYTES
    cloud_gbps = (cloud_bytes * 8) / (1024**3)
    edge_gbps = (edge_bytes * 8) / (1024**3)
    return cloud_gbps, edge_gbps


def compute_topological_integrity(qrels, query_id, retrieved_ids):
    """
    Measures 'Retrieval Integrity': Does the 1-bit Edge Topology matches the Cloud Topology?
    Uses 'Gold Standard' Qrels from FinMTEB.
    """
    ground_truth = set(qrels.get(query_id, []))
    
    if not ground_truth:
        return 0.0, False  # Skip invalid queries

    # Recall calculation
    retrieved_set = set(retrieved_ids)
    intersection = len(ground_truth.intersection(retrieved_set))
    recall = intersection / len(ground_truth)
    
    # Hit calculation (for MRR)
    is_hit = False
    if retrieved_ids and retrieved_ids[0] in ground_truth:
        is_hit = True
        
    return recall, is_hit


def compute_fidelity_with_qrels(qrels, query_id, retrieved_ids):
    return compute_topological_integrity(qrels, query_id, retrieved_ids)
