import os
import torch

N_SAMPLES = 100000        # The 100K target for your paper
TARGET_DOCS = 1000        # IEEE final benchmark target
VECTOR_DIM = 1536         # Qwen-2.5 GTE Dimension
RABITQ_EPSILON = 1.9      # 95% Confidence Bound for Rescoring
COMPRESSION_RATIO = 32.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
COLLECTION_NAME = "sentinel_100k_manifold"
GT_COLLECTION = f"{COLLECTION_NAME}_float32"
BQ_COLLECTION = COLLECTION_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINAL_RESULTS_FILE = "final_ieee_data.json"
RECALL_AT_K = 10

CLOUD_LOAD_GBPS = 160.0
SENTINEL_LOAD_GBPS = CLOUD_LOAD_GBPS / COMPRESSION_RATIO
BYTES_PER_FULL_VECTOR = VECTOR_DIM * 4
BYTES_PER_RABITQ_VECTOR = VECTOR_DIM / 8
DEFAULT_PERSONA = "Forensic Auditor"

CONCURRENT_NODES = 10000
EMBEDDING_BATCH_SIZE = 64
ENABLE_RERANKING = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 50
RERANK_BATCH_SIZE = 16

# --- FINANCIAL PERSONAS (Fin-E5 Methodology) ---
FINANCIAL_PERSONAS = {
    "Forensic Auditor": "Detect control failures, fraud indicators, anomalies",
    "Equity Analyst": "Assess revenue quality, margins, competitive positioning",
    "Risk Manager": "Identify market, credit, liquidity, operational risks",
    "Compliance Officer": "Verify regulatory adherence, governance, disclosures",
    "Tax Strategist": "Analyze tax efficiency, structuring, planning strategies"
}
