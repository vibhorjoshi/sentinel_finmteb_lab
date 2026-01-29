import json
import os
import re
import subprocess
from datetime import datetime

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "final_ieee_data.json")
LOG_FILE = os.path.join(RESULTS_DIR, "benchmark_run.log")

PIPELINE_STEPS = [
    ("Phase 0", "Smart subset loading"),
    ("Phase 1", "Vectorization"),
    ("Phase 2", "Index build"),
    ("Phase 3", "Retrieval + evaluation"),
    ("Phase 3B", "Multi-agent analysis"),
    ("Phase 4", "Export results"),
]

LEGACY_METRICS_MAP = {
    "recall@10": "fidelity_recall_at_10",
    "compression_ratio": "compression_ratio",
    "cloud_load_gbps": "projected_100k_cloud_load_gbps",
    "sentinel_load_gbps": "projected_100k_sentinel_load_gbps",
}


def _load_results():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def _read_logs():
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE, "r") as f:
        return f.read()


def _infer_pipeline_state(log_text):
    state = {key: "pending" for key, _ in PIPELINE_STEPS}
    if not log_text:
        return state
    if "PHASE 0" in log_text:
        state["Phase 0"] = "complete"
    if "PHASE 1: DOCUMENT VECTORIZATION" in log_text or "Encoding" in log_text:
        state["Phase 1"] = "complete"
    if "PHASE 2: INDEX BUILDING" in log_text:
        state["Phase 2"] = "complete"
    if "PHASE 3C: RETRIEVAL & COMPREHENSIVE EVALUATION" in log_text:
        state["Phase 3"] = "complete"
    if "PHASE 3B: MULTI-AGENT ANALYSIS" in log_text:
        state["Phase 3B"] = "complete"
    if "PHASE 4: COMPREHENSIVE EXPORT" in log_text:
        state["Phase 4"] = "complete"

    if "BENCHMARK COMPLETE" not in log_text:
        for key in state:
            if state[key] == "complete":
                continue
            if re.search(rf"{key}[:\s]", log_text, re.IGNORECASE):
                state[key] = "running"
                break
        else:
            for key in state:
                if state[key] == "complete":
                    continue
                state[key] = "running"
                break
    return state


def _extract_summary_metrics(results):
    if not results:
        return {}
    metrics = results.get("evaluation_metrics")
    if metrics:
        recall_at_k = metrics.get("recall_at_k", {})
        precision_at_k = metrics.get("precision_at_k", {})
        ndcg_at_k = metrics.get("ndcg_at_k", {})
        return {
            "recall@10": recall_at_k.get("10", 0),
            "precision@10": precision_at_k.get("10", 0),
            "ndcg@10": ndcg_at_k.get("10", 0),
            "map": metrics.get("map", 0),
            "legacy": False,
        }

    return {
        "recall@10": results.get(LEGACY_METRICS_MAP["recall@10"], 0),
        "precision@10": 0,
        "ndcg@10": 0,
        "map": 0,
        "legacy": True,
    }


def _start_benchmark(target_docs, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_handle = open(LOG_FILE, "w")
    env = os.environ.copy()
    env["SENTINEL_TARGET_DOCS"] = str(target_docs)
    env["SENTINEL_DEVICE"] = device
    process = subprocess.Popen(
        ["python", "-u", "run_ieee_final.py"],
        cwd=BASE_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return process


def _stop_benchmark(process):
    if process and process.poll() is None:
        process.terminate()


st.set_page_config(page_title="SENTINEL IEEE Benchmark", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 45%, #111827 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge.pending { background: #e2e8f0; color: #475569; }
    .badge.running { background: #fde68a; color: #92400e; }
    .badge.complete { background: #bbf7d0; color: #166534; }
    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        background: white;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.05);
    }
    .metric-label { color: #64748b; font-size: 0.8rem; }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .panel-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>SENTINEL IEEE Final Benchmark v2.0</h1>
        <p>Live orchestration dashboard for smart loading, RaBitQ vectorization, retrieval evaluation, and multi-agent analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "benchmark_process" not in st.session_state:
    st.session_state.benchmark_process = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None

with st.sidebar:
    st.markdown("## Live Orchestration Pipeline")
    log_text = _read_logs()
    pipeline_state = _infer_pipeline_state(log_text)
    for step_key, step_label in PIPELINE_STEPS:
        state = pipeline_state.get(step_key, "pending")
        st.markdown(
            f"<div style='margin-bottom:0.6rem;'>"
            f"<span class='badge {state}'>{state}</span> "
            f"<strong>{step_key}</strong> · {step_label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Run Configuration")
    target_docs = st.number_input(
        "Target documents (smart subset)",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100,
    )
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    if st.button("Start Benchmark", type="primary"):
        if st.session_state.benchmark_process and st.session_state.benchmark_process.poll() is None:
            st.warning("Benchmark is already running.")
        else:
            st.session_state.benchmark_process = _start_benchmark(target_docs, device)
            st.session_state.last_run = datetime.utcnow().isoformat()
            st.success("Benchmark started. Logs are streaming below.")

    if st.button("Stop Benchmark"):
        _stop_benchmark(st.session_state.benchmark_process)
        st.session_state.benchmark_process = None
        st.warning("Benchmark process terminated.")

    st.markdown("---")
    st.markdown("### Live Refresh")
    auto_refresh = st.toggle("Auto-refresh every 5s", value=True)
    if auto_refresh:
        st.caption("Refreshing logs & results automatically.")
        st.autorefresh(interval=5000, key="auto_refresh")

col_status, col_metrics = st.columns([1.1, 1.4])

with col_status:
    st.markdown("### Benchmark Status")
    process = st.session_state.benchmark_process
    if process and process.poll() is None:
        st.info("Benchmark is running.")
    elif process and process.poll() is not None:
        st.success("Benchmark completed.")
    else:
        st.write("No benchmark is currently running.")

    if st.session_state.last_run:
        st.write(f"Last run started at: {st.session_state.last_run} UTC")

    if log_text:
        st.markdown("#### Latest Log Snippet")
        st.code("\n".join(log_text.splitlines()[-12:]), language="text")
    else:
        st.write("No logs yet. Start a benchmark to generate logs.")

with col_metrics:
    st.markdown("### Latest Results")
    results = _load_results()
    if results:
        summary = _extract_summary_metrics(results)
        if summary.get("legacy"):
            st.info("Legacy results format detected. Showing available metrics.")

        st.markdown("#### Summary Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].markdown(
            f"<div class='metric-card'><div class='metric-label'>Recall@10</div>"
            f"<div class='metric-value'>{summary.get('recall@10', 0):.4f}</div></div>",
            unsafe_allow_html=True,
        )
        metric_cols[1].markdown(
            f"<div class='metric-card'><div class='metric-label'>Precision@10</div>"
            f"<div class='metric-value'>{summary.get('precision@10', 0):.4f}</div></div>",
            unsafe_allow_html=True,
        )
        metric_cols[2].markdown(
            f"<div class='metric-card'><div class='metric-label'>NDCG@10</div>"
            f"<div class='metric-value'>{summary.get('ndcg@10', 0):.4f}</div></div>",
            unsafe_allow_html=True,
        )
        metric_cols[3].markdown(
            f"<div class='metric-card'><div class='metric-label'>MAP</div>"
            f"<div class='metric-value'>{summary.get('map', 0):.4f}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Full Results")
        st.json(results)
    else:
        st.write("No benchmark results found. Run the benchmark to populate results.")

st.markdown("---")
st.markdown("### Live Logs")
st.text(log_text if log_text else "No logs yet. Start a benchmark to generate logs.")
