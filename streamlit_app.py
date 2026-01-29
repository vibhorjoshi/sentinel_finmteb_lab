import json
import os
import subprocess
from datetime import datetime

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "final_ieee_data.json")
LOG_FILE = os.path.join(RESULTS_DIR, "benchmark_run.log")


def _load_results():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


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

st.title("SENTINEL IEEE Final Benchmark (v2.0)")
st.caption("Run the IEEE benchmark and review metrics in a live Streamlit dashboard.")

if "benchmark_process" not in st.session_state:
    st.session_state.benchmark_process = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None

col_controls, col_status = st.columns([1, 2])

with col_controls:
    st.subheader("Run Configuration")
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

with col_status:
    st.subheader("Benchmark Status")
    process = st.session_state.benchmark_process
    if process and process.poll() is None:
        st.info("Benchmark is running.")
    elif process and process.poll() is not None:
        st.success("Benchmark completed.")
    else:
        st.write("No benchmark is currently running.")

    if st.session_state.last_run:
        st.write(f"Last run started at: {st.session_state.last_run} UTC")

st.divider()

st.subheader("Live Logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        st.text(f.read())
else:
    st.write("No logs yet. Start a benchmark to generate logs.")

st.divider()

st.subheader("Latest Results")
results = _load_results()
if results:
    st.json(results)

    metrics = results.get("evaluation_metrics", {})
    recall_at_k = metrics.get("recall_at_k", {})
    precision_at_k = metrics.get("precision_at_k", {})
    ndcg_at_k = metrics.get("ndcg_at_k", {})

    st.markdown("### Summary Metrics")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Recall@10", f"{recall_at_k.get('10', 0):.4f}")
    col_b.metric("Precision@10", f"{precision_at_k.get('10', 0):.4f}")
    col_c.metric("NDCG@10", f"{ndcg_at_k.get('10', 0):.4f}")
    col_d.metric("MAP", f"{metrics.get('map', 0):.4f}")
else:
    st.write("No benchmark results found. Run the benchmark to populate results.")
