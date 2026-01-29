# sentinel_finmteb_lab

## Streamlit Benchmark Dashboard

Run the IEEE final benchmark from a live Streamlit UI and view metrics/logs:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Use the **Start Benchmark** button to execute `run_ieee_final.py` and monitor output in the log panel. The latest metrics are read from `results/final_ieee_data.json` after the run completes.
