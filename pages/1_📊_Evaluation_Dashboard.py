import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.markdown("# 📊 System-Wide Evaluation Dashboard")
st.write("Review the offline evaluation results of various pipeline configurations against the ground truth dataset.")

csv_path = "evaluation/metrics_report.csv"

if not os.path.exists(csv_path):
    st.warning("Metrics report not found. Run `scripts/run_evals.py` first to generate the dataset.")
else:
    df = pd.read_csv(csv_path)
    
    # Clean up column names for display
    df["Config"] = df.apply(lambda row: f"{row['config_chunking']} | {row['config_index']} | MMR:{row['config_mmr']} | CE:{row['config_reranker']}", axis=1)
    
    st.markdown("### Aggregated Performance by Configuration")
    agg_df = df.groupby("Config").agg({
        "mrr": "mean",
        "precision": "mean",
        "recall": "mean",
        "trap_avoidance": "mean",
        "faithfulness": "mean",
        "relevancy": "mean",
        "context_precision": "mean",
        "latency": "mean"
    }).reset_index()
    
    # Display the aggregated metrics in a styled dataframe
    st.dataframe(
        agg_df.style.format({
            "mrr": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "trap_avoidance": "{:.1%}",
            "faithfulness": "{:.3f}",
            "relevancy": "{:.3f}",
            "context_precision": "{:.3f}",
            "latency": "{:.2f}s"
        }),
        use_container_width=True
    )
    
    st.markdown("### 📈 Metric Comparisons")
    metric_choice = st.selectbox("Select metric to visualize:", ["mrr", "recall", "faithfulness", "relevancy", "context_precision", "latency"])
    
    fig = px.bar(agg_df, x="Config", y=metric_choice, color="Config", title=f"Average {metric_choice.title()} by Configuration")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Raw Test Case Results")
    st.dataframe(df.drop(columns=["Config"]), use_container_width=True)
