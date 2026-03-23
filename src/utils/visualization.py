"""Visualization utilities for benchmark results."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger


def load_results(results_path: str = "reports/benchmark_results/all_results.json") -> list:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def load_comparison_matrix(
    matrix_path: str = "reports/model_comparison_matrix.csv",
) -> pd.DataFrame:
    """Load comparison matrix CSV."""
    return pd.read_csv(matrix_path)


def create_model_comparison_heatmap(
    matrix_path: str = "reports/model_comparison_matrix.csv",
    output_path: str = "reports/model_comparison_heatmap.html",
) -> go.Figure:
    """Create an interactive heatmap comparing models across metrics."""
    df = load_comparison_matrix(matrix_path)

    # Pivot: models as rows, metrics as columns
    metric_cols = [c for c in df.columns if c not in ("model", "benchmark")]
    pivot_data = df.groupby("model")[metric_cols].first().reset_index()

    # Create heatmap
    fig = px.imshow(
        pivot_data[metric_cols].values,
        x=metric_cols,
        y=pivot_data["model"].values,
        labels=dict(x="Metric", y="Model", color="Score"),
        color_continuous_scale="RdYlGn_r",
        title="Model Bias Comparison Matrix",
        aspect="auto",
    )

    fig.update_layout(
        width=1000,
        height=400,
        font=dict(size=12),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Heatmap saved to {output_path}")
    return fig


def create_benchmark_bar_chart(
    results: list,
    benchmark: str,
    metric_key: str,
    title: str | None = None,
    output_path: str | None = None,
) -> go.Figure:
    """Create a bar chart comparing models on a specific benchmark metric."""
    filtered = [r for r in results if r.get("benchmark") == benchmark and metric_key in r]

    if not filtered:
        logger.warning(f"No results for {benchmark}/{metric_key}")
        return None

    models = [r["model"] for r in filtered]
    values = [r[metric_key] for r in filtered]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models, y=values, marker_color=["#636EFA", "#EF553B", "#00CC96"][: len(models)]
            )
        ]
    )
    fig.update_layout(
        title=title or f"{benchmark} — {metric_key}",
        xaxis_title="Model",
        yaxis_title=metric_key,
        template="plotly_white",
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def create_stereotype_radar(
    results: list,
    output_path: str = "reports/stereotype_radar.html",
) -> go.Figure:
    """Create a radar chart of stereotype scores by bias type."""
    stereoset_results = [r for r in results if r.get("benchmark") == "StereoSet"]

    if not stereoset_results:
        logger.warning("No StereoSet results found")
        return None

    fig = go.Figure()
    for result in stereoset_results:
        per_type = result.get("per_bias_type", {})
        categories = list(per_type.keys())
        values = [per_type[c]["stereotype_score"] for c in categories]
        # Close the radar
        categories.append(categories[0])
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=result["model"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Stereotype Scores by Bias Type (50 = ideal)",
        template="plotly_white",
    )

    fig.write_html(output_path)
    logger.info(f"Radar chart saved to {output_path}")
    return fig


def create_sentiment_disparity_chart(
    results: list,
    output_path: str = "reports/sentiment_disparity_chart.html",
) -> go.Figure:
    """Create grouped bar chart of sentiment by demographic group."""
    sentiment_results = [r for r in results if r.get("benchmark") == "SentimentDisparity"]

    if not sentiment_results:
        logger.warning("No SentimentDisparity results found")
        return None

    fig = go.Figure()
    for result in sentiment_results:
        for category, data in result.get("per_category", {}).items():
            groups = list(data["per_group_means"].keys())
            means = list(data["per_group_means"].values())
            fig.add_trace(
                go.Bar(
                    name=f"{result['model']} - {category}",
                    x=groups,
                    y=means,
                )
            )

    fig.update_layout(
        title="Sentiment Polarity by Demographic Group",
        xaxis_title="Group",
        yaxis_title="Mean Sentiment Polarity",
        barmode="group",
        template="plotly_white",
    )

    fig.write_html(output_path)
    logger.info(f"Sentiment chart saved to {output_path}")
    return fig
