"""
Red Team Report Generator.

Produces HTML reports with interactive visualizations from
red-team assessment results. Includes:
- Executive summary with risk levels
- Jailbreak success heatmaps
- Bias elicitation grid heatmaps
- Escalation probe timelines
- Detailed per-attack breakdowns
"""

import json
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

from src.config import config


class RedTeamReportGenerator:
    """Generates HTML reports from red-team results."""

    def generate_html_report(
        self,
        report: Dict,
        output_dir: str = None,
    ) -> str:
        """Generate a full HTML report from red-team results.

        Returns the path to the generated HTML file.
        """
        output_dir = output_dir or str(Path(config.reports_dir) / "red_team")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        charts = []

        # 1. Jailbreak success rate comparison
        fig = self._jailbreak_comparison_chart(report)
        if fig:
            charts.append(("Jailbreak Success Rate by Model", fig))

        # 2. Attack category effectiveness
        fig = self._attack_category_chart(report)
        if fig:
            charts.append(("Attack Category Effectiveness", fig))

        # 3. Bias elicitation heatmaps
        for model_name, model_report in report.get("model_results", {}).items():
            grid = model_report.get("grid_probe", {})
            heatmap = grid.get("heatmap", {})
            if heatmap:
                fig = self._bias_heatmap(heatmap, model_name)
                if fig:
                    charts.append((f"Bias Heatmap — {model_name}", fig))

        # 4. Escalation timelines
        for model_name, model_report in report.get("model_results", {}).items():
            escalations = model_report.get("escalation", [])
            for esc in escalations:
                fig = self._escalation_timeline(esc, model_name)
                if fig:
                    charts.append((
                        f"Escalation — {model_name} ({esc['group']}/{esc['topic']})",
                        fig,
                    ))

        # Build HTML
        html_content = self._build_html(report, charts)
        output_path = Path(output_dir) / "red_team_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")
        return str(output_path)

    def _jailbreak_comparison_chart(self, report: Dict) -> go.Figure:
        """Bar chart comparing jailbreak success rates."""
        summary = report.get("summary", {})
        if not summary:
            return None

        models = list(summary.keys())
        rates = [
            (summary[m].get("jailbreak_success_rate") or 0) * 100
            for m in models
        ]
        risk_levels = [summary[m].get("risk_level", "?") for m in models]
        colors = [self._risk_color(rl) for rl in risk_levels]

        fig = go.Figure(data=[
            go.Bar(
                x=models, y=rates,
                marker_color=colors,
                text=[f"{r:.1f}%" for r in rates],
                textposition="outside",
            )
        ])
        fig.update_layout(
            yaxis_title="Jailbreak Success Rate (%)",
            yaxis_range=[0, max(rates + [20]) * 1.2],
            template="plotly_white",
        )
        fig.add_hline(
            y=config.red_team_max_success_rate * 100,
            line_dash="dash", line_color="red",
            annotation_text=f"Threshold ({config.red_team_max_success_rate:.0%})",
        )
        return fig

    def _attack_category_chart(self, report: Dict) -> go.Figure:
        """Grouped bar chart of attack success by category and model."""
        model_results = report.get("model_results", {})
        if not model_results:
            return None

        fig = go.Figure()
        for model_name, model_report in model_results.items():
            by_cat = model_report.get("jailbreak", {}).get("by_attack_category", {})
            if by_cat:
                categories = sorted(by_cat.keys())
                rates = [by_cat[c]["success_rate"] * 100 for c in categories]
                fig.add_trace(go.Bar(name=model_name, x=categories, y=rates))

        fig.update_layout(
            barmode="group",
            yaxis_title="Success Rate (%)",
            template="plotly_white",
        )
        return fig

    def _bias_heatmap(self, heatmap_data: Dict, model_name: str) -> go.Figure:
        """Create a group × topic bias score heatmap."""
        if not heatmap_data.get("values"):
            return None

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data["values"],
            x=heatmap_data["columns"],
            y=heatmap_data["index"],
            colorscale="RdYlGn_r",
            zmin=0, zmax=3,
            colorbar_title="Bias Score",
        ))
        fig.update_layout(
            title=f"Bias Elicitation Grid — {model_name}",
            xaxis_title="Topic",
            yaxis_title="Group",
            template="plotly_white",
        )
        return fig

    def _escalation_timeline(self, esc_data: Dict, model_name: str) -> go.Figure:
        """Create a line chart showing bias score across escalation levels."""
        results = esc_data.get("escalation_results", [])
        if not results:
            return None

        levels = [r["level"] for r in results]
        scores = [r["bias_score"] for r in results]
        labels = [r["bias_label"] for r in results]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=levels, y=scores,
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=12, color=scores, colorscale="RdYlGn_r", cmin=0, cmax=3),
            line=dict(width=2),
        ))
        fig.update_layout(
            xaxis_title="Escalation Level",
            yaxis_title="Bias Score",
            yaxis_range=[-0.5, 3.5],
            template="plotly_white",
        )
        broke = esc_data.get("broke_at_level")
        if broke is not None:
            fig.add_vline(
                x=broke, line_dash="dash", line_color="red",
                annotation_text="Model broke here",
            )
        return fig

    def _build_html(self, report: Dict, charts: list) -> str:
        """Build the complete HTML report."""
        summary = report.get("summary", {})

        # Summary table
        summary_rows = ""
        for model, data in summary.items():
            risk = data.get("risk_level", "?")
            color = self._risk_color(risk)
            jb_rate = data.get("jailbreak_success_rate")
            jb_str = f"{jb_rate:.1%}" if jb_rate is not None else "N/A"
            passes = "✅" if data.get("passes_threshold") else "❌"

            summary_rows += f"""
            <tr>
                <td><strong>{model}</strong></td>
                <td>{jb_str}</td>
                <td>{data.get('mean_bias_score', 'N/A')}</td>
                <td>{data.get('grid_overall_bias_rate', 'N/A')}</td>
                <td>{data.get('escalation_resilient', 'N/A')}</td>
                <td style="background-color:{color};color:white;font-weight:bold;text-align:center">{risk}</td>
                <td style="text-align:center">{passes}</td>
            </tr>"""

        # Chart divs
        chart_html = ""
        for i, (title, fig) in enumerate(charts):
            chart_div = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
            chart_html += f"""
            <div class="chart-container">
                <h3>{title}</h3>
                {chart_div}
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Red Team Assessment Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header p {{ margin: 5px 0; opacity: 0.8; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .risk-badge {{ padding: 4px 12px; border-radius: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔴 Red Team Assessment Report</h1>
            <p>Generated: {report.get('timestamp', 'N/A')}</p>
            <p>Models tested: {', '.join(report.get('models_tested', []))}</p>
            <p>Total attack prompts: {report.get('total_attack_prompts', 'N/A')}</p>
            <p>Judge model: {report.get('judge_model', 'N/A')}</p>
        </div>

        <div class="card">
            <h2>Executive Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Jailbreak Rate</th>
                        <th>Mean Bias Score</th>
                        <th>Grid Bias Rate</th>
                        <th>Escalation Resilience</th>
                        <th>Risk Level</th>
                        <th>Pass?</th>
                    </tr>
                </thead>
                <tbody>{summary_rows}</tbody>
            </table>
        </div>

        {chart_html}

        <div class="card" style="text-align:center; color:#888;">
            <p>Generated by LLM Bias Sentinel — Red Team Module</p>
            <p>All evaluations performed using local Ollama models (zero API cost)</p>
        </div>
    </div>
</body>
</html>"""

    def _risk_color(self, risk_level: str) -> str:
        """Map risk level to a color."""
        return {
            "HIGH": "#dc3545",
            "MEDIUM": "#fd7e14",
            "LOW": "#ffc107",
            "MINIMAL": "#28a745",
        }.get(risk_level, "#6c757d")
