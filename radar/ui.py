"""Styling and small HTML components for the Streamlit app."""
from __future__ import annotations

import html

import pandas as pd

ACCENT = "#0f5c4a"
INK = "#111827"
INK_2 = "#4b5563"
MUTED = "#6b7280"
LINE = "#e5e7eb"
SURFACE = "#ffffff"
MUTED_BAR = "#b8bec7"

CSS = f"""
<style>
#MainMenu, footer, [data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {INK}; }}

.brand {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.brand .mark {{ font-size: 1.55rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
.brand .mark span {{ color: {ACCENT}; }}
.brand .tag {{ font-size: .78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: {MUTED}; }}
.lede {{ color: {INK_2}; font-size: .95rem; margin: 4px 0 18px; max-width: 760px; }}

.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px;
  margin-bottom: 8px; }}
@media (max-width: 800px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 14px 16px; }}
.kpi .label {{ font-size: .78rem; color: {MUTED}; font-weight: 500; }}
.kpi .value {{ font-size: 1.6rem; font-weight: 600; color: {INK}; line-height: 1.25;
  font-variant-numeric: tabular-nums; }}
.kpi .note {{ font-size: .75rem; color: {MUTED}; }}

.panel {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px;
  padding: 20px 22px; margin-top: 8px; }}
.panel h3 {{ font-size: 1.05rem; margin: 0 0 4px; line-height: 1.4; }}
.meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px 24px; margin: 14px 0 4px; }}
.meta div {{ font-size: .85rem; color: {INK}; }}
.meta b {{ display: block; font-size: .72rem; font-weight: 600; color: {MUTED};
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }}
.pill {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: 2px 8px;
  border-radius: 999px; border: 1px solid {LINE}; color: {INK_2}; margin-right: 6px;
  background: #f9fafb; }}
.pill.accent {{ color: {ACCENT}; border-color: #b7d7cd; background: #eef6f3; }}
.pill.warn {{ color: #92400e; border-color: #f3d19c; background: #fffbeb; }}
.notice {{ border-left: 3px solid #d97706; background: #fffbeb; padding: 10px 14px;
  font-size: .85rem; color: #78350f; border-radius: 0 6px 6px 0; margin: 10px 0; }}
.section-label {{ font-size: .72rem; font-weight: 600; color: {MUTED}; text-transform: uppercase;
  letter-spacing: .06em; margin: 18px 0 6px; }}
.brief p, .brief li {{ font-size: .9rem; color: {INK}; }}
.chart-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin: 12px 0 6px; }}
.small {{ font-size: .8rem; color: {MUTED}; }}
.doclinks a {{ display: inline-block; margin: 0 14px 6px 0; font-size: .85rem; color: {ACCENT}; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 500; }}
</style>
"""


def esc(value) -> str:
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value))
                       else str(value))


def fmt_date(ts, with_time: bool = True) -> str:
    if ts is None or pd.isna(ts):
        return "Not stated"
    ts = pd.Timestamp(ts)
    if with_time and (ts.hour or ts.minute):
        return ts.strftime("%a %d %b %Y, %H:%M")
    return ts.strftime("%a %d %b %Y")


def kpis(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="kpi"><div class="label">{esc(l)}</div>'
                    f'<div class="value">{esc(v)}</div><div class="note">{esc(n)}</div></div>'
                    for l, v, n in items)
    return f'<div class="kpis">{cells}</div>'


def bar_layout(fig, height: int, x_title: str = ""):
    fig.update_layout(
        height=height, margin=dict(l=8, r=24, t=8, b=8), bargap=0.28,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, showlegend=False,
        font=dict(family="Source Sans Pro, Source Sans 3, system-ui, sans-serif", size=12, color=INK_2),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK, size=12)),
    )
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor="#f0f1f3", zeroline=False,
                     linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(title="", showgrid=False, automargin=True, linecolor=LINE, tickfont=dict(color=INK_2))
    fig.update_traces(marker_color=ACCENT, marker_cornerradius=4)
    return fig
