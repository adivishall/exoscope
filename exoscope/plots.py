"""Plotly figures. Colours follow fixed roles across the whole app:

    observations  muted grey     (raw data points)
    threshold     blue  (slot 1) (threshold detector, set A)
    BLS           orange (slot 2) (BLS detector, set B)
    shared/model  aqua  (slot 3) (A ∩ B, fitted models, binned data)

Backgrounds are left transparent so Streamlit's theme (light or dark) applies.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .detection.bls import BLSCandidate
from .detection.threshold import ThresholdResult
from .dmath.number_theory import phase_fold
from .dmath.relations import Relation
from .dmath.sets import EventSetComparison
from .processing import ProcessedLightCurve

C_OBS = "#898781"
C_THR = "#3987e5"
C_BLS = "#d95926"
C_SHARED = "#199e70"
C_VIOLET = "#9085e9"
C_WARN = "#fab219"
GRID = "rgba(137,135,129,0.25)"

DEFAULT_HEIGHT = 380


def _layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = DEFAULT_HEIGHT) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="closest",
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    return fig


def _scatter(n: int):
    return go.Scattergl if n > 3000 else go.Scatter


def time_label(time_format: str) -> str:
    return {"bkjd": "Time [BKJD = BJD − 2454833]", "btjd": "Time [BTJD = BJD − 2457000]"}.get(
        time_format, "Time [days]")


# ---------------------------------------------------------------- light curves
def fig_raw(time, flux, quality_ok, time_format: str, title: str = "Original light curve") -> go.Figure:
    fig = go.Figure()
    good = np.asarray(quality_ok) & np.isfinite(flux)
    S = _scatter(len(time))
    fig.add_trace(S(x=time[good], y=flux[good], mode="markers", name="Measurements",
                    marker=dict(size=3, color=C_OBS), hovertemplate="t=%{x:.4f}<br>flux=%{y:.6g}<extra></extra>"))
    bad = ~np.asarray(quality_ok) & np.isfinite(flux)
    if bad.any():
        fig.add_trace(S(x=time[bad], y=flux[bad], mode="markers", name="Quality-flagged",
                        marker=dict(size=5, color=C_WARN, symbol="x"),
                        hovertemplate="flagged<br>t=%{x:.4f}<br>flux=%{y:.6g}<extra></extra>"))
    return _layout(fig, title, time_label(time_format), "Flux")


def fig_detrend(lc: ProcessedLightCurve, time_format: str) -> go.Figure:
    fig = go.Figure()
    S = _scatter(len(lc.norm_time))
    fig.add_trace(S(x=lc.norm_time, y=lc.norm_flux, mode="markers", name="Normalised flux",
                    marker=dict(size=3, color=C_OBS), hovertemplate="t=%{x:.4f}<br>%{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=lc.norm_time, y=lc.trend, mode="lines", name="Fitted trend",
                             line=dict(color=C_SHARED, width=2), hoverinfo="skip"))
    if lc.clipped_time is not None and len(lc.clipped_time):
        trend_at = np.interp(lc.clipped_time, lc.norm_time, lc.trend)
        fig.add_trace(go.Scatter(x=lc.clipped_time, y=lc.clipped_flux * trend_at, mode="markers",
                                 name="Clipped outliers", marker=dict(size=7, color=C_WARN, symbol="x")))
    return _layout(fig, "Normalised flux and the trend removed by flattening", time_label(time_format),
                   "Relative flux")


def fig_detections(lc: ProcessedLightCurve, thr: ThresholdResult | None, cand: BLSCandidate | None,
                   time_format: str, xrange: tuple[float, float] | None = None,
                   highlight: Sequence[str] = ()) -> go.Figure:
    fig = go.Figure()
    S = _scatter(len(lc.time))
    fig.add_trace(S(x=lc.time, y=lc.flux, mode="markers", name="Processed flux",
                    marker=dict(size=3, color=C_OBS), hovertemplate="t=%{x:.4f}<br>%{y:.5f}<extra></extra>"))
    lo = float(np.nanpercentile(lc.flux, 0.2))
    hi = float(np.nanpercentile(lc.flux, 99.8))
    if thr is not None:
        fig.add_hline(y=thr.threshold_level, line=dict(color=C_THR, dash="dash", width=1.5),
                      annotation_text=f"threshold ({thr.settings.sigma_threshold:g}σ)",
                      annotation_position="bottom right", annotation_font_color=C_THR)
        if thr.events:
            ev = thr.events
            fig.add_trace(go.Scatter(
                x=[e.t_center for e in ev], y=[lo - 0.15 * (hi - lo)] * len(ev), mode="markers",
                name="Threshold events (A)", marker=dict(symbol="triangle-up", size=10, color=C_THR),
                customdata=[[e.event_id, e.depth * 1e6, e.duration * 24, e.snr] for e in ev],
                hovertemplate="%{customdata[0]}<br>t=%{x:.4f}<br>depth=%{customdata[1]:,.0f} ppm<br>"
                              "duration=%{customdata[2]:.1f} h<br>SNR=%{customdata[3]:.1f}<extra></extra>"))
    if cand is not None and cand.events:
        ev = cand.events
        fig.add_trace(go.Scatter(
            x=[e.t_center for e in ev], y=[hi + 0.15 * (hi - lo)] * len(ev), mode="markers",
            name="BLS transit epochs (B)", marker=dict(symbol="triangle-down", size=10, color=C_BLS),
            customdata=[[e.event_id, e.epoch, e.depth * 1e6, e.n_points] for e in ev],
            hovertemplate="%{customdata[0]} (epoch %{customdata[1]})<br>t=%{x:.4f}<br>"
                          "depth=%{customdata[2]:,.0f} ppm<br>%{customdata[3]} points<extra></extra>"))
    if highlight:
        events = (thr.events if thr else []) + (cand.events if cand else [])
        for e in events:
            if e.event_id in highlight:
                fig.add_vrect(x0=e.t_start, x1=e.t_end, fillcolor=C_SHARED, opacity=0.2, line_width=0)
    fig = _layout(fig, "Processed light curve with detections", time_label(time_format), "Relative flux")
    if xrange:
        fig.update_xaxes(range=list(xrange))
    return fig


def fig_folded(lc: ProcessedLightCurve, cand: BLSCandidate, window_durations: float = 3.0,
               n_bins: int = 60) -> go.Figure:
    ph = phase_fold(lc.time, cand.period, cand.t0)
    hours = ph * cand.period * 24
    half = window_durations * cand.duration * 24
    sel = np.abs(hours) <= half
    fig = go.Figure()
    fig.add_trace(_scatter(int(sel.sum()))(x=hours[sel], y=lc.flux[sel], mode="markers", name="Folded points",
                                           marker=dict(size=3, color=C_OBS),
                                           hovertemplate="%{x:.2f} h<br>%{y:.5f}<extra></extra>"))
    edges = np.linspace(-half, half, n_bins + 1)
    idx = np.digitize(hours[sel], edges) - 1
    centers, means = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 2:
            centers.append((edges[b] + edges[b + 1]) / 2)
            means.append(np.mean(lc.flux[sel][m]))
    fig.add_trace(go.Scatter(x=centers, y=means, mode="markers", name="Binned mean",
                             marker=dict(size=8, color=C_SHARED, line=dict(width=2, color="rgba(0,0,0,0)"))))
    xm = np.linspace(-half, half, 800)
    model = np.where(np.abs(xm) < cand.duration * 12, 1 - cand.depth, 1.0)
    fig.add_trace(go.Scatter(x=xm, y=model, mode="lines", name="BLS box model", line=dict(color=C_BLS, width=2)))
    return _layout(fig, f"Phase-folded at P = {cand.period:.5f} d", "Hours from mid-transit", "Relative flux")


def fig_periodogram(cand: BLSCandidate, references: Sequence = ()) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cand.periods, y=cand.power_spectrum, mode="lines", name="BLS power",
                             line=dict(color=C_THR, width=1.5),
                             hovertemplate="P=%{x:.5f} d<br>power=%{y:.4g}<extra></extra>"))
    pmin, pmax = float(np.min(cand.periods)), float(np.max(cand.periods))
    for mult, label in ((0.5, "P/2"), (2.0, "2P")):
        x = cand.period * mult
        if pmin <= x <= pmax:
            fig.add_vline(x=x, line=dict(color=C_OBS, dash="dot", width=1), annotation_text=label,
                          annotation_font_color=C_OBS)
    fig.add_vline(x=cand.period, line=dict(color=C_BLS, width=2),
                  annotation_text=f"peak {cand.period:.4f} d", annotation_font_color=C_BLS)
    for ref in references:
        if pmin <= ref.period_days <= pmax:
            fig.add_vline(x=ref.period_days, line=dict(color=C_SHARED, dash="dash", width=1.5),
                          annotation_text=f"{ref.name} (archive)", annotation_position="bottom left",
                          annotation_font_color=C_SHARED)
    fig = _layout(fig, "BLS periodogram", "Trial period [days]", "BLS power")
    fig.update_xaxes(type="log")
    return fig


# ---------------------------------------------------------------- number theory
def fig_intervals(times: Sequence[float], period: float | None, max_multiple: int = 6) -> go.Figure:
    d = np.diff(np.sort(np.asarray(times)))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d, nbinsx=max(10, min(80, len(d) * 2)), name="Consecutive intervals",
                               marker=dict(color=C_THR, line=dict(color="rgba(0,0,0,0)", width=2)),
                               hovertemplate="Δt %{x}<br>count %{y}<extra></extra>"))
    if period:
        for n in range(1, max_multiple + 1):
            if n * period <= (d.max() if len(d) else 0) * 1.05:
                fig.add_vline(x=n * period, line=dict(color=C_BLS, dash="dash", width=1.5),
                              annotation_text=f"{n}P", annotation_font_color=C_BLS)
    return _layout(fig, "Distribution of intervals between consecutive events", "Δt [days]", "Count", height=320)


def fig_mod_clock(times: Sequence[float], labels: Sequence[str], period: float, t0: float, tol_days: float) -> go.Figure:
    ph = phase_fold(np.asarray(times), period, t0)
    angle = (ph % 1.0) * 360
    ok = np.abs(ph * period) <= tol_days
    fig = go.Figure()
    for mask, name, color in ((ok, "≡ t₀ (mod P) within tolerance", C_SHARED), (~ok, "not congruent", C_BLS)):
        if mask.any():
            fig.add_trace(go.Scatterpolar(
                r=np.ones(mask.sum()), theta=angle[mask], mode="markers", name=name,
                marker=dict(size=12, color=color, line=dict(width=2, color="rgba(0,0,0,0)")),
                text=np.asarray(labels)[mask],
                customdata=np.c_[ph[mask] * period * 24],
                hovertemplate="%{text}<br>offset %{customdata[0]:+.2f} h<extra></extra>"))
    tol_deg = tol_days / period * 360
    fig.add_trace(go.Barpolar(r=[1.15], theta=[0], width=[2 * tol_deg], marker_color="rgba(25,158,112,0.18)",
                              name="tolerance window", hoverinfo="skip"))
    fig.update_layout(
        title=dict(text=f"Modular clock: phase of each event mod P = {period:.4f} d", x=0, font=dict(size=15)),
        polar=dict(radialaxis=dict(visible=False, range=[0, 1.2]),
                   angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GRID, ticksuffix="°")),
        height=420, margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def fig_oc(epochs: Sequence[int], residuals_days: Sequence[float], labels: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(epochs), y=np.asarray(residuals_days) * 24 * 60, mode="markers",
                             name="O − C", text=list(labels),
                             marker=dict(size=10, color=C_THR),
                             hovertemplate="%{text}<br>epoch n=%{x}<br>O−C=%{y:+.1f} min<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C_OBS, width=1))
    return _layout(fig, "Observed − calculated (O−C) transit times", "Epoch number n (integer)",
                   "O − C [minutes]", height=320)


# ---------------------------------------------------------------- set theory
def fig_venn(comp: EventSetComparison) -> go.Figure:
    s = comp.summary()
    fig = go.Figure()
    shapes = [
        dict(type="circle", x0=0.05, y0=0.1, x1=0.6, y1=0.9, line=dict(color=C_THR, width=3),
             fillcolor="rgba(57,135,229,0.15)"),
        dict(type="circle", x0=0.4, y0=0.1, x1=0.95, y1=0.9, line=dict(color=C_BLS, width=3),
             fillcolor="rgba(217,89,38,0.15)"),
        dict(type="rect", x0=0.0, y0=0.0, x1=1.0, y1=1.0, line=dict(color=C_OBS, width=1, dash="dot")),
    ]
    texts = [
        (0.22, 0.5, f"<b>{s['|A−B|']}</b><br>A − B<br>threshold only"),
        (0.5, 0.5, f"<b>{s['|A∩B|']}</b><br>A ∩ B<br>both"),
        (0.78, 0.5, f"<b>{s['|B−A|']}</b><br>B − A<br>BLS only"),
        (0.2, 0.93, f"A: threshold (|A| = {s['|A|']})"),
        (0.8, 0.93, f"B: BLS epochs (|B| = {s['|B|']})"),
        (0.5, 0.04, f"U − (A ∪ B) = {s['|(A∪B)ᶜ|']} predicted epochs in data gaps   ·   |U| = {s['|U|']}"),
    ]
    for x, y, t in texts:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=14))
    fig.update_layout(shapes=shapes, height=360, margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text="Venn diagram of detections", x=0, font=dict(size=15)),
                      xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_set_timeline(comp: EventSetComparison, time_format: str) -> go.Figure:
    regions = {"A ∩ B": (C_SHARED, 2), "A − B": (C_THR, 1), "B − A": (C_BLS, 3), "gap": (C_OBS, 0)}
    rows = comp.table()
    fig = go.Figure()
    for name, (color, y) in regions.items():
        sel = [r for r in rows if (r["region"].startswith("(A ∪ B)") if name == "gap" else r["region"] == name)]
        if not sel:
            continue
        fig.add_trace(go.Scatter(
            x=[r["t_center"] for r in sel], y=[y] * len(sel), mode="markers", name=name if name != "gap" else "(A ∪ B)ᶜ: gap",
            marker=dict(size=11, color=color, symbol="circle" if name != "gap" else "circle-open"),
            text=[f"{r['element']}: {r['threshold_event'] or '—'} / {r['bls_epoch'] or '—'}" for r in sel],
            hovertemplate="%{text}<br>t=%{x:.4f}<extra></extra>"))
    fig = _layout(fig, "Where each set element lies in time", time_label(time_format), "", height=280)
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["gap", "A − B", "A ∩ B", "B − A"], range=[-0.5, 3.5])
    return fig


# ---------------------------------------------------------------- graphs
def fig_event_graph(G: nx.Graph, components: list[list[str]], layout: str = "time",
                    selected: str | None = None) -> go.Figure:
    if layout == "time":
        pos = {n: (G.nodes[n]["t"], G.nodes[n]["depth"] * 1e6) for n in G.nodes}
    else:
        raw = nx.spring_layout(G, seed=7) if layout == "spring" else nx.circular_layout(
            sorted(G.nodes, key=lambda k: G.nodes[k]["t"]))
        pos = {n: (float(p[0]), float(p[1])) for n, p in raw.items()}
    ex, ey, mx, my, mtext = [], [], [], [], []
    for u, v, d in G.edges(data=True):
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        mtext.append(f"{u}–{v}<br>Δt = {d['dt']:.4f} d = {d['n']}·P {d['residual_hours']:+.2f} h")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(137,135,129,0.45)", width=1),
                             hoverinfo="skip", name="edges", showlegend=False))
    if G.number_of_edges() <= 80:  # edge tooltips; on dense graphs they would intercept vertex clicks
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=6, color="rgba(0,0,0,0)"),
                                 text=mtext, hovertemplate="%{text}<extra></extra>", showlegend=False))
    largest = set(components[0]) if components and len(components[0]) > 1 else set()
    groups = [
        ("Largest component", [n for n in G.nodes if n in largest], C_SHARED),
        ("Other components", [n for n in G.nodes if n not in largest and G.degree(n) > 0], C_THR),
        ("Isolated vertices", [n for n in G.nodes if G.degree(n) == 0], C_BLS),
    ]
    for name, nodes, color in groups:
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text" if len(G) <= 40 else "markers",
            name=name, text=nodes, textposition="top center", textfont=dict(size=10),
            marker=dict(size=[16 if n == selected else 11 for n in nodes], color=color,
                        line=dict(width=[3 if n == selected else 0 for n in nodes], color="#f0efec")),
            customdata=[[G.nodes[n]["t"], G.nodes[n]["depth"] * 1e6, G.degree(n), n] for n in nodes],
            hovertemplate="%{text}<br>t=%{customdata[0]:.4f}<br>depth=%{customdata[1]:,.0f} ppm<br>"
                          "degree=%{customdata[2]}<extra></extra>"))
    xt, yt = ("Event time [days]", "Depth [ppm]") if layout == "time" else ("", "")
    fig = _layout(fig, "Transit-event graph", xt, yt, height=440)
    if layout != "time":
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


def fig_adjacency(nodes: list[str], M: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=M, x=nodes, y=nodes, colorscale=[[0, "rgba(0,0,0,0)"], [1, "#3987e5"]],
                               showscale=False, xgap=2, ygap=2,
                               hovertemplate="%{y} – %{x}: %{z}<extra></extra>"))
    fig = _layout(fig, "Adjacency matrix (1 = edge)", "", "", height=max(320, 18 * len(nodes) + 100))
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------- relations
def fig_relation(rel: Relation, max_items: int = 14) -> go.Figure:
    dom = list(rel.domain)[:max_items]
    image = {y for x, y in rel.pairs if x in dom}
    cod = [y for y in rel.codomain if y in image] if rel.infinite_codomain else list(rel.codomain)[:max_items]
    ydom = {x: i for i, x in enumerate(reversed(dom))}
    span = max(len(dom), len(cod), 1)
    ycod = {y: (i + 0.5) * len(dom) / max(len(cod), 1) - 0.5 for i, y in enumerate(reversed(cod))}
    fig = go.Figure()
    in_deg = rel.in_degree()
    out_deg = rel.out_degree()
    for x, y in rel.pairs:
        if x in ydom and y in ycod:
            color = C_BLS if out_deg.get(x, 0) > 1 else C_THR
            fig.add_annotation(x=0.92, y=ycod[y], ax=0.08, ay=ydom[x], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.5, arrowcolor=color)
    fig.add_trace(go.Scatter(x=[0] * len(dom), y=[ydom[x] for x in dom], mode="markers+text", text=[str(x) for x in dom],
                             textposition="middle left", name=rel.domain_label,
                             marker=dict(size=12, color=C_THR), hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1] * len(cod), y=[ycod[y] for y in cod], mode="markers+text", text=[str(y) for y in cod],
                             textposition="middle right", name=rel.codomain_label,
                             marker=dict(size=12, color=C_SHARED),
                             customdata=[in_deg.get(y, 0) for y in cod],
                             hovertemplate="%{text}<br>preimages: %{customdata}<extra></extra>"))
    fig = _layout(fig, rel.name, "", "", height=max(300, 34 * span + 90))
    fig.update_xaxes(visible=False, range=[-0.9, 1.9])
    fig.update_yaxes(visible=False, range=[-1, len(dom)])
    return fig


def fig_relation_matrix(rel: Relation, max_items: int = 30) -> go.Figure:
    M = rel.matrix()[:max_items, :max_items]
    fig = go.Figure(go.Heatmap(z=M, x=[str(y) for y in rel.codomain[:max_items]], y=[str(x) for x in rel.domain[:max_items]],
                               colorscale=[[0, "rgba(0,0,0,0)"], [1, "#199e70"]], showscale=False, xgap=2, ygap=2,
                               hovertemplate="(%{y}, %{x}) ∈ R: %{z}<extra></extra>"))
    fig = _layout(fig, "Relation matrix", rel.codomain_label, rel.domain_label,
                  height=max(260, 22 * min(len(rel.domain), max_items) + 100))
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------- combinatorics / comparison
def fig_heatmap(pivot, title: str, xlabel: str, ylabel: str, value_label: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
                               colorscale=[[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]],
                               colorbar=dict(title=value_label), xgap=2, ygap=2, texttemplate="%{z:.3g}",
                               hovertemplate=f"{ylabel}=%{{y}}<br>{xlabel}=%{{x}}<br>{value_label}=%{{z:.3g}}<extra></extra>"))
    return _layout(fig, title, xlabel, ylabel, height=360)


def fig_method_comparison(rows: list[dict]) -> go.Figure:
    """Grouped bars: one group per metric, threshold vs BLS."""
    metrics = [r["metric"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=[r["threshold"] for r in rows], name="Threshold", marker_color=C_THR,
                         hovertemplate="%{x}: %{y:.3g}<extra>Threshold</extra>"))
    fig.add_trace(go.Bar(x=metrics, y=[r["bls"] for r in rows], name="BLS", marker_color=C_BLS,
                         hovertemplate="%{x}: %{y:.3g}<extra>BLS</extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    return _layout(fig, "Threshold vs BLS on this dataset", "", "Count", height=320)
