"""Render static PNG figures of real ExoScope results for the report and slides.

All figures are computed from the bundled real light curves (and the CSV tables
written by scripts/evaluate.py) — nothing is drawn by hand.

Usage:
    python scripts/make_figures.py        # writes docs/figures/*.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from exoscope.data.samples import load_sample  # noqa: E402
from exoscope.data.synthetic import SyntheticParams, generate_synthetic  # noqa: E402
from exoscope.dmath.graphs import EdgeRule, build_event_graph, summarize_graph  # noqa: E402
from exoscope.dmath.number_theory import phase_fold  # noqa: E402
from exoscope.legacy import legacy_detect  # noqa: E402
from exoscope.pipeline import run_analysis  # noqa: E402

OUT = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "legend.frameon": False, "figure.facecolor": "white",
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def folded(r, name: str, title: str, window: float = 3.0) -> None:
    b, lc = r.best, r.processed
    hours = phase_fold(lc.time, b.period, b.t0) * b.period * 24
    sel = np.abs(hours) <= window * b.duration * 24
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(hours[sel], (lc.flux[sel] - 1) * 1e6, s=3, color=GREY, alpha=0.5, label="measurements", rasterized=True)
    edges = np.linspace(hours[sel].min(), hours[sel].max(), 50)
    idx = np.digitize(hours[sel], edges)
    cx = [(edges[i - 1] + edges[i]) / 2 for i in range(1, len(edges)) if (idx == i).sum() > 2]
    cy = [np.mean(lc.flux[sel][idx == i] - 1) * 1e6 for i in range(1, len(edges)) if (idx == i).sum() > 2]
    ax.plot(cx, cy, "o", color=AQUA, ms=5, label="binned mean")
    xm = np.linspace(hours[sel].min(), hours[sel].max(), 600)
    ax.plot(xm, np.where(np.abs(xm) < b.duration * 12, -b.depth * 1e6, 0), color=ORANGE, lw=2, label="BLS box model")
    ax.set(xlabel="hours from mid-transit", ylabel="relative flux − 1 [ppm]", title=title)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k10 = run_analysis(load_sample("kepler10_q3"))
    k8 = run_analysis(load_sample("kepler8_q2"))

    folded(k10, "kepler10_folded.png",
           f"Kepler-10 b: {len(k10.best.events)} transits folded at P = {k10.best.period:.5f} d "
           f"(depth {k10.best.depth * 1e6:.0f} ppm)")
    folded(k8, "kepler8_folded.png", f"Kepler-8 b folded at P = {k8.best.period:.5f} d")

    # periodogram
    b = k10.best
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(b.periods, b.power_spectrum, color=BLUE, lw=0.8)
    ax.axvline(b.period, color=ORANGE, lw=1.5, label=f"peak {b.period:.5f} d")
    ax.axvline(0.8374907, color=AQUA, ls="--", lw=1.2, label="archive Kepler-10 b (0.83749 d)")
    for m, lab in ((0.5, "P/2"), (2, "2P")):
        ax.axvline(b.period * m, color=GREY, ls=":", lw=1)
        ax.text(b.period * m, ax.get_ylim()[1] * 0.92, lab, color=GREY, ha="center", fontsize=9)
    ax.set(xscale="log", xlabel="trial period [days]", ylabel="BLS power", title="BLS periodogram — Kepler-10, Quarter 3")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "kepler10_periodogram.png")

    # light curve with detections (zoomed 6 days)
    lc, thr = k10.processed, k10.threshold
    # a 6-day window that contains several threshold events (and therefore shows both kinds of detection)
    starts = [e.t_center - 0.5 for e in thr.events]
    t0 = max(starts, key=lambda a: sum(a < e.t_center < a + 6 for e in thr.events))
    sel = (lc.time > t0) & (lc.time < t0 + 6)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.scatter(lc.time[sel], (lc.flux[sel] - 1) * 1e6, s=4, color=GREY, rasterized=True, label="processed flux")
    ax.axhline((thr.threshold_level - 1) * 1e6, color=BLUE, ls="--", lw=1, label="threshold (3σ)")
    ev_a = [e.t_center for e in thr.events if t0 < e.t_center < t0 + 6]
    ev_b = [e.t_center for e in b.events if t0 < e.t_center < t0 + 6]
    lo = (lc.flux[sel].min() - 1) * 1e6
    hi = (lc.flux[sel].max() - 1) * 1e6
    ax.plot(ev_a, [lo - 40] * len(ev_a), "^", color=BLUE, ms=8, label="threshold events (A)")
    ax.plot(ev_b, [hi + 40] * len(ev_b), "v", color=ORANGE, ms=8, label="BLS epochs (B)")
    ax.set(xlabel="time [BKJD]", ylabel="relative flux − 1 [ppm]",
           title="Kepler-10, six days: BLS epochs (top) vs threshold events (bottom)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "kepler10_detections.png")

    # Venn
    s = k10.comparison.summary()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.add_patch(plt.Circle((0.36, 0.5), 0.3, color=BLUE, alpha=0.18, lw=2.5, ec=BLUE))
    ax.add_patch(plt.Circle((0.64, 0.5), 0.3, color=ORANGE, alpha=0.18, lw=2.5, ec=ORANGE))
    for x, t in ((0.2, f"{s['|A−B|']}\nA − B"), (0.5, f"{s['|A∩B|']}\nA ∩ B"), (0.8, f"{s['|B−A|']}\nB − A")):
        ax.text(x, 0.5, t, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.22, 0.9, f"A: threshold events ({s['|A|']})", ha="center", color=BLUE)
    ax.text(0.78, 0.9, f"B: BLS epochs ({s['|B|']})", ha="center", color=ORANGE)
    ax.text(0.5, 0.08, f"(A ∪ B)ᶜ = {s['|(A∪B)ᶜ|']} predicted epochs in data gaps · Jaccard = {s['jaccard']:.2f}",
            ha="center", fontsize=10)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Set comparison — Kepler-10 (τ = 2 h)")
    ax.axis("off")
    save(fig, "kepler10_venn.png")

    # modular clock: right vs wrong period
    times = np.array([e.t_center for e in thr.events])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), subplot_kw={"projection": "polar"})
    for ax, P, lab in ((axes[0], b.period, f"P = {b.period:.4f} d (BLS)"), (axes[1], b.period * 1.013, f"P = {b.period * 1.013:.4f} d (1.3% off)")):
        ang = (phase_fold(times, P, b.t0) % 1) * 2 * np.pi
        ax.scatter(ang, np.ones_like(ang), s=40, color=AQUA if P == b.period else ORANGE, alpha=0.8)
        ax.set(theta_zero_location="N", theta_direction=-1, yticks=[], ylim=(0, 1.2), title=lab)
        ax.title.set_fontsize(10)
    fig.suptitle("Modular clock: phase of each threshold event, (t − t₀) mod P", fontweight="bold")
    save(fig, "kepler10_modclock.png")

    # event graph (time × depth)
    G = build_event_graph(thr.events, EdgeRule(b.period, 2.0, 10))
    gs = summarize_graph(G)
    largest = set(gs.components[0])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for u, v in G.edges:
        ax.plot([G.nodes[u]["t"], G.nodes[v]["t"]], [G.nodes[u]["depth"] * 1e6, G.nodes[v]["depth"] * 1e6],
                color=GREY, lw=0.5, alpha=0.5)
    for n in G.nodes:
        c = AQUA if n in largest else (ORANGE if G.degree(n) == 0 else BLUE)
        ax.plot(G.nodes[n]["t"], G.nodes[n]["depth"] * 1e6, "o", color=c, ms=7)
    ax.plot([], [], "o", color=AQUA, label="largest component")
    ax.plot([], [], "o", color=BLUE, label="other components")
    ax.plot([], [], "o", color=ORANGE, label="isolated vertices")
    ax.set(xlabel="event time [BKJD]", ylabel="event depth [ppm]",
           title=f"Event graph — |V| = {gs.n_vertices}, |E| = {gs.n_edges}, {len(gs.components)} components, "
                 f"max clique {len(gs.max_clique)}")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "kepler10_graph.png")

    # v1 vs v2 on Kepler-8
    leg = legacy_detect(k8.processed.time, k8.processed.flux)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names = ["v1 median spacing", "v2 approximate GCD", "v2 BLS", "NASA archive"]
    vals = [leg.period_estimate, k8.threshold.period.period, k8.best.period, 3.5224991]
    bars = ax.barh(names, vals, color=[GREY, BLUE, ORANGE, AQUA])
    for bar, v in zip(bars, vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.4f} d", va="center", fontsize=10)
    ax.set(xlim=(0, 4.4), xlabel="period estimate [days]", title="Kepler-8 b: period from the same light curve")
    ax.invert_yaxis()
    save(fig, "kepler8_v1_vs_v2.png")

    # eclipsing binary imitation
    eb = run_analysis(generate_synthetic(SyntheticParams(odd_depth_ratio=0.5)))
    e = eb.best
    ph = phase_fold(eb.processed.time, e.period, e.t0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(ph, (eb.processed.flux - 1) * 1e6, s=3, color=GREY, rasterized=True)
    ax.axvspan(-0.03, 0.03, color=ORANGE, alpha=0.12)
    ax.axvspan(0.47, 0.5, color=BLUE, alpha=0.12)
    ax.axvspan(-0.5, -0.47, color=BLUE, alpha=0.12)
    ax.text(0, ax.get_ylim()[0] * 0.95, "primary", ha="center", color=ORANGE)
    ax.text(0.44, ax.get_ylim()[0] * 0.95, "secondary\n(phase 0.5)", ha="center", color=BLUE)
    ax.set(xlabel=f"phase at BLS period {e.period:.3f} d", ylabel="relative flux − 1 [ppm]",
           title="SYNTHETIC eclipsing binary — flagged: deep secondary eclipse")
    save(fig, "synthetic_eclipsing_binary.png")

    # evaluation CSVs (from scripts/evaluate.py)
    if (RES / "injection_summary.csv").exists():
        inj = pd.read_csv(RES / "injection_summary.csv")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for host, color in (("Kepler-10", BLUE), ("Kepler-8", ORANGE)):
            d = inj[inj["host"] == host]
            ax.plot(d["depth_ppm"], 100 * d["recovered_real_data"], "o-", color=color, lw=2, label=f"{host} — real data")
            ax.plot(d["depth_ppm"], 100 * d["recovered_white_noise"], "o--", color=color, lw=1.2, alpha=0.6,
                    label=f"{host} — white-noise twin")
        ax.set(xscale="log", xlabel="injected transit depth [ppm]", ylabel="recovered [%]", ylim=(-5, 105),
               title="Injection–recovery: what BLS can really detect")
        ax.set_xticks([50, 100, 200, 400, 800], ["50", "100", "200", "400", "800"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=8, loc="lower right")
        save(fig, "injection_recovery.png")
    if (RES / "real_targets.csv").exists():
        real = pd.read_csv(RES / "real_targets.csv").dropna(subset=["bls_err_%"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(real))
        ax.bar(x - 0.2, real["bls_err_%"], 0.4, color=ORANGE, label="BLS")
        ax.bar(x + 0.2, real["threshold_err_%"], 0.4, color=BLUE, label="threshold + approximate GCD")
        ax.set_xticks(x, real["target"], rotation=30, ha="right")
        ax.set(yscale="log", ylabel="period error vs archive [%]", title="9 known planets: period error (lower is better)")
        ax.legend(fontsize=9)
        save(fig, "real_targets_period_error.png")


if __name__ == "__main__":
    main()
