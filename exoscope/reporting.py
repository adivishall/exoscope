"""Exports: event/candidate CSVs, settings+metadata JSON, Markdown and HTML reports."""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd

from .dmath.graphs import EdgeRule, build_event_graph, summarize_graph
from .dmath.number_theory import interval_table
from .dmath.relations import event_algorithm_relation
from .dmath.sets import roster
from .pipeline import AnalysisResult, settings_dict_for_storage
from .storage import dumps

DISCLAIMER = ("ExoScope reports *potential transit candidates*. A periodic dip is consistent with a transiting "
              "planet but can also be caused by eclipsing binaries, blended stars, stellar variability or "
              "instrumental systematics. Planetary status requires independent follow-up.")


def events_frame(result: AnalysisResult) -> pd.DataFrame:
    events = (result.threshold.events if result.threshold else []) + (result.best.events if result.best else [])
    if not events:
        return pd.DataFrame(columns=["event_id", "method", "t_center", "depth_ppm", "duration_hours", "snr"])
    df = pd.DataFrame([e.to_dict() for e in events])
    cols = ["event_id", "method", "t_center", "t_start", "t_end", "depth_ppm", "duration_hours", "snr", "n_points", "epoch"]
    return df[cols].sort_values("t_center").reset_index(drop=True)


def candidates_frame(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    if result.bls:
        for c in result.bls.candidates:
            rows.append({"method": "BLS", **c.to_dict()})
    if result.threshold and result.threshold.period and result.threshold.period.members:
        p = result.threshold.period
        rows.append({"method": "Threshold + approximate GCD", "rank": 1, "period_days": p.period, "t0": p.t0,
                     "n_transits_observed": p.n_members, "period_err_days": p.period_err})
    return pd.DataFrame(rows)


def metadata_json(result: AnalysisResult) -> str:
    payload = {
        "exoscope_version": result.app_version,
        "created_at": result.created_at,
        "dataset": {"meta": result.dataset_meta, "summary": result.dataset_summary},
        "settings": settings_dict_for_storage(result),
        "preprocessing_steps": result.processed.steps,
        "results": result.headline(),
        "timings_s": result.timings,
        "set_comparison": result.comparison.summary(),
        "evaluation": {"verdict": result.evaluation.verdict, "criteria": result.evaluation.to_dicts()},
        "reference_check": result.reference_check,
        "truth_metrics": result.truth_metrics,
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(json.loads(dumps(payload)), indent=2)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "—" if not np.isfinite(v) else f"{v:.{digits}g}"
    return str(v)


def discrete_math_summary(result: AnalysisResult, experiment: dict | None = None) -> list[tuple[str, str]]:
    """(heading, explanation) pairs written from this analysis' actual numbers."""
    out = []
    comp = result.comparison
    s = comp.summary()
    out.append((
        "Set theory — comparing detectors",
        f"A (threshold events) has |A| = {s['|A|']}, B (BLS epochs with data) has |B| = {s['|B|']}, matched within "
        f"±{comp.tolerance_hours:g} h. |A ∩ B| = {s['|A∩B|']}, |A − B| = {s['|A−B|']}, |B − A| = {s['|B−A|']}, "
        f"|A ∪ B| = {s['|A∪B|']} = |A| + |B| − |A ∩ B| (inclusion–exclusion). The universe U adds "
        f"{s['|(A∪B)ᶜ|']} predicted epochs that fall in data gaps, so (A ∪ B)ᶜ = those gaps. "
        f"Jaccard similarity = {_fmt(s['jaccard'], 3)}. A − B = {roster(comp.a_minus_b(), comp, 8)}."
    ))
    best = result.best
    thr = result.threshold
    if best and thr and len(thr.events) >= 2:
        rows = interval_table([e.t_center for e in thr.events], best.period, result.settings.match_tolerance_hours / 24)
        ok = sum(r["consistent"] for r in rows)
        multiples = sorted({r["n"] for r in rows if r["consistent"]})
        out.append((
            "Number theory — periodicity",
            f"Of {len(rows)} consecutive threshold intervals, {ok} are integer multiples of P = {best.period:.5f} d "
            f"within the tolerance (multiples seen: n ∈ {{{', '.join(map(str, multiples[:10]))}}}; n > 1 means "
            "transits were missed in between). Event times are congruent modulo P: t ≡ t₀ (mod P)."
            + (f" The approximate-GCD estimate from threshold events alone gives P = {thr.period.period:.5f} d "
               f"({thr.period.note})" if thr.period and thr.period.members else "")
        ))
        rule = EdgeRule(best.period, result.settings.match_tolerance_hours)
        try:
            G = build_event_graph(thr.events, rule)
            gs = summarize_graph(G)
            out.append((
                "Graph theory — event network",
                f"With vertices = threshold events and edges = periodic consistency ({rule.describe()}), the graph has "
                f"|V| = {gs.n_vertices}, |E| = {gs.n_edges}, {len(gs.components)} connected component(s); the largest "
                f"has {len(gs.components[0]) if gs.components else 0} vertices, {len(gs.isolated)} are isolated, and "
                f"the largest clique has {len(gs.max_clique)} mutually consistent events. The consistency relation is "
                f"{'transitive' if gs.is_transitive else 'not transitive (tolerances accumulate)'}."
            ))
        except ValueError:
            pass
    rel = event_algorithm_relation(comp)
    if rel.domain:
        out.append((
            "Relations and functions",
            f"Observation → flux is a function (one flux per timestamp). Event ↔ algorithm here is "
            f"{rel.classify()} with {len(rel.pairs)} ordered pairs; it is "
            f"{'a function' if rel.is_function else 'not a function, because some events are related to both algorithms'}."
        ))
    s_set = result.settings
    text = (f"This run is one point of the configuration space (threshold k = {s_set.threshold.sigma_threshold:g}σ, "
            f"min points = {s_set.threshold.min_points}, flatten window = {s_set.preprocess.flatten_window_days:g} d).")
    if experiment:
        text += (f" In the Parameter Combination Explorer the chosen sets gave |S₁ × … × S_k| = {experiment['formula']}"
                 f" configurations, {experiment['n_valid']} satisfying the constraints; {experiment['n_run']} were run, "
                 f"needing only {experiment['preprocessing_runs']} distinct preprocessing runs (projection size).")
    else:
        text += (" Varying e.g. 4 thresholds × 3 minimum-point values × 2 windows would give 24 configurations by the "
                 "rule of product (run the Parameter Combination Explorer to include real results here).")
    out.append(("Combinatorics — configuration space", text))
    return out


def markdown_report(result: AnalysisResult, experiment: dict | None = None) -> str:
    m, summ = result.dataset_meta, result.dataset_summary
    L = [f"# ExoScope analysis report — {m.get('target', 'unknown')}", ""]
    if result.is_synthetic:
        L += ["> **SYNTHETIC DATA.** This light curve was generated by ExoScope and is not a real star.", ""]
    L += [f"> {DISCLAIMER}", "", "## Data", ""]
    L += [f"- Target: **{m.get('target')}**  ·  Mission: {m.get('mission', '—')}  ·  Dataset: {m.get('dataset', '—')}",
          f"- Source: {m.get('source')}" + (f" (product {m.get('product')})" if m.get("product") else ""),
          f"- Measurements: {summ.get('n_points')} total, {summ.get('n_usable')} usable "
          f"({summ.get('n_nan')} missing, {summ.get('n_quality_flagged')} quality-flagged)",
          f"- Baseline: {_fmt(summ.get('baseline_days'))} d, cadence {_fmt(summ.get('cadence_minutes'))} min, "
          f"duty cycle {_fmt(summ.get('duty_cycle'), 3)}", "", "## Preprocessing", ""]
    L += [f"{i}. **{s['step']}** — {s['detail']} ({s['points_before']} → {s['points_after']} points)"
          for i, s in enumerate(result.processed.steps, 1)]
    L += ["", f"Noise after processing: {result.processed.noise * 1e6:,.0f} ppm (robust σ).", "", "## Methods", "",
          f"- Threshold detector ({result.threshold.version if result.threshold else 'not run'}): points below "
          f"median − kσ grouped into events; period via approximate GCD (largest congruence class).",
          f"- BLS ({result.bls.version if result.bls else 'not run'}): "
          + (f"{result.bls.n_periods:,} trial periods between {result.settings.bls.period_min:g} and "
             f"{result.bls.period_max_used:.3g} d, durations {list(result.settings.bls.durations_hours)} h."
             if result.bls else ""), "", "## Results", ""]
    for k, v in result.headline().items():
        L.append(f"- {k.replace('_', ' ')}: {_fmt(v, 6)}")
    if result.reference_check:
        r = result.reference_check
        L += ["", f"**Reference check:** detected P = {r['detected_period']:.5f} d vs {r['planet']} "
              f"P = {r['reference_period']:.5f} d (NASA Exoplanet Archive): ratio ≈ {r['fraction']} "
              f"({r['kind']}), error {r['period_error_percent']:.3f}%."]
    if result.truth_metrics:
        L += ["", "**Against injected truth (synthetic):**"]
        for meth in ("threshold", "bls"):
            if meth in result.truth_metrics:
                t = result.truth_metrics[meth]
                L.append(f"- {meth}: precision {_fmt(t['precision'], 3)}, recall {_fmt(t['recall'], 3)}, "
                         f"F1 {_fmt(t['f1'], 3)}" + (f", period error {t['period_error_percent']:.3f}%"
                                                     if "period_error_percent" in t else ""))
    L += ["", "## Candidate evaluation", "", f"**{result.evaluation.verdict}** "
          f"({result.evaluation.n_pass}/{len(result.evaluation.applicable)} applicable checks passed — an evidence "
          "count, not a probability).", "", "| Check | Status | Value | Rule |", "|---|---|---|---|"]
    for c in result.evaluation.criteria:
        L.append(f"| {c.name} | {c.status} | {c.value} | {c.rule} |")
    L += ["", "## Discrete Mathematics in this analysis", ""]
    for head, text in discrete_math_summary(result, experiment):
        L += [f"### {head}", "", text, ""]
    L += ["## Limitations", "",
          "- Box-shaped BLS model; no limb darkening or full transit fit.",
          "- One light-curve segment (quarter/sector) at a time.",
          "- The evaluation checklist is a teaching aid, not a calibrated false-positive probability.",
          "- No pixel-level (centroid) or stellar-companion vetting.", "",
          f"_Generated by ExoScope {result.app_version} on {result.created_at}._"]
    return "\n".join(L)


def html_report(result: AnalysisResult, figures: list[tuple[str, Any]], experiment: dict | None = None) -> str:
    """Self-contained HTML (Plotly loaded from its CDN) with the Markdown content and figures."""
    md = markdown_report(result, experiment)
    try:
        import markdown as _md

        body = _md.markdown(md, extensions=["tables"])
    except Exception:  # markdown package missing: show the text as-is
        body = f"<pre>{html.escape(md)}</pre>"
    figs = "".join(
        f"<h3>{html.escape(t)}</h3>" + f.to_html(full_html=False, include_plotlyjs=False) for t, f in figures)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ExoScope report — {html.escape(str(result.dataset_meta.get('target', '')))}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:980px;margin:2rem auto;padding:0 16px;
line-height:1.55;color:#0b0b0b;background:#fcfcfb}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #e1e0d9;padding:4px 8px;text-align:left;font-size:14px}}
blockquote{{border-left:4px solid #3987e5;margin:1rem 0;padding:.25rem 1rem;background:#f0efec}}</style>
</head><body>{body}<h2>Figures</h2>{figs}</body></html>"""
