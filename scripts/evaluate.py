"""Reproducible evaluation of ExoScope's detectors. Writes docs/EVALUATION.md.

1. Synthetic benchmark (seeded): injected transits with known period and
   epochs → period recovery, epoch-level precision/recall/F1, runtime, for the
   legacy v1 detector, the improved threshold detector and BLS.
2. Null and false-positive tests: pure-noise light curves (how often a
   "significant" signal is reported) and eclipsing-binary imitations (how
   often the checklist flags a non-planetary origin).
3. Real data: bundled Kepler/TESS samples (and, with --online, more targets
   from MAST) compared with NASA Exoplanet Archive periods.

Usage:
    python scripts/evaluate.py            # synthetic + bundled samples (offline)
    python scripts/evaluate.py --online   # also download extra real targets
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from exoscope import __version__  # noqa: E402
from exoscope.data.samples import list_samples, load_sample  # noqa: E402
from exoscope.data.synthetic import SyntheticParams, generate_synthetic  # noqa: E402
from exoscope.data.targets import reference_planets  # noqa: E402
from exoscope.dmath.sets import detection_metrics  # noqa: E402
from exoscope.injection import Injection, inject, mask_transits, white_noise_twin  # noqa: E402
from exoscope.legacy import legacy_detect  # noqa: E402
from exoscope.pipeline import check_against_reference, default_settings_for, run_analysis  # noqa: E402

DEPTHS = [300, 1000, 3000, 10000]
NOISES = [300, 1000]
PERIODS = [1.3, 3.0, 7.1]
SEEDS = range(5)
TOL_H = 2.0
N_NULL = 30
ONLINE_TARGETS = [("TrES-2", "Kepler", 1), ("Kepler-6", "Kepler", 1), ("HAT-P-7", "Kepler", 1),
                  ("Kepler-17", "Kepler", 1), ("WASP-121", "TESS", 0), ("KELT-11", "TESS", 0)]


def recovered(p_found, p_true, rel=0.01) -> bool:
    return p_found is not None and np.isfinite(p_found) and abs(p_found - p_true) / p_true <= rel


def synthetic_benchmark() -> pd.DataFrame:
    rows = []
    for depth in DEPTHS:
        for noise in NOISES:
            for period in PERIODS:
                for seed in SEEDS:
                    d = generate_synthetic(SyntheticParams(period_days=period, depth_ppm=depth, noise_ppm=noise,
                                                           duration_hours=3.0, t0_days=0.7 + 0.1 * seed, seed=seed))
                    t = time.perf_counter()
                    r = run_analysis(d, default_settings_for(d))
                    total = time.perf_counter() - t
                    truth = [e["t_center"] for e in d.truth["epochs"] if e["observed"]]
                    thr, best = r.threshold, r.best
                    thr_p = thr.period.period if thr.period and thr.period.members and not thr.period.ambiguous else None
                    leg = legacy_detect(r.processed.time, r.processed.flux)
                    mt = detection_metrics([e.t_center for e in thr.events], truth, TOL_H / 24)
                    mb = detection_metrics([e.t_center for e in best.events], truth, TOL_H / 24)
                    rows.append({
                        "depth_ppm": depth, "noise_ppm": noise, "period": period, "seed": seed,
                        "snr_per_point": depth / noise,
                        "legacy_ok": recovered(leg.period_estimate, period),
                        "thr_ok": recovered(thr_p, period), "bls_ok": recovered(best.period, period),
                        "thr_err_%": 100 * abs(thr_p - period) / period if thr_p else np.nan,
                        "bls_err_%": 100 * abs(best.period - period) / period,
                        "thr_P": mt["precision"], "thr_R": mt["recall"], "thr_F1": mt["f1"],
                        "bls_P": mb["precision"], "bls_R": mb["recall"], "bls_F1": mb["f1"],
                        "thr_s": r.timings.get("threshold_s", np.nan), "bls_s": r.timings.get("bls_s", np.nan),
                        "total_s": total, "level": r.evaluation.level,
                    })
    return pd.DataFrame(rows)


def null_and_eb_tests() -> dict:
    noise_levels = []
    for seed in range(N_NULL):
        d = generate_synthetic(SyntheticParams(depth_ppm=1, noise_ppm=800, seed=100 + seed))
        noise_levels.append(run_analysis(d).evaluation.level)
    eb_levels = []
    for seed in range(N_NULL):
        d = generate_synthetic(SyntheticParams(depth_ppm=8000, odd_depth_ratio=0.5, seed=200 + seed))
        eb_levels.append(run_analysis(d).evaluation.level)
    return {"noise": noise_levels, "eb": eb_levels}


INJ_HOSTS = [("kepler10_q3", "Kepler-10"), ("kepler8_q2", "Kepler-8")]
INJ_DEPTHS_PPM = [50, 100, 200, 400, 800]
INJ_PERIODS = [1.7, 4.3, 7.9]
INJ_PHASES = 4


def injection_recovery() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject transits into real light curves (known planet masked) and into white-noise twins."""
    rows = []
    for sample_id, name in INJ_HOSTS:
        host = load_sample(sample_id)
        known = run_analysis(host).best  # mask the host's own planet before injecting
        masked = mask_transits(host, known.period, known.t0, known.duration, factor=2.5)
        twin = white_noise_twin(masked, seed=1)
        for depth in INJ_DEPTHS_PPM:
            for period in INJ_PERIODS:
                for k in range(INJ_PHASES):
                    inj = Injection(period, float(masked.time.min()) + (k + 0.37) / INJ_PHASES * period,
                                    depth * 1e-6, 3.0 / 24)
                    row = {"host": name, "depth_ppm": depth, "period": period, "phase": k}
                    for label, base in (("real", masked), ("white", twin)):
                        r = run_analysis(inject(base, inj), default_settings_for(base).with_overrides({"run_threshold": False}))
                        b = r.best
                        row[f"{label}_ok"] = bool(recovered(b.period, period) and b.depth_snr >= 7.1)
                        row[f"{label}_snr"] = b.depth_snr if recovered(b.period, period) else np.nan
                        row[f"{label}_noise_ppm"] = r.processed.noise * 1e6
                    rows.append(row)
    df = pd.DataFrame(rows)
    summary = df.groupby(["host", "depth_ppm"]).agg(
        injections=("phase", "size"), recovered_real_data=("real_ok", "mean"),
        recovered_white_noise=("white_ok", "mean"), median_snr_real=("real_snr", "median"),
        median_snr_white=("white_snr", "median"), noise_ppm_real=("real_noise_ppm", "median")).reset_index()
    return df, summary


def real_data(online: bool) -> pd.DataFrame:
    rows = []
    datasets = [(s["target"], s["mission"], s["dataset"], lambda sid=s["sample_id"]: load_sample(sid))
                for s in list_samples()]
    if online:
        from exoscope.data.retrieval import download_lightcurve, search_target

        for target, mission, idx in ONLINE_TARGETS:
            def loader(target=target, mission=mission, idx=idx):
                products = search_target(target, mission)
                return download_lightcurve(target, mission, min(idx, len(products) - 1), products=products)
            datasets.append((target, mission, "(downloaded)", loader))
    for target, mission, label, loader in datasets:
        try:
            d = loader()
        except Exception as exc:  # network problems: report, do not invent numbers
            rows.append({"target": target, "mission": mission, "dataset": label, "error": str(exc)[:80]})
            continue
        r = run_analysis(d)
        refs = reference_planets(target)
        ref = check_against_reference(r.best.period, refs)
        thr = r.threshold
        thr_p = thr.period.period if thr.period and thr.period.members and not thr.period.ambiguous else np.nan
        leg = legacy_detect(r.processed.time, r.processed.flux)
        rows.append({
            "target": target, "mission": mission, "dataset": d.meta.get("dataset", label),
            "reference_planet": ref["planet"] if ref else None,
            "reference_P_d": ref["reference_period"] if ref else np.nan,
            "bls_P_d": r.best.period, "bls_err_%": ref["period_error_percent"] if ref else np.nan,
            "relation": ref["kind"] if ref else None,
            "threshold_P_d": thr_p,
            "threshold_err_%": 100 * abs(thr_p - ref["reference_period"]) / ref["reference_period"] if ref else np.nan,
            "legacy_P_d": leg.period_estimate,
            "threshold_events": len(thr.events), "bls_transits": r.best.n_transits_observed,
            "jaccard": r.comparison.jaccard(), "bls_snr": r.best.depth_snr,
            "checks": f"{r.evaluation.n_pass}/{len(r.evaluation.applicable)}", "verdict": r.evaluation.level,
            "bls_runtime_s": r.timings.get("bls_s"),
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else format(v, floatfmt)
        return "—" if v is None else str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="also download extra real targets from MAST")
    args = ap.parse_args()

    t0 = time.perf_counter()
    syn = synthetic_benchmark()
    nt = null_and_eb_tests()
    real = real_data(args.online)
    inj_df, inj = injection_recovery()
    both = inj_df.dropna(subset=["real_snr", "white_snr"])
    beta = (both["white_snr"] / both["real_snr"]).median() if len(both) else float("nan")
    elapsed = time.perf_counter() - t0

    grp = syn.groupby(["depth_ppm", "noise_ppm"]).agg(
        runs=("seed", "size"), legacy_P_recovered=("legacy_ok", "mean"), threshold_P_recovered=("thr_ok", "mean"),
        bls_P_recovered=("bls_ok", "mean"), bls_median_err_pct=("bls_err_%", "median"),
        thr_precision=("thr_P", "mean"), thr_recall=("thr_R", "mean"), thr_F1=("thr_F1", "mean"),
        bls_precision=("bls_P", "mean"), bls_recall=("bls_R", "mean"), bls_F1=("bls_F1", "mean"),
    ).reset_index()
    overall = {
        "runs": len(syn),
        "legacy": syn["legacy_ok"].mean(), "threshold": syn["thr_ok"].mean(), "bls": syn["bls_ok"].mean(),
        "thr_runtime_ms": 1000 * syn["thr_s"].median(), "bls_runtime_s": syn["bls_s"].median(),
    }
    noise_sig = sum(lv != "noise" for lv in nt["noise"])
    eb_flag = sum(lv == "non-planetary" for lv in nt["eb"])

    md = ["# ExoScope evaluation\n",
          f"Generated by `scripts/evaluate.py` on {date.today().isoformat()} with ExoScope {__version__} "
          f"(total runtime {elapsed:.0f} s). Every number below is produced by that script; rerun it to reproduce.\n",
          "## 1. Synthetic benchmark\n",
          f"{len(syn)} seeded light curves: depths {DEPTHS} ppm × white noise {NOISES} ppm per point × periods "
          f"{PERIODS} d × seeds {list(SEEDS)}; 27-day baseline, 30-minute cadence, 3-hour trapezoidal transits, "
          "sinusoidal stellar variability, one data gap, outliers and missing values. Default ExoScope settings. "
          f"A period counts as recovered within 1%; events match the truth within ±{TOL_H:g} h.\n",
          f"**Overall period recovery:** legacy v1 {overall['legacy']:.0%} · improved threshold "
          f"{overall['threshold']:.0%} · BLS {overall['bls']:.0%}. Median runtime: threshold "
          f"{overall['thr_runtime_ms']:.1f} ms, BLS {overall['bls_runtime_s']:.2f} s.\n",
          md_table(grp, ".2f"), "",
          "## 2. Null and false-positive tests\n",
          f"- **Pure noise** ({N_NULL} light curves, 800 ppm noise, no transit): {noise_sig}/{N_NULL} were reported as a "
          "significant signal (checklist verdict other than “no significant signal”).",
          f"- **Eclipsing-binary imitation** ({N_NULL} light curves, alternating 8000/4000 ppm eclipses): {eb_flag}/{N_NULL} "
          "were flagged as “signs of a non-planetary origin” by the odd/even or secondary-eclipse checks.\n",
          "## 3. Real data vs NASA Exoplanet Archive\n",
          "Reference periods from the NASA Exoplanet Archive (pscomppars, retrieved 2026-10-28). One quarter/sector "
          "per target, default settings.\n",
          md_table(real, ".5g") if len(real) else "_No real datasets available._", "",
          "## 4. Injection–recovery on real light curves\n",
          "Simulated 3-hour transits were multiplied into **real** Kepler light curves after masking each star's own "
          f"planet (Kepler-10 b, Kepler-8 b) — depths {INJ_DEPTHS_PPM} ppm × periods {INJ_PERIODS} d × "
          f"{INJ_PHASES} phases per host. The same injections were made into a *white-noise twin* (same timestamps, "
          "same point-to-point scatter, Gaussian noise). Recovered = BLS best period within 1% of the injected period "
          "**and** SNR ≥ 7.1 (ExoScope's detection threshold). Data with injections are labelled “injected”, never "
          "“real observation”.\n",
          md_table(inj, ".3g"), "",
          f"Median ratio SNR(white twin) / SNR(real data) for injections found in both: **{beta:.2f}×**. Values above 1 "
          "show how much the real star's correlated (red) noise and residual systematics cost compared with ideal "
          "white noise — the effect Giacobbe et al. (2012) report as ~1.3× for their ground-based photometry.\n",
          "Raw per-run tables: `docs/results/*.csv`.\n",
          "## Limitations of this evaluation\n",
          "- Synthetic light curves use white noise; real stars have correlated (red) noise and instrumental "
          "systematics, so synthetic precision/recall are optimistic.",
          "- Real-data labels only cover *known* planets. Agreement with an archive period shows the method recovers "
          "known signals; it does not measure the false-positive rate on stars without planets.",
          "- The real-target sample is small and biased towards easy, deep hot Jupiters plus one small planet "
          "(Kepler-10 b).",
          "- “Legacy v1” is a reconstruction of the original method described in the project brief, applied to the "
          "same preprocessed light curve.",
          ]
    res = ROOT / "docs" / "results"
    res.mkdir(parents=True, exist_ok=True)
    syn.to_csv(res / "synthetic_benchmark.csv", index=False)
    grp.to_csv(res / "synthetic_summary.csv", index=False)
    real.to_csv(res / "real_targets.csv", index=False)
    inj_df.to_csv(res / "injection_recovery.csv", index=False)
    inj.to_csv(res / "injection_summary.csv", index=False)
    out = ROOT / "docs" / "EVALUATION.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
