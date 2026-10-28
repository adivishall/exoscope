"""Lightweight SQLite storage for analysis history and experiments.

Stores what is needed to understand and reproduce an analysis — dataset
identity, settings, algorithm versions, candidates and events — but not the
raw light curves (those are re-fetched from the local cache / MAST, or
regenerated from the stored synthetic parameters and seed).

The database path defaults to ``data/exoscope.db`` in the project folder and
can be changed with the ``EXOSCOPE_DB_PATH`` environment variable. All
queries are parameterised.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .pipeline import AnalysisResult, settings_dict_for_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "exoscope.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    app_version TEXT NOT NULL,
    target TEXT NOT NULL,
    mission TEXT,
    dataset TEXT,
    source TEXT NOT NULL,
    product TEXT,
    n_points INTEGER,
    settings_json TEXT NOT NULL,
    dataset_json TEXT NOT NULL,
    headline_json TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    comparison_json TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    rank INTEGER,
    period_days REAL, t0 REAL, duration_hours REAL, depth_ppm REAL,
    snr REAL, sde REAL, n_transits INTEGER, extra_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    event_id TEXT NOT NULL,
    t_center REAL, t_start REAL, t_end REAL, depth_ppm REAL, duration_hours REAL,
    snr REAL, n_points INTEGER, epoch INTEGER
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target TEXT NOT NULL,
    dataset TEXT,
    space_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    stats_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_analysis ON events(analysis_id);
CREATE INDEX IF NOT EXISTS idx_candidates_analysis ON candidates(analysis_id);
"""


def db_path() -> Path:
    return Path(os.environ.get("EXOSCOPE_DB_PATH", DEFAULT_DB))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def dumps(o: Any) -> str:
    return json.dumps(o, default=_json_default)


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _f(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def save_analysis(result: AnalysisResult, note: str = "", path: Path | None = None) -> int:
    meta = result.dataset_meta
    dataset_info = {"meta": meta, "summary": result.dataset_summary}
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO analyses (created_at, app_version, target, mission, dataset, source, product, n_points,
               settings_json, dataset_json, headline_json, evaluation_json, comparison_json, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.created_at, result.app_version, str(meta.get("target", "unknown")), meta.get("mission"),
                meta.get("dataset"), str(meta.get("source", "unknown")), meta.get("product"),
                result.processed.n_points, dumps(settings_dict_for_storage(result)), dumps(dataset_info),
                dumps({**result.headline(), "truth_metrics": result.truth_metrics,
                       "reference_check": result.reference_check, "timings": result.timings}),
                dumps(result.evaluation.to_dicts()), dumps(result.comparison.summary()), note[:500],
            ),
        )
        aid = int(cur.lastrowid)
        if result.bls:
            for c in result.bls.candidates:
                d = c.to_dict()
                conn.execute(
                    """INSERT INTO candidates (analysis_id, method, rank, period_days, t0, duration_hours, depth_ppm,
                       snr, sde, n_transits, extra_json) VALUES (?, 'bls', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (aid, c.rank, _f(c.period), _f(c.t0), _f(c.duration * 24), _f(c.depth * 1e6), _f(c.depth_snr),
                     _f(c.sde), c.n_transits_observed, dumps(d)),
                )
        if result.threshold and result.threshold.period and result.threshold.period.members:
            p = result.threshold.period
            conn.execute(
                """INSERT INTO candidates (analysis_id, method, rank, period_days, t0, n_transits, extra_json)
                   VALUES (?, 'threshold', 1, ?, ?, ?, ?)""",
                (aid, _f(p.period), _f(p.t0), p.n_members, dumps({"period_err": p.period_err, "note": p.note})),
            )
        events = (result.threshold.events if result.threshold else []) + (result.best.events if result.best else [])
        conn.executemany(
            """INSERT INTO events (analysis_id, method, event_id, t_center, t_start, t_end, depth_ppm, duration_hours,
               snr, n_points, epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(aid, e.method, e.event_id, _f(e.t_center), _f(e.t_start), _f(e.t_end), _f(e.depth * 1e6),
              _f(e.duration * 24), _f(e.snr), int(e.n_points), e.epoch) for e in events],
        )
    result.analysis_id = aid
    return aid


def list_analyses(limit: int = 200, path: Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, target, mission, dataset, source, n_points, headline_json, note "
            "FROM analyses ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["headline"] = json.loads(d.pop("headline_json"))
        out.append(d)
    return out


def load_analysis(analysis_id: int, path: Path | None = None) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (int(analysis_id),)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("settings_json", "dataset_json", "headline_json", "evaluation_json", "comparison_json"):
            d[k.removesuffix("_json")] = json.loads(d.pop(k)) if d.get(k) else None
        d["candidates"] = [dict(r) for r in conn.execute(
            "SELECT method, rank, period_days, t0, duration_hours, depth_ppm, snr, sde, n_transits "
            "FROM candidates WHERE analysis_id = ? ORDER BY method, rank", (int(analysis_id),))]
        d["events"] = [dict(r) for r in conn.execute(
            "SELECT method, event_id, t_center, depth_ppm, duration_hours, snr, n_points, epoch "
            "FROM events WHERE analysis_id = ? ORDER BY t_center", (int(analysis_id),))]
    return d


def delete_analysis(analysis_id: int, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (int(analysis_id),))


def save_experiment(target: str, dataset: str, space: dict, results: list[dict], stats: dict,
                    created_at: str, path: Path | None = None) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO experiments (created_at, target, dataset, space_json, results_json, stats_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, target, dataset, dumps(space), dumps(results), dumps(stats)),
        )
        return int(cur.lastrowid)


def list_experiments(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, target, dataset, space_json, results_json, stats_json "
            "FROM experiments ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("space_json", "results_json", "stats_json"):
            d[k.removesuffix("_json")] = json.loads(d.pop(k))
        out.append(d)
    return out
