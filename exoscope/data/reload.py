"""Recreate a dataset from stored metadata (used by the History page)."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from .model import DataError, LightCurveData
from .retrieval import download_lightcurve, search_target
from .samples import load_sample
from .synthetic import SyntheticParams, generate_synthetic


def reload_dataset(meta: dict[str, Any]) -> LightCurveData:
    source = meta.get("source")
    if source == "synthetic":
        raw = meta.get("synthetic_params") or {}
        names = {f.name for f in fields(SyntheticParams)}
        return generate_synthetic(SyntheticParams(**{k: v for k, v in raw.items() if k in names}))
    if source == "bundled" and meta.get("sample_id"):
        return load_sample(meta["sample_id"])
    if source == "mast":
        target, mission, product = meta.get("target"), meta.get("mission"), meta.get("product")
        products = search_target(target, mission)
        for p in products:
            if p["product"] == product:
                return download_lightcurve(target, mission, p["index"], products=products)
        raise DataError(f"Product {product} is no longer listed for {target}.")
    raise DataError(f"Cannot reload a dataset from source '{source}'.")
