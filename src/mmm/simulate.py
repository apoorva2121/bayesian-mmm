"""Generate weekly sales with known media effects, trend, seasonality, price and noise.

Writing the ground truth to disk is the point: the model is later scored on
whether its posteriors recover these exact numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .transforms import geometric_adstock_np, hill_np


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def simulate(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    s = cfg["simulation"]
    rng = np.random.default_rng(s["seed"])
    T = s["n_weeks"]
    t = np.arange(T)
    df = pd.DataFrame({"week": pd.date_range("2023-01-02", periods=T, freq="W-MON")})

    media_contrib = np.zeros(T)
    truth: dict[str, Any] = {"channels": {}, "contribution_share": {}}
    for ch, p in s["channels"].items():
        spend = np.clip(rng.normal(p["mean_spend"], p["spend_sd"], T), 0, None)
        # occasional dark weeks / bursts make identification realistic
        spend[rng.random(T) < 0.08] = 0.0
        spend[rng.random(T) < 0.05] *= 2.2
        df[f"spend_{ch}"] = spend.round(0)
        ad = geometric_adstock_np(spend, p["adstock_decay"], cfg["model"]["max_lag"])
        sat = hill_np(ad, p["hill_half_sat"], p["hill_slope"])
        contrib = p["beta"] * s["baseline_sales"] * 0.25 * sat
        df[f"true_contrib_{ch}"] = contrib
        media_contrib += contrib
        truth["channels"][ch] = {k: p[k] for k in ("adstock_decay", "hill_half_sat", "hill_slope", "beta")}

    price = np.clip(1.0 + rng.normal(0, 0.05, T) + 0.06 * np.sin(2 * np.pi * t / 26), 0.8, 1.25)
    df["price_index"] = price
    base = s["baseline_sales"] + s["trend_per_week"] * t
    season = s["seasonality_amplitude"] * np.sin(2 * np.pi * t / 52.18)
    price_effect = s["baseline_sales"] * (np.exp(s["price_elasticity"] * np.log(price)) - 1.0)
    noise = rng.normal(0, s["noise_sd"], T)
    df["sales"] = np.clip(base + season + price_effect + media_contrib + noise, 1, None).round(0)

    total_media = media_contrib.sum()
    for ch in s["channels"]:
        truth["contribution_share"][ch] = float(df[f"true_contrib_{ch}"].sum() / total_media)
    truth.update({
        "baseline_sales": s["baseline_sales"], "trend_per_week": s["trend_per_week"],
        "seasonality_amplitude": s["seasonality_amplitude"], "price_elasticity": s["price_elasticity"],
        "noise_sd": s["noise_sd"], "media_share_of_sales": float(total_media / df["sales"].sum()),
    })
    return df, truth


def main() -> None:
    cfg = load_config()
    df, truth = simulate(cfg)
    Path(cfg["paths"]["data"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["paths"]["data"], index=False)
    Path(cfg["paths"]["truth"]).write_text(json.dumps(truth, indent=2))
    print(f"{len(df)} weeks -> {cfg['paths']['data']}; media share of sales {truth['media_share_of_sales']:.1%}")
    print(pd.Series(truth["contribution_share"]).round(3).to_string())


if __name__ == "__main__":
    main()
