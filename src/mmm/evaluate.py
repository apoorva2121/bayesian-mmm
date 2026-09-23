"""Score the fitted model against the known ground truth.

Three questions a marketing team actually asks, each with a pass/fail:
1. Did we recover each channel's carryover (decay) and saturation shape?
2. Did we recover each channel's share of media-driven sales?
3. Do the response curves give a sane budget reallocation?
"""

from __future__ import annotations

import json
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from .transforms import geometric_adstock_np, hill_np


def _hdi(draws: np.ndarray, prob: float) -> tuple[float, float]:
    """Shortest interval containing `prob` of the draws (library-independent)."""
    x = np.sort(np.asarray(draws).ravel())
    n = len(x); k = max(int(np.floor(prob * n)), 1)
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def recovery_table(idata: az.InferenceData, truth: dict[str, Any], channels: list[str],
                   df: pd.DataFrame, hdi_prob: float) -> pd.DataFrame:
    rows = []
    post = idata.posterior
    for ch in channels:
        spend_mean = df[f"spend_{ch}"].mean()
        tr = truth["channels"][ch]
        checks = {
            "adstock_decay": (post[f"decay_{ch}"].values.ravel(), tr["adstock_decay"]),
            # half_sat is fitted in units of mean spend -> convert back to currency
            "hill_half_sat": (post[f"half_sat_{ch}"].values.ravel() * spend_mean, tr["hill_half_sat"]),
            "hill_slope": (post[f"slope_{ch}"].values.ravel(), tr["hill_slope"]),
        }
        for name, (draws, true_val) in checks.items():
            lo, hi = _hdi(draws, hdi_prob)
            rows.append({"channel": ch, "parameter": name, "true": true_val,
                         "posterior_median": float(np.median(draws)),
                         f"hdi_{int(hdi_prob*100)}_low": float(lo), f"hdi_{int(hdi_prob*100)}_high": float(hi),
                         "recovered": bool(lo <= true_val <= hi)})
    return pd.DataFrame(rows)


def contribution_share_table(idata: az.InferenceData, truth: dict[str, Any], channels: list[str]) -> pd.DataFrame:
    post = idata.posterior
    totals = {ch: post[f"contrib_{ch}"].sum(dim="contrib_{}_dim_0".format(ch)) for ch in channels}
    grand = sum(totals.values())
    rows = []
    for ch in channels:
        share = (totals[ch] / grand).values.ravel()
        lo, hi = _hdi(share, 0.94)
        rows.append({"channel": ch, "true_share": truth["contribution_share"][ch],
                     "posterior_median": float(np.median(share)), "hdi_low": float(lo), "hdi_high": float(hi),
                     "recovered": bool(lo <= truth["contribution_share"][ch] <= hi)})
    return pd.DataFrame(rows)


def marginal_roi(idata: az.InferenceData, df: pd.DataFrame, channels: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Posterior-median marginal sales per unit spend at the current mean spend level."""
    post = idata.posterior
    y_scale = df["sales"].mean()
    rows = []
    for ch in channels:
        spend_mean = df[f"spend_{ch}"].mean()
        decay = float(post[f"decay_{ch}"].median()); ks = float(post[f"half_sat_{ch}"].median())
        slope = float(post[f"slope_{ch}"].median()); beta = float(post[f"beta_{ch}"].median())
        # steady-state adstock of a constant spend equals that spend (weights sum to 1)
        x0 = 1.0; dx = 0.01
        r0 = beta * hill_np(np.array([x0]), ks, slope)[0] * y_scale
        r1 = beta * hill_np(np.array([x0 + dx]), ks, slope)[0] * y_scale
        rows.append({"channel": ch, "mean_weekly_spend": spend_mean,
                     "marginal_sales_per_currency": (r1 - r0) / (dx * spend_mean)})
    return pd.DataFrame(rows).sort_values("marginal_sales_per_currency", ascending=False)
