"""PyMC market mix model: baseline + trend + Fourier seasonality + log-price +
per-channel adstock -> Hill saturation -> scaled contribution.

Priors are weakly informative and stated in the README; nothing is set to the
true value. Spend is scaled by its channel mean inside the model so that the
half-saturation prior is in interpretable units.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .transforms import geometric_adstock_pt, hill_pt


def build_model(df: pd.DataFrame, channels: list[str], cfg: dict[str, Any]) -> pm.Model:
    T = len(df)
    t = np.arange(T)
    y = df["sales"].to_numpy(dtype=float)
    y_scale = y.mean()
    log_price = np.log(df["price_index"].to_numpy(dtype=float))
    fourier = np.column_stack([
        np.sin(2 * np.pi * t / 52.18), np.cos(2 * np.pi * t / 52.18),
        np.sin(4 * np.pi * t / 52.18), np.cos(4 * np.pi * t / 52.18),
    ])
    max_lag = cfg["model"]["max_lag"]

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=1.0, sigma=0.5)                 # in units of mean sales
        trend = pm.Normal("trend", mu=0.0, sigma=0.01)                        # per week, mean-sales units
        seas = pm.Normal("seasonality", mu=0.0, sigma=0.2, shape=fourier.shape[1])
        price_elast = pm.Normal("price_elasticity", mu=-1.0, sigma=1.0)

        mu = intercept + trend * t + pt.dot(fourier, seas) + price_elast * log_price

        for ch in channels:
            x = df[f"spend_{ch}"].to_numpy(dtype=float)
            x_scaled = x / x.mean()
            decay = pm.Beta(f"decay_{ch}", alpha=2, beta=2)
            half_sat = pm.Gamma(f"half_sat_{ch}", alpha=3, beta=3)         # ~1 = mean spend
            slope = pm.Gamma(f"slope_{ch}", alpha=3, beta=1.5)             # ~2
            beta = pm.HalfNormal(f"beta_{ch}", sigma=0.5)                  # contribution at full saturation, mean-sales units
            ad = geometric_adstock_pt(pt.as_tensor_variable(x_scaled), decay, max_lag)
            sat = hill_pt(ad, half_sat, slope)
            pm.Deterministic(f"contrib_{ch}", beta * sat * y_scale)
            mu = mu + beta * sat

        sigma = pm.HalfNormal("sigma", sigma=0.2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y / y_scale)
    return model


def fit(model: pm.Model, cfg: dict[str, Any]) -> az.InferenceData:
    m = cfg["model"]
    with model:
        idata = pm.sample(draws=m["draws"], tune=m["tune"], chains=m["chains"],
                          target_accept=m["target_accept"], random_seed=m["seed"],
                          progressbar=False)
    return idata
