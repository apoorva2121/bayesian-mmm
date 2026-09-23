"""Simulate -> fit -> evaluate recovery -> write reports.

Usage:
    python -m mmm.run [--config config.yaml] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import pandas as pd

from .evaluate import contribution_share_table, marginal_roi, recovery_table
from .model import build_model, fit
from .simulate import load_config, simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="fewer draws for a smoke test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["model"].update({"draws": 300, "tune": 300, "chains": 2})
    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["data"]).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, truth = simulate(cfg)
    df.to_parquet(cfg["paths"]["data"], index=False)
    Path(cfg["paths"]["truth"]).write_text(json.dumps(truth, indent=2))
    channels = list(cfg["simulation"]["channels"].keys())

    model = build_model(df, channels, cfg)
    idata = fit(model, cfg)
    summ = az.summary(idata, var_names=["intercept", "trend", "price_elasticity", "sigma"]
                      + [f"decay_{c}" for c in channels] + [f"beta_{c}" for c in channels])
    diverging = int(idata.sample_stats["diverging"].sum())
    print(f"sampled in {time.time() - t0:.0f}s | divergences: {diverging} | max r_hat: {summ['r_hat'].max():.3f}")

    rec = recovery_table(idata, truth, channels, df, cfg["evaluation"]["hdi_prob"])
    share = contribution_share_table(idata, truth, channels)
    roi = marginal_roi(idata, df, channels, cfg)
    print("\nParameter recovery:\n", rec.round(3).to_string(index=False))
    print(f"\nrecovered {rec['recovered'].sum()}/{len(rec)} shape parameters")
    print("\nContribution share recovery:\n", share.round(3).to_string(index=False))
    print("\nMarginal ROI at current spend:\n", roi.round(2).to_string(index=False))

    rec.to_csv(rep / "parameter_recovery.csv", index=False)
    share.to_csv(rep / "contribution_share.csv", index=False)
    roi.to_csv(rep / "marginal_roi.csv", index=False)
    summ.to_csv(rep / "posterior_summary.csv")
    (rep / "results.json").write_text(json.dumps({
        "divergences": diverging, "max_r_hat": float(summ["r_hat"].max()),
        "shape_params_recovered": int(rec["recovered"].sum()), "shape_params_total": int(len(rec)),
        "share_recovered": int(share["recovered"].sum()), "share_total": int(len(share)),
        "price_elasticity_true": truth["price_elasticity"],
        "price_elasticity_posterior_mean": float(summ.loc["price_elasticity", "mean"]),
        "runtime_seconds": round(time.time() - t0, 1), "quick": args.quick,
    }, indent=2))
    print(f"\nreports -> {rep}")


if __name__ == "__main__":
    main()
