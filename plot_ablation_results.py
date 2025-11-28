#!/usr/bin/env python3
"""
plot_ablation_results.py

Reads the output of run_ablation.py and produces:
  • Training curves (loss + proxy FID) for every run
  • Bar charts comparing final FID scores grouped by ablation key

It is safe to run while training is still in progress:
  - Missing metrics.jsonl / summary.csv are skipped.
  - Runs without final_fid are skipped in the bar plots.

All sweeps, training curves + final FID bars:

python plot_ablation_results.py \
  --runs_dir runs/ablations \
  --figures_dir figures \
  --all


Only one ablation, everything:

python plot_ablation_results.py \
  --runs_dir runs/ablations \
  --figures_dir figures \
  --ablation embed_t_r_name


Only training curves (no bar charts), for all ablations:

python plot_ablation_results.py \
  --runs_dir runs/ablations \
  --figures_dir figures \
  --all \
  --training-curves
  
"""
import json
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------
# Helpers: metrics loading
# ----------------------


def _load_jsonl(path: Path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_metrics_for_run(run_dir: Path):
    """
    Load metrics for a run from:
      - run_dir/metrics.jsonl  OR
      - run_dir/checkpoints/metrics.jsonl

    Returns list[dict] (possibly empty).
    """
    candidates = [
        run_dir / "metrics.jsonl",
        run_dir / "checkpoints" / "metrics.jsonl",
    ]
    for p in candidates:
        if p.exists():
            try:
                return _load_jsonl(p)
            except Exception:
                return []
    return []


# ----------------------
# Plotting: training curves
# ----------------------


def plot_training_curves(run_dir: Path, metrics: list, figdir: Path):
    """Create mean_loss + fid_k curves for one run (if metrics exist)."""
    if not metrics:
        return  # nothing to plot

    # x-axis: prefer global_step, else epoch, else index
    steps = []
    for i, m in enumerate(metrics):
        if "global_step" in m:
            steps.append(m["global_step"])
        elif "step" in m:
            steps.append(m["step"])
        elif "epoch" in m:
            steps.append(m["epoch"])
        else:
            steps.append(i)

    # Loss: mean_loss (your case), fallback to "loss" if needed
    losses = [m.get("mean_loss", m.get("loss")) for m in metrics]

    # Detect which fid_* key is present (e.g. "fid_2", "fid_1000", etc.)
    fid_key = None
    for m in metrics:
        for k in m.keys():
            if isinstance(k, str) and k.startswith("fid_"):
                fid_key = k
                break
        if fid_key is not None:
            break

    if fid_key is not None:
        fids = [m.get(fid_key) for m in metrics]
    else:
        fids = [None] * len(metrics)

    figdir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    if any(l is not None for l in losses):
        plt.figure(figsize=(6, 4))
        plt.plot(steps, losses, marker="o", label="mean_loss")
        plt.xlabel("global_step / epoch")
        plt.ylabel("mean_loss")
        plt.title(f"Loss Curve\n{run_dir.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figdir / "loss_curve.png")
        plt.close()

    # FID_k curve
    if fid_key is not None and any(f is not None for f in fids):
        plt.figure(figsize=(6, 4))
        plt.plot(steps, fids, marker="o", color="orange", label=fid_key)
        plt.xlabel("global_step / epoch")
        plt.ylabel(fid_key)
        plt.title(f"{fid_key} Curve\n{run_dir.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figdir / f"{fid_key}_curve.png")
        plt.close()


# ----------------------
# Plotting: final FID bars from summary.csv
# ----------------------


def plot_final_fid_bar_for_sweep(ablation_key: str, sweep_dir: Path, figdir: Path):
    """
    Create bar chart of final FID for this ablation sweep using sweep_dir/summary.csv.
    If summary.csv or usable rows are missing, silently skip.
    """
    summary_csv = sweep_dir / "summary.csv"
    if not summary_csv.exists():
        return

    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return

    if "sweep_val" not in df.columns or "final_fid" not in df.columns:
        return

    df = df.dropna(subset=["final_fid"])
    if df.empty:
        return

    labels = [str(v) for v in df["sweep_val"]]
    scores = df["final_fid"].astype(float).values

    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.bar(labels, scores)
    plt.xlabel(f"Ablation parameter: {ablation_key}")
    plt.ylabel("Final FID")
    plt.title(f"Final FID comparison for {ablation_key}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / f"final_fid_bar_{ablation_key}.png")
    plt.close()


# ----------------------
# Main CLI
# ----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_dir",
        type=str,
        default="runs/ablations",
        help="Directory containing ablation sweeps (as created by run_ablation.py).",
    )
    ap.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Where to save generated plots.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process all ablation sweeps under runs_dir.",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Only process this ablation key (subdirectory name under runs_dir).",
    )
    ap.add_argument(
        "--training-curves",
        action="store_true",
        help="Only generate per-run training curves; skip final FID bar plots.",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"[WARN] runs_dir does not exist: {runs_dir}")
        return

    # Find available sweeps
    all_sweeps = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())

    if args.ablation is not None:
        if args.ablation not in all_sweeps:
            print(
                f"[WARN] ablation '{args.ablation}' not found in {runs_dir}. "
                f"Available: {all_sweeps}"
            )
            return
        sweep_keys = [args.ablation]
    else:
        # Default: process all sweeps (so --all is mostly a hint)
        sweep_keys = all_sweeps

    if not sweep_keys:
        print(f"[INFO] No ablation subdirectories found under {runs_dir}.")
        return

    for ablation_key in sweep_keys:
        sweep_dir = runs_dir / ablation_key
        if not sweep_dir.is_dir():
            continue

        sweep_fig_dir = figures_dir / ablation_key
        sweep_fig_dir.mkdir(parents=True, exist_ok=True)

        # 1) Final FID bar plot (unless user asked for training-curves only)
        if not args.training_curves:
            plot_final_fid_bar_for_sweep(ablation_key, sweep_dir, sweep_fig_dir)

        # 2) Per-run training curves
        for run_dir in sweep_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metrics = load_metrics_for_run(run_dir)
            if not metrics:
                continue

            run_fig_dir = sweep_fig_dir / run_dir.name
            plot_training_curves(run_dir, metrics, run_fig_dir)

    print(f"[INFO] Plots saved under: {figures_dir}")


if __name__ == "__main__":
    main()
