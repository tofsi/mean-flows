#!/usr/bin/env python3
"""
run_ablation.py

CLI launcher for Mean Flows Table 1 ablations, matching your current train.py.

What it does:
  - Loads defaults/sweeps from ablation_configs.py
  - For each run, creates:
        <outdir>/<sweep_key>/<sweep_key>=<value>/
    and runs training INSIDE that directory so that:
        checkpoints/metrics.jsonl   and checkpoints/checkpoint_* are per-run.
  - After training, computes FINAL FID-K (default 50k) and saves it.
  - Writes global + per-sweep summaries:
        <outdir>/summary.jsonl
        <outdir>/summary.csv
        <outdir>/<sweep_key>/summary.csv

  Reloading from checkpoints and skipping completed runs:
  - If a run has a DONE file, it is skipped (training assumed complete).
  - If checkpoints/ exists but DONE is missing, training is resumed from latest
    checkpoint (via a resume_from_checkpoint flag passed to Trainer).
  - If --clear_checkpoints is given, checkpoints and DONE are deleted at the
    start of each run so it restarts from scratch.

Usage:
  python run_ablation.py --all
  python run_ablation.py --ablation time_sampler_params
  python run_ablation.py --set epochs=20 --ablation p
  python run_ablation.py --set p=0.5 --set ratio_r_not_eq_t=0.5
  python run_ablation.py --dry_run

  # force re-run from scratch (wipe checkpoints/DONE)
  python run_ablation.py --ablation embed_t_r_name --clear_checkpoints
"""

import argparse, copy, json, os, time, re, ast, shutil

from flax.training import checkpoints

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path

from ablation_configs import ABLATIONS
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from train import TrainingParams, Trainer


# --- embed_t_r_name -> lambda (Table 1c) ---
EMBED_FN_MAP = {
    "tr": (lambda t, r: (t, r)),
    "t_tr": (lambda t, r: (t, t - r)),
    "tr_t_tr": (lambda t, r: (t, r, t - r)),
    "t_tr_only": (lambda t, r: (t - r,)),
}

# --- dimensionality of above embeddings ---
EMBED_DIM_MAP = {
    "tr": 2,
    "t_tr": 2,
    "tr_t_tr": 3,
    "t_tr_only": 1,
}


# --- JVP tangent label -> (dr_tangent, dt_tangent) (Table 1b, your ordering) ---
JVP_TANGENT_MAP = {
    "(v,0,1)": (False, True),
    "(v,0,0)": (False, False),
    "(v,1,0)": (True, False),
    "(v,1,1)": (True, True),
}


def parse_set_kv(kv: str):
    """
    Parse --set key=value.

    Supports:
      - JSON (numbers, lists, dicts, true/false/null)
      - bare numbers (e.g. 0.25, 10)
      - Python literals (tuples, lists, None, etc.)
      - strings (fallback)
    """
    if "=" not in kv:
        raise ValueError("--set requires key=value")
    k, v = kv.split("=", 1)
    v = v.strip()

    # Explicit handling for None (case-insensitive)
    if v.lower() == "none":
        return k, None

    # 1) Try JSON (handles numbers, bool, null, lists, dicts)
    try:
        parsed = json.loads(v)
        return k, parsed
    except Exception:
        pass

    # 2) Try plain numbers (backwards compatible)
    try:
        if "." in v or "e" in v.lower():
            return k, float(v)
        return k, int(v)
    except Exception:
        pass

    # 3) Try Python literals: tuples, lists, "None", etc.
    try:
        parsed = ast.literal_eval(v)
        return k, parsed
    except Exception:
        # 4) Fallback: leave as string
        return k, v


def make_run_name(sweep_key=None, sweep_val=None):
    if sweep_key is None:
        return "single"
    if sweep_val is None:
        val_str = "None"
    elif isinstance(sweep_val, (tuple, list)):
        val_str = "_".join(str(x) for x in sweep_val)
    else:
        val_str = str(sweep_val)
    val_str = re.sub(r"[^A-Za-z0-9\-\._]+", "_", val_str)
    return f"{sweep_key}={val_str}"


def cfg_to_training_params(cfg: dict):
    cfg = copy.deepcopy(cfg)

    # --- kill old key if it exists ---
    # (prevents your exact error)
    cfg.pop("time_sampler", None)

    # --- embed_t_r_name -> embed_t_r callable ---
    name = cfg.get("embed_t_r_name")
    if name not in EMBED_FN_MAP:
        raise ValueError(f"Unknown embed_t_r_name: {name}")
    cfg["embed_t_r"] = EMBED_FN_MAP[name]
    cfg["time_embed_dim"] = EMBED_DIM_MAP[name]

    # --- jvp_tangent -> jvp_computation tuple ---
    jt = cfg.get("jvp_tangent")
    if jt not in JVP_TANGENT_MAP:
        raise ValueError(f"Unknown jvp_tangent: {jt}")
    cfg["jvp_computation"] = JVP_TANGENT_MAP[jt]

    # remove sweep-only label
    cfg.pop("jvp_tangent", None)

    # --- required fields per TrainingParams ---
    required = {
        "architecture",
        "epochs",
        "lr",
        "beta1",
        "beta2",
        "ema_decay",
        "p",
        "omega",
        "ratio_r_not_eq_t",
        "jvp_computation",
        "embed_t_r_name",
        "embed_t_r",
        "time_embed_dim",
        "time_sampler_params",
    }
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing required TrainingParams fields: {missing}")

    # Pass exactly the dataclass fields (no extras)
    return TrainingParams(
        architecture=cfg["architecture"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        beta1=cfg["beta1"],
        beta2=cfg["beta2"],
        ema_decay=cfg["ema_decay"],
        p=cfg["p"],
        omega=cfg["omega"],
        ratio_r_not_eq_t=cfg["ratio_r_not_eq_t"],
        jvp_computation=cfg["jvp_computation"],
        embed_t_r_name=cfg["embed_t_r_name"],
        embed_t_r=cfg["embed_t_r"],
        time_embed_dim=cfg["time_embed_dim"],
        time_sampler_params=cfg["time_sampler_params"],
    )


def append_summary(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path):
    if not jsonl_path.exists():
        return
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    # flatten tuple fields for csv
    flat_rows = []
    for r in rows:
        fr = {}
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                fr[k] = "_".join(map(str, v))
            else:
                fr[k] = v
        flat_rows.append(fr)

    import csv

    keys = sorted({k for r in flat_rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat_rows)


def clear_run_checkpoints(run_dir: Path):
    """Delete checkpoints, DONE marker and final_fid.json for a run."""
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    done_file = run_dir / "DONE"
    if done_file.exists():
        done_file.unlink()
    final_fid_file = run_dir / "final_fid.json"
    if final_fid_file.exists():
        final_fid_file.unlink()


def run_one(
    cfg: dict, sweep_out: Path, run_name: str, final_fid_k: int, clear_ckpts=False
):
    """
    Run training in its own directory so checkpoints/metrics are per-run.
    Then compute final FID-K and save it.
    """
    run_dir = sweep_out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if clear_ckpts:
        print(f"[run_ablation] Clearing checkpoints for {run_name}")
        clear_run_checkpoints(run_dir)

    done_file = run_dir / "DONE"
    final_fid_file = run_dir / "final_fid.json"
    done_file = run_dir / "DONE"
    final_fid_file = run_dir / "final_fid.json"

    # Case 1: training already finished for this config
    if done_file.exists():
        print(f"[run_ablation] Skipping {run_name}: DONE marker exists.")
        final_fid = None
        if final_fid_file.exists():
            try:
                with open(final_fid_file, "r") as f:
                    final_fid = json.load(f).get("final_fid")
            except Exception:
                final_fid = None
        return final_fid, str(run_dir)

    tp = cfg_to_training_params(cfg)
    resume = False
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print(f"[run_ablation] Found existing checkpoint: {latest}")
        resume = True

    trainer = Trainer(
        tp,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoints=resume,
    )

    # run inside run_dir so Trainer writes checkpoints there
    old_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        trained_params = trainer.train()

        final_fid = None
        if final_fid_k and final_fid_k > 0:
            final_fid = trainer.eval_fid(trained_params, num_samples=final_fid_k)

        # save final fid explicitly
        with open("final_fid.json", "w") as f:
            json.dump({"final_fid_k": final_fid_k, "final_fid": final_fid}, f, indent=2)

    finally:
        os.chdir(old_cwd)

    # mark completion
    with open(run_dir / "DONE", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    return final_fid, str(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=[k for k in ABLATIONS.keys() if k != "default"],
        help="Which ablation sweep to run (e.g., p, omega, time_sampler_params).",
    )
    ap.add_argument("--all", action="store_true", help="Run all Table 1 sweeps.")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a setting: key=value. Can be repeated.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="runs/ablations",
        help="Base directory for runs.",
    )
    ap.add_argument(
        "--final_fid_k",
        type=int,
        default=1000,
        help="Compute and record final FID-K after training (0 to skip).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configs that would run, but do not train.",
    )
    ap.add_argument(
        "--clear_checkpoints",
        action="store_true",
        help="Before each run, delete its checkpoints/DONE/final_fid so it restarts from scratch.",
    )
    args = ap.parse_args()

    base_cfg = copy.deepcopy(ABLATIONS["default"])

    # manual overrides
    for kv in args.set:
        k, v = parse_set_kv(kv)
        if k not in base_cfg and k not in ABLATIONS:
            print(f"Warning: overriding unknown key '{k}'")
        base_cfg[k] = v

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweeps = []
    if args.all:
        for k, spec in ABLATIONS.items():
            if k == "default":
                continue
            sweeps.append((k, spec["values"]))
    elif args.ablation:
        sweeps.append((args.ablation, ABLATIONS[args.ablation]["values"]))

    summary_jsonl = outdir / "summary.jsonl"
    if summary_jsonl.exists():
        summary_jsonl.unlink()

    if not sweeps:
        print("Running single config:")
        print(json.dumps(base_cfg, indent=2))
        if not args.dry_run:
            final_fid, run_dir = run_one(base_cfg, outdir, "single", args.final_fid_k)
            row = {
                "sweep_key": None,
                "sweep_val": None,
                "run_dir": run_dir,
                "final_fid": final_fid,
                **base_cfg,
            }
            append_summary(summary_jsonl, row)
            write_csv_from_jsonl(summary_jsonl, outdir / "summary.csv")
        return

    for sweep_key, values in sweeps:
        sweep_out = outdir / sweep_key
        sweep_out.mkdir(parents=True, exist_ok=True)
        sweep_summary = sweep_out / "summary.jsonl"
        if sweep_summary.exists():
            sweep_summary.unlink()

        print(f"\n=== Sweep: {sweep_key} ({len(values)} runs) ===")
        for v in values:
            cfg = copy.deepcopy(base_cfg)
            cfg[sweep_key] = v
            run_name = make_run_name(sweep_key, v)
            print("Config:", run_name)

            if args.dry_run:
                continue

            final_fid, run_dir = run_one(cfg, sweep_out, run_name, args.final_fid_k)
            row = {
                "sweep_key": sweep_key,
                "sweep_val": v,
                "run_dir": run_dir,
                "final_fid": final_fid,
                **cfg,
            }
            append_summary(summary_jsonl, row)
            append_summary(sweep_summary, row)

        write_csv_from_jsonl(sweep_summary, sweep_out / "summary.csv")

    write_csv_from_jsonl(summary_jsonl, outdir / "summary.csv")


if __name__ == "__main__":
    main()
