from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path


FACTORS = {
    "embed_excitation_dim": [2, 4, 8],
    "embed_channel_dim": [8, 12, 24],
    "low_hidden": ["32,64,32", "64,128,64", "128,128,64"],
    "mod_gamma": [0.1, 0.2, 0.4],
    "dropout_prob": [0.02, 0.05, 0.10],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 5e-5, 1e-4],
}


L27 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1],
    [1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2],
    [1, 3, 3, 3, 1, 1, 1, 3, 3, 3, 2, 2, 2],
    [1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 3, 3, 3],
    [1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1],
    [2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    [2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    [2, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2],
    [2, 2, 3, 1, 1, 2, 3, 2, 3, 1, 3, 1, 2],
    [2, 2, 3, 1, 2, 3, 1, 3, 1, 2, 1, 2, 3],
    [2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 2, 3, 1],
    [2, 3, 1, 2, 1, 2, 3, 3, 1, 2, 2, 3, 1],
    [2, 3, 1, 2, 2, 3, 1, 1, 2, 3, 3, 1, 2],
    [2, 3, 1, 2, 3, 1, 2, 2, 3, 1, 1, 2, 3],
    [3, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2],
    [3, 1, 3, 2, 2, 1, 3, 2, 1, 3, 2, 1, 3],
    [3, 1, 3, 2, 3, 2, 1, 3, 2, 1, 3, 2, 1],
    [3, 2, 1, 3, 1, 3, 2, 2, 1, 3, 3, 2, 1],
    [3, 2, 1, 3, 2, 1, 3, 3, 2, 1, 1, 3, 2],
    [3, 2, 1, 3, 3, 2, 1, 1, 3, 2, 2, 1, 3],
    [3, 3, 2, 1, 1, 3, 2, 3, 2, 1, 2, 1, 3],
    [3, 3, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2, 1],
    [3, 3, 2, 1, 3, 2, 1, 2, 1, 3, 1, 3, 2],
]


def _config_for_row(row: list[int]) -> dict:
    names = list(FACTORS)
    return {name: FACTORS[name][row[i] - 1] for i, name in enumerate(names)}


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_run_summary(run_dir: Path) -> tuple[list[float], list[float]]:
    summary_path = run_dir / "kfold_summary.csv"
    best_vals: list[float] = []
    final_trains: list[float] = []
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                best_vals.append(_float_or_nan(row.get("best_metric", "")))
                final_trains.append(_float_or_nan(row.get("final_train_loss", "")))
        return best_vals, final_trains

    for fold in range(5):
        log_path = run_dir / f"fold_{fold}" / "train_log.csv"
        if not log_path.exists():
            best_vals.append(float("nan"))
            final_trains.append(float("nan"))
            continue
        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        val_values = [_float_or_nan(r.get("loss_val_all", r.get("loss_val_total", ""))) for r in rows]
        train_values = [_float_or_nan(r.get("loss_train_all", r.get("loss_train_total", ""))) for r in rows]
        val_values = [v for v in val_values if math.isfinite(v)]
        train_values = [v for v in train_values if math.isfinite(v)]
        best_vals.append(min(val_values) if val_values else float("nan"))
        final_trains.append(train_values[-1] if train_values else float("nan"))
    return best_vals, final_trains


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def _std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _network_capacity(cfg: dict) -> int:
    hidden = [int(x) for x in str(cfg["low_hidden"]).split(",") if x]
    return int(cfg["embed_excitation_dim"]) + int(cfg["embed_channel_dim"]) + sum(hidden)


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _factor_effects(summary_rows: list[dict]) -> list[dict]:
    out = []
    for factor, levels in FACTORS.items():
        for level in levels:
            rows = [r for r in summary_rows if str(r[factor]) == str(level)]
            out.append(
                {
                    "factor": factor,
                    "level": level,
                    "mean_best_val_loss": _mean([float(r["mean_best_val_loss"]) for r in rows]),
                    "mean_final_train_loss": _mean([float(r["mean_final_train_loss"]) for r in rows]),
                    "mean_generalization_gap": _mean([float(r["generalization_gap"]) for r in rows]),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run L27 Taguchi search for fixed-SF base training.")
    parser.add_argument("--out_root", type=str, default="taguchi_runs")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for idx, row in enumerate(L27, start=1):
        cfg = _config_for_row(row)
        run_dir = out_root / f"run_{idx:03d}"
        cmd = [
            args.python,
            "-m",
            "surrogate_model.cli.train",
            "--kfold",
            "5",
            "--val_split_strategy",
            "height_mach_kfold",
            "--pole_target",
            "freq_damping",
            "--residue_target",
            "polar",
            "--epochs",
            str(args.epochs),
            "--early_stop_patience",
            "0",
            "--embed_excitation_dim",
            str(cfg["embed_excitation_dim"]),
            "--embed_channel_dim",
            str(cfg["embed_channel_dim"]),
            "--low_hidden",
            str(cfg["low_hidden"]),
            "--use_modulation",
            "--mod_gamma",
            str(cfg["mod_gamma"]),
            "--dropout_prob",
            str(cfg["dropout_prob"]),
            "--lr",
            str(cfg["lr"]),
            "--weight_decay",
            str(cfg["weight_decay"]),
            "--out_dir",
            str(run_dir),
        ]
        print(f"[Taguchi] run_{idx:03d}: {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

        best_vals, final_trains = _read_run_summary(run_dir)
        row_out = {"run_id": f"run_{idx:03d}", **cfg}
        for fold in range(5):
            row_out[f"fold{fold}_best_val_loss"] = best_vals[fold] if fold < len(best_vals) else float("nan")
            row_out[f"fold{fold}_final_train_loss"] = final_trains[fold] if fold < len(final_trains) else float("nan")
        mean_best = _mean(best_vals)
        mean_train = _mean(final_trains)
        row_out["mean_best_val_loss"] = mean_best
        row_out["mean_final_train_loss"] = mean_train
        row_out["std_best_val_loss"] = _std(best_vals)
        row_out["generalization_gap"] = mean_best - mean_train if math.isfinite(mean_best) and math.isfinite(mean_train) else float("nan")
        row_out["network_capacity_score"] = _network_capacity(cfg)
        summary_rows.append(row_out)
        _write_csv(out_root / "taguchi_summary.csv", summary_rows)

    summary_rows.sort(
        key=lambda r: (
            float(r["mean_best_val_loss"]),
            float(r["std_best_val_loss"]),
            float(r["generalization_gap"]),
            int(r["network_capacity_score"]),
        )
    )
    _write_csv(out_root / "taguchi_summary.csv", summary_rows)
    _write_csv(out_root / "taguchi_factor_effects.csv", _factor_effects(summary_rows))

    best = summary_rows[0]
    with open(out_root / "taguchi_best_config.txt", "w", encoding="utf-8") as f:
        for key, value in best.items():
            f.write(f"{key}: {value}\n")
    print(f"[Taguchi] best={best['run_id']} mean_best_val_loss={best['mean_best_val_loss']}", flush=True)


if __name__ == "__main__":
    main()
