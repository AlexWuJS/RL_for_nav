import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MODE_COLORS = {
    "baseline": "#2f6fdd",
    "shield_only": "#2ca02c",
    "mppi_dbas": "#d62728",
    "trust_mppi": "#9467bd",
    "trust_mppi_dbas": "#ff7f0e",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def discover_metric_files(result_dir: str) -> Dict[str, str]:
    files = {}
    for name in os.listdir(result_dir):
        if not name.endswith("_metrics.csv"):
            continue
        mode = name[: -len("_metrics.csv")]
        files[mode] = os.path.join(result_dir, name)
    return dict(sorted(files.items()))


def load_metric_rows(result_dir: str) -> Dict[str, List[Dict[str, str]]]:
    return {mode: read_csv_rows(path) for mode, path in discover_metric_files(result_dir).items()}


def load_summary(result_dir: str) -> Dict:
    path = os.path.join(result_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def color_for(mode: str) -> str:
    return MODE_COLORS.get(mode, None)


def plot_summary_bars(summary: Dict, modes: Iterable[str], output_dir: str) -> None:
    metrics = [
        ("mean_success", "Success Rate"),
        ("mean_collision", "Collision Rate"),
        ("mean_out_of_bounds", "Out of Bounds Rate"),
        ("mean_reward", "Mean Reward"),
        ("mean_min_obstacle_distance", "Min Obstacle Distance"),
        ("mean_mean_frenet_abs_d", "Mean |Frenet d|"),
    ]
    modes = [mode for mode in modes if mode in summary]
    if not modes:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    x = np.arange(len(modes))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(summary.get(mode, {}).get(key, 0.0)) for mode in modes]
        ax.bar(x, values, color=[color_for(mode) for mode in modes])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_bars.png"), dpi=160)
    plt.close(fig)


def plot_episode_curves(rows_by_mode: Dict[str, List[Dict[str, str]]], output_dir: str) -> None:
    metrics = [
        ("reward", "Episode Reward"),
        ("min_obstacle_distance", "Episode Min Obstacle Distance"),
        ("mean_frenet_abs_d", "Episode Mean |Frenet d|"),
        ("mean_action_delta_norm", "Mean Action Delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics):
        for mode, rows in rows_by_mode.items():
            if not rows:
                continue
            xs = [int(as_float(row, "episode", idx)) for idx, row in enumerate(rows)]
            ys = [as_float(row, key) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=mode, color=color_for(mode))
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "episode_curves.png"), dpi=160)
    plt.close(fig)


def plot_terminal_outcomes(rows_by_mode: Dict[str, List[Dict[str, str]]], output_dir: str) -> None:
    modes = list(rows_by_mode.keys())
    if not modes:
        return
    success = [np.mean([as_float(row, "success") for row in rows]) if rows else 0.0 for rows in rows_by_mode.values()]
    collision = [np.mean([as_float(row, "collision") for row in rows]) if rows else 0.0 for rows in rows_by_mode.values()]
    out_of_bounds = [np.mean([as_float(row, "out_of_bounds") for row in rows]) if rows else 0.0 for rows in rows_by_mode.values()]
    timeout = [np.mean([as_float(row, "timeout") for row in rows]) if rows else 0.0 for rows in rows_by_mode.values()]

    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(modes))
    for label, values, color in [
        ("success", success, "#2ca02c"),
        ("collision", collision, "#d62728"),
        ("out_of_bounds", out_of_bounds, "#ff7f0e"),
        ("timeout", timeout, "#7f7f7f"),
    ]:
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Terminal Outcome Rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "terminal_outcomes.png"), dpi=160)
    plt.close(fig)


def trace_files(result_dir: str) -> Dict[str, List[str]]:
    trace_dir = os.path.join(result_dir, "traces")
    files_by_mode = defaultdict(list)
    if not os.path.isdir(trace_dir):
        return files_by_mode
    for name in os.listdir(trace_dir):
        if not name.endswith(".csv") or "_episode_" not in name:
            continue
        mode = name.split("_episode_")[0]
        files_by_mode[mode].append(os.path.join(trace_dir, name))
    for files in files_by_mode.values():
        files.sort()
    return files_by_mode


def mean_trace_series(files: List[str], key: str, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(max_steps, dtype=float)
    counts = np.zeros(max_steps, dtype=float)
    for path in files:
        for row in read_csv_rows(path):
            step = int(as_float(row, "step", -1))
            if step < 0 or step >= max_steps:
                continue
            sums[step] += as_float(row, key)
            counts[step] += 1.0
    xs = np.arange(max_steps)
    ys = np.divide(sums, counts, out=np.full(max_steps, np.nan), where=counts > 0)
    valid = counts > 0
    return xs[valid], ys[valid]


def plot_trace_curves(result_dir: str, output_dir: str, max_steps: int) -> None:
    files_by_mode = trace_files(result_dir)
    if not files_by_mode:
        return
    metrics = [
        ("min_scan_distance", "Mean Min Laser Distance"),
        ("frenet_abs_d", "Mean |Frenet d|"),
        ("action_delta_norm", "Mean Action Delta Norm"),
        ("mppi_active", "MPPI Active Fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics):
        for mode, files in files_by_mode.items():
            xs, ys = mean_trace_series(files, key, max_steps)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, label=mode, color=color_for(mode), linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trace_mean_curves.png"), dpi=160)
    plt.close(fig)


def plot_action_source(result_dir: str, output_dir: str, max_steps: int) -> None:
    files_by_mode = trace_files(result_dir)
    modes = [mode for mode in files_by_mode if mode != "baseline"]
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 1, figsize=(13, 4 * len(modes)), squeeze=False)
    for ax, mode in zip(axes.flatten(), modes):
        files = files_by_mode[mode]
        series = {}
        for source in ("sac", "mppi", "fallback"):
            sums = np.zeros(max_steps, dtype=float)
            counts = np.zeros(max_steps, dtype=float)
            for path in files:
                for row in read_csv_rows(path):
                    step = int(as_float(row, "step", -1))
                    if step < 0 or step >= max_steps:
                        continue
                    sums[step] += 1.0 if row.get("action_source", "sac") == source else 0.0
                    counts[step] += 1.0
            series[source] = np.divide(sums, counts, out=np.zeros(max_steps), where=counts > 0)
        xs = np.arange(max_steps)
        ax.stackplot(xs, series["sac"], series["mppi"], series["fallback"], labels=["sac", "mppi", "fallback"], colors=["#9ecae1", "#fb6a4a", "#74c476"])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Action Source Fraction: {mode}")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "action_source_stack.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison curves from compare_sac_mppi.py outputs.")
    parser.add_argument("--result-dir", default="./comparison_results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=180)
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(result_dir, "plots"))
    ensure_dir(output_dir)

    rows_by_mode = load_metric_rows(result_dir)
    summary = load_summary(result_dir)
    if not rows_by_mode and not summary:
        raise FileNotFoundError(f"No comparison metrics found in {result_dir}")

    modes = list(rows_by_mode.keys()) or [mode for mode in summary.keys() if mode != "paired"]
    plot_summary_bars(summary, modes, output_dir)
    plot_episode_curves(rows_by_mode, output_dir)
    plot_terminal_outcomes(rows_by_mode, output_dir)
    plot_trace_curves(result_dir, output_dir, args.max_steps)
    plot_action_source(result_dir, output_dir, args.max_steps)
    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
