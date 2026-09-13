"""Regenerate the paper-style DRL training curves from the released epoch records."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from scripts.verify_reference_results import RESULT_ROOT, verify

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "training_curves"
TASK_LAYOUT = {
    "sz_air_ppo": (0, 0, "a", "SZ Air", "PPO"),
    "mz_hydro_ppo": (0, 1, "b", "MZ Hydro", "PPO"),
    "mz_air_ppo": (0, 2, "c", "MZ Air", "PPO"),
    "mz_hydro_mappo": (1, 1, "e", "MZ Hydro", "MAPPO"),
    "mz_air_mappo": (1, 2, "f", "MZ Air", "MAPPO"),
}
TASK_ORDER = (
    "sz_air_ppo",
    "mz_hydro_ppo",
    "mz_hydro_mappo",
    "mz_air_ppo",
    "mz_air_mappo",
)
SEEDS = (42, 1337, 2026)
COLORS = {"PPO": "#8DA0CB", "MAPPO": "#E78AC3"}
LINE_STYLES = {42: "-", 1337: "--", 2026: ":"}


def _load_rows() -> dict[tuple[str, int], list[dict[str, str]]]:
    path = RESULT_ROOT / "training_epochs.csv"
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[(row["task"], int(row["seed"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["epoch"]))
    return dict(grouped)


def plot(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create PDF, SVG, and PNG training-curve files under *output_dir*."""

    verification = verify()
    if verification["training_rows"] != 4225:
        raise RuntimeError("the verified release does not contain 4,225 training rows")
    runs = _load_rows()

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "h3c-drl-training-v1.1.0",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 7.2))
    figure.subplots_adjust(
        left=0.10, right=0.985, bottom=0.095, top=0.875, wspace=0.34, hspace=0.36
    )

    for task, (row_index, column_index, letter, case, algorithm) in TASK_LAYOUT.items():
        axis = axes[row_index, column_index]
        for seed in SEEDS:
            rows = runs[(task, seed)]
            epochs = [int(row["epoch"]) for row in rows]
            rewards = [float(row["reward_mean"]) for row in rows]
            axis.plot(
                epochs,
                rewards,
                color=COLORS[algorithm],
                linestyle=LINE_STYLES[seed],
                linewidth=1.5,
                alpha=0.9,
            )
            axis.plot(
                epochs[-1],
                rewards[-1],
                marker="x",
                color="black",
                linestyle="none",
                markersize=7,
                markeredgewidth=1.5,
            )
        maximum_epoch = max(int(runs[(task, seed)][-1]["epoch"]) for seed in SEEDS)
        axis.set_title(f"{letter}  {case} · {algorithm}", loc="left", pad=12)
        axis.set_ylabel("Cumulative reward")
        axis.set_xlim(0, maximum_epoch * 1.04)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.margins(y=0.13)
        axis.grid(linestyle=":", alpha=0.3)

    duration_axis = axes[1, 0]
    for task_index, task in enumerate(TASK_ORDER):
        algorithm = TASK_LAYOUT[task][4]
        for seed_index, seed in enumerate(SEEDS):
            stop_epoch = int(runs[(task, seed)][-1]["epoch"])
            position = task_index + (seed_index - 1) * 0.22
            duration_axis.plot(
                [0, stop_epoch],
                [position, position],
                color=COLORS[algorithm],
                linestyle=LINE_STYLES[seed],
                linewidth=1.5,
            )
            duration_axis.plot(
                stop_epoch,
                position,
                marker="x",
                color="black",
                linestyle="none",
                markersize=6,
            )
    duration_axis.set_title("d  Training duration", loc="left", pad=12)
    duration_axis.set_yticks(
        range(5),
        [
            "SZ Air · PPO",
            "MZ Hydro · PPO",
            "MZ Hydro · MAPPO",
            "MZ Air · PPO",
            "MZ Air · MAPPO",
        ],
    )
    duration_axis.set_ylim(4.6, -0.6)
    maximum_stop = max(int(rows[-1]["epoch"]) for rows in runs.values())
    duration_axis.set_xlim(0, maximum_stop * 1.04)
    duration_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    duration_axis.grid(axis="x", linestyle=":", alpha=0.3)

    for axis in axes[1]:
        axis.set_xlabel("Epoch")
    handles = [
        Line2D(
            [],
            [],
            color="#555555",
            linestyle=LINE_STYLES[seed],
            linewidth=1.5,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    handles.append(Line2D([], [], color="black", marker="x", linestyle="none", label="Early stop"))
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.54, 0.985),
        handlelength=3.2,
        columnspacing=2.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "DRL_epoch_cumulative_reward"
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    figure.savefig(
        pdf_path,
        metadata={"Title": "DRL epoch cumulative reward", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(svg_path, metadata={"Title": "DRL epoch cumulative reward", "Date": None})
    figure.savefig(png_path, dpi=300, metadata={"Title": "DRL epoch cumulative reward"})
    plt.close(figure)
    return pdf_path, svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    outputs = plot(args.output_dir.resolve())
    print("Generated training curves from 4,225 released epoch rows:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
