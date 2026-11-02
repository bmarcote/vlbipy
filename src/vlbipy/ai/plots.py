"""Figures of the AI-mode reports: antenna gains over time (raw and calibrated), subband and hand levels.

Same conventions as :mod:`vlbipy.plotting`: hands always red (RR) / green (LL), sources in the fixed source
palette, one panel per antenna with shared axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..plotting import SOURCE_COLORS, style_axis, subplot_grid


def _gain_grid(gains: dict, antennas: list[str], title: str, path: Path) -> Path:
    """One panel per antenna: relative gain (%) over time, one colour per field, both hands."""
    rows, cols = subplot_grid(len(antennas), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.4 * rows), sharex=True, sharey=True, squeeze=False)
    t0 = min(min(g["time"]) for g in gains.values() if g["time"])
    for k, name in enumerate(antennas):
        ax = axes[k // cols][k % cols]
        for f, (field, g) in enumerate(gains.items()):
            x = np.asarray(g["x"], dtype=float)
            hours = (np.asarray(g["time"]) - t0) / 3600.0
            for h, hand in enumerate(("RR", "LL")[:x.shape[2]]):
                ax.plot(hours, 100 * (np.exp(x[:, k, h]) - 1), ls="-" if h == 0 else ":", marker=".", ms=3, lw=0.8,
                        color=SOURCE_COLORS[f % len(SOURCE_COLORS)], label=f"{field} {hand}" if k == 0 else None)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        style_axis(ax)
    for k in range(len(antennas), rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("hours from start")
    for row in axes:
        row[0].set_ylabel("gain vs median (%)")
    fig.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _levels(report: dict, title: str, path: Path) -> Path:
    """Subband offsets per hand and the RR/LL ratio, per antenna (antenna-based solutions, %)."""
    antennas = report["antennas"]
    spws = report["spws"]
    sb = np.asarray(report["subband_offsets"], dtype=float)          # (nspw, nant, nhand)
    hr = np.asarray(report["hand_ratios"], dtype=float)              # (nspw, nant)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    width = 0.8 / max(len(antennas), 1)
    for k, name in enumerate(antennas):
        offset = (k - len(antennas) / 2) * width
        color = SOURCE_COLORS[k % len(SOURCE_COLORS)]
        cycle = k // len(SOURCE_COLORS)                    # more antennas than colours: change the marker
        for h, hand in enumerate(("RR", "LL")[:sb.shape[2]]):
            marker = ("o", "s")[h] if cycle == 0 else ("^", "v")[h]
            a1.plot(np.array(spws) + offset, 100 * (np.exp(sb[:, k, h]) - 1), marker, ms=3, color=color,
                    label=name if h == 0 else None)
        a2.plot(np.array(spws) + offset, 100 * (np.exp(hr[:, k]) - 1), ("o-", "^--")[min(cycle, 1)], ms=3, lw=0.8,
                color=color, label=name)
    for ax, label in ((a1, "subband level vs antenna median (%)"), (a2, "RR / LL amplitude ratio - 1 (%)")):
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("subband")
        ax.set_ylabel(label)
        style_axis(ax)
    a2.legend(fontsize=7, ncol=3, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_prediagnostic(report: dict, plot_dir, code: str) -> list[Path]:
    """Raw antenna gains over time on every calibrator, and the raw subband/hand levels."""
    plot_dir = Path(plot_dir)
    out = []
    if report.get("gains"):
        out.append(_gain_grid(report["gains"], report["antennas"],
                              f"{code} - raw data: antenna gain over time (phase-independent amplitude)",
                              plot_dir / f"{code}.ai.raw_gains.png"))
    out.append(_levels(report, f"{code} - raw data: subband and hand levels", plot_dir / f"{code}.ai.raw_levels.png"))
    return out


def plot_postdiagnostic(report: dict, plot_dir, code: str, name: str = "postdiagnostic") -> list[Path]:
    """Calibrated antenna gains over time on every calibrator, and the residual subband/hand levels."""
    plot_dir = Path(plot_dir)
    out = []
    if report.get("gains"):
        out.append(_gain_grid(report["gains"], report["antennas"],
                              f"{code} - calibrated data: residual antenna gain over time",
                              plot_dir / f"{code}.ai.{name}_gains.png"))
    out.append(_levels(report, f"{code} - calibrated data: residual subband and hand levels",
                       plot_dir / f"{code}.ai.{name}_levels.png"))
    return out
