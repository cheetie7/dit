"""Window-weight diagnostics; all plotted values are observed labels/losses."""
import json
from pathlib import Path

import numpy as np


def summarize_rows(rows, min_weight=None):
    advantage = np.asarray([r["advantage"] for r in rows])
    weight = np.asarray([r["weight"] for r in rows])
    mass = {}
    for row in rows:
        key = str(row["trajectory_id"])
        mass[key] = mass.get(key, 0.)+row["weight"]
    result = {f"advantage/{key}": float(value) for key, value in {
        "mean": advantage.mean(), "std": advantage.std(), "min": advantage.min(),
        "max": advantage.max(), "median": np.median(advantage)}.items()}
    result.update({f"weight/{key}": float(value) for key, value in {
        "mean": weight.mean(), "std": weight.std(), "min": weight.min(), "max": weight.max(),
        "clip_fraction": np.mean([r["clipped"] for r in rows]),
        "effective_sample_size": weight.sum()**2/np.square(weight).sum(),
        "ess_ratio": weight.sum()**2/np.square(weight).sum()/len(rows),
        "largest_trajectory_mass_fraction": max(mass.values())/weight.sum()}.items()})
    if min_weight is not None:
        result["weight/minimum_fraction"] = float(np.isclose(weight, min_weight).mean())
    result["weight/trajectory_mass_fraction"] = {key: value/weight.sum() for key, value in mass.items()}
    return result


def write_analysis(rows, directory, min_weight=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    advantage = np.asarray([r["advantage"] for r in rows])
    weight = np.asarray([r["weight"] for r in rows])
    loss = np.asarray([r["loss"] for r in rows])
    # Equal advantages stay together; do not arbitrarily assign tied labels to bins.
    edges = np.quantile(advantage, np.linspace(0, 1, 5))
    bins = np.searchsorted(edges[1:-1], advantage, side="right")
    quantiles = [{"bin": i, "count": int((bins == i).sum()),
                  "mean_loss": float(loss[bins == i].mean()) if (bins == i).any() else None}
                 for i in range(4)]
    report = {"statistics": summarize_rows(rows, min_weight), "quantile_edges": edges.tolist(),
              "loss_by_advantage_quantile": quantiles, "rows": rows}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].hist(advantage, bins=20)
    axes[0, 0].set(xlabel="Raw advantage", ylabel="Window count", title="Advantage distribution")
    axes[0, 1].hist(weight, bins=20)
    axes[0, 1].set(xlabel="Sample weight", ylabel="Window count", title="Weight distribution")
    axes[1, 0].scatter(advantage, weight, s=9, alpha=0.6)
    axes[1, 0].set(xlabel="Raw advantage", ylabel="Sample weight", title="Advantage to weight")
    axes[1, 1].bar([q["bin"] for q in quantiles], [q["mean_loss"] or 0 for q in quantiles])
    axes[1, 1].set(xlabel="Advantage quantile", ylabel="Unweighted mean FM loss", title="Loss by advantage quantile")
    axes[1, 1].set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    axes[1, 1].set_ylim(0, max(1e-6, float(loss.max()))*1.2)
    for q in quantiles:
        axes[1, 1].annotate(f"n={q['count']}", (q["bin"], q["mean_loss"] or 0), xytext=(0, 4), textcoords="offset points", ha="center")
    fig.savefig(directory / "distributions.png", dpi=140)
    plt.close(fig)
