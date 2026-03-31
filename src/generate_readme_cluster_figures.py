"""
README용 사용자 군집화 시각화 생성 엔트리포인트.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import _set_korean_font

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READ_ME_FIG_DIR = PROJECT_ROOT / "reports" / "figures_readme"
READ_ME_FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "짧은 저녁형": "#2F6DB3",
    "짧은 오전형": "#3FA35C",
    "장거리형": "#E3A018",
}
ORDER = ["짧은 저녁형", "짧은 오전형", "장거리형"]


def infer_cluster_names(df: pd.DataFrame) -> dict[int, str]:
    profile = df.groupby("cluster")[["use_min", "hour"]].mean()
    long_cluster = int(profile["use_min"].idxmax())

    remaining = profile.drop(index=long_cluster).sort_values("hour")
    morning_cluster = int(remaining.index[0])
    evening_cluster = int(remaining.index[1])

    return {
        evening_cluster: "짧은 저녁형",
        morning_cluster: "짧은 오전형",
        long_cluster: "장거리형",
    }


def plot_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    sampled_frames = []
    for name in ORDER:
        grp = df[df["cluster_name"] == name]
        sampled_frames.append(grp.sample(n=min(len(grp), 6000), random_state=42))
    sample = pd.concat(sampled_frames, ignore_index=True)

    for name in ORDER:
        grp = sample[sample["cluster_name"] == name]
        ax.scatter(
            grp["use_min"].clip(0, 120),
            grp["use_m"].clip(0, 8000),
            s=10,
            alpha=0.20,
            c=COLORS[name],
            label=name,
            edgecolors="none",
        )
        cx = grp["use_min"].clip(0, 120).mean()
        cy = grp["use_m"].clip(0, 8000).mean()
        ax.scatter(cx, cy, marker="*", s=260, c=COLORS[name], edgecolors="black", linewidths=1)

    ax.set_title("이용시간 vs 이동거리 - 개선 군집 산포도", fontsize=15)
    ax.set_xlabel("이용시간 (분)")
    ax.set_ylabel("이동거리 (m)")
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 8200)
    ax.grid(alpha=0.15)
    ax.legend(title="군집")
    plt.tight_layout()
    fig.savefig(READ_ME_FIG_DIR / "cluster_scatter_2d.png", dpi=150)
    plt.close(fig)


def plot_hour_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)

    for ax, name in zip(axes, ORDER):
        grp = df[df["cluster_name"] == name]
        pct = grp["hour"].value_counts(normalize=True).sort_index().reindex(range(24), fill_value=0) * 100
        ax.bar(pct.index, pct.values, color=COLORS[name], alpha=0.9, width=0.8)
        ax.set_title(f"{name}\n(n={len(grp):,})")
        ax.set_xlabel("시간")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.grid(axis="y", alpha=0.15)

    axes[0].set_ylabel("비율 (%)")
    fig.suptitle("군집별 24시간 이용 분포", fontsize=15)
    plt.tight_layout()
    fig.savefig(READ_ME_FIG_DIR / "cluster_hour_dist.png", dpi=150)
    plt.close(fig)


def plot_dow_distribution(df: pd.DataFrame) -> None:
    labels = ["월", "화", "수", "목", "금", "토", "일"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)

    for ax, name in zip(axes, ORDER):
        grp = df[df["cluster_name"] == name]
        pct = grp["dow"].value_counts(normalize=True).sort_index().reindex(range(7), fill_value=0) * 100
        colors = [COLORS[name]] * 5 + ["#F28E8E", "#F28E8E"]
        ax.bar(labels, pct.values, color=colors, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("요일")
        ax.grid(axis="y", alpha=0.15)

    axes[0].set_ylabel("비율 (%)")
    fig.suptitle("군집별 요일별 이용 분포 (주말 강조)", fontsize=15)
    plt.tight_layout()
    fig.savefig(READ_ME_FIG_DIR / "cluster_dow_dist.png", dpi=150)
    plt.close(fig)


def main() -> None:
    _set_korean_font()
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "user_clusters_improved.parquet")
    name_map = infer_cluster_names(df)
    df["cluster_name"] = df["cluster"].map(name_map)

    plot_scatter(df)
    plot_hour_distribution(df)
    plot_dow_distribution(df)

    print("Saved README figures:")
    for name in ["cluster_scatter_2d.png", "cluster_hour_dist.png", "cluster_dow_dist.png"]:
        print(READ_ME_FIG_DIR / name)


if __name__ == "__main__":
    main()
