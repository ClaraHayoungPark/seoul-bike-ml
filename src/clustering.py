"""
clustering.py
─────────────
사용자 행동 기반 군집화 실험 유틸리티.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROJECT_ROOT, load_processed, save_processed
from src.evaluation import _set_korean_font
from src.feature_engineering import add_user_features, build_trip_features

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ClusteringArtifacts:
    model_name: str
    n_clusters: int
    metrics: dict[str, float]
    feature_cols: list[str]
    scaler: StandardScaler
    estimator: KMeans | GaussianMixture
    clustered_frame: pd.DataFrame
    profile: pd.DataFrame


def _cyclical_encode(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * values.astype(float) / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        },
        index=values.index,
    )


def build_behavior_clustering_frame(
    parquet_name: str = "rentals_clean",
    sample_n: int = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    군집화용 사용자 행동 피처를 생성합니다.

    Notes
    -----
    - `is_weekend`와 `bike_type`은 군집 생성에 사용하지 않습니다.
    - `hour`, `dow`는 주기형(sin/cos)으로 변환합니다.
    - 긴 꼬리 분포를 완화하기 위해 log1p를 사용합니다.
    """
    df = load_processed(parquet_name)
    df = add_user_features(df)
    df = build_trip_features(df)

    df["hour"] = df["rent_dt"].dt.hour
    df["dow"] = df["rent_dt"].dt.dayofweek

    df["use_min"] = df["use_min"].clip(lower=1)
    df["use_m"] = df["use_m"].clip(lower=0)

    behavior = pd.DataFrame(index=df.index)
    behavior["log_use_min"] = np.log1p(df["use_min"].clip(upper=df["use_min"].quantile(0.995)))
    behavior["log_use_m"] = np.log1p(df["use_m"].clip(upper=df["use_m"].quantile(0.995)))
    behavior["speed_kmh"] = df["speed_kmh"].clip(upper=df["speed_kmh"].quantile(0.995))
    behavior["is_round_trip"] = df["is_round_trip"].fillna(0).astype(int)

    behavior = behavior.join(_cyclical_encode(df["hour"], period=24, prefix="hour"))
    behavior = behavior.join(_cyclical_encode(df["dow"], period=7, prefix="dow"))

    profile_cols = [
        "use_min",
        "use_m",
        "speed_kmh",
        "hour",
        "dow",
        "age",
        "gender_enc",
        "user_type",
        "is_round_trip",
        "rent_dt",
    ]
    available_profile_cols = [col for col in profile_cols if col in df.columns]
    available_profile_cols = [
        col for col in available_profile_cols if col not in behavior.columns
    ]
    result = pd.concat([behavior, df[available_profile_cols]], axis=1).dropna(
        subset=behavior.columns.tolist()
    )

    if sample_n and sample_n < len(result):
        result = result.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    return result


def _cluster_balance(labels: np.ndarray) -> float:
    counts = pd.Series(labels).value_counts(normalize=True)
    return float(counts.min())


def _sample_silhouette(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    n = len(labels)
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        return float(silhouette_score(X_scaled[idx], labels[idx]))
    return float(silhouette_score(X_scaled, labels))


def _labels_from_estimator(estimator, X_scaled: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(X_scaled)
    estimator.fit(X_scaled)
    return estimator.predict(X_scaled)


def evaluate_cluster_candidates(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    cluster_range: range = range(2, 7),
    random_states: tuple[int, ...] = (21, 42),
) -> pd.DataFrame:
    """
    KMeans / GaussianMixture 후보를 평가합니다.
    """
    X = df_features[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows: list[dict[str, float | int | str]] = []
    for n_clusters in cluster_range:
        model_specs = [
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_states[0], n_init=20)),
            (
                "gmm",
                GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    random_state=random_states[0],
                    n_init=5,
                ),
            ),
        ]
        for model_name, base_estimator in model_specs:
            print(f"평가 중: model={model_name}, k={n_clusters}")
            estimator = clone(base_estimator)
            labels = _labels_from_estimator(estimator, X_scaled)

            stability_scores = []
            for seed_a, seed_b in combinations(random_states, 2):
                est_a = clone(base_estimator).set_params(random_state=seed_a)
                est_b = clone(base_estimator).set_params(random_state=seed_b)
                labels_a = _labels_from_estimator(est_a, X_scaled)
                labels_b = _labels_from_estimator(est_b, X_scaled)
                stability_scores.append(adjusted_rand_score(labels_a, labels_b))

            rows.append(
                {
                    "model": model_name,
                    "n_clusters": n_clusters,
                    "silhouette": _sample_silhouette(X_scaled, labels),
                    "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                    "cluster_balance": _cluster_balance(labels),
                    "stability_ari": float(np.mean(stability_scores)),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["silhouette", "stability_ari", "cluster_balance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def fit_best_clustering(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_clusters: int,
    random_state: int = 42,
) -> ClusteringArtifacts:
    X = df_features[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_name == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = estimator.fit_predict(X_scaled)
    elif model_name == "gmm":
        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
        )
        labels = estimator.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    clustered = df_features.copy()
    clustered["cluster"] = labels

    metrics = {
        "silhouette": _sample_silhouette(X_scaled, labels),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "cluster_balance": _cluster_balance(labels),
    }
    profile = build_cluster_profile(clustered)

    return ClusteringArtifacts(
        model_name=model_name,
        n_clusters=n_clusters,
        metrics=metrics,
        feature_cols=feature_cols,
        scaler=scaler,
        estimator=estimator,
        clustered_frame=clustered,
        profile=profile,
    )


def build_cluster_profile(df_clustered: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    profile_cols = [
        "use_min",
        "use_m",
        "speed_kmh",
        "hour",
        "dow",
        "age",
        "is_round_trip",
    ]
    available = [col for col in profile_cols if col in df_clustered.columns]
    profile = (
        df_clustered.groupby(cluster_col)[available]
        .agg(["mean", "median"])
        .round(2)
    )
    profile["size"] = df_clustered.groupby(cluster_col).size()
    profile["share"] = (profile["size"] / len(df_clustered)).round(4)
    return profile.sort_values("size", ascending=False)


def flatten_profile_columns(profile: pd.DataFrame) -> pd.DataFrame:
    flattened = profile.copy()
    flattened.columns = [
        "_".join(str(part) for part in col if str(part) != "").strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in flattened.columns
    ]
    return flattened.reset_index()


def save_clustering_artifacts(
    artifacts: ClusteringArtifacts,
    model_dir: Path | None = None,
    parquet_name: str = "user_clusters_improved",
) -> None:
    model_dir = model_dir or (PROJECT_ROOT / "models" / "clustering")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{artifacts.model_name}_k{artifacts.n_clusters}_improved.pkl"
    scaler_path = model_dir / f"{artifacts.model_name}_k{artifacts.n_clusters}_improved_scaler.pkl"
    joblib.dump(artifacts.estimator, model_path)
    joblib.dump(artifacts.scaler, scaler_path)
    save_processed(artifacts.clustered_frame, parquet_name)


def plot_model_comparison(results: pd.DataFrame, save_name: str = "cluster_model_comparison") -> Path:
    _set_korean_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("silhouette", "Silhouette"),
        ("stability_ari", "Stability (ARI)"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("cluster_balance", "Min Cluster Share"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        for model_name, grp in results.groupby("model"):
            grp = grp.sort_values("n_clusters")
            ax.plot(grp["n_clusters"], grp[metric], marker="o", label=model_name)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.grid(alpha=0.25)
        if metric == "davies_bouldin":
            ax.invert_yaxis()
        if metric in {"silhouette", "stability_ari", "cluster_balance"}:
            ax.set_ylim(bottom=0)
    axes[0, 0].legend()
    plt.tight_layout()

    path = FIGURES_DIR / f"{save_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_results_table(results: pd.DataFrame, name: str = "cluster_model_comparison") -> Path:
    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    results.to_csv(path, index=False)
    return path
