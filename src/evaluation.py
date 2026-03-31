"""
evaluation.py
─────────────
공통 평가 지표 및 시각화 함수 모음.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정 (macOS)
def _set_korean_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()


# ── 회귀 평가 ──────────────────────────────────────────────────────────────────
def regression_report(y_true, y_pred, label: str = "") -> dict:
    """MAE, RMSE, MAPE, R² 지표를 계산하여 출력하고 dict로 반환합니다."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)

    # MAPE (0으로 나누기 방지)
    mask = y_true_arr != 0
    mape = (
        np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
        if mask.any()
        else np.nan
    )

    result = {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R²": r2}
    header = f"[{label}] " if label else ""
    print(f"{header}MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.4f}")
    return result


# ── 분류 평가 ─────────────────────────────────────────────────────────────────
def topk_accuracy(y_true, y_prob, k: int = 5) -> float:
    """Top-K 정확도를 계산합니다. y_prob: (n_samples, n_classes) 행렬."""
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


def classification_report_extended(y_true, y_pred, y_prob=None, top_k: int = 5) -> dict:
    """Accuracy, Top-K Accuracy, Macro F1, Weighted F1 를 출력합니다."""
    from sklearn.metrics import accuracy_score, f1_score
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    result = {"Accuracy": acc, "Macro F1": f1_mac, "Weighted F1": f1_wt}

    if y_prob is not None:
        tk = topk_accuracy(y_true, y_prob, k=top_k)
        result[f"Top-{top_k} Accuracy"] = tk
        print(f"Accuracy={acc:.4f}  Top-{top_k}={tk:.4f}  Macro-F1={f1_mac:.4f}  Weighted-F1={f1_wt:.4f}")
    else:
        print(f"Accuracy={acc:.4f}  Macro-F1={f1_mac:.4f}  Weighted-F1={f1_wt:.4f}")
    return result


# ── 시각화: 실제 vs 예측 ───────────────────────────────────────────────────────
def plot_actual_vs_predicted(
    y_true, y_pred,
    title: str = "Actual vs Predicted",
    xlabel: str = "Index",
    ylabel: str = "Value",
    save_name: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시계열 비교
    axes[0].plot(y_true, label="Actual", alpha=0.7)
    axes[0].plot(y_pred, label="Predicted", alpha=0.7)
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    # 산점도
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    lim = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    axes[1].plot(lim, lim, "r--", lw=1)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Scatter: Actual vs Predicted")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.show()


# ── 시각화: 피처 중요도 ────────────────────────────────────────────────────────
def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_name: str | None = None,
):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names   = [feature_names[i] for i in indices]
    vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(names, vals, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.show()


# ── 시각화: 클러스터 프로파일 ─────────────────────────────────────────────────
def plot_cluster_profiles(
    df_clustered: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "cluster",
    save_name: str | None = None,
):
    """박스플롯으로 클러스터별 피처 분포를 비교합니다."""
    n_feats = len(feature_cols)
    n_cols  = 3
    n_rows  = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(feature_cols):
        df_clustered.boxplot(column=feat, by=cluster_col, ax=axes[i])
        axes[i].set_title(feat)
        axes[i].set_xlabel("Cluster")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Cluster Profiles")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.show()


# ── 시각화: 이상치 타임라인 ───────────────────────────────────────────────────
def plot_anomaly_timeline(
    df: pd.DataFrame,
    dt_col: str = "rent_dt",
    score_col: str = "anomaly_score",
    label_col: str | None = None,
    title: str = "Anomaly Timeline",
    save_name: str | None = None,
):
    fig, ax = plt.subplots(figsize=(14, 5))

    if label_col and label_col in df.columns:
        normal   = df[df[label_col] == 0]
        anomaly  = df[df[label_col] == 1]
        ax.scatter(normal[dt_col], normal[score_col], s=5, alpha=0.3, label="Normal")
        ax.scatter(anomaly[dt_col], anomaly[score_col], s=20, c="red", alpha=0.7, label="Anomaly")
        ax.legend()
    else:
        ax.scatter(df[dt_col], df[score_col], s=5, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Anomaly Score")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.show()
