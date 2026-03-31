"""
개선된 사용자 군집화 실험 실행 엔트리포인트.
"""

from __future__ import annotations

from pathlib import Path

from src.clustering import (
    build_behavior_clustering_frame,
    evaluate_cluster_candidates,
    flatten_profile_columns,
    fit_best_clustering,
    save_clustering_artifacts,
    save_results_table,
)


def main() -> None:
    feature_cols = [
        "log_use_min",
        "log_use_m",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    df_features = build_behavior_clustering_frame()
    print(f"군집화 입력 데이터: {len(df_features):,}행")
    print(f"사용 피처: {feature_cols}")

    results = evaluate_cluster_candidates(
        df_features,
        feature_cols=feature_cols,
        cluster_range=range(3, 6),
    )
    print("\n=== 후보 비교 결과 ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = (
        results.sort_values(
            ["cluster_balance", "silhouette", "stability_ari"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )
    print(
        "\n선택된 실험:"
        f" model={best['model']}, k={int(best['n_clusters'])},"
        f" silhouette={best['silhouette']:.4f},"
        f" balance={best['cluster_balance']:.4f},"
        f" stability_ari={best['stability_ari']:.4f}"
    )

    artifacts = fit_best_clustering(
        df_features,
        feature_cols=feature_cols,
        model_name=str(best["model"]),
        n_clusters=int(best["n_clusters"]),
    )
    save_clustering_artifacts(artifacts)

    table_path = save_results_table(results)
    profile_path = Path("reports/tables/cluster_profile_improved.csv")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    flatten_profile_columns(artifacts.profile).to_csv(profile_path, index=False)

    print("\n=== 최종 모델 프로파일 ===")
    print(artifacts.profile.to_string())
    print(f"비교 표 저장: {table_path}")
    print(f"프로파일 표 저장: {profile_path}")


if __name__ == "__main__":
    main()
