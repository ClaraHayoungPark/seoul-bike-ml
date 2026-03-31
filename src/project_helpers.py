"""
프로젝트 1/2 노트북에서 반복되는 유틸리티를 모아둔 헬퍼 모듈.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import STL

from src.evaluation import _set_korean_font, regression_report
from src.feature_engineering import (
    add_cyclical_features,
    add_holiday_features,
    build_lag_features,
    complete_hourly_panel,
)

LGBM_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
NET_FLOW_LAGS = [1, 24, 168]
CYCLICAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
BASE_DEMAND_COLS = [
    "stn_enc",
    "hour",
    "dow",
    "day",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
]
HOLIDAYS = pd.to_datetime(["2025-10-03", "2025-10-09", "2025-12-25"])
HOLIDAY_SET = set(HOLIDAYS)


def setup_korean_matplotlib() -> None:
    _set_korean_font()


def classify_stl_pattern(
    dt: pd.Timestamp,
    resid: float,
    holidays: set[pd.Timestamp] = HOLIDAY_SET,
) -> str:
    date = dt.normalize()
    if date in holidays:
        return "공휴일"
    if dt.dayofweek >= 5:
        return "주말"
    if dt.hour == 8 and resid < 0:
        return "평일 출근 급감"
    if 17 <= dt.hour <= 19 and resid > 0:
        return "평일 저녁 급증"
    return "기타 급감" if resid < 0 else "기타 급증"


def collect_stl_anomalies(
    hourly: pd.DataFrame,
    station_ids: list[str],
    period: int = 24,
    sigma: float = 3.0,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for stn_id in station_ids:
            series = (
                hourly.loc[hourly["stn_id"] == stn_id, ["datetime_hour", "rent_count"]]
                .set_index("datetime_hour")["rent_count"]
                .asfreq("h")
                .fillna(0)
            )
            if len(series) < period * 2:
                continue

            result = STL(series, period=period, robust=True).fit()
            resid_std = result.resid.std()
            if resid_std == 0:
                continue

            mask = result.resid.abs() > sigma * resid_std
            for dt, resid in result.resid[mask].items():
                records.append(
                    {
                        "stn_id": stn_id,
                        "datetime": dt,
                        "resid": resid,
                        "pattern": classify_stl_pattern(dt, resid),
                    }
                )

    stl_all = pd.DataFrame(records)
    if stl_all.empty:
        return stl_all

    stl_all["direction"] = np.where(stl_all["resid"] > 0, "급증", "급감")
    stl_all["date"] = stl_all["datetime"].dt.date
    stl_all["hour"] = stl_all["datetime"].dt.hour
    return stl_all


def build_demand_features(
    hourly: pd.DataFrame,
    top_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    top_stns = hourly.groupby("stn_id")["rent_count"].sum().nlargest(top_n).index
    df_top = hourly[hourly["stn_id"].isin(top_stns)].copy()
    df_panel = complete_hourly_panel(df_top, station_ids=top_stns)
    df_feat = build_lag_features(df_panel, target_col="rent_count")

    dt = df_feat["datetime_hour"].dt
    df_feat = df_feat.assign(
        hour=dt.hour,
        dow=dt.dayofweek,
        day=dt.day,
        month=dt.month,
    )
    df_feat["is_weekend"] = (df_feat["dow"] >= 5).astype(int)
    df_feat = add_holiday_features(df_feat, dt_col="datetime_hour")
    df_feat = add_cyclical_features(df_feat)
    df_feat["stn_enc"] = LabelEncoder().fit_transform(df_feat["stn_id"])

    lag_cols = [col for col in df_feat.columns if col.startswith(("lag_", "roll_"))]
    for lag in NET_FLOW_LAGS:
        df_feat[f"net_flow_lag_{lag}h"] = df_feat.groupby("stn_id")["net_flow"].shift(lag)
    netflow_cols = [f"net_flow_lag_{lag}h" for lag in NET_FLOW_LAGS]

    df_feat = df_feat.dropna(subset=lag_cols + netflow_cols)
    return df_top, df_panel, df_feat, lag_cols, netflow_cols


def build_ablation_steps(
    lag_cols: list[str],
    netflow_cols: list[str],
) -> list[tuple[str, list[str]]]:
    return [
        ("Step 1: 기본", BASE_DEMAND_COLS),
        ("Step 2: + lag", BASE_DEMAND_COLS + lag_cols),
        ("Step 3: + net_flow lag", BASE_DEMAND_COLS + lag_cols + netflow_cols),
        (
            "Step 4: + 순환 인코딩",
            BASE_DEMAND_COLS + lag_cols + netflow_cols + CYCLICAL_COLS,
        ),
    ]


def split_demand_dataset(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    split_ratio: float = 0.75,
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    unique_times = sorted(df_feat["datetime_hour"].unique())
    split_time = unique_times[int(len(unique_times) * split_ratio)]
    df_train = df_feat[df_feat["datetime_hour"] < split_time].copy()
    df_test = df_feat[df_feat["datetime_hour"] >= split_time].copy()
    X_train = df_train[feature_cols]
    y_train = df_train["rent_count"]
    X_test = df_test[feature_cols]
    y_test = df_test["rent_count"]
    return split_time, df_train, df_test, X_train, y_train, X_test, y_test


def print_demand_split_summary(
    split_time: pd.Timestamp,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    print(f"split 시점: {split_time}")
    print(f"학습 데이터: {len(df_train):,}행  |  테스트: {len(df_test):,}행")
    print(f"학습/테스트 대여소 수: {df_train['stn_id'].nunique()} / {df_test['stn_id'].nunique()}")
    print(f"피처 수: {len(feature_cols)}")
    print(f"시간 범위: {df_train['datetime_hour'].min()} ~ {df_test['datetime_hour'].max()}")


def evaluate_demand_baselines(
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline_mean = np.full(len(y_test), y_train.mean())
    r1 = regression_report(y_test, baseline_mean, "평균 예측")

    lag168 = df_test["lag_168h"].fillna(y_train.mean())
    r2 = regression_report(y_test, lag168, "7일전 래그")

    print(f"\n베이스라인 비교: 평균 MAE={r1['MAE']:.3f}, 7일전래그 MAE={r2['MAE']:.3f}")
    return r1, r2


def fit_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, object] | None = None,
    log_period: int = -1,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**(params or LGBM_PARAMS))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(log_period),
        ],
    )
    return model


def run_ablation_study(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: pd.DataFrame,
    y_test: pd.Series,
    steps: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    results: list[dict[str, str | float | int]] = []
    for name, feature_cols in steps:
        model = fit_lgbm_regressor(
            df_train[feature_cols],
            y_train,
            df_test[feature_cols],
            y_test,
        )
        pred = model.predict(df_test[feature_cols]).clip(0)
        mae = mean_absolute_error(y_test, pred)
        results.append({"단계": name, "MAE": round(mae, 3), "피처 수": len(feature_cols)})
        print(f"{name:<30}  피처 수={len(feature_cols):>3}  MAE={mae:.3f}")

    abl_df = pd.DataFrame(results)
    abl_df["MAE 개선"] = abl_df["MAE"].diff().map(
        lambda x: f"{x:+.3f}" if pd.notna(x) else "-"
    )
    abl_df["개선율(%)"] = (
        (abl_df["MAE"].shift(1) - abl_df["MAE"]) / abl_df["MAE"].shift(1) * 100
    ).map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    return abl_df


def summarize_model_results(results: list[dict[str, float]], index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(results, index=index).round(4)
