"""
feature_engineering.py
───────────────────────
공통 피처 엔지니어링 함수 모음.
모든 ML 노트북에서 import하여 사용합니다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 대여소 ID 인코딩 ───────────────────────────────────────────────────────────
def encode_station_id(
    df: pd.DataFrame,
    col: str = "rent_stn_id",
    min_count: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    대여소 ID를 정수로 레이블 인코딩합니다.
    출현 빈도가 min_count 미만인 대여소는 'RARE'로 그룹화합니다.

    Returns
    -------
    (df_with_new_col, mapping_dict)
    """
    df = df.copy()
    counts = df[col].value_counts()
    rare_stations = counts[counts < min_count].index
    df[f"{col}_grouped"] = df[col].where(~df[col].isin(rare_stations), other="RARE")

    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[f"{col}_grouped"].fillna("UNKNOWN"))

    mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    return df, mapping


def save_station_mapping(mapping: dict, name: str = "station_mapping") -> None:
    path = PROJECT_ROOT / "models" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"💾 Station mapping saved → {path}")


# ── 사용자 피처 ────────────────────────────────────────────────────────────────
def add_user_features(df: pd.DataFrame, ref_year: int = 2025) -> pd.DataFrame:
    """
    생년/성별 기반 사용자 피처를 추가합니다.
    - age : 나이 (ref_year - birth_year)
    - age_group : 연령 그룹 (youth / young_adult / adult / middle / senior)
    - gender_enc : M→1, F→0, NaN→-1
    """
    df = df.copy()

    if "birth_year" in df.columns:
        df["age"] = ref_year - df["birth_year"]
        bins   = [0, 24, 34, 49, 64, 200]
        labels = ["youth", "young_adult", "adult", "middle", "senior"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    if "gender" in df.columns:
        df["gender_enc"] = df["gender"].map({"M": 1, "F": 0}).fillna(-1).astype(int)

    if "bike_type" in df.columns:
        df["bike_type_enc"] = df["bike_type"].map(
            {"일반자전거": 0, "새싹자전거": 1, "전기자전거": 2}
        ).fillna(-1).astype(int)

    return df


# ── 이동 피처 ──────────────────────────────────────────────────────────────────
def build_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    이동 관련 파생 피처를 추가합니다.
    - speed_kmh     : 평균 속도 (km/h)
    - is_round_trip : 대여소 == 반납소 여부
    - dist_bucket   : 거리 구간 레이블
    """
    df = df.copy()

    # 평균 속도 (km/h), 비현실적 값 클리핑
    has_time = df["use_min"] > 0
    df["speed_kmh"] = np.where(
        has_time,
        (df["use_m"] / 1000) / (df["use_min"] / 60),
        np.nan,
    )
    df["speed_kmh"] = df["speed_kmh"].clip(0, 40)

    # 왕복 여부
    if "rent_stn_id" in df.columns and "rtrn_stn_id" in df.columns:
        df["is_round_trip"] = (
            df["rent_stn_id"] == df["rtrn_stn_id"]
        ).astype(int)

    # 거리 구간
    bins   = [-1, 500, 2000, 5000, 50001]
    labels = ["short", "medium", "long", "very_long"]
    df["dist_bucket"] = pd.cut(df["use_m"], bins=bins, labels=labels)

    return df


# ── 공휴일 피처 ───────────────────────────────────────────────────────────────
def add_holiday_features(df: pd.DataFrame, dt_col: str = "rent_dt") -> pd.DataFrame:
    """
    한국 공휴일 및 전날(이브) 피처를 추가합니다.
    - is_holiday     : 공휴일 당일 여부 (0/1)
    - is_holiday_eve : 공휴일 전날 여부 (0/1) — 전날 저녁 이용 급증 패턴 반영
    """
    HOLIDAYS = pd.to_datetime([
        "2025-10-03",  # 개천절
        "2025-10-09",  # 한글날
        "2025-12-25",  # 크리스마스
    ])
    df = df.copy()
    dates = df[dt_col].dt.normalize()
    df["is_holiday"] = dates.isin(HOLIDAYS).astype(int)
    df["is_holiday_eve"] = dates.isin(
        [h - pd.Timedelta(days=1) for h in HOLIDAYS]
    ).astype(int)
    return df


# ── 시간/요일 순환 인코딩 ──────────────────────────────────────────────────────
def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    hour, dow를 sin/cos로 순환 인코딩합니다.
    예) 23시와 0시의 거리 = 1시간 (숫자 인코딩에서는 23 차이가 나는 문제 해결).
    """
    df = df.copy()
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "dow" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


# ── 래그 피처 (수요 예측용) ───────────────────────────────────────────────────
def build_lag_features(
    hourly_df: pd.DataFrame,
    target_col: str = "rent_count",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    대여소별 시간별 수요 DataFrame에 래그·롤링 피처를 추가합니다.

    Parameters
    ----------
    hourly_df : columns = [stn_id, datetime_hour, rent_count, ...]
    lags      : 래그 시간 목록 (기본: [1, 2, 3, 24, 48, 168])
    rolling_windows : 롤링 평균 윈도우 (기본: [3, 6, 24])
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168, 336, 720]
    if rolling_windows is None:
        rolling_windows = [3, 6, 24]

    df = hourly_df.copy().sort_values(["stn_id", "datetime_hour"])

    grp = df.groupby("stn_id")[target_col]

    for lag in lags:
        df[f"lag_{lag}h"] = grp.shift(lag)

    for w in rolling_windows:
        df[f"roll_mean_{w}h"] = grp.shift(1).rolling(w).mean().reset_index(level=0, drop=True)

    return df


def complete_hourly_panel(
    hourly_df: pd.DataFrame,
    station_ids: list[str] | pd.Index | None = None,
    time_col: str = "datetime_hour",
    station_col: str = "stn_id",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    대여소×시간 전체 패널을 생성하고, 관측이 없는 시간대는 0으로 채웁니다.

    수요 예측에서는 래그 피처가 긴 구간(예: 720h)을 사용하므로,
    이벤트가 없던 시간대를 누락한 sparse 집계 그대로 쓰면 대부분의 대여소가
    전처리 단계에서 탈락할 수 있습니다.
    """
    if value_cols is None:
        value_cols = ["rent_count", "rtrn_count", "net_flow"]

    df = hourly_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if station_ids is None:
        station_ids = df[station_col].sort_values().unique()

    full_hours = pd.date_range(df[time_col].min(), df[time_col].max(), freq="h")
    full_index = pd.MultiIndex.from_product(
        [station_ids, full_hours],
        names=[station_col, time_col],
    )

    panel = (
        df.set_index([station_col, time_col])[value_cols]
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values([station_col, time_col])
        .reset_index(drop=True)
    )

    return panel


# ── 대여소별 통계 피처 (회귀 타겟 인코딩) ────────────────────────────────────
def add_station_stats(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.DataFrame:
    """
    학습 데이터에서 대여소별 평균 이용시간/거리를 계산하여
    타겟 df에 피처로 추가합니다 (타겟 인코딩).

    추가 피처:
    - stn_avg_time     : 대여소별 평균 이용시간
    - stn_avg_dist     : 대여소별 평균 이용거리
    - stn_hour_avg_time: 대여소×시간대별 평균 이용시간
    - stn_weekend_avg_time: 대여소×주말여부별 평균 이용시간

    Parameters
    ----------
    df_train  : 통계를 계산할 기준 데이터 (학습 데이터)
    df_target : 피처를 붙일 대상 데이터 (학습/테스트 둘 다 가능)
    """
    df_target = df_target.copy()

    global_avg_time = df_train["use_min"].mean()
    global_avg_dist = df_train["use_m"].mean()

    # 대여소별 평균
    stn_stats = df_train.groupby("rent_stn_id").agg(
        stn_avg_time=("use_min", "mean"),
        stn_avg_dist=("use_m", "mean"),
    ).reset_index()
    df_target = df_target.merge(stn_stats, on="rent_stn_id", how="left")
    df_target["stn_avg_time"] = df_target["stn_avg_time"].fillna(global_avg_time)
    df_target["stn_avg_dist"] = df_target["stn_avg_dist"].fillna(global_avg_dist)

    # 대여소×시간대 평균
    stn_hour = df_train.groupby(["rent_stn_id", "hour"])["use_min"].mean().reset_index()
    stn_hour = stn_hour.rename(columns={"use_min": "stn_hour_avg_time"})
    df_target = df_target.merge(stn_hour, on=["rent_stn_id", "hour"], how="left")
    df_target["stn_hour_avg_time"] = df_target["stn_hour_avg_time"].fillna(global_avg_time)

    # 대여소×주말 평균
    stn_weekend = df_train.groupby(["rent_stn_id", "is_weekend"])["use_min"].mean().reset_index()
    stn_weekend = stn_weekend.rename(columns={"use_min": "stn_weekend_avg_time"})
    df_target = df_target.merge(stn_weekend, on=["rent_stn_id", "is_weekend"], how="left")
    df_target["stn_weekend_avg_time"] = df_target["stn_weekend_avg_time"].fillna(global_avg_time)

    return df_target



# ── 반납 대여소 예측 피처 ─────────────────────────────────────────────────────
def build_return_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    반납 대여소 예측에 필요한 피처를 선택/구성합니다.
    전제: add_user_features(), encode_station_id() 가 이미 적용된 df.
    """
    feature_cols = [
        col for col in [
            "rent_stn_id_enc",
            "hour",
            "dow",
            "is_weekend",
            "gender_enc",
            "age_group",
            "bike_type_enc",
            "time_slot",
        ]
        if col in df.columns
    ]
    return df[feature_cols + ["rtrn_stn_id"]].dropna(subset=["rtrn_stn_id"])
