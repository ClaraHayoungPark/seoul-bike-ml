"""
data_loader.py
──────────────
서울시 공공자전거(따릉이) 대여 데이터를 읽고 전처리하는 공통 모듈.
모든 노트북에서 import하여 사용합니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 월별 파일 목록 (오래된 순)
RAW_CSV_FILES = sorted(RAW_DIR.glob("seoul_bike_rentals_*.csv"))

# ── 컬럼 한글명 → 영문 별칭 매핑 ───────────────────────────────────────────────
COL_MAP = {
    "자전거번호":       "bike_id",
    "대여일시":         "rent_dt",
    "대여 대여소번호":  "rent_stn_no",
    "대여 대여소명":    "rent_stn_name",
    "대여거치대":       "rent_dock",
    "반납일시":         "rtrn_dt",
    "반납대여소번호":   "rtrn_stn_no",
    "반납대여소명":     "rtrn_stn_name",
    "반납거치대":       "rtrn_dock",
    "이용시간(분)":     "use_min",
    "이용거리(M)":      "use_m",
    "생년":             "birth_year",
    "성별":             "gender",
    "이용자종류":       "user_type",
    "대여대여소ID":     "rent_stn_id",
    "반납대여소ID":     "rtrn_stn_id",
    "자전거구분":       "bike_type",
}

DTYPE_SPEC = {
    "자전거번호":       "str",
    "대여 대여소번호":  "str",
    "대여 대여소명":    "str",
    "반납대여소번호":   "str",
    "반납대여소명":     "str",
    "대여대여소ID":     "str",
    "반납대여소ID":     "str",
    "성별":             "str",
    "이용자종류":       "str",
    "자전거구분":       "str",
}


# ── 원본 로딩 ──────────────────────────────────────────────────────────────────
def _read_one(filepath: Path, rename: bool = True) -> pd.DataFrame:
    """단일 CSV 파일을 읽어 DataFrame으로 반환합니다."""
    print(f"📂 Loading: {filepath}")
    df = pd.read_csv(
        filepath,
        encoding="cp949",
        encoding_errors="replace",
        na_values=["\\N", "NULL", "null", ""],
        dtype=DTYPE_SPEC,
        low_memory=False,
    )
    if rename:
        df = df.rename(columns=COL_MAP)
    print(f"   ↳ shape: {df.shape}")
    return df


def load_raw(
    path: str | Path | None = None,
    sample_n: int | None = None,
    rename: bool = True,
) -> pd.DataFrame:
    """
    원본 CSV를 읽어 DataFrame으로 반환합니다.
    path 미지정 시 data/raw/seoul_bike_rentals_*.csv 파일을 모두 합칩니다.

    Parameters
    ----------
    path : 특정 파일 경로. None이면 RAW_CSV_FILES 전체를 concat.
    sample_n : 랜덤 샘플 행 수 (None이면 전체 로드).
    rename : True이면 컬럼명을 영문으로 변환.

    Returns
    -------
    pd.DataFrame
    """
    if path:
        df = _read_one(Path(path), rename=rename)
    else:
        if not RAW_CSV_FILES:
            raise FileNotFoundError(f"data/raw/ 에 seoul_bike_rentals_*.csv 파일이 없습니다.")
        dfs = [_read_one(f, rename=rename) for f in RAW_CSV_FILES]
        df = pd.concat(dfs, ignore_index=True)
        print(f"   ↳ 합산 shape: {df.shape}")

    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"   ↳ sampled {sample_n:,} rows")

    return df


# ── 날짜 파싱 ──────────────────────────────────────────────────────────────────
def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """대여/반납 일시를 datetime으로 변환하고 이용시간 검증 컬럼을 추가합니다."""
    df = df.copy()
    for col in ["rent_dt", "rtrn_dt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    if "rent_dt" in df.columns and "rtrn_dt" in df.columns:
        df["use_min_calc"] = (
            (df["rtrn_dt"] - df["rent_dt"]).dt.total_seconds() / 60
        ).round(1)

    return df


# ── 데이터 정제 ────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 처리, 이상값 클리핑, 타입 정규화 등 기본 정제를 수행합니다.
    """
    df = df.copy()

    # 대여일시 없으면 제거
    if "rent_dt" in df.columns:
        df = df.dropna(subset=["rent_dt"])

    # 이용시간: 0 초과 480분 이하 (8시간 기준)
    if "use_min" in df.columns:
        df["use_min"] = pd.to_numeric(df["use_min"], errors="coerce")
        df["use_min_outlier"] = (df["use_min"] < 0) | (df["use_min"] > 480)
        df["use_min"] = df["use_min"].clip(lower=0, upper=480)

    # 이용거리: 0 이상 50000m 이하
    if "use_m" in df.columns:
        df["use_m"] = pd.to_numeric(df["use_m"], errors="coerce")
        df["use_m_outlier"] = (df["use_m"] < 0) | (df["use_m"] > 50000)
        df["use_m"] = df["use_m"].clip(lower=0, upper=50000)

    # 성별 정규화: M/F 이외는 NaN
    if "gender" in df.columns:
        df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]), other=np.nan)

    # 이용자종류 결측 → 'unknown'
    if "user_type" in df.columns:
        df["user_type"] = df["user_type"].fillna("unknown")

    # 생년 정규화
    if "birth_year" in df.columns:
        df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
        df["birth_year"] = df["birth_year"].where(
            df["birth_year"].between(1930, 2015), other=np.nan
        )

    # 대여소 이름 공백 제거
    for col in ["rent_stn_name", "rtrn_stn_name"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 자전거구분 정규화 (비정상 값 제거)
    if "bike_type" in df.columns:
        valid_types = ["일반자전거", "새싹자전거", "전기자전거"]
        df["bike_type"] = df["bike_type"].where(
            df["bike_type"].isin(valid_types), other=np.nan
        )

    print(f"✅ clean() 완료 → shape: {df.shape}")
    return df


# ── 시간 피처 추가 ─────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """대여일시 기반 시간 관련 피처를 추가합니다."""
    df = df.copy()
    dt = df["rent_dt"]

    df["hour"]         = dt.dt.hour
    df["dow"]          = dt.dt.dayofweek       # 0=월요일, 6=일요일
    df["day"]          = dt.dt.day
    df["week"]         = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)

    # 시간대 구분
    conditions = [
        dt.dt.hour.between(0, 5),
        dt.dt.hour.between(6, 8),
        dt.dt.hour.between(9, 11),
        dt.dt.hour.between(12, 16),
        dt.dt.hour.between(17, 20),
        dt.dt.hour.between(21, 23),
    ]
    labels = ["새벽", "출근", "오전", "오후", "퇴근", "저녁"]
    df["time_slot"] = np.select(conditions, labels, default="저녁")

    return df


# ── 대여소별 시간별 집계 ───────────────────────────────────────────────────────
def build_station_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    대여소별 시간별 대여/반납 건수를 집계합니다.

    Returns
    -------
    DataFrame with columns:
        stn_id, datetime_hour, rent_count, rtrn_count, net_flow
    """
    df = df.copy()
    df["datetime_hour"] = df["rent_dt"].dt.floor("h")

    rent = (
        df.groupby(["rent_stn_id", "datetime_hour"])
        .size()
        .reset_index(name="rent_count")
        .rename(columns={"rent_stn_id": "stn_id"})
    )

    rtrn = (
        df.dropna(subset=["rtrn_stn_id"])
        .groupby(["rtrn_stn_id", "datetime_hour"])
        .size()
        .reset_index(name="rtrn_count")
        .rename(columns={"rtrn_stn_id": "stn_id"})
    )

    hourly = rent.merge(rtrn, on=["stn_id", "datetime_hour"], how="outer").fillna(0)
    hourly["net_flow"] = hourly["rent_count"] - hourly["rtrn_count"]
    hourly = hourly.sort_values(["stn_id", "datetime_hour"]).reset_index(drop=True)

    print(f"✅ build_station_hourly() → shape: {hourly.shape}")
    return hourly


# ── Parquet 저장/로드 ─────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, name: str) -> None:
    """DataFrame을 data/processed/{name}.parquet으로 저장합니다."""
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"💾 Saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_processed(name: str) -> pd.DataFrame:
    """data/processed/{name}.parquet을 읽어 반환합니다."""
    path = PROCESSED_DIR / f"{name}.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"📂 Loaded {name}.parquet → shape: {df.shape}")
    return df
