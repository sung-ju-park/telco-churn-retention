"""원본 CSV를 점검하고 SQLite 세 테이블로 나눠 넣은 뒤, SQL로 피처를 뽑는다."""
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config


def load_raw(path: Path = config.RAW_CSV) -> tuple[pd.DataFrame, dict]:
    """원본을 읽고 품질 점검 결과를 함께 돌려준다."""
    df = pd.read_csv(path)
    checks = {
        "rows": len(df),
        "unique_customers": int(df["customerID"].nunique()),
        "churn_rate": float((df["Churn"] == "Yes").mean()),
    }
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = total.isna()
    checks["total_charges_blank"] = int(blank.sum())
    checks["total_charges_blank_tenure"] = sorted(df.loc[blank, "tenure"].unique().tolist())
    checks["total_charges_blank_churned"] = int((df.loc[blank, "Churn"] == "Yes").sum())
    # 누적 요금과 '가입 기간 × 현재 월 요금'의 차이: 요금제 변경 흔적
    gap = total - df["tenure"] * df["MonthlyCharges"]
    ok = gap.notna()
    checks["charge_gap_abs_median"] = float(gap[ok].abs().median())
    auc = roc_auc_score(df.loc[ok, "Churn"] == "Yes", gap[ok])
    checks["charge_gap_churn_auc"] = float(max(auc, 1 - auc))  # 0.5에 가까우면 해지와 관계없음
    df["TotalCharges"] = total
    return df, checks


def build_db(df: pd.DataFrame, db_path: Path = config.DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript((config.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        customers = df[["customerID", "gender", "SeniorCitizen", "Partner", "Dependents"]]
        accounts = df[["customerID", "tenure", "Contract", "PaperlessBilling", "PaymentMethod",
                       "MonthlyCharges", "TotalCharges"]].assign(churn=(df["Churn"] == "Yes").astype(int))
        services = df[["customerID", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                       "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]]
        con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers.itertuples(index=False))
        con.executemany(
            "INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)",
            [tuple(None if pd.isna(v) else v for v in row) for row in accounts.itertuples(index=False)],
        )
        con.executemany("INSERT INTO services VALUES (?,?,?,?,?,?,?,?,?,?)", services.itertuples(index=False))


def query(sql_file: str, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    sql = (config.SQL_DIR / sql_file).read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)
