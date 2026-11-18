from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "churn.db"
SQL_DIR = ROOT / "sql"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
N_BOOT = 2000

NUMERIC = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges", "auto_pay", "n_addons"]
CATEGORICAL = [
    "gender", "partner", "dependents", "contract", "paperless_billing", "payment_method",
    "phone_service", "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
]
TARGET = "churn"

# ── 해지 방어 비용 가정 (데이터에 없는 값이라 모두 가정이다) ─────────────────
# 기본값은 극단을 피한 중간값으로 두고, 결론이 가정에 따라 어떻게 바뀌는지는 민감도 표로 함께 본다.
# 해지하면 잃는 매출: 월 요금 × LOST_MONTHS 개월
LOST_MONTHS = 12
# 방어 혜택(할인 쿠폰 등) 1건 비용. 연락한 고객은 해지 여부와 관계없이 혜택을 받는다고 본다.
OFFER_COST = 40.0
# 혜택을 받은 해지 예정 고객 중 실제로 남는 비율
SAVE_RATE = 0.20
