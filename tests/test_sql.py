import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn import config, db  # noqa: E402


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    if not config.RAW_CSV.exists():
        pytest.skip("원본 데이터가 없습니다 (python scripts/download_data.py)")
    path = tmp_path_factory.mktemp("db") / "churn.db"
    raw, _ = db.load_raw()
    db.build_db(raw, path)
    return db.query("features.sql", path)


def test_one_row_per_customer(features):
    assert len(features) == 7043
    assert features["customer_id"].is_unique


def test_blank_total_charges_filled_with_zero_only_for_new_customers(features):
    zero = features[features["total_charges"] == 0]
    assert len(zero) == 11
    assert (zero["tenure_months"] == 0).all()


def test_n_addons_range(features):
    assert features["n_addons"].between(0, 6).all()
