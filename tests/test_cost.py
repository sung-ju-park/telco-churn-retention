import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import cost, monitoring  # noqa: E402


def test_cost_rule_contacts_only_positive_expected_profit():
    # 월 요금 50, 12개월, 남는 비율 0.2 → 지킬 수 있는 기대 매출 = p × 120. 혜택 40이면 p > 1/3 일 때만 연락
    p = np.array([0.30, 0.34, 0.90])
    contact = cost.cost_rule(p, np.full(3, 50.0), offer_cost=40, save_rate=0.2, lost_months=12)
    assert contact.tolist() == [False, True, True]


def test_high_value_customer_contacted_at_lower_probability():
    # 같은 확률이라도 요금이 높은 고객은 연락 대상이 된다
    p = np.array([0.2, 0.2])
    contact = cost.cost_rule(p, np.array([30.0, 120.0]), offer_cost=40, save_rate=0.2, lost_months=12)
    assert contact.tolist() == [False, True]


def test_realized_profit_counts_offer_cost_for_everyone_contacted():
    y = np.array([1, 0, 0])
    r = cost.realized_profit(y, np.array([True, True, False]), np.array([100.0, 100.0, 100.0]),
                             offer_cost=40, save_rate=0.2, lost_months=12)
    assert r["saved_revenue"] == 0.2 * 100 * 12
    assert r["offer_cost"] == 80
    assert r["profit"] == 240 - 80


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert monitoring.psi(a, a) < 1e-9
    assert monitoring.psi(a, a + 1.0) > 0.25
