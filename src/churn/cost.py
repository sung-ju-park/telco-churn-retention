"""해지 방어 비용 기준으로 연락할 고객을 정한다.

고객 i에게 방어 혜택을 주면
  기대 이익 = p_i × SAVE_RATE × (월 요금_i × LOST_MONTHS) − OFFER_COST
이 값이 0보다 크면 연락한다. 즉 고객마다 기준 확률이 다르다.
  기준 확률_i = OFFER_COST / (SAVE_RATE × 월 요금_i × LOST_MONTHS)
요금이 높은 고객은 낮은 확률에서도 연락할 가치가 있다.

평가(실제 결과 기준) 지킨 매출 − 혜택 비용 = Σ_연락한 해지 고객 (SAVE_RATE × 잃을 매출) − 연락 수 × OFFER_COST

매출 기준이고 원가(망 도매대가 등)는 빼지 않았다. 마진만 남기려면 LOST_MONTHS에 마진율을 곱한 것과 같다.
대상을 고를 때와 평가할 때 같은 가정을 쓰므로, '가정이 맞다면'의 비교다. 가정 자체의 불확실성은 sensitivity()로 본다.
"""
import numpy as np
import pandas as pd

from . import config


def customer_value(monthly: np.ndarray, lost_months: float = config.LOST_MONTHS) -> np.ndarray:
    return np.asarray(monthly, dtype=float) * lost_months


def realized_profit(y, contact, monthly, offer_cost=config.OFFER_COST, save_rate=config.SAVE_RATE,
                    lost_months=config.LOST_MONTHS) -> dict:
    y, contact = np.asarray(y).astype(bool), np.asarray(contact).astype(bool)
    value = customer_value(monthly, lost_months)
    saved_revenue = save_rate * value[contact & y].sum()
    cost = offer_cost * contact.sum()
    return {
        "contacted": int(contact.sum()),
        "contact_rate": float(contact.mean()),
        "churners_reached": int((contact & y).sum()),
        "recall": float((contact & y).sum() / max(y.sum(), 1)),
        "precision": float((contact & y).sum() / max(contact.sum(), 1)),
        "saved_revenue": float(saved_revenue),
        "offer_cost": float(cost),
        "profit": float(saved_revenue - cost),
    }


def cost_rule(p, monthly, offer_cost=config.OFFER_COST, save_rate=config.SAVE_RATE,
              lost_months=config.LOST_MONTHS) -> np.ndarray:
    """기대 이익이 0보다 큰 고객만 연락한다."""
    return np.asarray(p) * save_rate * customer_value(monthly, lost_months) > offer_cost


def top_share_contact(p, share: float) -> np.ndarray:
    """해지 확률 상위 share 비율에게 연락."""
    p = np.asarray(p)
    contact = np.zeros(len(p), bool)
    contact[np.argsort(-p)[: int(round(len(p) * share))]] = True
    return contact


def best_share(y_tr, p_tr, monthly_tr, offer_cost=config.OFFER_COST, save_rate=config.SAVE_RATE,
               lost_months=config.LOST_MONTHS) -> float:
    """학습 데이터의 교차검증 예측으로 '확률 상위 몇 %에 연락할지'를 고른다(평가 데이터는 보지 않음)."""
    shares = np.round(np.arange(0, 1.0001, 0.01), 2)
    profits = [realized_profit(y_tr, top_share_contact(p_tr, s), monthly_tr, offer_cost, save_rate, lost_months)["profit"]
               for s in shares]
    return float(shares[int(np.argmax(profits))])


def gain_table(y, p, monthly, shares=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """해지 확률 상위 N%에 연락하면 해지 고객을 몇 명 잡는지. 사업 담당자용 표."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    rows = []
    for s in shares:
        k = int(round(len(y) * s))
        contact = np.zeros(len(y), bool)
        contact[order[:k]] = True
        r = realized_profit(y, contact, monthly)
        rows.append({"top_share": s, "contacted": k, "churners_reached": r["churners_reached"],
                     "recall": r["recall"], "precision": r["precision"],
                     "lift": r["precision"] / y.mean(), "profit": r["profit"]})
    return pd.DataFrame(rows)


def sensitivity(y, monthly, p_main, p_weighted, y_tr, p_tr, monthly_tr,
                save_rates=(0.05, 0.1, 0.2, 0.3, 0.5), offer_costs=(10, 20, 40, 80),
                lost_months=config.LOST_MONTHS) -> pd.DataFrame:
    """가정이 바뀌면 연락 인원과 결과가 어떻게 달라지는지.
    결과에는 '남는 비율 × 잃는 매출 기간'의 곱만 들어가므로 기간은 고정하고 남는 비율만 바꾼다."""
    rows = []
    n = len(y)
    for sr in save_rates:
        for oc in offer_costs:
            args = (oc, sr, lost_months)
            rule = realized_profit(y, cost_rule(p_main, monthly, *args), monthly, *args)
            share = best_share(y_tr, p_tr, monthly_tr, *args)
            topk = realized_profit(y, top_share_contact(p_main, share), monthly, *args)["profit"]
            all_ = realized_profit(y, np.ones(n, bool), monthly, *args)["profit"]
            weighted = realized_profit(y, cost_rule(p_weighted, monthly, *args), monthly, *args)["profit"]
            rows.append({"save_rate": sr, "offer_cost": oc, "contact_rate": rule["contact_rate"],
                         "recall": rule["recall"], "profit": rule["profit"],
                         "profit_topk": topk, "topk_share": share,
                         "profit_contact_all": all_, "profit_trivial_best": max(all_, 0.0),
                         "profit_weighted_probs": weighted})
    return pd.DataFrame(rows)


def bootstrap_profit_diff(y, contact_a, contact_b, monthly, n: int = config.N_BOOT) -> dict:
    """평가 고객을 다시 뽑아 두 정책의 결과 차이(a - b)의 95% 신뢰구간을 구한다.
    평가 표본의 변동만 반영하고, 비용 가정이나 학습의 변동은 반영하지 않는다."""
    rng = np.random.default_rng(config.SEED)
    y, a, b = np.asarray(y).astype(bool), np.asarray(contact_a), np.asarray(contact_b)
    value = config.SAVE_RATE * customer_value(monthly)
    gain_a = np.where(a & y, value, 0.0) - np.where(a, config.OFFER_COST, 0.0)
    gain_b = np.where(b & y, value, 0.0) - np.where(b, config.OFFER_COST, 0.0)
    d = gain_a - gain_b
    boots = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.sum()), "ci_low": float(lo), "ci_high": float(hi)}
