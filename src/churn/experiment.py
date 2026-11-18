"""'혜택을 받은 해지 예정자 중 몇 %가 남는가'를 실제로 재는 비교 집단 실험 설계.

표본 크기는 두 가지로 따로 본다.
- 효과가 있는지(남는 비율 > 0) 확인하는 데 필요한 인원
- 남는 비율을 원하는 정밀도(예: ±5%p)로 추정하는 데 필요한 인원 — 보통 훨씬 많다
통화 연결률은 데이터에 없어 60%로 가정했다 (연결되지 않은 고객은 처치를 받지 못한다).

연락 대상으로 뽑힌 고객을 무작위로 나눠 일부는 혜택을 주고(처치군) 일부는 주지 않는다(대조군).
두 집단의 해지율 차이로 남는 비율을 잰다. 대조군은 혜택을 못 받으니 그만큼 놓치는 매출이 비용이다.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def n_per_group(p_control: float, p_treat: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """두 비율 차이 검정(양측)에 필요한 집단별 인원. 정규 근사."""
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p_bar = (p_control + p_treat) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
           z_b * np.sqrt(p_control * (1 - p_control) + p_treat * (1 - p_treat))) ** 2
    return int(np.ceil(num / (p_control - p_treat) ** 2))


def save_rate_ci_halfwidth(base_churn: float, save_rate: float, n_treat: int, n_control: int) -> float:
    """남는 비율 추정값(1 − 처치군 해지율 / 대조군 해지율)의 95% 신뢰구간 반폭. 델타 방법 근사."""
    p_c, p_t = base_churn, base_churn * (1 - save_rate)
    ratio = p_t / p_c
    var = ratio ** 2 * ((1 - p_t) / (n_treat * p_t) + (1 - p_c) / (n_control * p_c))
    return float(1.96 * np.sqrt(var))


def n_for_precision(base_churn: float, save_rate: float, half_width: float, control_share: float = 0.5) -> int:
    """남는 비율을 ±half_width 안으로 추정하는 데 필요한 '통화가 연결된' 대상자 수."""
    n = 100
    while save_rate_ci_halfwidth(base_churn, save_rate, int(n * (1 - control_share)), int(n * control_share)) > half_width:
        n = int(n * 1.05) + 1
    return n


def design_table(base_churn: float, mean_risk_value: float, offer_cost: float, save_rates=(0.10, 0.20, 0.30),
                 control_shares=(0.5, 0.2), connect_rate: float = 0.6) -> pd.DataFrame:
    """base_churn: 연락 대상자의 해지율(혜택 없을 때),
    mean_risk_value: 대상자의 평균 '해지 확률 × 해지하면 잃는 매출'.
    대조군 비율을 줄이면 대조군 비용은 줄지만 필요한 전체 대상자는 늘어난다."""
    rows = []
    for s in save_rates:
        p_treat = base_churn * (1 - s)
        n_equal = n_per_group(base_churn, p_treat)
        for share in control_shares:
            # 집단 크기가 다를 때: 필요한 총원 ≈ 같은 크기일 때 총원 × 1 / (4 r (1 - r))
            total = int(np.ceil(2 * n_equal / (4 * share * (1 - share))))
            control = int(np.ceil(total * share))
            # 대조군 비용(기대값) = 대조군 수 × (남는 비율 × 해지 확률 × 잃는 매출 − 아낀 혜택 비용)
            hold_cost = control * (s * mean_risk_value - offer_cost)
            ci = save_rate_ci_halfwidth(base_churn, s, total - control, control)
            rows.append({"save_rate": s, "treat_churn": p_treat, "control_share": share,
                         "total_needed": total, "control_n": control, "control_cost": hold_cost,
                         "save_rate_ci_halfwidth": ci,
                         # 처치군은 통화가 연결돼야 혜택을 받는다. 연결률을 감안해 연락 시도해야 할 인원
                         "total_to_call": int(np.ceil(total / connect_rate)), "connect_rate": connect_rate})
    return pd.DataFrame(rows)
