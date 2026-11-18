"""상담원에게 넘길 연락 대상 리스트.

예측 점수와 고객별 연락 사유를 SQLite scores 테이블에 넣고, sql/call_list.sql 로 대상을 뽑는다.

연락 사유는 로지스틱 회귀에서 그 고객의 해지 확률을 평균 고객보다 끌어올린 항목(기여도) 상위 2개다.
- 성별·나이·가족 같은 인적 사항은 상담에서 다룰 수 없어 사유에서 뺀다.
- 가입 기간·누적 요금·월 요금은 서로 강하게 얽혀 계수 하나만 떼어 읽으면 방향이 뒤집히므로
  세 항목의 기여도를 합쳐 '가입 기간·요금' 하나로 본다.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, cost

# 인적 사항, 그리고 상담원이 고객에게 해지 이유로 꺼내기 어려운 단순 가입 항목은 사유에서 뺀다
EXCLUDE = {"gender", "senior_citizen", "partner", "dependents",
           "multiple_lines", "phone_service", "streaming_tv", "streaming_movies"}
GROUPS = {"tenure_months": "tenure_charges", "total_charges": "tenure_charges", "monthly_charges": "tenure_charges"}

PAYMENT = {"Electronic check": "전자 수표", "Mailed check": "우편 수표",
           "Bank transfer (automatic)": "계좌 자동이체", "Credit card (automatic)": "카드 자동결제"}
CONTRACT = {"Month-to-month": "월 단위 계약", "One year": "1년 약정", "Two year": "2년 약정"}
INTERNET = {"Fiber optic": "광섬유 인터넷", "DSL": "DSL 인터넷", "No": "인터넷 미가입"}
SERVICE = {"online_security": "보안 서비스", "online_backup": "백업 서비스", "device_protection": "기기 보호",
           "tech_support": "기술 지원", "streaming_tv": "스트리밍 TV", "streaming_movies": "스트리밍 영화",
           "multiple_lines": "추가 회선", "phone_service": "전화 서비스"}

# 상담 포인트는 예시다 (실제 혜택 종류는 회사 정책에 따라 바뀐다).
# 사유 순위와 관계없이, 고객이 가진 조건 중 상담에서 바로 바꿀 수 있는 것(월 단위 계약, 수동 결제,
# 보안·기술지원 미가입)을 먼저 보고, 없으면 상위 사유로 정한다.
TALK = {"contract": "약정 전환 혜택 안내", "tenure_charges": "가입 초기 불편·요금 만족도 확인",
        "internet_service": "인터넷 품질·요금 만족도 확인", "payment_method": "자동이체 전환 안내",
        "paperless_billing": "청구서 확인 여부 점검", "auto_pay": "자동이체 전환 안내",
        "n_addons": "부가 서비스 체험 안내"}


def _reason_text(var: str, row: pd.Series) -> str:
    if var == "tenure_charges":
        return f"가입 {int(row['tenure_months'])}개월·월 ${row['monthly_charges']:.0f}"
    if var == "contract":
        return CONTRACT[row["contract"]]
    if var == "internet_service":
        return INTERNET[row["internet_service"]]
    if var == "payment_method":
        return f"{PAYMENT[row['payment_method']]} 결제"
    if var == "paperless_billing":
        return "종이 청구서 없음" if row["paperless_billing"] == "Yes" else "종이 청구서 사용"
    if var == "auto_pay":
        return "자동이체 아님" if row["auto_pay"] == 0 else "자동이체"
    if var == "n_addons":
        return f"부가 서비스 {int(row['n_addons'])}개"
    if var in SERVICE:
        v = row[var]
        return f"{SERVICE[var]} 미가입" if v == "No" else (f"{SERVICE[var]} 가입" if v == "Yes" else f"{SERVICE[var]} 해당 없음")
    return var


def _talk_point(row: pd.Series, top_var: str | None) -> str:
    if row["contract"] == "Month-to-month":
        return "약정 전환 혜택 안내"
    if row["auto_pay"] == 0:
        return "자동이체 전환 안내"
    if row["internet_service"] != "No" and (row["tech_support"] == "No" or row["online_security"] == "No"):
        return "기술 지원·보안 서비스 체험 안내"
    return TALK.get(top_var, "이용 만족도 확인") if top_var else "이용 만족도 확인"


def reason_codes(model, X_ref: pd.DataFrame, X: pd.DataFrame, top: int = 2) -> pd.DataFrame:
    """고객별로 해지 확률을 평균 고객보다 끌어올린 항목 상위 top개."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = prep.get_feature_names_out()
    z = prep.transform(X) - prep.transform(X_ref).mean(axis=0)
    contrib = z * clf.coef_[0]
    # 변환된 열 이름 → 원래 변수 → 묶음
    var_of = []
    for n in names:
        raw = n.split("__", 1)[1]
        base = next(c for c in config.NUMERIC + config.CATEGORICAL if raw == c or raw.startswith(c + "_"))
        var_of.append(GROUPS.get(base, base))
    by_var = pd.DataFrame(contrib, columns=var_of, index=X.index).T.groupby(level=0).sum().T
    by_var = by_var.drop(columns=[c for c in by_var.columns if c in EXCLUDE])
    rows = []
    for idx, r in by_var.iterrows():
        pos = r[r > 0].sort_values(ascending=False).head(top)
        texts = [_reason_text(v, X.loc[idx]) for v in pos.index]
        texts += [""] * (top - len(texts))
        talk = _talk_point(X.loc[idx], pos.index[0] if len(pos) else None)
        rows.append({"reason_1": texts[0], "reason_2": texts[1], "talk_point": talk,
                     "top_var": pos.index[0] if len(pos) else ""})
    return pd.DataFrame(rows, index=X.index)


def write_scores(customer_ids, p, monthly, reasons: pd.DataFrame, db_path: Path = config.DB_PATH) -> None:
    value = cost.customer_value(monthly)
    expected = np.asarray(p) * config.SAVE_RATE * value - config.OFFER_COST
    t = pd.DataFrame({"customer_id": np.asarray(customer_ids), "churn_prob": np.asarray(p),
                      "value_at_risk": value, "expected_gain": expected,
                      "reason_1": reasons["reason_1"].to_numpy(), "reason_2": reasons["reason_2"].to_numpy(),
                      "talk_point": reasons["talk_point"].to_numpy(),
                      # 혜택이 '1개월 요금 할인'이면 기준 확률은 1 / (남는 비율 × 잃는 매출 기간) 하나다
                      "in_1month_discount_list": (np.asarray(p) > 1 / (config.SAVE_RATE * config.LOST_MONTHS)).astype(int)})
    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS scores")
        t.to_sql("scores", con, index=False)
