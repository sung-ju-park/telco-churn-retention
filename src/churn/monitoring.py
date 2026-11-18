"""운영 중 성능 모니터링 시뮬레이션.

이 데이터에는 시점 정보가 없어 실제 시간에 따른 변화를 볼 수 없다.
그래서 평가 데이터를 일부러 바꿔 '이런 변화가 생기면 어떤 지표가 먼저 반응하는가'를 본다.
모든 시나리오는 인위적으로 만든 것이다.

해지 여부는 한 달 뒤에야 확정되므로, 운영에서는 정답 없이 볼 수 있는 지표(PSI, 평균 예측 확률)를
먼저 보고, 정답이 쌓이면 AUROC와 실제 해지율로 확인하는 순서를 가정한다.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

PRICE_UP = 1.15
CONCEPT_ODDS = 1.8


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index (숫자형). 기준 분포의 10분위 구간으로 나눠 비교한다.
    관행상 0.1 미만 안정, 0.1~0.25 주의, 0.25 이상 큰 변화로 본다."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    return _psi(e, a)


def psi_categorical(expected, actual) -> float:
    """범주형 변수의 PSI. 범주별 비율을 비교한다."""
    cats = sorted(set(expected) | set(actual))
    e = pd.Series(expected).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = pd.Series(actual).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(e, a)


def _psi(e, a) -> float:
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _resample(df: pd.DataFrame, weights: np.ndarray, rng) -> pd.DataFrame:
    idx = rng.choice(len(df), size=len(df), replace=True, p=weights / weights.sum())
    return df.iloc[idx].reset_index(drop=True)


def scenarios(X: pd.DataFrame, y, p_model, rng) -> dict[str, tuple[pd.DataFrame, np.ndarray | None]]:
    """(바뀐 입력, 바뀐 정답 또는 None). None은 실제 해지가 어떻게 바뀔지 알 수 없는 경우.
    p_model: 평가 고객의 모델 예측 확률. '해지 확률 자체 상승' 시나리오에서 정답을 만들 때만 쓴다."""
    base = X.assign(churn=np.asarray(y), _p=np.asarray(p_model)).reset_index(drop=True)
    out = {"기준 (평가 데이터 그대로)": (base, base["churn"].to_numpy())}

    # 1) 고객 구성 변화: 고객을 다시 뽑는다. 고객 유형별 해지 확률은 그대로다.
    w = np.where(base["tenure_months"] <= 12, 3.0, 1.0)
    d = _resample(base, w, rng)
    out["신규 고객 비중 증가 (가입 12개월 이하 3배)"] = (d, d["churn"].to_numpy())

    w = np.where(base["contract"] == "Month-to-month", 2.0, 1.0)
    d = _resample(base, w, rng)
    out["월 단위 계약 비중 증가 (2배)"] = (d, d["churn"].to_numpy())

    # 2) 같은 고객인데 해지 확률 자체가 오르는 경우 (경쟁사 프로모션 등).
    #    입력은 그대로 두고, 모든 고객의 해지 오즈를 CONCEPT_ODDS배로 올린 확률로 정답을 다시 만든다.
    #    원래 해지한 고객은 그대로 두고, 남은 고객 중 일부를 해지로 바꿔 E[새 정답] = 새 확률이 되게 한다.
    #    (모델 확률을 실제 확률의 대리값으로 쓰는 가정이 들어간다)
    p = base["_p"].to_numpy()
    odds = p / (1 - p) * CONCEPT_ODDS
    p_new = odds / (1 + odds)
    flip = (p_new - p) / (1 - p)
    yy = base["churn"].to_numpy().copy()
    yy[(yy == 0) & (rng.random(len(yy)) < flip)] = 1
    out[f"같은 고객의 해지 확률 상승 (오즈 {CONCEPT_ODDS}배)"] = (base, yy)

    # 3) 요금 인상: 입력만 바뀌고 실제로 해지가 얼마나 늘지는 알 수 없으므로 정답 없음.
    #    인상 직후라 누적 요금은 그대로 두고 월 요금만 올린다.
    d = base.copy()
    d["monthly_charges"] = d["monthly_charges"] * PRICE_UP
    out[f"월 요금 {PRICE_UP - 1:.0%} 인상 (정답 없음)"] = (d, None)
    return out


def monitor_table(model, X_ref: pd.DataFrame, X_test: pd.DataFrame, y_test, features: list[str],
                  rng, n_rep: int = 50) -> pd.DataFrame:
    """시나리오마다 n_rep번 다시 만들어 지표 평균을 낸다."""
    p_ref = model.predict_proba(X_ref)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    acc: dict[str, list] = {}
    for _ in range(n_rep):
        for name, (d, yy) in scenarios(X_test, y_test, p_test, rng).items():
            p = model.predict_proba(d[features])[:, 1]
            acc.setdefault(name, []).append({
                "score_psi": psi(p_ref, p),
                "tenure_psi": psi(X_ref["tenure_months"], d["tenure_months"]),
                "charges_psi": psi(X_ref["monthly_charges"], d["monthly_charges"]),
                "contract_psi": psi_categorical(X_ref["contract"], d["contract"]),
                "mean_pred": p.mean(),
                "actual_rate": np.nan if yy is None else yy.mean(),
                "auc": np.nan if yy is None else roc_auc_score(yy, p),
            })
    t = pd.DataFrame([{"scenario": k, **pd.DataFrame(v).mean().to_dict()} for k, v in acc.items()])
    t["calib_gap"] = t["mean_pred"] - t["actual_rate"]
    return t


def charges_explained_by_services(data: pd.DataFrame) -> float:
    """월 요금이 가입 서비스 구성만으로 얼마나 설명되는지(R²). 1에 가까우면 '같은 서비스, 다른 요금' 고객이
    데이터에 거의 없다는 뜻이라, 모델은 요금만 오르는 상황을 배운 적이 없다."""
    svc = ["phone_service", "multiple_lines", "internet_service", "online_security", "online_backup",
           "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    X = pd.get_dummies(data[svc], drop_first=True)
    return float(LinearRegression().fit(X, data["monthly_charges"]).score(X, data["monthly_charges"]))


def price_response(models: dict, X: pd.DataFrame, mults=(1.0, 1.05, 1.10, 1.15, 1.20)) -> pd.DataFrame:
    """월 요금만 올렸을 때 모델별 평균 해지 확률. 학습 데이터에 없던 상황에서 모델이 어떻게 반응하는지 본다."""
    rows = []
    for mult in mults:
        d = X.copy()
        d["monthly_charges"] = d["monthly_charges"] * mult
        rows.append({"price_mult": mult, **{k: float(m.predict_proba(d)[:, 1].mean()) for k, m in models.items()}})
    return pd.DataFrame(rows)
