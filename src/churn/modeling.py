"""모델 파이프라인, 누수 점검, 모델 비교, 확률 보정 점검."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config


def preprocessor(numeric=config.NUMERIC, categorical=config.CATEGORICAL) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric)),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), list(categorical)),
    ])


def make_model(kind: str, weighted: bool = False, pos_ratio: float | None = None) -> Pipeline:
    """kind: 'logit' 또는 'xgb'. weighted=True 면 해지 고객에 가중치를 줘 0/1 비율을 맞춘다."""
    if kind == "logit":
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced" if weighted else None)
    elif kind == "xgb":
        spw = (1 - pos_ratio) / pos_ratio if (weighted and pos_ratio) else 1.0
        clf = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss",
            random_state=config.SEED, n_jobs=4,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("prep", preprocessor()), ("clf", clf)])


def cv_folds():
    return StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)


def cv_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """fold별 AUROC와 교차검증 예측 확률(out-of-fold)을 함께 돌려준다."""
    aucs, oof = [], np.zeros(len(y))
    for tr, va in cv_folds().split(X, y):
        m = model.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "fold_aucs": aucs, "oof": oof}


# ── 누수 점검 ────────────────────────────────────────────────────────────────
def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """변수 하나만으로 해지를 얼마나 맞히는지(교차검증 AUROC). 한 변수가 지나치게 잘 맞히면 누수를 의심한다."""
    rows = []
    for col in config.NUMERIC + config.CATEGORICAL:
        num = [col] if col in config.NUMERIC else []
        cat = [col] if col in config.CATEGORICAL else []
        model = Pipeline([("prep", preprocessor(num, cat)), ("clf", LogisticRegression(max_iter=2000))])
        p = cross_val_predict(model, X[[col]], y, cv=cv_folds(), method="predict_proba")[:, 1]
        rows.append({"feature": col, "auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# ── 모델 비교 ────────────────────────────────────────────────────────────────
def bootstrap_auc_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = config.N_BOOT) -> dict:
    """평가 데이터를 고객 단위로 다시 뽑아 AUROC 차이(a - b)의 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(config.SEED)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), "ci_low": float(lo), "ci_high": float(hi)}


# ── 확률 보정 ────────────────────────────────────────────────────────────────
def calibration_summary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """확률이 실제 해지율과 맞는지. 평균 예측 vs 실제 해지율, Brier, ECE(구간별 차이의 가중 평균)."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    return {
        "mean_pred": float(p.mean()), "actual_rate": float(y.mean()),
        "ratio": float(p.mean() / y.mean()), "brier": float(brier_score_loss(y, p)),
        "ece": float(ece), "auc": float(roc_auc_score(y, p)),
    }


def platt_calibrated(model: Pipeline) -> CalibratedClassifierCV:
    """학습 데이터 안에서 교차검증으로 로지스틱 보정(Platt)을 붙인다."""
    return CalibratedClassifierCV(model, method="sigmoid", cv=cv_folds())


def logit_odds_table(model: Pipeline, top: int = 8) -> pd.DataFrame:
    """로지스틱 회귀 계수를 '해지 확률을 높이는/낮추는 요인' 표로 바꾼다 (표준화·원핫 기준)."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    t["feature"] = t["feature"].str.replace("num__", "").str.replace("cat__", "")
    t = t.reindex(t["coef"].abs().sort_values(ascending=False).index)
    return t.head(top).reset_index(drop=True)


def calibration_by_band(y, probs: dict[str, np.ndarray], monthly, n_bands: int = 4) -> pd.DataFrame:
    """월 요금 구간별 평균 예측 확률 vs 실제 해지율. 비용 기준 규칙은 요금 구간마다 기준이 달라서
    전체 평균뿐 아니라 구간별로도 확률이 맞아야 한다."""
    y, monthly = np.asarray(y), np.asarray(monthly)
    band = pd.qcut(monthly, n_bands, labels=[f"{i + 1}분위" for i in range(n_bands)])
    t = pd.DataFrame({"band": band, "y": y, **{k: np.asarray(v) for k, v in probs.items()}, "monthly": monthly})
    g = t.groupby("band", observed=True)
    out = g.agg(customers=("y", "size"), monthly_min=("monthly", "min"), monthly_max=("monthly", "max"),
                actual_rate=("y", "mean"), **{f"pred_{k}": (k, "mean") for k in probs})
    return out.reset_index()
