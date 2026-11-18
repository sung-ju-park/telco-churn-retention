"""전체 분석을 처음부터 다시 실행한다.

    python scripts/download_data.py
    python run_pipeline.py

결과: reports/metrics.json, reports/tables/*.csv, reports/figures/*.png
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from churn import calllist, config, cost, db, experiment, modeling, monitoring, plotting  # noqa: E402

warnings.filterwarnings("ignore")
FEATURES = config.NUMERIC + config.CATEGORICAL
TABLES = config.REPORT_DIR / "tables"


def r(x, n=4):
    if isinstance(x, dict):
        return {k: r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    plotting.setup_style()
    TABLES.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # 1. 데이터 점검 → SQLite → SQL 피처
    raw, checks = db.load_raw()
    db.build_db(raw)
    data = db.query("features.sql")
    seg = db.query("segment_churn.sql")
    seg.to_csv(TABLES / "segment_churn.csv", index=False)
    out["data_checks"] = checks
    plotting.segment_heatmap(seg, config.FIG_DIR / "01_segment_churn.png")

    X, y = data[FEATURES], data[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    out["split"] = {"train": len(y_tr), "train_churn": int(y_tr.sum()), "test": len(y_te), "test_churn": int(y_te.sum())}
    pos = float(y_tr.mean())

    # 2. 누수 점검
    leak = modeling.single_feature_auc(X_tr, y_tr)
    leak.to_csv(TABLES / "single_feature_auc.csv", index=False)
    out["leakage"] = {"max_feature": leak.iloc[0]["feature"], "max_auc": leak.iloc[0]["auc"],
                      "flag_threshold": 0.85, "flagged": leak[leak["auc"] >= 0.85]["feature"].tolist()}
    plotting.single_feature_auc(leak, config.FIG_DIR / "02_single_feature_auc.png")

    # 3. 모델 비교 (가중치 유무 × 로지스틱/XGBoost)
    fitted, test_p, cv, oof = {}, {}, {}, {}
    for kind in ("logit", "xgb"):
        for weighted in (False, True):
            key = f"{kind}{'_weighted' if weighted else ''}"
            m = modeling.make_model(kind, weighted, pos)
            s = modeling.cv_scores(m, X_tr, y_tr)
            cv[key] = {"auc_mean": s["auc_mean"], "auc_std": s["auc_std"]}
            oof[key] = s["oof"]
            fitted[key] = modeling.make_model(kind, weighted, pos).fit(X_tr, y_tr)
            test_p[key] = fitted[key].predict_proba(X_te)[:, 1]
    out["cv_auc"] = cv
    diff = modeling.bootstrap_auc_diff(y_te, test_p["xgb"], test_p["logit"])
    diff["equivalence_margin"] = 0.01
    diff["within_margin"] = bool(diff["ci_low"] > -0.01 and diff["ci_high"] < 0.01)
    out["test_auc_diff_xgb_minus_logit"] = diff

    # 4. 확률 보정: 가중치가 확률을 부풀리는지, Platt 보정으로 되돌아오는지
    platt = modeling.platt_calibrated(modeling.make_model("logit", True, pos)).fit(X_tr, y_tr)
    test_p["logit_weighted_platt"] = platt.predict_proba(X_te)[:, 1]
    out["calibration"] = {k: modeling.calibration_summary(y_te, p) for k, p in test_p.items()}
    plotting.calibration_plot(y_te.to_numpy(), {
        "가중치 없음": test_p["logit"],
        "해지 가중치": test_p["logit_weighted"],
        "해지 가중치 + 보정": test_p["logit_weighted_platt"],
    }, config.FIG_DIR / "03_calibration.png")
    band = modeling.calibration_by_band(y_te, {"logit": test_p["logit"], "logit_weighted": test_p["logit_weighted"],
                                               "logit_weighted_platt": test_p["logit_weighted_platt"]},
                                        X_te["monthly_charges"])
    band.to_csv(TABLES / "calibration_by_charge_band.csv", index=False)
    out["calibration_by_charge_band"] = band.to_dict(orient="records")
    odds = modeling.logit_odds_table(fitted["logit"], top=10)
    odds.to_csv(TABLES / "logit_top_factors.csv", index=False)

    # 5. 비용 기준 연락 정책 (최종 모델: 가중치 없는 로지스틱 회귀)
    p_main, p_w, p_wp = test_p["logit"], test_p["logit_weighted"], test_p["logit_weighted_platt"]
    monthly = X_te["monthly_charges"].to_numpy()
    yv = y_te.to_numpy()
    n = len(yv)
    share = cost.best_share(y_tr.to_numpy(), oof["logit"], X_tr["monthly_charges"].to_numpy())
    contacts = {
        "아무에게도 연락 안 함": np.zeros(n, bool),
        "전원 연락": np.ones(n, bool),
        "확률 0.5 이상": p_main >= 0.5,
        f"확률 상위 {share:.0%} (학습 데이터로 정함)": cost.top_share_contact(p_main, share),
        "비용 기준 · 가중치 확률 (보정 안 함)": cost.cost_rule(p_w, monthly),
        "비용 기준 · 가중치 + 보정 확률": cost.cost_rule(p_wp, monthly),
        "비용 기준 (채택)": cost.cost_rule(p_main, monthly),
    }
    policies = pd.DataFrame({k: cost.realized_profit(yv, c, monthly) for k, c in contacts.items()}).T
    policies.to_csv(TABLES / "policy_profit.csv")
    rule_main = contacts["비용 기준 (채택)"]
    by_prob = cost.top_share_contact(p_main, rule_main.sum() / n)
    out["cost"] = {
        "assumptions": {"lost_months": config.LOST_MONTHS, "offer_cost": config.OFFER_COST, "save_rate": config.SAVE_RATE},
        "topk_share_from_train": share,
        "policies": policies.to_dict(orient="index"),
        "same_count_prob_only": cost.realized_profit(yv, by_prob, monthly),
        "diff_rule_vs_topk": cost.bootstrap_profit_diff(yv, rule_main, contacts[f"확률 상위 {share:.0%} (학습 데이터로 정함)"], monthly),
        "diff_rule_vs_prob_only_same_count": cost.bootstrap_profit_diff(yv, rule_main, by_prob, monthly),
        "diff_rule_vs_all": cost.bootstrap_profit_diff(yv, rule_main, contacts["전원 연락"], monthly),
        "diff_rule_vs_half": cost.bootstrap_profit_diff(yv, rule_main, contacts["확률 0.5 이상"], monthly),
        "diff_rule_vs_weighted_raw": cost.bootstrap_profit_diff(yv, rule_main, contacts["비용 기준 · 가중치 확률 (보정 안 함)"], monthly),
    }
    a = out["cost"]["assumptions"]
    plotting.policy_profit(
        policies, f"가정: 해지 시 월 요금 × {a['lost_months']}개월 손실 · 혜택 1건 ${a['offer_cost']:.0f} · "
                  f"혜택 받은 해지 예정자의 {a['save_rate']:.0%}가 남음", config.FIG_DIR / "04_policy_profit.png")
    gains = cost.gain_table(yv, p_main, monthly)
    gains.to_csv(TABLES / "gain_table.csv", index=False)
    out["gain_table"] = gains.to_dict(orient="records")
    plotting.gain_curve(yv, p_main, config.FIG_DIR / "05_gain_curve.png")

    sens = cost.sensitivity(yv, monthly, p_main, p_w, y_tr.to_numpy(), oof["logit"], X_tr["monthly_charges"].to_numpy())
    sens.to_csv(TABLES / "sensitivity.csv", index=False)
    loss = sens["profit"] - sens["profit_weighted_probs"]
    out["sensitivity"] = {
        "cases": len(sens),
        "rule_beats_trivial_best": int((sens["profit"] > sens["profit_trivial_best"]).sum()),
        "rule_beats_topk": int((sens["profit"] > sens["profit_topk"]).sum()),
        "rule_minus_topk_median": float((sens["profit"] - sens["profit_topk"]).median()),
        "rule_minus_topk_min": float((sens["profit"] - sens["profit_topk"]).min()),
        "rule_minus_topk_max": float((sens["profit"] - sens["profit_topk"]).max()),
        "weighted_worse_cases": int((loss > 0).sum()),
        "weighted_max_loss": float(loss.max()),
    }
    plotting.sensitivity_heatmap(sens, config.LOST_MONTHS, config.FIG_DIR / "06_sensitivity.png")

    # 5-1. 상담원용 연락 대상 리스트 (평가 고객을 '이번 달 대상'으로 가정)
    reasons = calllist.reason_codes(fitted["logit"], X_tr, X_te)
    calllist.write_scores(data.loc[X_te.index, "customer_id"], p_main, monthly, reasons)
    calls = db.query("call_list.sql")
    calls.to_csv(TABLES / "call_list.csv", index=False, encoding="utf-8-sig")  # 엑셀에서 바로 열리게
    top_reason = reasons.loc[rule_main, "reason_1"].str.replace(r"가입 \d+개월·월 \$\d+", "가입 기간·요금", regex=True)
    out["call_list"] = {
        "rows": len(calls),
        "matches_cost_rule": bool(len(calls) == int(rule_main.sum())),
        "top_reason_share": top_reason.value_counts(normalize=True).round(3).to_dict(),
        "talk_point_counts": calls["talk_point"].value_counts().to_dict(),
        "in_1month_discount_list": int(calls["in_1month_discount_list"].sum()),
        "head": calls.head(10).to_dict(orient="records"),
    }

    # 5-2. 남는 비율을 재는 비교 집단 실험 설계
    risk_value = p_main * cost.customer_value(monthly)
    base_churn = float(p_main[rule_main].mean())
    design = experiment.design_table(base_churn, float(risk_value[rule_main].mean()), config.OFFER_COST)
    design.to_csv(TABLES / "experiment_design.csv", index=False)
    out["experiment"] = {"eligible_mean_pred": base_churn, "eligible_actual_rate": float(yv[rule_main].mean()),
                         "table": design.to_dict(orient="records"),
                         "n_connected_for_pm5pp": {f"{s:.0%}": experiment.n_for_precision(base_churn, s, 0.05)
                                                   for s in (0.1, 0.2, 0.3)}}

    # 5-3. 혜택이 요금 비례 할인일 때
    prop = cost.proportional_offer_table(yv, p_main, monthly, y_tr.to_numpy(), oof["logit"],
                                         X_tr["monthly_charges"].to_numpy())
    prop.to_csv(TABLES / "proportional_offer.csv", index=False)
    out["proportional_offer"] = prop.to_dict(orient="records")

    # 6. 운영 모니터링 시뮬레이션
    mon = monitoring.monitor_table(fitted["logit"], X_tr, X_te, y_te, FEATURES, np.random.default_rng(config.SEED))
    mon.to_csv(TABLES / "monitoring.csv", index=False)
    out["monitoring"] = mon.to_dict(orient="records")
    plotting.monitoring_plot(mon, config.FIG_DIR / "07_monitoring.png")
    out["monthly_charges_r2_from_services"] = monitoring.charges_explained_by_services(data)
    price = monitoring.price_response({"로지스틱 회귀": fitted["logit"], "XGBoost": fitted["xgb"]}, X_te)
    price.to_csv(TABLES / "price_response.csv", index=False)
    out["price_response"] = price.to_dict(orient="records")
    plotting.price_response_plot(price, config.FIG_DIR / "08_price_response.png")

    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(r(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(r(out, 3), ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
