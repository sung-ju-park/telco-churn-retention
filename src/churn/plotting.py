"""결과 그림. 모든 그림은 같은 색 체계와 글꼴을 쓴다."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, RED = "#898781", "#e34948"

KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]


def setup_style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in KOREAN_FONTS if f in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 24, "axes.axisbelow": True, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _subtitle(ax, text: str) -> None:
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom")


def segment_heatmap(seg: pd.DataFrame, path: Path) -> Path:
    pv = seg.pivot(index="contract", columns="tenure_band", values="churn_rate_pct")
    n = seg.pivot(index="contract", columns="tenure_band", values="customers")
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.imshow(pv.values, cmap="Oranges", vmin=0, vmax=60, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(pv.shape[1]), [c.split(") ")[1] for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), ["월 단위", "1년 약정", "2년 약정"])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.0f}%\n({n.values[i, j]:,}명)", ha="center", va="center", fontsize=9,
                    color="white" if v > 35 else INK)
    ax.set_title("계약 형태 × 가입 기간별 해지율")
    _subtitle(ax, "sql/segment_churn.sql 결과. 월 단위 계약 신규 고객의 절반 이상이 해지")
    return _save(fig, path)


def single_feature_auc(t: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    t = t.iloc[::-1]
    ax.barh(t["feature"], t["auc"], color=[BLUE if a >= 0.7 else AXIS for a in t["auc"]])
    ax.axvline(0.85, color=RED, ls="--", lw=1)
    ax.text(0.853, 0.5, "이 이상이면 누수 의심", color=RED, fontsize=8.5, va="center")
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("변수 하나만으로 맞힌 AUROC (학습 데이터 5-fold)")
    for y, a in enumerate(t["auc"]):
        ax.text(a + 0.004, y, f"{a:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_title("누수 점검: 혼자서 해지를 너무 잘 맞히는 변수가 있는가")
    _subtitle(ax, "가장 높은 가입 기간도 0.74. 해지 이후에야 알 수 있는 값으로 보이는 변수는 없음")
    return _save(fig, path)


def calibration_plot(y, curves: dict[str, np.ndarray], path: Path) -> Path:
    colors = [GRAY, RED, BLUE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    for (name, p), c in zip(curves.items(), colors):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        a1.plot(mean, frac, marker="o", ms=4, color=c, label=name)
    a1.set_xlabel("예측 해지 확률")
    a1.set_ylabel("실제 해지율")
    a1.legend(loc="upper left")
    a1.set_title("해지 가중치를 주면 확률이 부풀어 오른다")
    _subtitle(a1, "점선 위에 있어야 '확률 0.3 = 실제로 30%가 해지'")
    names = list(curves)
    means = [curves[n].mean() for n in names]
    a2.bar(range(len(names)), means, color=colors[: len(names)])
    a2.axhline(np.mean(y), color=INK, lw=1)
    for i, m in enumerate(means):
        a2.text(i, m + 0.008, f"{m:.1%}", ha="center", fontsize=9)
    a2.set_xticks(range(len(names)), [n.replace(" + ", "\n+ ") for n in names], fontsize=8.5)
    a2.set_ylim(0, max(means) * 1.25)
    a2.set_title("평균 예측 해지 확률")
    _subtitle(a2, f"검은 선 = 평가 데이터 실제 해지율 {np.mean(y):.1%}")
    fig.tight_layout()
    return _save(fig, path)


def policy_profit(t: pd.DataFrame, assumptions: str, path: Path) -> Path:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    colors = [BLUE if "채택" in i else (RED if "보정 안 함" in i else AXIS) for i in t.index]
    ax.barh(range(len(t)), t["profit"], color=colors)
    ax.set_yticks(range(len(t)), t.index)
    for y, (p, c) in enumerate(zip(t["profit"], t["contacted"])):
        ax.text(max(p, 0) + 500, y, f"${p:,.0f}  ({int(c):,}명 연락)", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("평가 데이터 1,409명 기준 지킨 매출 − 혜택 비용 (달러, 원가 미반영)")
    ax.set_xlim(min(t["profit"].min() * 1.1, 0), t["profit"].max() * 1.45)
    ax.set_title("누구에게 연락할 것인가: 연락 정책별 결과")
    _subtitle(ax, assumptions)
    return _save(fig, path)


def gain_curve(y, p, path: Path) -> Path:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    share = np.arange(1, len(y) + 1) / len(y)
    caught = np.cumsum(y[order]) / y.sum()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(share, caught, color=BLUE, label="모델 순서대로 연락")
    ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1, label="무작위로 연락")
    for s in (0.1, 0.2, 0.3):
        k = int(round(len(y) * s)) - 1
        ax.plot(share[k], caught[k], "o", color=BLUE, ms=5)
        ax.text(share[k] + 0.015, caught[k] - 0.05, f"상위 {s:.0%} → 해지자 {caught[k]:.0%}", fontsize=8.5)
    ax.set_xlabel("연락한 고객 비율 (해지 확률 높은 순)")
    ax.set_ylabel("잡아낸 해지 고객 비율")
    ax.legend(loc="lower right")
    ax.set_title("해지 확률 상위 몇 %에 연락하면 해지자를 얼마나 잡는가")
    _subtitle(ax, f"평가 데이터 해지 고객 {int(y.sum())}명 기준")
    return _save(fig, path)


def sensitivity_heatmap(s: pd.DataFrame, lost_months: int, path: Path) -> Path:
    gain = s.pivot(index="save_rate", columns="offer_cost", values="profit") - \
        s.pivot(index="save_rate", columns="offer_cost", values="profit_trivial_best")
    vs_k = s.pivot(index="save_rate", columns="offer_cost", values="profit") - \
        s.pivot(index="save_rate", columns="offer_cost", values="profit_topk")
    rate = s.pivot(index="save_rate", columns="offer_cost", values="contact_rate")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.imshow(np.log1p(gain.values / 1000), cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(gain.shape[1]), [f"${c}" for c in gain.columns])
    ax.set_yticks(range(gain.shape[0]), [f"{r:.0%}" for r in gain.index])
    ax.set_xlabel("방어 혜택 1건 비용")
    ax.set_ylabel("혜택 받은 해지 예정자 중 남는 비율", labelpad=8)
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            g, k = gain.values[i, j], vs_k.values[i, j]
            ax.text(j, i, f"+${g:,.0f}\n상위k 대비 {k:+,.0f}\n연락 {rate.values[i, j]:.0%}", ha="center", va="center",
                    fontsize=7.8, color="white" if g > 30000 else INK)
    ax.set_title("가정이 바뀌면: 비용 기준 규칙의 결과")
    _subtitle(ax, f"위: 전원 연락·무연락 중 나은 쪽 대비 · 가운데: 학습 데이터로 정한 확률 상위 k% 대비 (잃는 매출 = 월 요금 × {lost_months}개월)")
    return _save(fig, path)


def monitoring_plot(t: pd.DataFrame, path: Path) -> Path:
    t = t.reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(t))[::-1] * 1.25
    labels = t["scenario"].tolist()
    cols = [("score_psi", "예측 확률 분포", BLUE), ("tenure_psi", "가입 기간", AQUA),
            ("contract_psi", "계약 형태", ORANGE), ("charges_psi", "월 요금", GRAY)]
    h = 0.24
    for i, (c, name, color) in enumerate(cols):
        a1.barh(y + (1.5 - i) * h, t[c].clip(upper=0.6), height=h, color=color, label=name)
    a1.axvline(0.1, color=ORANGE, ls="--", lw=1)
    a1.axvline(0.25, color=RED, ls="--", lw=1)
    a1.set_yticks(y, labels)
    a1.set_xlabel("PSI (0.1 주의 · 0.25 큰 변화, 0.6 이상은 잘라서 표시)")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title("정답 없이 바로 볼 수 있는 지표")
    _subtitle(a1, "해지 여부가 확정되기 전에도 계산 가능")
    gap = (t["mean_pred"] - t["actual_rate"]).fillna(0)
    a2.barh(y, gap, height=0.6, color=[RED if abs(g) > 0.03 else AXIS for g in gap])
    for yy, g, mp, auc, has in zip(y, gap, t["mean_pred"], t["auc"], t["actual_rate"].notna()):
        txt = f"{g:+.1%}p · AUROC {auc:.3f}" if has else f"정답 없음 · 평균 예측 {mp:.1%}"
        a2.text(0.002 if g >= 0 else g - 0.002, yy, txt, va="center", ha="left" if g >= 0 else "right",
                fontsize=8.5, color=INK_2)
    a2.axvline(0, color=AXIS, lw=1)
    a2.set_xlim(-0.22, 0.10)
    a2.set_xlabel("평균 예측 확률 − 실제 해지율")
    a2.set_title("한 달 뒤 정답이 쌓여야 볼 수 있는 지표")
    _subtitle(a2, "입력이 그대로여도 확률이 틀어지는 경우가 있다")
    fig.tight_layout()
    return _save(fig, path)


def price_response_plot(t: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, c in zip([c for c in t.columns if c != "price_mult"], [ORANGE, BLUE]):
        ax.plot((t["price_mult"] - 1) * 100, t[name] * 100, marker="o", ms=4, color=c, label=name)
    ax.set_xlabel("월 요금 인상률 (%)")
    ax.set_ylabel("평균 예측 해지 확률 (%)")
    ax.legend()
    ax.set_title("요금을 올리면? 두 모델의 답이 반대")
    _subtitle(ax, "AUROC가 같은 두 모델도 학습 데이터에 없던 상황에서는 다르게 반응")
    return _save(fig, path)
