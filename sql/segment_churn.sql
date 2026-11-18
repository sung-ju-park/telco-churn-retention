-- 계약 형태 × 가입 기간 구간별 해지율. 어느 고객층에 방어 예산을 써야 하는지 먼저 본다.

SELECT
    a.contract,
    CASE
        WHEN a.tenure_months <= 6  THEN '01) 0-6개월'
        WHEN a.tenure_months <= 12 THEN '02) 7-12개월'
        WHEN a.tenure_months <= 24 THEN '03) 13-24개월'
        ELSE                            '04) 25개월 이상'
    END                                   AS tenure_band,
    COUNT(*)                              AS customers,
    SUM(a.churn)                          AS churned,
    ROUND(AVG(a.churn) * 100, 1)          AS churn_rate_pct,
    ROUND(AVG(a.monthly_charges), 1)      AS avg_monthly_charges
FROM accounts AS a
GROUP BY a.contract, tenure_band
ORDER BY a.contract, tenure_band;
