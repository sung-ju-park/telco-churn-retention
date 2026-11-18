-- 상담원에게 넘길 이번 달 연락 대상 리스트.
-- scores 테이블은 run_pipeline.py 가 만든다 (평가 데이터 고객을 '이번 달 대상 고객'으로 가정).
-- 기대값(expected_gain) = 해지 확률 × 남는 비율 × 잃는 매출 − 혜택 비용. 0보다 큰 고객만, 기대값이 큰 순서로.
-- 정답(churn)은 상담원에게 보이지 않도록 뽑지 않는다.

SELECT
    ROW_NUMBER() OVER (ORDER BY s.expected_gain DESC)  AS priority,
    s.customer_id,
    a.contract,
    a.tenure_months,
    a.monthly_charges,
    ROUND(s.churn_prob * 100, 1)                       AS churn_prob_pct,
    ROUND(s.expected_gain, 1)                          AS expected_gain,
    s.reason_1,
    s.reason_2,
    s.talk_point,
    s.in_1month_discount_list           -- 혜택이 1개월 요금 할인이라면 대상인지 (1/0)
FROM scores   AS s
JOIN accounts AS a ON a.customer_id = s.customer_id
WHERE s.expected_gain > 0
ORDER BY s.expected_gain DESC;
