-- 세 테이블을 고객 단위로 묶어 모델 입력 피처를 만든다.
-- 파생 피처는 운영 시점에 이미 알고 있는 값(가입 기간, 요금, 가입 서비스)으로만 만든다.

SELECT
    c.customer_id,
    -- 고객 정보
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    -- 계약·요금
    a.tenure_months,
    a.contract,
    a.paperless_billing,
    a.payment_method,
    a.monthly_charges,
    COALESCE(a.total_charges, 0.0)                         AS total_charges,
    -- 자동이체 여부: 결제 수단 이름에 automatic 이 들어가면 1
    CASE WHEN a.payment_method LIKE '%automatic%' THEN 1 ELSE 0 END AS auto_pay,
    -- 가입 서비스
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    -- 부가 서비스 개수(보안·백업·기기보호·기술지원·TV·영화)
    (  (s.online_security   = 'Yes') + (s.online_backup    = 'Yes')
     + (s.device_protection = 'Yes') + (s.tech_support     = 'Yes')
     + (s.streaming_tv      = 'Yes') + (s.streaming_movies = 'Yes'))  AS n_addons,
    -- 정답
    a.churn
FROM customers  AS c
JOIN accounts   AS a ON a.customer_id = c.customer_id
JOIN services   AS s ON s.customer_id = c.customer_id
ORDER BY c.customer_id;
