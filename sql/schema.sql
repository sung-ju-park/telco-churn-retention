-- 원본 CSV 한 장을 고객 / 계약·요금 / 가입 서비스 세 테이블로 나눈다.
-- 실제 통신사에서 이 정보들이 서로 다른 시스템에 있다고 보고, 분석용 피처는 SQL로 다시 모은다.

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS services;

CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    gender          TEXT NOT NULL,
    senior_citizen  INTEGER NOT NULL,      -- 0/1
    partner         TEXT NOT NULL,         -- Yes/No
    dependents      TEXT NOT NULL          -- Yes/No
);

CREATE TABLE accounts (
    customer_id        TEXT PRIMARY KEY REFERENCES customers(customer_id),
    tenure_months      INTEGER NOT NULL,   -- 가입 기간(개월)
    contract           TEXT NOT NULL,      -- Month-to-month / One year / Two year
    paperless_billing  TEXT NOT NULL,
    payment_method     TEXT NOT NULL,
    monthly_charges    REAL NOT NULL,
    total_charges      REAL,               -- 가입 0개월 고객은 원본이 공백 → NULL
    churn              INTEGER NOT NULL    -- 지난 한 달 안에 해지했으면 1
);

CREATE TABLE services (
    customer_id        TEXT PRIMARY KEY REFERENCES customers(customer_id),
    phone_service      TEXT NOT NULL,
    multiple_lines     TEXT NOT NULL,
    internet_service   TEXT NOT NULL,      -- DSL / Fiber optic / No
    online_security    TEXT NOT NULL,
    online_backup      TEXT NOT NULL,
    device_protection  TEXT NOT NULL,
    tech_support       TEXT NOT NULL,
    streaming_tv       TEXT NOT NULL,
    streaming_movies   TEXT NOT NULL
);
