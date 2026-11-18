"""IBM이 공개한 통신사 고객 이탈 데이터를 받아 data/raw/ 에 저장한다.

    python scripts/download_data.py

자동 다운로드가 안 되면 아래 저장소의 data/Telco-Customer-Churn.csv 를
직접 받아 data/raw/ 폴더에 넣으면 된다. (Apache License 2.0)
https://github.com/IBM/telco-customer-churn-on-icp4d
"""
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
EXPECTED_ROWS = 7043


def main() -> None:
    if OUT.exists():
        print(f"이미 있습니다: {OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"내려받는 중: {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            OUT.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"다운로드 실패: {exc}\n위 주석의 저장소에서 직접 받아 {OUT} 에 넣어 주세요.")

    import pandas as pd

    n = len(pd.read_csv(OUT))
    if n != EXPECTED_ROWS:
        sys.exit(f"행 수가 예상과 다릅니다: {n} (예상 {EXPECTED_ROWS})")
    print(f"저장 완료: {OUT} ({n}행)")


if __name__ == "__main__":
    main()
