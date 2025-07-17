# add_type_col.py  ────────────────
"""
gen_result.csv ↔ KPG193_ver1_2.m  ──>  gen_result_with_type.csv
    • status==1 발전기만 대상
    • m 파일의 '; % 연료' 주석으로부터 type 열 추가
"""

import re
from pathlib import Path
import pandas as pd

# ────────── 사용자 설정 ──────────
GEN_CSV   = Path("./pf/gen_result.csv")          # PF 결과 (기존)
MATPOWER_M = Path("./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m")
OUT_CSV   = Path("./pf/gen_result_with_type.csv")  # type 추가본 저장
# ────────────────────────────────

# 1) PF 결과 읽기 ----------------------------------------------------------
df = pd.read_csv(GEN_CSV)
print(f"CSV 로드: {GEN_CSV}  (rows={len(df)})")

# 2) .m 파일에서 발전기 블록 파싱 -----------------------------------------
fuel_types = []                # status==1인 발전기의 fuel (CSV 행 순서대로)

inside = False                 # mpc.gen 구역 플래그
with MATPOWER_M.open(encoding="utf-8") as f:
    for line in f:
        # 블록 시작 찾기
        if not inside:
            if re.match(r"\s*mpc\.gen\s*=\s*\[", line):
                inside = True
            continue

        # 블록 끝이면 중단
        if re.match(r"\s*\];", line):
            break
        if not line.strip():          # 빈 줄 패스
            continue

        if ';' not in line:           # 숫자 줄은 반드시 ';' 가 있음
            continue

        num_part, comment_part = line.split(';', 1)

        tokens = num_part.strip().split()
        if len(tokens) < 8:           # 필드 부족 → 주석·공백 라인
            continue

        status = int(tokens[7])       # 8번째 필드
        if status == 0:               # 오프라인 발전기 → 건너뜀
            continue

        # 주석에서 연료명 추출 (Coal, LNG, Nuclear …)
        m = re.search(r'%\s*([A-Za-z]+)', comment_part)
        fuel = m.group(1) if m else "Unknown"
        fuel_types.append(fuel)

# 3) 행 수 검증 ------------------------------------------------------------
if len(fuel_types) != len(df):
    raise ValueError(
        f"❌ 행 개수 불일치: CSV {len(df)} vs .m에서 추출 {len(fuel_types)}"
    )

# 4) type 열 추가 & 저장 ---------------------------------------------------
df["type"] = fuel_types
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, float_format="%.6f")

print(f"✅  {OUT_CSV.name} 저장 완료 (rows={len(df)})")
print("   연료 구성:", dict(pd.Series(fuel_types).value_counts()))
