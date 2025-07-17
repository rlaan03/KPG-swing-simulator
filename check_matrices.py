#!/usr/bin/env python3
# check_matrix.py ― Verify that a saved B-matrix CSV is sound
#
# Usage:
#   python check_matrix.py ./data/B_193.csv
#   (if no path is given, defaults to ./data/B.csv)

import sys
from pathlib import Path
import numpy as np

# ── 1. 입력 경로 설정 ──────────────────────────────────────────
csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/B_43.csv")
if not csv_path.exists():
    sys.exit(f"❌  file not found: {csv_path}")

print(f"● loading {csv_path}")

# ── 2. CSV 로드 ────────────────────────────────────────────────
try:
    B = np.loadtxt(csv_path, delimiter=",")
except Exception as e:
    sys.exit(f"❌  failed to read CSV: {e}")

n, m = B.shape
print(f"  └─ shape = ({n}, {m})")

# ── 3. 기본 검증 ───────────────────────────────────────────────
ok = True

# 3-1. 정사각 행렬 여부
if n != m:
    print("⚠️  matrix is not square!")
    ok = False

# 3-2. 실수 여부
if not np.isrealobj(B):
    print("⚠️  values are not purely real!")
    ok = False

# 3-3. 대칭성
max_sym_err = np.max(np.abs(B - B.T))
print(f"    symmetry max error  : {max_sym_err:.3e}")
if max_sym_err > 1e-8:
    print("⚠️  matrix is not symmetric within 1e-8!")
    ok = False

# 3-4. 행 합 ≈ 0  (Laplacian 조건)
row_sums = B.sum(axis=1)
max_row_err = np.max(np.abs(row_sums))
print(f"    row-sum max |ΣB_ij| : {max_row_err:.3e}")
if max_row_err > 1e-8:
    print("⚠️  row sums deviate from zero by >1e-8!")
    ok = False

# 3-5. 대각 원소 부호 (옵션) : 보통 음수
if np.any(np.diag(B) >= 0):
    print("⚠️  some diagonal entries are not negative (check convention)")

# ── 4. 결과 보고 ───────────────────────────────────────────────
if ok:
    print("✅  matrix passes basic checks!")
else:
    print("❗  matrix FAILED one or more checks")
