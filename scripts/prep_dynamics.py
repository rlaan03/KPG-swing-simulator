# prep_dynamics.py  ― 재작성본
import pandas as pd, numpy as np, re, pathlib, textwrap

CSV   = "./data/gen_result_with_type.csv"
MFILE = "./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m"

############################
# 1) PF 결과 (107행)
############################
gen = pd.read_csv(CSV)          # gen_id, bus, Pg, Qg, type

############################
# 2) .m 파일에서 Pmax 파싱
############################
m_txt  = pathlib.Path(MFILE).read_text()

# mpc.gen 블록만 잘라오기
raw_blk = re.search(r"mpc\.gen\s*=\s*\[\s*(.*?)\];", m_txt, re.S).group(1)

clean_lines = []
status_col, pmax_col = 7, 8      # 0-based index

for line in raw_blk.splitlines():
    line = line.split('%')[0]    # 코멘트 제거
    line = line.replace(';', '') # 세미콜론 제거
    line = line.strip()
    if not line:
        continue
    clean_lines.append(line)

# → float 배열 (rows × 21)
gen_mat = np.loadtxt(clean_lines)

# ▸ status==1 행만 취한다면:
active   = gen_mat[:, status_col] == 1
Pmax_all = gen_mat[active, pmax_col]

# sanity check
if len(Pmax_all) != len(gen):
    raise ValueError(f"행 개수 불일치: PF {len(gen)} vs active gens {len(Pmax_all)}")

gen["S_base_MVA"] = Pmax_all

############################
# 3) 관성 H, 4) M, D 계수
############################
H_typ = {"Coal": 5.0, "LNG": 3.0, "Nuclear": 6.0}
gen["H_s"] = gen["type"].map(H_typ).fillna(4.0)      # H [s]

OMEGA = 2 * np.pi * 60           # ω_syn [rad/s]
gen["M"] = 2 * gen["H_s"] * gen["S_base_MVA"] / OMEGA

# ── D 값 부여 ───────────────────────────────────────────
# (A) 비례 댐핑 : D = 2 ζ M   (ζ ≈ 1 s⁻¹ 권장)
ζ = 1.0
gen["D"] = 2 * ζ * gen["M"]

# --- (B) 타입별 고정값 예시 -----------------------------
# D_typ = {"Coal": 0.05, "LNG": 0.04, "Nuclear": 0.06}   # pu·s/rad
# gen["D"] = gen["type"].map(D_typ).fillna(0.05)

# 둘 다 쓰고 싶으면 gen["D"] = (A)+(B) 처럼 합산도 가능
#########################################################


out = "./data/dyn_params.csv"
gen.to_csv(out, index=False)
print(f"✅  dyn_params.csv 저장 완료 (rows={len(gen)})")
print("   연료별 H_s (s) 분포:\n", gen.groupby('type')["H_s"].describe()[["count","mean"]])
