###############################################################################
# make_B.jl  (전체 코드 ─ 수정 & 재정리)
###############################################################################
using PowerModels, SparseArrays, DataFrames, CSV

case_path = "./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m"
out_csv   = "./pf/B_gen.csv"

# 1) 로드 & dcline 제거
pm_data = PowerModels.parse_file(case_path)
pm_data["dcline"] = Dict{Int,Any}()          # dcline → 빈 dict

# 2) Y-bus 계산
am    = PowerModels.calc_admittance_matrix(pm_data)      # AdmittanceMatrix
Bbus  = sparse(imag.(am.matrix))                         # susceptance only

# 3) 온라인 발전기 버스 index 가져오기
status_key(g) = get(g, "status", get(g, "gen_status", 1))  # ← 핵심 수정
gen_buses  = [g["gen_bus"] for g in values(pm_data["gen"]) if status_key(g) == 1]

bus_to_idx = am.bus_to_idx
idx        = [bus_to_idx[b] for b in gen_buses]

# 4) 발전기-전용 B 행렬
B_gen = Bbus[idx, idx]                # sparse (|G|×|G|)

# 5) CSV 저장 (dense → DataFrame)
df = DataFrame(Matrix(B_gen), :auto)      # ← :auto 추가
CSV.write(out_csv, df; writeheader=false)
println("✅  B_gen.csv 저장 완료   size = ", size(B_gen))

