using PowerModels, Ipopt
using DataFrames, CSV

# 1) 케이스 읽고 PF
case_path = "./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m"
case   = PowerModels.parse_file(case_path)
result = PowerModels.solve_ac_pf(case, Ipopt.Optimizer)

bus_sol = result["solution"]["bus"]
gen_sol = result["solution"]["gen"]

###############
# 2) 발전기 DF
###############
gen_ids = collect(keys(gen_sol))         # "32", "33", …
gen_df  = DataFrame(
    gen_id = gen_ids,
    bus    = [Int(case["gen"][gid]["gen_bus"]) for gid in gen_ids],
    Pg     = [gen_sol[gid]["pg"]               for gid in gen_ids],
    Qg     = [gen_sol[gid]["qg"]               for gid in gen_ids],
)

############
# 3) 버스 DF
############
# 버스 솔루션 안에 실제 필드명을 한 번 점검해 보세요:
# @show keys(first(values(bus_sol)))  # 보통 "vm", "va" 일 것

bus_ids = collect(keys(bus_sol))
bus_df  = DataFrame(
    bus_id = bus_ids,
    Vm     = [bus_sol[bid]["vm"] for bid in bus_ids],
    Va     = [bus_sol[bid]["va"] for bid in bus_ids],
)

###############
# 4) CSV 저장
###############
mkpath("./pf")                     # 폴더가 없으면 생성
CSV.write("./pf/gen_result.csv", gen_df)
CSV.write("./pf/bus_result.csv", bus_df)

println("✅  gen_result.csv / bus_result.csv 저장 완료")
