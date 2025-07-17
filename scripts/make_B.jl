#!/usr/bin/env julia
# make_B_noPM.jl  ―  PowerModels 없이 Ybus → Kron-reduced B 행렬

using SparseArrays, LinearAlgebra

########################  사용자 설정  ########################
const case_path = raw"./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m"
const out_csv   = raw"./data/B_gen.csv"
################################################################

"----- 1) MATPOWER .m 로드 --------------------------------------------------"

# 아주 단순한 파서 ― MATPOWER 문법 가정 (숫자·;·[]·% 주석)
function read_matpower_tables(fname)
    raw = read(fname, String)

    # 버스
    bus_txt = match(r"mpc\.bus\s*=\s*\[\s*(.*?)\s*\];"ms, raw).captures[1]
    bus_mat = [parse.(Float64, split(strip(l), r"\s+")) for l in split(bus_txt, ';') if !isempty(strip(l))]
    bus     = hcat(bus_mat...)'          # NB×13

    # branch
    br_txt = match(r"mpc\.branch\s*=\s*\[\s*(.*?)\s*\];"ms, raw).captures[1]
    br_mat = [parse.(Float64, split(strip(l), r"\s+")) for l in split(br_txt, ';') if !isempty(strip(l))]
    branch = hcat(br_mat...)'            # NL×17 (우린 앞 9개만 씀)

    # gen (발전기 버스 집합용)
    gen_txt = match(r"mpc\.gen\s*=\s*\[\s*(.*?)\s*\];"ms, raw).captures[1]
    gen_mat = [parse.(Float64, split(strip(l), r"\s+")) for l in split(gen_txt, ';') if !isempty(strip(l))]
    gen     = hcat(gen_mat...)'          # NG×21

    # baseMVA
    base = parse(Float64, match(r"mpc\.baseMVA\s*=\s*(\d+(?:\.\d+)?)", raw).captures[1])

    return bus, branch, gen, base
end

bus, branch, gen, baseMVA = read_matpower_tables(case_path)

NB = size(bus,1)
bus_id = Int.(bus[:,1])
bus_to_idx = Dict(b=>i for (i,b) in enumerate(bus_id))

"----- 2) Y-bus 구성 ---------------------------------------------------------"

row = Int[]; col = Int[]; val = ComplexF64[]
diag = zeros(ComplexF64, NB)

deg2rad(θ) = θ*pi/180

for r in eachrow(branch)
    f = bus_to_idx[Int(r[1])]
    t = bus_to_idx[Int(r[2])]
    status = r[11]         # 0 → skip
    status ≈ 0 && continue

    r_s  = r[3];  x_s = r[4];   b_s = r[5]
    tap  = r[9] ≈ 0 ? 1.0 : r[9]
    ang  = deg2rad(r[10])
    tapc = tap*exp(im*ang)

    y = inv(r_s + im*x_s)
    bsh = im*b_s

    # 대각
    diag[f] += y/abs2(tapc) + bsh/2/abs2(tapc)
    diag[t] += y + bsh/2

    # 오프 대각
    yft = -y/conj(tapc)
    push!(row, f); push!(col, t); push!(val, yft)
    push!(row, t); push!(col, f); push!(val, yft)
end

# 버스 shunt(Gs, Bs)
Gs = bus[:,4] ./ baseMVA
Bs = bus[:,5] ./ baseMVA
diag .+= complex.(Gs, Bs)

# 희소 Ybus
Ybus = sparse([1:NB; row], [1:NB; col], [diag; val], NB, NB)

"----- 3) 발전기 버스 추출 & Kron 축소 --------------------------------------"

gen_buses = unique(sort(Int.(gen[gen[:,8] .== 1, 1])))  # status==1
gen_idx   = [bus_to_idx[b] for b in gen_buses]
load_idx  = setdiff(1:NB, gen_idx)

Ygg = Ybus[gen_idx, gen_idx]
Ygl = Ybus[gen_idx, load_idx]
Ylg = Ybus[load_idx, gen_idx]
Yll = Matrix(Ybus[load_idx, load_idx])   # dense로 변환

Yred = Ygg - Ygl * (Yll \ Ylg)
Bred = imag.(Matrix(Yred))

"----- 4) CSV 저장 -----------------------------------------------------------"

open(out_csv, "w") do io
    println(io, "bus_id," * join(gen_buses, ","))
    for (i,b) in enumerate(gen_buses)
        println(io, string(b) * "," * join(Bred[i,:], ","))
    end
end

println("✅  $(out_csv)  생성 완료  (size = $(size(Bred,1))×$(size(Bred,2)))")
