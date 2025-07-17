#!/usr/bin/env python3
# simcore/no_disturbance.py
# -----------------------------------------------------------
# 10-second no-disturbance swing simulation (41 active-gen buses)
# ▸ 평형각 δ* 계산 → Pe(δ*) = Pm
# ▸ δ/ω 궤적, Kuramoto R(t) 그래프, R(t) CSV 저장
# -----------------------------------------------------------
from pathlib import Path
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.ticker as mtick      # ← 파일 맨 위 import 추가


# ─── 파일 경로 ──────────────────────────────────────────────
DATA   = Path("data")
B_CSV  = DATA / "B_41_labeled.csv"
DYN_CSV= DATA / "dyn_params.csv"
CASE_M = Path("./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m")

# ── MATPOWER block() util  (PF angle 읽기 전용) ─────────────
_m = CASE_M.read_text(encoding="utf-8", errors="ignore")
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
def block(key):
    pat=rf"(?:mpc\.)?{key}\s*=\s*\[([\s\S]+?)\];"
    rows=[]
    for ln in re.search(pat,_m,re.I).group(1).splitlines():
        ln=ln.split("%",1)[0].strip().rstrip(";")
        if ln: rows.append([float(x) for x in re.findall(NUM,ln)])
    n=max(map(len,rows)); arr=np.zeros((len(rows),n))
    for i,r in enumerate(rows): arr[i,:len(r)] = r
    return arr

# ── 시스템 데이터 로드 ──────────────────────────────────────
def load_system():
    """B, M⁻¹, D, Pm(총 ΔP=0 교정), δ_guess 반환"""

    # ── 1. B 라플라시안 & 발전기-버스 리스트 ───────────────
    dfB   = pd.read_csv(B_CSV, index_col=0)
    B     = dfB.values
    buses = dfB.index.astype(int)                # 41개 버스

    # ── 2. 관성·댐핑·Pg  (버스 합계) ──────────────────────
    dyn = pd.read_csv(DYN_CSV); dyn["bus"] = dyn["bus"].astype(int)
    agg = (dyn[dyn["bus"].isin(buses)]
           .groupby("bus").agg({"M":"sum", "D":"sum", "Pg":"sum"})
           .reindex(buses))
    if agg.isna().any(axis=None):
        miss = agg[agg.isna().any(1)].index.to_list()
        raise SystemExit(f"dyn_params 누락 버스 → {miss}")

    Mvec  = agg["M"].values
    M_inv = np.diag(1 / Mvec)
    Dmat  = np.diag(agg["D"].values)
    Pm    = agg["Pg"].values.copy()              # (41,)

    # ── 3. PF 각도 θ → δ_guess (41개) ────────────────────
    θ = np.deg2rad(block("bus")[:, 7]); θ -= θ.mean()
    bus_all = block("bus")[:, 0].astype(int)
    δ_guess = θ[np.isin(bus_all, buses)]

    # ── 4. 총 ΔP 보정 :  Σ(Pm - Pe0) = 0 로 맞춤 ─────────
    Pe0 = (B * np.sin(δ_guess[:, None] - δ_guess[None, :])).sum(1)
    Delta_total = (Pm - Pe0).sum()              # 시스템 잔여 전력
    weight = Mvec / Mvec.sum()                  # 관성 비례 분배
    Pm -= weight * Delta_total                  # Pm 교정 완료

    return B, M_inv, Dmat, Pm, δ_guess


# ── 평형각 δ* 찾기 (ref-bus 고정, 나머지 40개 미지수) ───────
def solve_equilibrium(B, Pm, δ0):
    n=len(Pm); ref=0; mask=np.arange(n)!=ref
    def mismatch(x):
        δ = δ0.copy(); δ[mask]=x
        Pe = (B*np.sin(δ[:,None]-δ[None,:])).sum(1)
        return (Pm-Pe)[mask]
    sol = root(mismatch, δ0[mask], method='lm', tol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    δ = δ0.copy(); δ[mask]=sol.x
    δ -= δ.mean()
    return δ


# ── 스윙 RHS ───────────────────────────────────────────────
def rhs(t,y,K,M_inv,D,Pm):
    n=len(Pm); δ,ω = y[:n], y[n:]
    Pe = (K*np.sin(δ[:,None]-δ[None,:])).sum(1)
    return np.r_[ω, M_inv@(Pm-Pe-D@ω)]

def R_order(δ):
    return np.abs(np.exp(1j*δ).mean(axis=0))

# ── 메인 실행 + 결과 저장 ──────────────────────────────────
def run(t_end=10.0, dt=0.01):
    B,M_inv,D,Pm,δ_guess = load_system()
    δ0 = solve_equilibrium(B,Pm,δ_guess)
    n  = len(Pm);  y0 = np.r_[δ0, np.zeros(n)]

    sol = solve_ivp(rhs, (0,t_end), y0,
                    t_eval=np.arange(0,t_end+dt,dt),
                    args=(B,M_inv,D,Pm),
                    method="RK45", rtol=1e-9, atol=1e-11)
    δ,ω = sol.y[:n], sol.y[n:]
    R   = R_order(δ)

    # 궤적 플롯 -------------------------------------------------
    fig,ax=plt.subplots(2,1,sharex=True,figsize=(8,6))
    ax[0].plot(sol.t,δ.T,lw=.6); ax[0].set_ylabel("δ (rad)")
    ax[0].set_title("Angle deviations (no disturbance)")
    ax[1].plot(sol.t,ω.T,lw=.6); ax[1].set_ylabel("ω (rad/s)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(DATA/"no_dist_delta_omega.png",dpi=250)

    # ---- R(t) plot -------------------------------------------------
    figR = plt.figure(figsize=(6,2.4))
    axR  = figR.add_subplot(111)            # ← 축 핸들 얻기
    axR.plot(sol.t, R, lw=1)

    pad = 0.05*max(R.max()-R.min(), 1e-4)
    axR.set_ylim(R.min()-pad, R.max()+pad)

    # ── 새 줄: y축 숫자 형식 고정(소수 둘째 자리) ───────────────
    axR.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    axR.set_ylabel("R(t)")
    axR.set_xlabel("time (s)")
    axR.set_title("Kuramoto Order Parameter")

    figR.tight_layout()
    figR.savefig(DATA/"no_dist_R.png", dpi=250)

    # R(t) CSV 저장
    pd.DataFrame({"t":sol.t,"R":R}
                ).to_csv(DATA/"no_dist_R_timeseries.csv", index=False,
                          float_format="%.10g")

    # 콘솔 지표
    print(f"max|δ|={np.abs(δ).max():.3e} rad")
    print(f"max|ω|={np.abs(ω).max():.3e} rad/s")
    print(f"R min={R.min():.6f}   final={R[-1]:.6f}")

    plt.show()

if __name__=="__main__":
    run()
