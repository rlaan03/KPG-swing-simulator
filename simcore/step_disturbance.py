#!/usr/bin/env python3
# simcore/step_disturbance.py
# -----------------------------------------------------------
# Swing simulation with *multiple* step disturbances
#   • no_disturbance.py 로드 로직 + 평형각 δ* 그대로 사용
#   • (bus_id, t_step, ΔPm) 목록을 한 번에 지정할 수 있음
#   • ω(t) 그래프, Kuramoto R(t), RoCoF 요약, 시계열 CSV 저장
# -----------------------------------------------------------
from pathlib import Path
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.ticker as mtick
TWO_PI = 2 * np.pi


# ── 시나리오 파라미터 ────────────────────────────────────────
disturbances = [                 # (bus_id,  t_step[s],  ΔPm[pu])
    (72,  5.0, -59.0),          # 5 s  : 163번 발전기 출력 –0.50 pu
    # (126,  7.5, +0.30),          # 7.5 s: 126번 발전기 출력 +0.30 pu
    # (139, 15.0, +0.20),          # 15 s : 139번 발전기 출력 –0.20 pu
]                                # 원하는 만큼 추가
tot_time = 21.0   # [s] 시뮬 길이
dt       = 0.3   # [s] 출력 간격

# ── 공통 경로 / MATPOWER 파서 ────────────────────────────────
DATA    = Path("data")
B_CSV   = DATA / "B_41_labeled.csv"
DYN_CSV = DATA / "dyn_params.csv"
CASE_M  = Path("./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m")

_m   = CASE_M.read_text(encoding="utf-8", errors="ignore")
NUM  = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
def block(key):
    pat=rf"(?:mpc\.)?{key}\s*=\s*\[([\s\S]+?)\];"
    rows=[]
    for ln in re.search(pat,_m,re.I).group(1).splitlines():
        ln=ln.split("%",1)[0].strip().rstrip(";")
        if ln: rows.append([float(x) for x in re.findall(NUM,ln)])
    n=max(map(len,rows)); arr=np.zeros((len(rows),n))
    for i,r in enumerate(rows): arr[i,:len(r)] = r
    return arr

# ── 시스템 로드 ─────────────────────────────────────────────
def load_system():
    dfB   = pd.read_csv(B_CSV, index_col=0)
    B     = dfB.values
    buses = dfB.index.astype(int)          # (41,)

    dyn = pd.read_csv(DYN_CSV); dyn["bus"]=dyn["bus"].astype(int)
    agg = (dyn[dyn["bus"].isin(buses)]
           .groupby("bus").agg({"M":"sum","D":"sum","Pg":"sum"})
           ).reindex(buses)

    Mvec = agg["M"].values
    Dvec = agg["D"].values
    Pm   = agg["Pg"].values.copy()

    θ = np.deg2rad(block("bus")[:,7]); θ -= θ.mean()
    δ_guess = θ[np.isin(block("bus")[:,0].astype(int), buses)]

    # 총 ΔPΣ = 0 보정
    Pe0 = (B*np.sin(δ_guess[:,None]-δ_guess[None,:])).sum(1)
    Pm -= (Pm-Pe0).sum()*Mvec/Mvec.sum()

    return B, np.diag(1/Mvec), np.diag(Dvec), Pm, δ_guess, buses

# ── 평형각 δ* ───────────────────────────────────────────────
def solve_equilibrium(B,Pm,δ0):
    n=len(Pm); ref=0; mask=np.arange(n)!=ref
    def F(x):
        δ=δ0.copy(); δ[mask]=x
        Pe=(B*np.sin(δ[:,None]-δ[None,:])).sum(1)
        return (Pm-Pe)[mask]
    sol=root(F,δ0[mask],method='lm',tol=1e-12)
    δ=δ0.copy(); δ[mask]=sol.x; δ-=δ.mean()
    return δ

# ── Pm(t) : 여러 버스·각기 다른 시각 스텝 ──────────────────
def Pm_time(t, Pm0, events):
    """
    events: list of (idx, t_step, dPm)
    반환 shape: (n,)  또는 (n,T)
    """
    if np.isscalar(t):
        out = Pm0.copy()
        for idx, ts, dp in events:
            if t >= ts: out[idx] += dp
        return out
    base = np.repeat(Pm0[:,None], t.size, axis=1)
    for idx, ts, dp in events:
        base[idx, t >= ts] += dp
    return base

# ── 스윙 RHS ───────────────────────────────────────────────
def rhs(t,y,K,M_inv,D,Pm0,events):
    n=len(Pm0); δ,ω = y[:n], y[n:]
    Pe = (K*np.sin(δ[:,None]-δ[None,:])).sum(1)
    Pm = Pm_time(t, Pm0, events)
    return np.r_[ω, M_inv@(Pm-Pe-D@ω)]

def R_order(δ): return np.abs(np.exp(1j*δ).mean(axis=0))
def RoCoF(ω):   return np.gradient(ω, dt, axis=1)

# ── 메인 실행 ──────────────────────────────────────────────
def run():
    B,M_inv,D,Pm0,δ_guess,buses = load_system()
    δ0 = solve_equilibrium(B,Pm0,δ_guess)
    n  = len(Pm0)
    y0 = np.r_[δ0, np.zeros(n)]

    # disturbances → (idx, t_step, dPm)
    events=[]
    for bid, ts, dp in disturbances:
        if bid not in buses:
            raise ValueError(f"bus_id {bid} not in 41-bus set {list(buses)}")
        events.append( (np.where(buses==bid)[0][0], ts, dp) )

    sol = solve_ivp(
        rhs, (0, tot_time), y0,
        t_eval=np.arange(0, tot_time+dt, dt),
        args=(B,M_inv,D,Pm0,events),
        method='RK45',rtol=1e-9,atol=1e-11
    )

    δ, ω = sol.y[:n], sol.y[n:]
    f_Hz = ω / TWO_PI            # (n, T)  rad/s → Hz
    R    = R_order(δ)
    roc  = RoCoF(ω)
    roc_Hz = roc / TWO_PI        # rad/s²  →  Hz/s


    # δ(t) & ω(t) 2-subplot  ────────────────────────────────────
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(sol.t, δ.T, lw=0.6)
    ax[0].set_ylabel("δ (rad)")
    ax[0].set_title(f"δ/ω  multiple steps: {disturbances}")

    ax[1].plot(sol.t, ω.T, lw=0.6)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("ω (rad/s)")

    fig.tight_layout()
    fig.savefig(DATA/"multi_step_delta_omega.png", dpi=300)


    # # ω(t) → Δf(t) plot  --------------------------------------
    # fig, ax = plt.subplots(figsize=(8, 4))
    # ax.plot(sol.t, f_Hz.T, lw=0.7)
    # ax.set_title(f"Frequency deviation Δf(t)  multiple steps: {disturbances}")
    # ax.set_xlabel("time (s)")
    # ax.set_ylabel("Δf (Hz)")
    # fig.tight_layout()
    # fig.savefig(DATA/"multi_step_freq.png", dpi=300)


    # R(t) plot
    figR, axR = plt.subplots(figsize=(6,2.4))
    axR.plot(sol.t, R, lw=1)
    pad = 0.05*max(R.ptp(),1e-4)
    axR.set_ylim(R.min()-pad, R.max()+pad)
    axR.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    axR.set_ylabel("R(t)"); axR.set_xlabel("time (s)")
    axR.set_title("Kuramoto Order Parameter")
    figR.tight_layout(); figR.savefig(DATA/"multi_step_R.png", dpi=300)
    plt.show()

    # 콘솔 요약
    print(f"Max |RoCoF| : {np.max(np.abs(roc_Hz)):.3e} Hz/s²")
    print(f"Final R(t_end): {R[-1]:.6f}")

    # R(t) CSV
    pd.DataFrame({"t":sol.t,"R":R}).to_csv(
        DATA/"multi_step_R_timeseries.csv", index=False, float_format="%.10g"
    )

if __name__ == "__main__":
    run()
