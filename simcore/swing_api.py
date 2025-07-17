# simcore/swing_api.py
"""
Swing-Equation step-disturbance API
* 순수 계산 모듈 ― Streamlit 등 UI 코드 금지
"""
from typing import List, Tuple, Dict, Any
import numpy as np
from .step_disturbance import (
    load_system, solve_equilibrium, rhs,
    R_order, RoCoF, TWO_PI, dt as default_dt
)
from scipy.integrate import solve_ivp


def run_step_disturbance(
    events: List[Tuple[int, float, float]],   # (bus_id, t_step, ΔPm)
    tot_time: float = 21.0,
    dt: float = default_dt
) -> Dict[str, Any]:
    """Return dict with t, freq, R, roc, buses."""
    B, M_inv, D, Pm0, δ_guess, buses = load_system()
    δ0 = solve_equilibrium(B, Pm0, δ_guess)
    n  = len(Pm0)
    y0 = np.concatenate([δ0, np.zeros(n)])

    # bus_id → index
    idx_events = []
    for bid, ts, dp in events:
        try:
            idx = np.where(buses == bid)[0][0]
        except IndexError:
            raise ValueError(f"bus_id {bid} not in reduced system")
        idx_events.append((idx, ts, dp))

    # t_eval: 끝값 정확히 tot_time 포함
    steps  = int(round(tot_time / dt))
    t_eval = np.linspace(0, tot_time, steps + 1)

    sol = solve_ivp(
        rhs, (0, tot_time), y0,
        t_eval=t_eval,
        args=(B, M_inv, D, Pm0, idx_events),
        method="RK45", rtol=1e-9, atol=1e-11
    )

    δ, ω = sol.y[:n], sol.y[n:]
    f_Hz = ω / TWO_PI
    R    = R_order(δ)
    roc  = RoCoF(ω) / TWO_PI

    return {"t": sol.t, "freq": f_Hz, "R": R, "roc": roc, "buses": buses}
