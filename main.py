import matplotlib.pyplot as plt
from simcore import simulate_swing

sol, n_gen = simulate_swing(t_end=0.5)
delta = sol.y[:n_gen, :]
omega = sol.y[n_gen:, :]

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i in range(n_gen):
    axs[0].plot(sol.t, delta[i], label=f"Gen{i+1}")
    axs[1].plot(sol.t, omega[i], label=f"Gen{i+1}")
axs[0].set_ylabel("Rotor Angle δ [rad]")
axs[1].set_ylabel("Rotor Speed ω [rad/s]")
axs[1].set_xlabel("Time [s]")
axs[0].set_title("Swing Simulation: Rotor Angles")
axs[1].set_title("Swing Simulation: Rotor Speeds")
# axs[0].legend(fontsize=8, ncol=4, loc="upper right")
plt.tight_layout()
plt.show()
