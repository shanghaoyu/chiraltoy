## a srg solver in momentum space,
## written by hu rongzhe.

import sys
sys.path.append("../deps")
sys.path.append("../lib")
import utility
import profiler
import time
import chiral_potential as chiral_potential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import constants as const
from scipy.integrate import solve_ivp


utility.header_message()

################################################################################################################

utility.section_message("Initialization")

mp = 938.2720
mn = 939.5654
mN = 2 * (mp * mn) / (mp + mn)

potential_type = "n3loem"
potential = chiral_potential.two_nucleon_potential(potential_type)
print(f"potential type : {potential_type}\n")

# SRG flow parameters which you want to get, in fm^(4).
Ns = 4
S_eval_points = np.logspace(-4, -0.8, Ns)


# from fm^(4) to MeV^(-2).
units_factor = mN**2 / const.hbarc**4
S_eval_points = units_factor * S_eval_points

# Total number of integration points Nq~100 would be enough for most cases.
# For Chiral EFT interactions, use q_max 6-8 fm^(-1).
# For standard OBE interactions such as AV18, use q_max 20-30 fm^(-1).
Nq = 100
q_max = 8.0


# we use linear mesh points.
q_min = 1e-8
Delta_q = (q_max - q_min) / Nq * const.hbarc
mesh_q = const.hbarc * np.linspace(q_min, q_max, Nq)
weight_q = np.full(Nq, Delta_q)

J = 1


# np interaction in 3S1-3S1 channel.
def potential_mm_initial(pp: float, p: float):
    ll, l, j, s, tz = J - 1, J - 1, J, 1, 0
    return potential.potential(ll, l, pp, p, j, s, tz)


def potential_mp_initial(pp: float, p: float):
    ll, l, j, s, tz = J - 1, J + 1, J, 1, 0
    return potential.potential(ll, l, pp, p, j, s, tz)


def potential_pm_initial(pp: float, p: float):
    ll, l, j, s, tz = J + 1, J - 1, J, 1, 0
    return potential.potential(ll, l, pp, p, j, s, tz)


def potential_pp_initial(pp: float, p: float):
    ll, l, j, s, tz = J + 1, J + 1, J, 1, 0
    return potential.potential(ll, l, pp, p, j, s, tz)


t1 = time.time()
V0_mm = np.zeros((Nq, Nq))
V0_mp = np.zeros((Nq, Nq))
V0_pm = np.zeros((Nq, Nq))
V0_pp = np.zeros((Nq, Nq))
Tkin = np.zeros((Nq, Nq))
for idx_p, p in enumerate(mesh_q):
    Tkin[idx_p, idx_p] = p**2 / mN
    for idx_pp, pp in enumerate(mesh_q):
        w_p = weight_q[idx_p]
        w_pp = weight_q[idx_pp]
        V0_mm[idx_pp, idx_p] = (
            np.sqrt(w_pp) * pp * potential_mm_initial(pp, p) * np.sqrt(w_p) * p
        )
        V0_mp[idx_pp, idx_p] = (
            np.sqrt(w_pp) * pp * potential_mp_initial(pp, p) * np.sqrt(w_p) * p
        )
        V0_pm[idx_pp, idx_p] = (
            np.sqrt(w_pp) * pp * potential_pm_initial(pp, p) * np.sqrt(w_p) * p
        )
        V0_pp[idx_pp, idx_p] = (
            np.sqrt(w_pp) * pp * potential_pp_initial(pp, p) * np.sqrt(w_p) * p
        )
Tkin_coupled = np.block([[Tkin, np.zeros((Nq, Nq))], [np.zeros((Nq, Nq)), Tkin]])
V0_coupled = np.block([[V0_mm, V0_mp], [V0_pm, V0_pp]])

t2 = time.time()
profiler.add_timing("Initialize V0", t2 - t1)
print("initialize V0 complete.")

################################################################################################################


################################################################################################################
utility.section_message("Solving SRG Flow Equations With Scipy")


# commutator, arguments A and B matrices
def commutator(A, B):
    return np.matmul(A, B) - np.matmul(B, A)


# commutator for coupled channels including reshaping to [Nrows,Nrows] matrices, arguments A and B are 1d arrays
def commutator_reshape_coupled(A: np.ndarray, B: np.ndarray, Nrows: int):
    A_reshape = np.reshape(A, (2 * Nrows, 2 * Nrows))
    B_reshape = np.reshape(B, (2 * Nrows, 2 * Nrows))
    return np.matmul(A_reshape, B_reshape) - np.matmul(B_reshape, A_reshape)


def dVds_coupled(s: float, Vcoupled: np.ndarray, Nrows: int, Tkincoupled: np.ndarray):
    Vcoupled_reshape = np.reshape(Vcoupled, (2 * Nrows, 2 * Nrows))
    ret = np.reshape(
        commutator_reshape_coupled(
            commutator_reshape_coupled(Tkincoupled, Vcoupled_reshape, Nrows),
            Tkincoupled + Vcoupled_reshape,
            Nrows,
        ),
        Vcoupled.size,
    )
    return ret


t3 = time.time()
solution = solve_ivp(
    dVds_coupled,
    [0, 1.1 * S_eval_points.max()],
    V0_coupled.ravel(),
    method="RK23",
    t_eval=S_eval_points,
    args=(Nq, Tkin_coupled),
    rtol=1e-5,
    atol=1e-5,
)
if not solution.success:
    error_message = "The ODE solver failed with error message:\n" + solution.message
    sys.exit(error_message)
t4 = time.time()
profiler.add_timing("SRG Flow", t4 - t3)
print("Solving SRG flow complete.")

################################################################################################################


################################################################################################################
utility.section_message("Plot")

fig, axes = plt.subplots(1, Ns, figsize=(5 * Ns, 5), sharey=True)

for idx_s, s_plot in enumerate(S_eval_points):
    S_to_plot = s_plot / units_factor  # in fm^(4).
    Lambda = str(round(pow(S_to_plot, -1 / 4), 1))

    potential_plot = solution.y[:, idx_s].reshape((2 * Nq, 2 * Nq))

    potential_plot_without_factor = np.zeros_like(V0_coupled)
    for idxpp, pp in enumerate(mesh_q):
        for idxp, p in enumerate(mesh_q):
            w_p = weight_q[idxp]
            w_pp = weight_q[idxpp]
            potential_plot_without_factor[idxpp][idxp] = potential_plot[idxpp][idxp] / (
                np.sqrt(w_pp * w_p) * pp * p
            )
            potential_plot_without_factor[idxpp + Nq][idxp] = potential_plot[
                idxpp + Nq
            ][idxp] / (np.sqrt(w_pp * w_p) * pp * p)
            potential_plot_without_factor[idxpp][idxp + Nq] = potential_plot[idxpp][
                idxp + Nq
            ] / (np.sqrt(w_pp * w_p) * pp * p)
            potential_plot_without_factor[idxpp + Nq][idxp + Nq] = potential_plot[
                idxpp + Nq
            ][idxp + Nq] / (np.sqrt(w_pp * w_p) * pp * p)

    potential_plot_mm = potential_plot_without_factor[:Nq, :Nq]
    potential_plot_mp = potential_plot_without_factor[:Nq, Nq:]
    potential_plot_pm = potential_plot_without_factor[Nq:, :Nq]
    potential_plot_pp = potential_plot_without_factor[Nq:, Nq:]

    mtx = potential_plot_mp
    pp, p = np.meshgrid(mesh_q, mesh_q)

    norm = TwoSlopeNorm(vmin=mtx.min(), vcenter=0, vmax=mtx.max())
    c = axes[idx_s].imshow(
        mtx,
        cmap="RdBu_r",
        interpolation="bicubic",
        extent=(p.min(), p.max(), pp.min(), pp.max()),
        origin="lower",
        norm=norm,
    )
    axes[idx_s].set_xlabel(r"$p$ (MeV)")
    axes[idx_s].set_title(r"$\lambda=$" + Lambda + r"$\;\mathrm{fm}^{-1}$")


axes[0].set_ylabel(r"$p'$ (MeV)")
plt.subplots_adjust(wspace=0, right=1)

cbar = fig.colorbar(c, ax=axes, orientation="vertical", pad=0.05)
cbar.set_label("$V(p',p)\,\mathrm{(MeV^{-2})}$")
fig_name = f"srg-coupled-channel-e1-{potential_type}.png"
plt.savefig(fig_name, dpi=600)
print(f"fig has been saved as: {fig_name}")
plt.show()

################################################################################################################

################################################################################################################

utility.section_message("Timings")

profiler.print_timings()

################################################################################################################

utility.footer_message()
