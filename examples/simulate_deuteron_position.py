# a deuteron solver in position space,
# written by wang xing and hu rongzhe.
# for details, see https://nukephysik101.wordpress.com/2023/02/27/deuteron-wave-function-using-av18/

import sys

sys.path.append("../lib")
import utility
import profiler
import time
import chiral_potential as chiral_potential
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache


utility.header_message()

################################################################################################################

utility.section_message("Initialization")

# this is a chiral interaction in position space.
potential_type = "em500"
print(f"potential type : {potential_type}\n")

# initialize an object for the chiral interaction.
potential = chiral_potential.two_nucleon_potential(potential_type)

################################################################################################################


################################################################################################################

utility.section_message("Setting Parameters")

# defines necessary things.

# some constants.
Mp = 938.2720
Mn = 939.5654
mu = Mp * Mn / (Mp + Mn)
Mslash = (Mp + Mn) / 2
Mhat = 2 * mu
deltaM = Mn - Mp
hc = 197.32698

# r meshes.
rmin = 1e-16
rmax = 40
Nr = 800
r_meshes = np.linspace(rmin, rmax, Nr)

print(f"    Proton Mass       : " + "%3.4f" % Mp + " MeV")
print(f"    Neutron Mass      : " + "%3.4f" % Mn + " MeV")
print(f"    Constant hc       : " + "%3.5f" % hc + " MeV fm")
print(f"    r min             : " + "%.0e" % rmin + " fm")
print(f"    r max             : " + "%4.1f" % rmax + " fm")
print(f"    r mesh numbers    : " + "%d" % Nr)


# here defines 4 potentials for deuteron, r in fm and V in MeV.
@lru_cache(maxsize=None)
def V00(r):
    ll, l, s, j, tz = 0, 0, 1, 1, 0
    return potential.potential_local(ll, l, s, j, tz, r)


@lru_cache(maxsize=None)
def V02(r):
    ll, l, s, j, tz = 0, 2, 1, 1, 0
    return potential.potential_local(ll, l, s, j, tz, r)


@lru_cache(maxsize=None)
def V20(r):
    ll, l, s, j, tz = 2, 0, 1, 1, 0
    return potential.potential_local(ll, l, s, j, tz, r)


@lru_cache(maxsize=None)
def V22(r):
    ll, l, s, j, tz = 2, 2, 1, 1, 0
    return potential.potential_local(ll, l, s, j, tz, r)


# Numerov method.
@lru_cache(maxsize=None)
def solve_schrodinger(E, pri1, pri2):
    h = r_meshes[1] - r_meshes[0]
    l = len(r_meshes)
    u = np.zeros(l)
    w = np.zeros(l)

    u[0] = 0
    w[0] = 0
    u[1] = pri1
    w[1] = pri2

    for i in range(1, len(r_meshes) - 1):
        rf = r_meshes[i + 1]
        r = r_meshes[i]
        ri = r_meshes[i - 1]
        t1 = time.time()
        V00ri = V00(ri)
        V00rf = V00(rf)
        V00r = V00(r)
        V22ri = V22(ri)
        V22rf = V22(rf)
        V22r = V22(r)
        V02ri = V02(ri)
        V02rf = V02(rf)
        V02r = V02(r)
        V20ri = V20(ri)
        V20rf = V20(rf)
        V20r = V20(r)
        t2 = time.time()
        profiler.add_timing("Cal V(r)", t2 - t1)

        cu1 = 1 + h**2 / 6 * mu * (E - V00rf) / hc**2
        cw2 = 1 + h**2 / 6 * mu * (E - V22rf) / hc**2 - h**2 / 2 / rf**2
        a1 = (
            -2 * u[i] * (1 - 5 / 6 * h**2 * mu * (E - V00r) / hc**2)
            + u[i - 1] * (1 + h**2 / 6 * mu * (E - V00ri) / hc**2)
            - h**2 / 6 * mu * (10 * V02r * w[i] + V02ri * w[i - 1]) / hc**2
        )
        a2 = (
            -2
            * w[i]
            * (1 - 5 / 6 * h**2 * mu * (E - V22r) / hc**2 + 2.5 * h**2 / r**2)
            + w[i - 1]
            * (1 + h**2 / 6 * mu * (E - V22ri) / hc**2 - h**2 / 2 / ri**2)
            - h**2 / 6 * mu * (10 * V20r * u[i] + V20ri * u[i - 1]) / hc**2
        )
        cw1 = h**2 / 6 * mu * V02rf / hc**2
        cu2 = h**2 * mu / 6 * V20rf / hc**2
        u[i + 1] = (-a1 - cw1 / cw2 * a2) / (cu1 - cw1 * cu2 / cw2)
        w[i + 1] = (-a2 - cu2 / cu1 * a1) / (cw2 - cu2 * cw1 / cu1)
        t3 = time.time()
        profiler.add_timing("Numerov", t3 - t2)

    u_p = (u[-1] - u[-2]) / h
    w_p = (w[-1] - w[-2]) / h

    return u, w, u_p, w_p


@lru_cache(maxsize=None)
def u_asymptotic(E, r):
    gamma = 0.2315380  # accurate value, in fm^(-1).
    # Bd = np.abs(E)  # binding energy
    # gamma = np.sqrt(2 * mu * Bd) / hc
    u = np.exp(-gamma * r)
    up = -gamma * np.exp(-gamma * r)
    return u, up


@lru_cache(maxsize=None)
def w_asymptotic(E, r):
    gamma = 0.2315380  # accurate value, in fm^(-1).
    # Bd = np.abs(E)  # binding energy
    # gamma = np.sqrt(2 * mu * Bd) / hc
    w = np.exp(-gamma * r) * (1 + 3 / gamma / r + 3 / (gamma * r) ** 2)
    wp = np.exp(-gamma * r) * (-gamma + 3 / r - 6 / gamma**2 / r**3)
    return w, wp


@lru_cache(maxsize=None)
def solve_detA(E):
    t1 = time.time()
    A = np.zeros((4, 4))
    u1, w1, u_p1, w_p1 = solve_schrodinger(E, 0.1, 1)
    u2, w2, u_p2, w_p2 = solve_schrodinger(E, 1, 0.1)
    rf = r_meshes[-1]
    u, u_p = u_asymptotic(E, rf)
    w, w_p = w_asymptotic(E, rf)
    A[0][0], A[0][1], A[0][2] = u1[-1], u2[-1], u
    A[1][0], A[1][1], A[1][3] = w1[-1], w2[-1], w
    A[2][0], A[2][1], A[2][2] = u_p1, u_p2, u_p
    A[3][0], A[3][1], A[3][3] = w_p1, w_p2, w_p
    detaA = np.linalg.det(A)
    t2 = time.time()
    profiler.add_timing("Cal det(A)", t2 - t1)
    return detaA


def draw_detA(Emin, Emax, n):
    list_E = np.linspace(Emin, Emax, n)
    list_detA = [solve_detA(E) for E in list_E]
    plt.figure(figsize=(8, 6), dpi=160)
    plt.scatter(list_E, list_detA)
    plt.xlabel(r"$E\;(\mathrm{MeV})$")
    plt.ylabel(r"$\mathrm{det}(A)$")
    plt.show()


iter = 1


# solve energy in [El, Er] using bisection with a precision.
def solve_E(El, Er, precision, verbose=True, Maxsteps=100):
    global iter
    Emid = (El + Er) / 2
    Al = solve_detA(El)
    Ar = solve_detA(Er)
    if verbose:
        print(
            "iteration step " + str(iter) + ":    ",
            "( Eleft , Eright ) = " + "( " + "%2.7f" % El + " , " + "%2.7f" % Er + " )",
            "    delta = " + "%2.7f" % abs(El - Er),
        )
    if Al * Ar > 0:
        exit(f"no answer for bisec in [{El},{Er}] !")
    if iter >= Maxsteps:
        print(f"\nnot converged after {iter} steps!\nreturn results of the last step.")
        return Emid
    if abs(El - Er) < precision:
        print(f"\nconverged after {iter} steps with precision " + "%2.7f" % precision)
        return Emid
    Am = solve_detA(Emid)
    iter = iter + 1
    if Am * Al < 0:
        return solve_E(El, Emid, precision)
    else:
        return solve_E(Emid, Er, precision)


# solve wave functions.
def solve_phi(E):
    u1, w1, u_p1, w_p1 = solve_schrodinger(E, 0.1, 1)
    u2, w2, u_p2, w_p2 = solve_schrodinger(E, 1, 0.1)
    a = -w2[-1] / w1[-1]
    uf = a * u1 + u2
    wf = a * w1 + w2

    norm = np.sqrt(np.trapz(uf**2 + wf**2, r_meshes))
    uf /= norm
    wf /= norm

    return uf, wf


################################################################################################################


################################################################################################################
utility.section_message("Solving Deuteron by Bisection")

Eleft, Eright = -5, -0
precision = 1e-7

# draw_detA(Eleft, Eright, 20)


E0 = solve_E(Eleft, Eright, precision)

################################################################################################################


################################################################################################################
utility.section_message("Calculating Deuteron Observables")

t1 = time.time()

print(f"    E = " + "%2.6f" % E0 + " MeV")

ur, wr = solve_phi(E0)
psi_norm = np.sqrt(np.trapz(ur**2 + wr**2, r_meshes))
print(f"    psi_norm = " + "%2.6f" % psi_norm)

r2 = np.trapz((r_meshes / 2) ** 2 * (ur**2 + wr**2), r_meshes)
print(f"    r = " + "%2.3f" % np.sqrt(r2) + " fm")
Pd = np.trapz((wr**2), r_meshes)
print(f"    Pd = " + "%2.2f" % (100 * Pd) + " %")
Q = 1 / 20.0 * np.trapz(r_meshes**2 * wr * (np.sqrt(8) * ur - wr), r_meshes)
print(f"    Q = " + "%2.3f" % Q + " fm^2")

rindex = 100
As = ur[rindex] / u_asymptotic(E0, r_meshes[rindex])[0]
Ad = wr[rindex] / w_asymptotic(E0, r_meshes[rindex])[0]
eta = Ad / As
print(f"    As = " + "%2.4f" % As + " fm^(-1/2)")
print(f"    Ad = " + "%2.4f" % Ad + " fm^(-1/2)")
print(f"    eta = " + "%2.4f" % eta)

t2 = time.time()
profiler.add_timing("Cal Observables", t2 - t1)
################################################################################################################


################################################################################################################
utility.section_message("Plotting Wave Functions")

plt.figure(dpi=240)
plt.plot(r_meshes, ur/r_meshes, color="C0", label=r"$\Psi_{0}$")
plt.plot(r_meshes, wr/r_meshes, color="C1", label=r"$\Psi_{2}$")
plt.legend()
plt.xlim(0.1,25)
plt.xlabel(r"$r\;(\mathrm{fm})$")
plt.ylabel(r"$\mathrm{Deuteron\;Wave\;Functions}\;(\mathrm{fm^{-1/2}})$")
plt.savefig('./result/deuteron_phir.png',dpi=512)
plt.show()

################################################################################################################

utility.section_message("Timings")

profiler.print_timings()

################################################################################################################

utility.footer_message()
