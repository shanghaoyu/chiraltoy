import constants as const
import numpy as np
import WignerSymbol as ws
import scipy
import sys
import os
import time
import utility
import profiler


class nn_studio:
    def __init__(self, jmin, jmax, tzmin, tzmax, Np=75, mesh_type="gauleg_infinite"):
        self.jmin = jmin
        self.jmax = jmax
        self.tzmin = tzmin
        self.tzmax = tzmax

        self.basis = self.setup_NN_basis()
        self.channels = self.setup_NN_channels()

        self.Np = Np
        if mesh_type == "gauleg_infinite":
            self.pmesh, self.wmesh = self.gauss_legendre_inf_mesh()
        elif mesh_type == "gauleg_finite":
            self.pmesh, self.wmesh = self.gauss_legendre_line_mesh(1e-16, 1200)

        self.Tlabs = None

        # potential
        self.V = None

        # Tmatrices
        self.Tmtx = []

        # phase shifts
        self.phase_shifts = []

        # theta in degree
        self.theta = np.linspace(1, 180, 180)

        # spin matrixs
        self.m11 = []
        self.m10 = []
        self.mpm = []
        self.m01 = []
        self.m00 = []
        self.mss = []

        # spin observables
        self.spin_obs = {}

    def setup_NN_basis(self):
        basis = []
        for tz in range(self.tzmin, self.tzmax + 1, 1):
            for J in range(self.jmin, self.jmax + 1, 1):
                for S in range(0, 2, 1):
                    for L in range(abs(J - S), J + S + 1, 1):
                        for T in range(abs(tz), 2, 1):
                            if (L + S + T) % 2 != 0:
                                basis_state = {}
                                basis_state["tz"] = tz
                                basis_state["l"] = L
                                basis_state["pi"] = (-1) ** L
                                basis_state["s"] = S
                                basis_state["j"] = J
                                basis_state["t"] = T
                                basis.append(basis_state)
        return basis

    def setup_NN_channels(self):
        from itertools import groupby
        from operator import itemgetter

        states = []

        for bra in self.basis:
            for ket in self.basis:
                if self.kroenecker_delta(bra, ket, "j", "tz", "s", "pi"):
                    state = {}

                    state["l"] = bra["l"]
                    state["ll"] = ket["l"]

                    state["s"] = bra["s"]
                    state["j"] = bra["j"]
                    state["t"] = bra["t"]
                    state["tz"] = bra["tz"]
                    state["pi"] = bra["pi"]
                    states.append(state)

        grouper = itemgetter("s", "j", "tz", "pi")
        NN_channels = []

        for key, grp in groupby(sorted(states, key=grouper), grouper):
            NN_channels.append(list(grp))

        for chn_idx, chn in enumerate(NN_channels):
            for block in chn:
                block.update({"chn_idx": chn_idx})

        return NN_channels

    def lookup_channel_idx(self, **kwargs):
        matching_indices = []
        channels = []
        for idx, chn in enumerate(self.channels):
            for block in chn:
                # this will return a channel if any of the partial-wave couplings of a block match
                if kwargs.items() <= block.items():
                    matching_indices.append(idx)
                    channels.append(chn)

        matching_indices = list(dict.fromkeys(matching_indices))

        return matching_indices, channels

    def linear_mesh(self):
        return np.linspace(1e-6, 650, self.Np)

    def gauss_legendre_line_mesh(self, a, b):
        x, w = np.polynomial.legendre.leggauss(self.Np)
        # Translate x values from the interval [-1, 1] to [a, b]
        t = 0.5 * (x + 1) * (b - a) + a
        u = w * 0.5 * (b - a)

        return t, u

    def gauss_legendre_inf_mesh(self):
        scale = 100.0

        x, w = np.polynomial.legendre.leggauss(self.Np)

        # Translate x values from the interval [-1, 1] to [0, inf)
        pi_over_4 = np.pi / 4.0

        t = scale * np.tan(pi_over_4 * (x + 1.0))
        u = scale * pi_over_4 / np.cos(pi_over_4 * (x + 1.0)) ** 2 * w

        return t, u

    @staticmethod
    # a static method is bound to a class rather than the objects for that class
    def triag(a, b, ab):
        if ab < abs(a - b):
            return False
        if ab > a + b:
            return False
        return True

    @staticmethod
    # a static method is bound to a class rather than the objects for that class
    def kroenecker_delta(bra, ket, *args):
        for ar in args:
            if bra[ar] != ket[ar]:
                return False
        return True

    def lab2rel(self, Tlab, tz):
        if tz == -1:
            mu = const.Mp / 2
            ko2 = 0.5 * const.Mp * Tlab  #! I correct the factor from 2 to 0.5.
        elif tz == 0:
            mu = const.Mp * const.Mn / (const.Mp + const.Mn)
            ko2 = (
                const.Mp**2
                * Tlab
                * (Tlab + 2 * const.Mn)
                / ((const.Mp + const.Mn) ** 2 + 2 * Tlab * const.Mp)
            )
        elif tz == +1:
            mu = const.Mn / 2
            ko2 = 0.5 * const.Mn * Tlab  #! I correct the factor from 2 to 0.5.
        else:
            exit("unknown isospin projection")

        if ko2 < 0:
            ko = np.complex(0, np.sqrt(np.abs(ko2)))
        else:
            ko = np.sqrt(ko2)

        return ko, mu

    @staticmethod
    # a static method is bound to a class rather than the objects for that class
    # note that this idx convention is different with Machleidt's codes.
    def map_to_coup_idx(ll, l, s, j):
        if l == ll:
            if l < j:
                # --
                coup = True
                idx = 3
            elif l > j:
                # ++
                coup = True
                idx = 2
            else:
                if s == 1:
                    coup = False
                    idx = 1
                else:
                    coup = False
                    idx = 0
        else:
            if l < j:
                # -+
                coup = True
                idx = 5
            else:
                # +-
                coup = True
                idx = 4

        return coup, idx

    def Vmtx(self, this_mesh, ll, l, s, j, t, tz):
        mtx = np.zeros((len(this_mesh), len(this_mesh)))
        for pidx, p in enumerate(this_mesh):
            for ppidx, pp in enumerate(this_mesh):
                mtx[ppidx][pidx] = self.V.potential(ll, l, pp, p, j, s, tz)
        return np.array(mtx)

    def setup_Vmtx(self, this_channel, ko=False):
        if ko == False:
            this_mesh = self.pmesh
        else:
            this_mesh = np.hstack((self.pmesh, ko))
        m = []

        for idx, block in enumerate(this_channel):
            l = block["l"]
            ll = block["ll"]
            s = block["s"]
            j = block["j"]
            t = block["t"]
            tz = block["tz"]

            mtx = np.copy(self.Vmtx(this_mesh, ll, l, s, j, t, tz))

            m.append(mtx)

        if len(this_channel) > 1:
            # note that m[1], m[2] show be switched compared with original code.
            V = np.copy(np.vstack((np.hstack((m[0], m[2])), np.hstack((m[1], m[3])))))
        else:
            V = np.copy(m[0])
        return V, m

    def setup_G0_vector(self, ko, mu):
        G = np.zeros((2 * self.Np + 2), dtype=complex)

        # note that we index from zero, and the N+1 point is at self.Np
        G[0 : self.Np] = (
            self.wmesh * self.pmesh**2 / (ko**2 - self.pmesh**2)
        )  # Gaussian integral

        # print('   G0 pole subtraction')
        G[self.Np] = (
            -np.sum(self.wmesh / (ko**2 - self.pmesh**2)) * ko**2
        )  # 'Principal value'
        G[self.Np] -= 1j * ko * (np.pi / 2)

        # python vec[0:n] is the first n elements, i.e., 0,1,2,3,...,n-1
        G[self.Np + 1 : 2 * self.Np + 2] = G[0 : self.Np + 1]
        return G * 2 * mu

    def setup_GV_kernel(self, channel, Vmtx, ko, mu):
        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks) * (self.Np + 1))
        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko, mu)

        g = np.copy(G0[0:Np_chn])
        GV = np.zeros((len(g), len(g)), dtype=complex)

        for g_idx, g_elem in enumerate(g):
            GV[g_idx, :] = g_elem * Vmtx[g_idx, :]

        return GV

    def setup_VG_kernel(self, channel, Vmtx, ko, mu):
        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks) * (self.Np + 1))

        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko, mu)
        g = np.copy(G0[0:Np_chn])
        VG = np.zeros((len(g), len(g)), dtype=complex)

        for g_idx, g_elem in enumerate(g):
            VG[:, g_idx] = g_elem * Vmtx[:, g_idx]

        return VG

    def solve_lippmann_schwinger(self, channel, Vmtx, ko, mu):
        # matrix inversion:
        # T = V + VGT
        # (1-VG)T = V
        # T = (1-VG)^{-1}V

        VG = self.setup_VG_kernel(channel, Vmtx, ko, mu)
        VG = np.eye(VG.shape[0]) - VG
        # golden rule of linear algebra: avoid matrix inversion if you can
        # T = np.matmul(np.linalg.inv(VG),Vmtx)
        T = np.linalg.solve(VG, Vmtx)

        return T

    @staticmethod
    # a static method is bound to a class rather than the objects for that class
    def compute_phase_shifts(ko, mu, on_shell_T):
        rad2deg = 180.0 / np.pi

        fac = np.pi * mu * ko

        if len(on_shell_T) == 3:
            T11 = on_shell_T[0]
            T12 = on_shell_T[1]
            T22 = on_shell_T[2]

            # Blatt-Biedenharn (BB) convention
            twoEpsilonJ_BB = np.arctan(2 * T12 / (T11 - T22))  # mixing parameter
            delta_plus_BB = (
                -0.5
                * 1j
                * np.log(
                    1
                    - 1j * fac * (T11 + T22)
                    + 1j * fac * (2 * T12) / np.sin(twoEpsilonJ_BB)
                )
            )
            delta_minus_BB = (
                -0.5
                * 1j
                * np.log(
                    1
                    - 1j * fac * (T11 + T22)
                    - 1j * fac * (2 * T12) / np.sin(twoEpsilonJ_BB)
                )
            )

            # this version has a numerical instability that I should fix.
            # Stapp convention (bar-phase shifts) in terms of Blatt-Biedenharn convention
            # twoEpsilonJ = np.arcsin(
            #     np.sin(twoEpsilonJ_BB) * np.sin(delta_minus_BB - delta_plus_BB)
            # )  # mixing parameter
            # delta_minus = 0.5 * (
            #     delta_plus_BB
            #     + delta_minus_BB
            #     + np.arcsin(np.tan(twoEpsilonJ) / np.tan(twoEpsilonJ_BB))
            # )
            # delta_plus = 0.5 * (
            #     delta_plus_BB
            #     + delta_minus_BB
            #     - np.arcsin(np.tan(twoEpsilonJ) / np.tan(twoEpsilonJ_BB))
            # )
            # epsilon = 0.5 * twoEpsilonJ

            # numerially stable conversion
            cos2e = np.cos(twoEpsilonJ_BB / 2) * np.cos(twoEpsilonJ_BB / 2)
            cos_2dp = np.cos(2 * delta_plus_BB)
            cos_2dm = np.cos(2 * delta_minus_BB)
            sin_2dp = np.sin(2 * delta_plus_BB)
            sin_2dm = np.sin(2 * delta_minus_BB)

            aR = np.real(cos2e * cos_2dm + (1 - cos2e) * cos_2dp)
            aI = np.real(cos2e * sin_2dm + (1 - cos2e) * sin_2dp)
            delta_minus = 0.5 * np.arctan2(aI, aR)

            aR = np.real(cos2e * cos_2dp + (1 - cos2e) * cos_2dm)
            aI = np.real(cos2e * sin_2dp + (1 - cos2e) * sin_2dm)
            delta_plus = 0.5 * np.arctan2(aI, aR)

            tmp = 0.5 * np.sin(twoEpsilonJ_BB)
            aR = tmp * (cos_2dm - cos_2dp)
            aI = tmp * (sin_2dm - sin_2dp)
            tmp = delta_plus + delta_minus
            epsilon = 0.5 * np.arcsin(aI * np.cos(tmp) - aR * np.sin(tmp))

            if ko < 150:
                if delta_minus * rad2deg < 0:
                    delta_minus += np.pi
                    epsilon *= -1.0
            return [
                np.real(delta_minus * rad2deg),
                np.real(delta_plus * rad2deg),
                np.real(epsilon * rad2deg),
            ]

        else:
            # uncoupled
            T = on_shell_T[0]
            Z = 1 - fac * 2j * T
            # S=exp(2i*delta)
            delta = (-0.5 * 1j) * np.log(Z)

            return np.real(delta * rad2deg)

    def compute_Tmtx(self, channels, verbose=False):
        if verbose:
            print(f"computing T-matrices for")

        self.Tmtx = []
        self.phase_shifts = []

        for idx, channel in enumerate(channels):
            if verbose:
                print(f"channel = {channel}")

            phase_shifts_for_this_channel = []

            nof_blocks = len(channel)

            for Tlab in self.Tlabs:
                if verbose:
                    print(f"Tlab = {Tlab} MeV")

                ko, mu = self.lab2rel(Tlab, channel[0]["tz"])
                t1 = time.time()
                Vmtx = self.setup_Vmtx(channel, ko)[
                    0
                ]  # get only V, not the list of submatrices
                t2 = time.time()
                profiler.add_timing("Setup V Matrix", t2 - t1)
                this_T = self.solve_lippmann_schwinger(channel, Vmtx, ko, mu)
                t3 = time.time()
                profiler.add_timing("Solve LS", t3 - t2)
                self.Tmtx.append(this_T)

                Np = this_T.shape[0]
                # extract the on-shell T elements
                if nof_blocks > 1:
                    # coupled
                    Np = int((Np - 2) / 2)
                    T11 = this_T[Np, Np]
                    T12 = this_T[2 * Np + 1, Np]
                    T22 = this_T[2 * Np + 1, 2 * Np + 1]
                    on_shell_T = [T11, T12, T22]
                else:
                    # uncoupled
                    Np = Np - 1
                    T11 = this_T[Np, Np]
                    on_shell_T = [T11]

                this_phase_shift = self.compute_phase_shifts(ko, mu, on_shell_T)
                t4 = time.time()
                profiler.add_timing("Solve Phase Shifts", t4 - t3)
                phase_shifts_for_this_channel.append(this_phase_shift)

            self.phase_shifts.append(np.array(phase_shifts_for_this_channel))

    def lookup_phase_shift(self, L, Ll, S, J, Tz, koid):
        rad2deg = 180.0 / np.pi
        temp = 0
        indices, channels = self.lookup_channel_idx(l=L, ll=Ll, s=S, j=J, tz=Tz)
        if L == Ll and L == 1 and S == 1 and J == 0:
            temp = self.phase_shifts[indices[0]][koid]
        elif L == Ll and L == J:
            temp = self.phase_shifts[indices[0]][koid]
        else:
            tempp = self.phase_shifts[indices[0]][koid]
            if L == Ll and L < J:
                temp = tempp[0]
            elif L == Ll and L > J:
                temp = tempp[1]
            else:
                temp = tempp[2]
        return temp / rad2deg

    @staticmethod
    # this convention is the same with Mathematica.
    def sph_harm(l, m, theta, phi):
        if l < np.abs(m):
            return 0
        # different convention in scipy's sph_harm.
        return scipy.special.sph_harm(m, l, phi, theta)

    def get_spin_coeff(self, Sp, S, mp, m, the, lp, l, j):
        fac1 = np.sqrt(4 * np.pi * (2 * l + 1))
        fac2 = (1j) ** (l - lp)
        # note ws.CG() use double of the real angular momentum quantum number to avoid half integers
        fac3 = ws.CG(2 * l, 2 * S, 2 * j, 0, 2 * m, 2 * m)
        fac4 = ws.CG(2 * lp, 2 * Sp, 2 * j, 2 * (m - mp), 2 * mp, 2 * m)
        fac5 = self.sph_harm(lp, m - mp, the, 0)
        fac = fac1 * fac2 * fac3 * fac4 * fac5
        return fac

    def compute_m11(self, the, koid, ko):
        Sp, S, mp, m = 1, 1, 1, 1
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 3P0
        fac3p0 = self.get_spin_coeff(Sp, S, mp, m, the, 1, 1, 0)
        del3p0 = self.lookup_phase_shift(1, 1, 1, 0, tz, koid)
        amp3p0 = (np.exp(2 * 1j * del3p0) - 1) / (2 * 1j * ko)
        temp = temp + fac3p0 * amp3p0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled triplet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 1, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1
                # coupled triplet
                dm = self.lookup_phase_shift(jtemp - 1, jtemp - 1, 1, jtemp, tz, koid)
                dp = self.lookup_phase_shift(jtemp + 1, jtemp + 1, 1, jtemp, tz, koid)
                de = self.lookup_phase_shift(jtemp - 1, jtemp + 1, 1, jtemp, tz, koid)
                amp_m = (np.cos(2 * de) * np.exp(2 * 1j * dm) - 1) / (2 * 1j * ko)
                amp_p = (np.cos(2 * de) * np.exp(2 * 1j * dp) - 1) / (2 * 1j * ko)
                amp_e = 1j * np.sin(2 * de) * np.exp(1j * (dm + dp)) / (2 * 1j * ko)
                for l in {jtemp - 1, jtemp + 1}:
                    for ll in {jtemp - 1, jtemp + 1}:
                        fac3 = self.get_spin_coeff(Sp, S, mp, m, the, ll, l, jtemp)
                        if l == ll and l < jtemp:
                            temp = temp + fac3 * amp_m
                        elif l == ll and l > jtemp:
                            temp = temp + fac3 * amp_p
                        else:
                            temp = temp + fac3 * amp_e

        return temp

    def compute_m10(self, the, koid, ko):
        Sp, S, mp, m = 1, 1, 1, 0
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 3P0
        fac3p0 = self.get_spin_coeff(Sp, S, mp, m, the, 1, 1, 0)
        del3p0 = self.lookup_phase_shift(1, 1, 1, 0, tz, koid)
        amp3p0 = (np.exp(2 * 1j * del3p0) - 1) / (2 * 1j * ko)
        temp = temp + fac3p0 * amp3p0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled triplet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 1, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1
                # coupled triplet
                dm = self.lookup_phase_shift(jtemp - 1, jtemp - 1, 1, jtemp, tz, koid)
                dp = self.lookup_phase_shift(jtemp + 1, jtemp + 1, 1, jtemp, tz, koid)
                de = self.lookup_phase_shift(jtemp - 1, jtemp + 1, 1, jtemp, tz, koid)
                amp_m = (np.cos(2 * de) * np.exp(2 * 1j * dm) - 1) / (2 * 1j * ko)
                amp_p = (np.cos(2 * de) * np.exp(2 * 1j * dp) - 1) / (2 * 1j * ko)
                amp_e = 1j * np.sin(2 * de) * np.exp(1j * (dm + dp)) / (2 * 1j * ko)
                for l in {jtemp - 1, jtemp + 1}:
                    for ll in {jtemp - 1, jtemp + 1}:
                        fac3 = self.get_spin_coeff(Sp, S, mp, m, the, ll, l, jtemp)
                        if l == ll and l < jtemp:
                            temp = temp + fac3 * amp_m
                        elif l == ll and l > jtemp:
                            temp = temp + fac3 * amp_p
                        else:
                            temp = temp + fac3 * amp_e

        return temp

    def compute_mpm(self, the, koid, ko):
        Sp, S, mp, m = 1, 1, 1, -1
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 3P0
        fac3p0 = self.get_spin_coeff(Sp, S, mp, m, the, 1, 1, 0)
        del3p0 = self.lookup_phase_shift(1, 1, 1, 0, tz, koid)
        amp3p0 = (np.exp(2 * 1j * del3p0) - 1) / (2 * 1j * ko)
        temp = temp + fac3p0 * amp3p0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled triplet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 1, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1
                # coupled triplet
                dm = self.lookup_phase_shift(jtemp - 1, jtemp - 1, 1, jtemp, tz, koid)
                dp = self.lookup_phase_shift(jtemp + 1, jtemp + 1, 1, jtemp, tz, koid)
                de = self.lookup_phase_shift(jtemp - 1, jtemp + 1, 1, jtemp, tz, koid)
                amp_m = (np.cos(2 * de) * np.exp(2 * 1j * dm) - 1) / (2 * 1j * ko)
                amp_p = (np.cos(2 * de) * np.exp(2 * 1j * dp) - 1) / (2 * 1j * ko)
                amp_e = 1j * np.sin(2 * de) * np.exp(1j * (dm + dp)) / (2 * 1j * ko)
                for l in {jtemp - 1, jtemp + 1}:
                    for ll in {jtemp - 1, jtemp + 1}:
                        fac3 = self.get_spin_coeff(Sp, S, mp, m, the, ll, l, jtemp)
                        if l == ll and l < jtemp:
                            temp = temp + fac3 * amp_m
                        elif l == ll and l > jtemp:
                            temp = temp + fac3 * amp_p
                        else:
                            temp = temp + fac3 * amp_e

        return temp

    def compute_m01(self, the, koid, ko):
        Sp, S, mp, m = 1, 1, 0, 1
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 3P0
        fac3p0 = self.get_spin_coeff(Sp, S, mp, m, the, 1, 1, 0)
        del3p0 = self.lookup_phase_shift(1, 1, 1, 0, tz, koid)
        amp3p0 = (np.exp(2 * 1j * del3p0) - 1) / (2 * 1j * ko)
        temp = temp + fac3p0 * amp3p0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled triplet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 1, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1
                # coupled triplet
                dm = self.lookup_phase_shift(jtemp - 1, jtemp - 1, 1, jtemp, tz, koid)
                dp = self.lookup_phase_shift(jtemp + 1, jtemp + 1, 1, jtemp, tz, koid)
                de = self.lookup_phase_shift(jtemp - 1, jtemp + 1, 1, jtemp, tz, koid)
                amp_m = (np.cos(2 * de) * np.exp(2 * 1j * dm) - 1) / (2 * 1j * ko)
                amp_p = (np.cos(2 * de) * np.exp(2 * 1j * dp) - 1) / (2 * 1j * ko)
                amp_e = 1j * np.sin(2 * de) * np.exp(1j * (dm + dp)) / (2 * 1j * ko)
                for l in {jtemp - 1, jtemp + 1}:
                    for ll in {jtemp - 1, jtemp + 1}:
                        fac3 = self.get_spin_coeff(Sp, S, mp, m, the, ll, l, jtemp)
                        if l == ll and l < jtemp:
                            temp = temp + fac3 * amp_m
                        elif l == ll and l > jtemp:
                            temp = temp + fac3 * amp_p
                        else:
                            temp = temp + fac3 * amp_e

        return temp

    def compute_m00(self, the, koid, ko):
        Sp, S, mp, m = 1, 1, 0, 0
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 3P0
        fac3p0 = self.get_spin_coeff(Sp, S, mp, m, the, 1, 1, 0)
        del3p0 = self.lookup_phase_shift(1, 1, 1, 0, tz, koid)
        amp3p0 = (np.exp(2 * 1j * del3p0) - 1) / (2 * 1j * ko)
        temp = temp + fac3p0 * amp3p0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled triplet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 1, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1
                # coupled triplet
                dm = self.lookup_phase_shift(jtemp - 1, jtemp - 1, 1, jtemp, tz, koid)
                dp = self.lookup_phase_shift(jtemp + 1, jtemp + 1, 1, jtemp, tz, koid)
                de = self.lookup_phase_shift(jtemp - 1, jtemp + 1, 1, jtemp, tz, koid)
                amp_m = (np.cos(2 * de) * np.exp(2 * 1j * dm) - 1) / (2 * 1j * ko)
                amp_p = (np.cos(2 * de) * np.exp(2 * 1j * dp) - 1) / (2 * 1j * ko)
                amp_e = 1j * np.sin(2 * de) * np.exp(1j * (dm + dp)) / (2 * 1j * ko)
                for l in {jtemp - 1, jtemp + 1}:
                    for ll in {jtemp - 1, jtemp + 1}:
                        fac3 = self.get_spin_coeff(Sp, S, mp, m, the, ll, l, jtemp)
                        if l == ll and l < jtemp:
                            temp = temp + fac3 * amp_m
                        elif l == ll and l > jtemp:
                            temp = temp + fac3 * amp_p
                        else:
                            temp = temp + fac3 * amp_e

        return temp

    def compute_mss(self, the, koid, ko):
        Sp, S, mp, m = 0, 0, 0, 0
        temp = 0
        jmax = self.jmax
        tz = 0  # only consider np case for now.

        # j=0 channels: 1S0
        fac1s0 = self.get_spin_coeff(Sp, S, mp, m, the, 0, 0, 0)
        del1s0 = self.lookup_phase_shift(0, 0, 0, 0, tz, koid)
        amp1s0 = (np.exp(2 * 1j * del1s0) - 1) / (2 * 1j * ko)
        temp = temp + fac1s0 * amp1s0

        # J>=1 channels
        if jmax > 0:
            for jtemp in range(1, jmax + 1, 1):
                # uncoupled singlet
                fac1 = self.get_spin_coeff(Sp, S, mp, m, the, jtemp, jtemp, jtemp)
                del1 = self.lookup_phase_shift(jtemp, jtemp, 0, jtemp, tz, koid)
                amp1 = (np.exp(2 * 1j * del1) - 1) / (2 * 1j * ko)
                temp = temp + fac1 * amp1

        return temp

    def build_m_matrix(self):
        ws.init(20, "Jmax", 3)
        self.m11 = []
        self.m10 = []
        self.mpm = []
        self.m01 = []
        self.m00 = []
        self.mss = []
        for idx, Tlab in enumerate(self.Tlabs):
            tz = 0
            ko, mu = self.lab2rel(Tlab, tz)
            temp_m11 = []
            temp_m10 = []
            temp_mpm = []
            temp_m01 = []
            temp_m00 = []
            temp_mss = []
            for the in self.theta:
                # from degree to rad
                the = the * np.pi / 180
                m11 = self.compute_m11(the, idx, ko)
                m10 = self.compute_m10(the, idx, ko)
                mpm = self.compute_mpm(the, idx, ko)
                m01 = self.compute_m01(the, idx, ko)
                m00 = self.compute_m00(the, idx, ko)
                mss = self.compute_mss(the, idx, ko)
                temp_m11.append(m11)
                temp_m10.append(m10)
                temp_mpm.append(mpm)
                temp_m01.append(m01)
                temp_m00.append(m00)
                temp_mss.append(mss)
            self.m11.append(temp_m11)
            self.m10.append(temp_m10)
            self.mpm.append(temp_mpm)
            self.m01.append(temp_m01)
            self.m00.append(temp_m00)
            self.mss.append(temp_mss)

    def cal_spin_observables(self):
        self.spin_obs = {}
        self.spin_obs["DSG"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["D"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["P"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["A"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["R"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["Rp"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["Axx"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["Azz"] = np.zeros((len(self.Tlabs), len(self.theta)))
        self.spin_obs["Axz"] = np.zeros((len(self.Tlabs), len(self.theta)))
        for idx, Tlab in enumerate(self.Tlabs):
            for idx_the, the in enumerate(self.theta):
                m11 = self.m11[idx][idx_the]
                m10 = self.m10[idx][idx_the]
                mpm = self.mpm[idx][idx_the]
                m01 = self.m01[idx][idx_the]
                m00 = self.m00[idx][idx_the]
                mss = self.mss[idx][idx_the]
                I0 = (
                    0.5 * np.abs(m11) ** 2
                    + 0.5 * np.abs(m10) ** 2
                    + 0.5 * np.abs(mpm) ** 2
                    + 0.5 * np.abs(m01) ** 2
                    + 0.25 * np.abs(m00) ** 2
                    + 0.25 * np.abs(mss) ** 2
                )
                I01D = (
                    0.25 * np.abs(m11 + mpm - mss) ** 2
                    + 0.25 * np.abs(m11 - mpm - m00) ** 2
                    + 0.5 * np.abs(m10 + m01) ** 2
                )  # I0(1-D)
                I0P = (
                    np.sqrt(2)
                    / 4.0
                    * np.real(1j * (m10 - m01) * np.conj(m11 - mpm + m00))
                )
                I0A = (
                    -0.5
                    * np.real(
                        (m00 + np.sqrt(2) * (np.cos(the) + 1) / np.sin(the) * m10)
                        * np.conj(m11 + mpm + mss)
                        - np.sqrt(2) / np.sin(the) * (m10 + m01) * np.conj(m11 + mpm)
                    )
                    * np.sin(the / 2.0)
                )
                I0R = (
                    0.5
                    * np.real(
                        (m00 + np.sqrt(2) * (np.cos(the) - 1) / np.sin(the) * m10)
                        * np.conj(m11 + mpm + mss)
                        + np.sqrt(2) / np.sin(the) * (m10 + m01) * np.conj(mss)
                    )
                    * np.cos(the / 2)
                )
                I0Rp = (
                    0.5
                    * np.real(
                        (m00 + np.sqrt(2) * (np.cos(the) + 1) / np.sin(the) * m10)
                        * np.conj(m11 + mpm + mss)
                        - np.sqrt(2) / np.sin(the) * (m10 + m01) * np.conj(mss)
                    )
                    * np.sin(the / 2)
                )
                I0Axx = (
                    0.25 * np.abs(m00) ** 2
                    - 0.25 * np.abs(mss) ** 2
                    - 0.5 * np.abs(m01) ** 2
                    + 0.5 * np.abs(m10) ** 2
                    + np.real(m11 * np.conj(mpm))
                )
                I0Azz = (
                    0.5 * np.abs(m11) ** 2
                    - 0.25 * np.abs(m00) ** 2
                    - 0.25 * np.abs(mss) ** 2
                    + 0.5 * np.abs(m01) ** 2
                    - 0.5 * np.abs(m10) ** 2
                    + 0.5 * np.abs(mpm) ** 2
                )
                I0Axz = 0.25 * np.tan(the) * (
                    np.abs(m11 - mpm) ** 2 - np.abs(m00) ** 2
                ) - 0.5 / np.tan(the) * (np.abs(m01) ** 2 - np.abs(m10) ** 2)
                self.spin_obs["DSG"][idx][idx_the] = I0
                self.spin_obs["D"][idx][idx_the] = 1 - I01D / I0
                self.spin_obs["P"][idx][idx_the] = I0P / I0
                self.spin_obs["A"][idx][idx_the] = I0A / I0
                self.spin_obs["R"][idx][idx_the] = I0R / I0
                self.spin_obs["Rp"][idx][idx_the] = I0Rp / I0
                self.spin_obs["Axx"][idx][idx_the] = I0Axx / I0
                self.spin_obs["Azz"][idx][idx_the] = I0Azz / I0
                self.spin_obs["Axz"][idx][idx_the] = I0Axz / I0

    @staticmethod
    def write_arrays_to_file(filename, column_names, arrays, width=24, precision=4):
        num_columns = len(arrays)
        num_rows = len(arrays[0])
        formatted_rows = []

        for row_idx in range(num_rows):
            row_data = []
            for col_idx in range(num_columns):
                value = arrays[col_idx][row_idx]
                formatted_value = format(value, f".{precision}f")
                row_data.append(formatted_value.ljust(width))
            formatted_rows.append(" ".join(row_data))

        with open(filename, "w") as file:
            names_line = " ".join(name.ljust(width) for name in column_names) + "\n"
            file.write(names_line)
            file.write("\n".join(formatted_rows))

    # Writter of observables
    def store_observables(self):
        dir = "result"  # observables files are generated in this dir.
        if not os.path.exists(dir):
            os.makedirs(dir)
        for idx, tlab in enumerate(self.Tlabs):
            file_name = (
                "./"
                + dir
                + "/"
                + f"spin_observables_{self.V.chiral_type}_tlab{tlab}_Jmax{self.jmax}.txt"
            )
            name_list = ["theta", "DSG", "D", "P", "A", "R", "Rp", "Axx", "Azz", "Axz"]
            self.write_arrays_to_file(
                file_name,
                name_list,
                [
                    self.theta,
                    self.spin_obs["DSG"][idx],
                    self.spin_obs["D"][idx],
                    self.spin_obs["P"][idx],
                    self.spin_obs["A"][idx],
                    self.spin_obs["R"][idx],
                    self.spin_obs["Rp"][idx],
                    self.spin_obs["Axx"][idx],
                    self.spin_obs["Azz"][idx],
                    self.spin_obs["Axz"][idx],
                ],
            )

    def make_partial_wave_name(self):
        orbit_table = "SPDFGHIJKLMNOQRTUVWXYZ"
        name = []
        name.append("1S0")
        name.append("3P0")
        for jtemp in range(1, self.jmax + 1, 1):
            name.append("1" + orbit_table[jtemp] + str(jtemp))
            name.append("3" + orbit_table[jtemp] + str(jtemp))
        for jtemp in range(1, self.jmax + 1, 1):
            name.append("3" + orbit_table[jtemp - 1] + str(jtemp))
            name.append("3" + orbit_table[jtemp + 1] + str(jtemp))
            name.append("E" + str(jtemp))
        return name

    # Writter of phase shifts
    def store_phase_shifts(self):
        dir = "result"  # observables files are generated in this dir.
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = (
            "./" + dir + "/" + f"phase_shifts_{self.V.chiral_type}_Jmax{self.jmax}.txt"
        )
        name_list = ["Tlab"] + self.make_partial_wave_name()
        phases = [self.Tlabs]
        indice, channel = self.lookup_channel_idx(l=0, ll=0, s=0, j=0)
        phases.append(self.phase_shifts[indice[0]])
        indice, channel = self.lookup_channel_idx(l=1, ll=1, s=1, j=0)
        phases.append(self.phase_shifts[indice[0]])
        for jtemp in range(1, self.jmax + 1, 1):
            indice, channel = self.lookup_channel_idx(l=jtemp, ll=jtemp, s=0, j=jtemp)
            phases.append(self.phase_shifts[indice[0]])
            indice, channel = self.lookup_channel_idx(l=jtemp, ll=jtemp, s=1, j=jtemp)
            phases.append(self.phase_shifts[indice[0]])
        for jtemp in range(1, self.jmax + 1, 1):
            indice, channel = self.lookup_channel_idx(
                l=jtemp + 1, ll=jtemp - 1, s=1, j=jtemp
            )
            phases.append(self.phase_shifts[indice[0]][:, 0])
            phases.append(self.phase_shifts[indice[0]][:, 1])
            phases.append(self.phase_shifts[indice[0]][:, 2])
        self.write_arrays_to_file(file_name, name_list, phases)
