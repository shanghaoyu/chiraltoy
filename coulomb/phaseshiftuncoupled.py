import sys

sys.path.append("../deps")
sys.path.append("../lib")
import gausslegendremesh as gl
import constants as const
import numpy as np
import chiral_potential as chiral_potential
import time
import profiler
import mpmath

def F(l, eta, z):
    return mpmath.coulombf(l, eta, z)

def G(l, eta, z):
    return mpmath.coulombg(l, eta, z)

def dF(l, eta, z):
    return mpmath.diff(lambda z: F(l, eta, z), z)

def dG(l, eta, z):
    return mpmath.diff(lambda z: G(l, eta, z), z)

class phaseshift_uncouple:
    def __init__(self,potential,j,s,tz,Np=100):
        self.Np=Np
        self.pmesh,self.wmesh=gl.gauss_legendre_inf_mesh(Np)
        self.Tlabs=None
        self.phaseshifts=[]
        self.tz=tz
        self.j=j
        self.s=s
        self.V= chiral_potential.two_nucleon_potential(potential, add_cutoff_coulomb=False)
        if tz == -1:
            self.mu = const.Mp / 2
        elif tz == 0:
            self.mu = const.Mp * const.Mn / (const.Mp + const.Mn)
        elif tz == 1:
            self.mu = const.Mn / 2
        else:
            exit("unknown isospin projection")
        self.Tmtx =None
        
    def lab2rel(self, Tlab):
        tz=self.tz
        if tz == -1:
            ko2 = 0.5 * const.Mp * Tlab  #! I correct the factor from 2 to 0.5.
        elif tz == 0:
            ko2 = (
                const.Mp**2
                * Tlab
                * (Tlab + 2 * const.Mn)
                / ((const.Mp + const.Mn) ** 2 + 2 * Tlab * const.Mp)
            )
        elif tz == 1:
            ko2 = 0.5 * const.Mn * Tlab  #! I correct the factor from 2 to 0.5.
        else:
            exit("unknown isospin projection")

        if ko2 < 0:
            exit("Tlab less than zero")
        else:
            ko = np.sqrt(ko2)

        return ko
    
    def Vmtx(self, ko):
        this_mesh = np.hstack((self.pmesh, ko))
        mtx = np.zeros((len(this_mesh), len(this_mesh)))
        for pidx, p in enumerate(this_mesh):
            for ppidx, pp in enumerate(this_mesh):
                if self.j==0 and self.s==1:
                    mtx[ppidx][pidx] = self.V.potential(1, 1, pp, p, self.j, self.s, self.tz)
                else:
                    mtx[ppidx][pidx] = self.V.potential(self.j, self.j, pp, p, self.j, self.s, self.tz)
        return np.array(mtx)
    
    def Coulomb_rel(self,ko):
        this_mesh = np.hstack((self.pmesh, ko))
        mtx = np.zeros((len(this_mesh), len(this_mesh)))
        for pidx, p in enumerate(this_mesh):
            for ppidx, pp in enumerate(this_mesh):
                if self.j==0 and self.s==1:
                    mtx[ppidx][pidx] = self.V.potential_cutoff_relcoulomb(1, 1, pp, p, self.j, self.s, self.tz,ko)
                else:
                    mtx[ppidx][pidx] = self.V.potential_cutoff_relcoulomb(self.j, self.j, pp, p, self.j, self.s, self.tz,ko)
        return np.array(mtx)
    
    def setup_G0_vector(self, ko):
        mu=self.mu
        G = np.zeros(( self.Np + 1), dtype=complex)

        # note that we index from zero, and the N+1 point is at self.Np
        G[0 : self.Np] = (
            self.wmesh * self.pmesh**2 / (ko**2 - self.pmesh**2)
        )  # Gaussian integral

        # print('   G0 pole subtraction')
        G[self.Np] = (
            -np.sum(self.wmesh / (ko**2 - self.pmesh**2)) * ko**2
        )  # 'Principal value'
        G[self.Np] -= 1j * ko * (np.pi / 2)
        return G * 2 * mu
    
    def setup_VG_kernel(self,Vmtx, ko):
        # Go-vector dim(u) = 2*len(p)+2
        g = self.setup_G0_vector(ko)
        VG = np.zeros((len(g), len(g)), dtype=complex)

        for g_idx, g_elem in enumerate(g):
            VG[:, g_idx] = g_elem * Vmtx[:, g_idx]

        return VG
    
    def solve_lippmann_schwinger(self,Vmtx, ko):
        # matrix inversion:
        # T = V + VGT
        # (1-VG)T = V
        # T = (1-VG)^{-1}V

        VG = self.setup_VG_kernel(Vmtx, ko)
        VG = np.eye(VG.shape[0]) - VG
        # golden rule of linear algebra: avoid matrix inversion if you can
        # T = np.matmul(np.linalg.inv(VG),Vmtx)
        T = np.linalg.solve(VG, Vmtx)

        return T
    
    def compute_phase_shifts(self,ko, on_shell_T):
        rad2deg = 180.0 / np.pi
        mu=self.mu
        fac = np.pi * mu * ko

        # uncoupled
        T = on_shell_T
        Z = 1 - fac * 2j * T
        # S=exp(2i*delta)
        delta = (-0.5 * 1j) * np.log(Z)

        return np.real(delta * rad2deg)
    
    def get_coulomb_phase(self,ko, deltas):
        if self.j==0 and self.s==1:
            l=1
        else:
            l = self.j
        fac = np.pi / 180.0
        R = 10.0 / const.hbarc
        alpha = 1.0 / 137.035989
        q = ko
        z = q * R
        factor_rel = (1 + 2 * ko*ko / const.Mp**2) / (np.sqrt(1 + ko*ko / const.Mp**2))
        eta = self.mu * alpha * factor_rel / q
        tdS = np.tan(deltas * fac)
        AL0= (F(l,0,z)+G(l,0,z)*tdS)/(dF(l,0,z)+dG(l,0,z)*tdS)
        tdC = float((AL0*dF(l,eta,z)-F(l,eta,z)) / (G(l,eta,z)-AL0*dG(l,eta,z)))
        deltaC = np.arctan(tdC)
        return deltaC / fac
    
    def compute_Tmtx_phaseshifts(self,verbose=False):
        if verbose:
            print(f"computing T-matrices for")

        self.Tmtx = []
        self.phase_shifts = []

        for Tlab in self.Tlabs:
            if verbose:
                print(f"Tlab = {Tlab} MeV")

            ko = self.lab2rel(Tlab)
            t1 = time.time()
            Vmtx = self.Vmtx(ko)
            if self.tz==-1:
                Vcoul=self.Coulomb_rel(ko)
                Vmtx=Vmtx+Vcoul
            t2 = time.time()
            profiler.add_timing("Setup V Matrix", t2 - t1)
            self.Tmtx = self.solve_lippmann_schwinger(Vmtx, ko)
            t3 = time.time()
            profiler.add_timing("Solve LS", t3 - t2)

            Np = self.Tmtx.shape[0]
            Np = Np - 1
            T11 = self.Tmtx[Np, Np]
            on_shell_T =T11

            this_phase_shift = self.compute_phase_shifts(ko, on_shell_T)
            deltaf=this_phase_shift
            if self.tz==-1:
                deltaf=self.get_coulomb_phase(ko,this_phase_shift)
            
            t4 = time.time()
            profiler.add_timing("Solve Phase Shifts", t4 - t3)
            self.phase_shifts.append(deltaf)
    






        


