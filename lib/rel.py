import numpy as np
from scipy.special import gamma, genlaguerre
import chiral_potential as chiral_potential


hbarc=197.32705
mu=938.91852
Nmax=30
def Rnl_p(n, l, pi1, b):
    """
    Calculate the radial wave function R_nl(π1).
    
    Parameters:
    n : int
        Quantum number n.
    l : int
        Quantum number l.
    pi1 : float
        π1 parameter.
    b : float
        b parameter.
        
    Returns:
    float
        The value of the radial wave function.
    """
    if pi1 <= 0 or b <= 0:
        return 0.0  # Avoid invalid values
    
    # Prefactor calculation
    prefactor = ((-1)**n) * np.sqrt((2 * b * gamma(n + 1)) / gamma(n + l + 1.5))
    
    # Avoid overflow in exponential and large pi1 values
    pi1_squared_b_squared = pi1**2 * b**2
    if pi1_squared_b_squared > 700:  # Prevent overflow in exp
        exp_factor = 0.0
    else:
        exp_factor = np.exp(-pi1_squared_b_squared / 2)
    
    # Compute radial part
    try:
        radial_part = (b * pi1)**l * b * exp_factor * genlaguerre(n, l + 0.5)(pi1_squared_b_squared)
    except (OverflowError, ValueError):
        radial_part = 0.0

    return prefactor * radial_part

def gauss_legendre_line_mesh(Np, a, b):
    """
    Generate points and weights for Gauss-Legendre integration over a finite interval [a, b].
    
    Parameters:
    Np : int
        Number of Gauss-Legendre points.
    a : float
        Start of the interval.
    b : float
        End of the interval.
        
    Returns:
    t : ndarray
        The transformed points for integration over [a, b].
    u : ndarray
        The corresponding weights.
    """
    x, w = np.polynomial.legendre.leggauss(Np)
    t = 0.5 * (x + 1) * (b - a) + a
    u = w * 0.5 * (b - a)
    return t, u

def V_nplp_nl(n_p, l_p, n, l, S, J, func, b=1.0, Np=100, a=0, b_max=10):
    """
    Optimized calculation of the matrix element using Einstein summation.
    
    Parameters:
    n_p : int
        Quantum number n'
    l_p : int
        Quantum number l'
    n : int
        Quantum number n
    l : int
        Quantum number l
    S : int
        Spin quantum number S
    J : int
        Total angular momentum quantum number J
    func : callable
        A function representing V(l', l, S, J, π', π).
    b : float, optional
        Parameter b for the radial wave function (default 1.0).
    Np : int, optional
        Number of Gauss-Legendre points (default 100).
    a : float, optional
        Lower bound of the integral (default 0).
    b_max : float, optional
        Upper bound of the integral (default 10).
    
    Returns:
    float
        The calculated matrix element.
    """
    # Generate Gauss-Legendre points and weights over [a, b_max]
    pi1_p_points, pi1_p_weights = gauss_legendre_line_mesh(Np, a, b_max)
    pi1_points, pi1_weights = gauss_legendre_line_mesh(Np, a, b_max)
    
    # Precompute Rnl values for all points
    Rn_p_l_p_values = np.array([Rnl_p(n_p, l_p, pi1_p, b) for pi1_p in pi1_p_points])
    Rnl_values = np.array([Rnl_p(n, l, pi1, b) for pi1 in pi1_points])
    
    # Use Einstein summation to handle the double integral
    Pi1_P, Pi1 = np.meshgrid(pi1_p_points, pi1_points, indexing='ij')
    
    # Calculate the potential function on the mesh
    V_rel_mesh = np.vectorize(func)(l_p, l,  Pi1_P*hbarc, Pi1*hbarc,S, J,0)*hbarc**3
  
    
    # Compute the integrand using Einstein summation (einsum)
    integrand = (Pi1_P**2) * (Pi1**2) * Rn_p_l_p_values[:, None] * Rnl_values[None, :] * V_rel_mesh
    result = np.einsum('ij,i,j->', integrand, pi1_p_weights, pi1_weights)
    
    if l_p != l:
        factor=-1
    else:
        factor=1
    return result*factor

def T_nplp_nl(n_p, l_p, n, l,b=1.0, Np=100, a=0, b_max=10):
    if l_p == l:
        p_points, p_weights = gauss_legendre_line_mesh(Np, a, b_max)
        Rnl_values = np.array([Rnl_p(n, l, pi, b) for pi in p_points])
        Rnpl_values = np.array([Rnl_p(n_p, l, pi, b) for pi in p_points])
        result = 1/mu*np.sum(p_weights*Rnpl_values*Rnl_values*p_points**4)
    else:
        result=0
    factor=hbarc**2
    return factor*result


T0_matrix_HO = np.zeros((Nmax, Nmax))
T2_matrix_HO = np.zeros((Nmax, Nmax))
Vpp_HO=np.zeros((Nmax, Nmax))
Vpm_HO=np.zeros((Nmax, Nmax))
Vmp_HO=np.zeros((Nmax, Nmax))
Vmm_HO=np.zeros((Nmax, Nmax))

pot=chiral_potential.two_nucleon_potential('n3loemn500')
func=pot.potential
for n_p in range(Nmax):
    for n in range(Nmax):
        T0_matrix_HO[n_p, n] = T_nplp_nl(n_p, 0, n, 0,b=1.0, Np=100, a=0, b_max=10)
        T2_matrix_HO[n_p, n] = T_nplp_nl(n_p, 2, n, 2,b=1.0, Np=100, a=0, b_max=10)
        Vpp_HO[n_p, n] = V_nplp_nl(n_p, 2, n, 2, 1, 1, func, b=1.0, Np=100, a=0, b_max=10)
        Vpm_HO[n_p, n] = V_nplp_nl(n_p, 2, n, 0, 1, 1, func, b=1.0, Np=100, a=0, b_max=10)
        Vmp_HO[n_p, n] = V_nplp_nl(n_p, 0, n, 2, 1, 1, func, b=1.0, Np=100, a=0, b_max=10)
        Vmm_HO[n_p, n] = V_nplp_nl(n_p, 0, n, 0, 1, 1, func, b=1.0, Np=100, a=0, b_max=10)
print('22')
print(Vpp_HO)
print('20')
print(Vpm_HO)
print('02')
print(Vmp_HO)
print('00')
print(Vmm_HO)
print(T0_matrix_HO)
print(T2_matrix_HO)
Hmtx=np.block([[Vmm_HO+T0_matrix_HO,Vmp_HO],[Vpm_HO,Vpp_HO+T2_matrix_HO]])
eigenvalues = np.linalg.eigvalsh(Hmtx)

print("Symmetric matrix:")
print(Hmtx)
print("\nEigenvalues sorted from small to large:")
print(eigenvalues)