import numpy as np
from scipy.special import gamma, genlaguerre
import chiral_potential as chiral_potential
def Rnl(n, l, pi1, b):
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
def overlap_integral(n_p, l_p, n, l, b=1.0, Np=100, a=0, b_max=10):
    """
    Calculates the overlap integral for radial wavefunctions.
    
    Parameters:
    n_p : int
        Quantum number n'
    l_p : int
        Quantum number l'
    n : int
        Quantum number n
    l : int
        Quantum number l
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
        The calculated overlap integral.
    """
    # Generate Gauss-Legendre points and weights over [a, b_max]
    pi1_points, pi1_weights = gauss_legendre_line_mesh(Np, a, b_max)
    
    # Precompute Rnl values for all points
    Rn_p_l_p_values = np.array([Rnl(n_p, l_p, pi1, b) for pi1 in pi1_points])
    Rnl_values = np.array([Rnl(n, l, pi1, b) for pi1 in pi1_points])
    
    # Compute the integrand
    integrand = (pi1_points**2) * Rn_p_l_p_values * Rnl_values
    result = np.einsum('i,i->', integrand, pi1_weights)
    
    return result

# Call the overlap integral function
result_same_nl = overlap_integral(0, 0, 0, 0, b=1.0, Np=100, a=0, b_max=10)
result_diff_nl = overlap_integral(2, 0, 1, 0, b=1.0, Np=100, a=0, b_max=10)

print("Overlap when n'=n and l'=l:", result_same_nl)
print("Overlap when n'≠n or l'≠l:", result_diff_nl)
def V(n_p, l_p, n, l, S, J, func, b=1.0, Np=100, a=0, b_max=10):
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
    Rn_p_l_p_values = np.array([Rnl(n_p, l_p, pi1_p, b) for pi1_p in pi1_p_points])
    Rnl_values = np.array([Rnl(n, l, pi1, b) for pi1 in pi1_points])
    
    
    # Calculate the potential function on the mesh
    V_rel_mesh = func(l_p, l, S, J, pi1_points, pi1_weights)
    
    # Compute the integrand using Einstein summation (einsum)
    # integrand = (Pi1_P**2) * (Pi1**2) * Rn_p_l_p_values[:, None] * Rnl_values[None, :] * V_rel_mesh
    result = np.einsum('ij,i,j,i,j,i,j->', V_rel_mesh,pi1_p_points**2,pi1_points**2,pi1_p_weights,pi1_weights,Rn_p_l_p_values,Rnl_values)
    
    return result
hbarc=197.32705
mu=938.91852
def T_matrix_pi(l_p, l, S, J, ks,ws):
    T=np.diag(1 *np.sqrt(2)/ws*hbarc**2/mu)
    return T

pot=chiral_potential.two_nucleon_potential('n3loemn500')
def V_matrix(l_p, l, S, J, ks,ws):
    
    X, Y = np.meshgrid(ks, ks)
    V_rel_mesh = np.vectorize(pot.potential)(l_p, l,X*hbarc/np.sqrt(2), Y*hbarc/np.sqrt(2),S, J,0 )

Nmax=30
T_matrix_HO = np.zeros((Nmax, Nmax))
Vpp_HO=np.zeros((Nmax, Nmax))
Vpm_HO=np.zeros((Nmax, Nmax))
Vmp_HO=np.zeros((Nmax, Nmax))
Vmm_HO=np.zeros((Nmax, Nmax))

for n_p in range(Nmax):
    for n in range(Nmax):
        print(n_p,n)
        T_matrix_HO[n_p, n] = V(n_p, 0, n, 0, 0, 0, T_matrix_pi, Np=100, a=0, b_max=10)*hbarc**2
        Vpp_HO[n_p, n] = V(n_p, 2, n, 2, 1, 1, T_matrix_pi, Np=100, a=0, b_max=10)*hbarc**3
        Vpm_HO[n_p, n] = -V(n_p, 2, n, 0, 1, 1, T_matrix_pi, Np=100, a=0, b_max=10)*hbarc**3
        Vmp_HO[n_p, n] = -V(n_p, 0, n, 2, 1, 1, T_matrix_pi, Np=100, a=0, b_max=10)*hbarc**3
        Vmm_HO[n_p, n] = V(n_p, 0, n, 0, 1, 1, T_matrix_pi, Np=100, a=0, b_max=10)*hbarc**3


Vmtx=np.block([[Vmm_HO+T_matrix_HO,Vmp_HO],[Vpm_HO,Vpp_HO+T_matrix_HO]])
eigenvalues = np.linalg.eigvalsh(Vmtx)

print("Symmetric matrix:")
print(Vmtx)
print("\nEigenvalues sorted from small to large:")
print(eigenvalues)