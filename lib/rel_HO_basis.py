import numpy as np
from scipy.special import gamma, genlaguerre
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt
hbarc=197.32705
mu=938.91852
Nmax=10
Np=100
a=0
b_max=15
b=1.4

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

pot=chiral_potential.two_nucleon_potential('n3loemn500')
func=pot.potential

pi1_p_points, pi1_p_weights = gauss_legendre_line_mesh(Np, a, b_max)
Rn0_values=[]
Rn2_values=[]
for Ni in range(Nmax):
    Rn0_values.append(np.array([Rnl_p(Ni, 0, pi1_p, b) for pi1_p in pi1_p_points]))
    Rn2_values.append(-np.array([Rnl_p(Ni, 2, pi1_p, b) for pi1_p in pi1_p_points]))

Rn0_values = np.array(Rn0_values)
Rn2_values = np.array(Rn2_values)



Pi1_P, Pi1 = np.meshgrid(pi1_p_points, pi1_p_points, indexing='ij')
V00=np.vectorize(func)(0, 0,  Pi1_P*hbarc, Pi1*hbarc,1, 1,0)*hbarc**3
V02=-np.vectorize(func)(0, 2,  Pi1_P*hbarc, Pi1*hbarc,1, 1,0)*hbarc**3
V20=-np.vectorize(func)(2, 0,  Pi1_P*hbarc, Pi1*hbarc,1, 1,0)*hbarc**3
V22=np.vectorize(func)(2, 2,  Pi1_P*hbarc, Pi1*hbarc,1, 1,0)*hbarc**3

Vmm = np.einsum('mi,i,i,ik,k,k,nk->mn', Rn0_values,pi1_p_weights, pi1_p_points**2, V00,pi1_p_weights,pi1_p_points**2,Rn0_values)
Vmp = np.einsum('mi,i,i,ik,k,k,nk->mn', Rn0_values,pi1_p_weights, pi1_p_points**2, V02,pi1_p_weights,pi1_p_points**2,Rn2_values)
Vpm = np.einsum('mi,i,i,ik,k,k,nk->mn', Rn2_values,pi1_p_weights, pi1_p_points**2, V20,pi1_p_weights,pi1_p_points**2,Rn0_values)
Vpp = np.einsum('mi,i,i,ik,k,k,nk->mn', Rn2_values,pi1_p_weights, pi1_p_points**2, V22,pi1_p_weights,pi1_p_points**2,Rn2_values)

T00=1/mu*np.einsum('mi,i,i,ni->mn',Rn0_values,pi1_p_points**4,pi1_p_weights,Rn0_values)*hbarc**2
T22=1/mu*np.einsum('mi,i,i,ni->mn',Rn2_values,pi1_p_points**4,pi1_p_weights,Rn2_values)*hbarc**2



Hmtx=np.block([[Vmm+T00,Vmp],[Vpm,Vpp+T22]])
eigenvalues, eigenvectors = np.linalg.eigh(Hmtx)


print("\nEigenvalues sorted from small to large:")
print(eigenvalues[0])




########################################################################################
def Rnl_r(n, l, r, b):
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
    if r <= 0 or b <= 0:
        return 0.0  # Avoid invalid values
    
    # Prefactor calculation
    prefactor = np.sqrt((2 / b**3 * gamma(n + 1)) / gamma(n + l + 1.5))
    
    # Avoid overflow in exponential and large pi1 values
    r_squared_over_b_squared = r**2 / b**2
    if r_squared_over_b_squared > 700:  # Prevent overflow in exp
        exp_factor = 0.0
    else:
        exp_factor = np.exp(-r_squared_over_b_squared / 2)
    
    # Compute radial part
    try:
        radial_part = ( r/b)**l  * exp_factor * genlaguerre(n, l + 0.5)(r_squared_over_b_squared)
    except (OverflowError, ValueError):
        radial_part = 0.0

    return prefactor * radial_part


def phi_0(r):
    phi=0
    for Ni in range(Nmax):
        phi=phi+eigenvectors[Ni][0]*Rnl_r(Ni,0,r,b)
    return phi

def phi_2(r):
    phi=0
    for Ni in range(Nmax):
        phi=phi+eigenvectors[Ni+Nmax][0]*Rnl_r(Ni,2,r,b)
    return phi

# Create an array of r values for the x-axis
r_values = np.linspace(0.1, 5, 500)

# Compute phi_0 and phi_2 for each r value
phi_0_values = np.array([phi_0(r) for r in r_values])
phi_2_values = np.array([phi_2(r) for r in r_values])

# Plotting
plt.figure()
plt.plot(r_values, phi_0_values, color='red', label='phi_0')
plt.plot(r_values, phi_2_values, color='blue', label='phi_2')
plt.xlim(0, 5)
plt.xlabel('r')
plt.ylabel('phi')
plt.legend()
plt.title('phi_0 (red) and phi_2 (blue)')
plt.show()
plt.savefig(f'wavefunction-Nmax{Nmax}-b{b}.png',dpi=512)