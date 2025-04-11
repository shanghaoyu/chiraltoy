import numpy as np
from functools import lru_cache


# Calculation of the Laguerre polynomial L(n, alpha, x)
# ---------------------------------------------- -
# The following recurrence relation is used :
# (i+1).L(i+1, alpha, x) = (-x + alpha + 1 + 2i).L(i, alpha, x) - (i+a).L(i-1, alpha, x) for i > 0.
# L(0, alpha, x) = 1. This value is directly returned if the degree of the polynomial is 0.
# L(1, alpha, x) = alpha + 1 - x. This value is directly returned if the degree of the polynomial is 1.
#
#
# Variables:
# ----------
# degree : degree of the Laguerre polynomial
# i, poly_im1, poly_i, poly_ip1 : Laguerre polynomials of degree i-1, i, and i+1, with n > 1.
# i goes from 1 to n - 1.
# alpha : parameter of the Laguerre polynomial
# minus_x_plus_alpha_plus_one : -x + alpha + 1
# x : variable of the Laguerre polynomial.
@lru_cache(maxsize=None)
def laguerre_poly(degree: int, alpha: float, x: float):
    if degree == 0:
        return 1.0

    minus_x_plus_alpha_plus_one = -x + alpha + 1.0

    poly_im1 = 0.0
    poly_i = 1.0
    poly_ip1 = minus_x_plus_alpha_plus_one

    for i in range(1, degree):
        poly_im1, poly_i, poly_ip1 = (
            poly_i,
            poly_ip1,
            ((minus_x_plus_alpha_plus_one + 2 * i) * poly_i - (i + alpha) * poly_im1)
            / (i + 1.0),
        )

    return poly_ip1


@lru_cache(maxsize=None)
def gauss_legendre_line_mesh(a, b, N: int):
    x, w = np.polynomial.legendre.leggauss(N)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5 * (x + 1) * (b - a) + a
    u = w * 0.5 * (b - a)
    return t, u


@lru_cache(maxsize=None)
def hat(j):
    two_j = int(2 * j)
    return np.sqrt(two_j + 1)


@lru_cache(maxsize=None)
def dirac_delta_int(m: int, n: int):
    if m == n:
        return 1
    else:
        return 0


def dirac_delta_float(m: float, n: float):
    precision = 1e-5
    if np.abs(m - n) <= precision:
        return 1
    else:
        return 0
