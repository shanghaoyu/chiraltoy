import numpy as np
import math
import WignerSymbol as ws
from functools import lru_cache
from numba import jit


@lru_cache(maxsize=None)
def is_triangle(j1: float, j2: float, j3: float):
    precision = 1e-6
    boo1 = np.abs((j1 + j2 + j3) - int(j1 + j2 + j3)) < precision
    boo2 = int(np.abs(j1 - j2) - j3) <= 0
    boo3 = int(np.abs(j1 + j2) - j3) >= 0
    boo = boo1 and boo2 and boo3
    return boo


@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, j, m):
    dj1 = int(2.0 * j1)
    dj2 = int(2.0 * j2)
    dj = int(2.0 * j)
    dm1 = int(2.0 * m1)
    dm2 = int(2.0 * m2)
    dm = int(2.0 * m)
    result = ws.CG(dj1, dj2, dj, dm1, dm2, dm)
    return result


@lru_cache(maxsize=None)
def f6j(j1, j2, j3, j4, j5, j6):
    dj1 = int(2.0 * j1)
    dj2 = int(2.0 * j2)
    dj3 = int(2.0 * j3)
    dj4 = int(2.0 * j4)
    dj5 = int(2.0 * j5)
    dj6 = int(2.0 * j6)
    result = ws.f6j(dj1, dj2, dj3, dj4, dj5, dj6)
    return result


@lru_cache(maxsize=None)
def f9j(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    dj1 = int(2.0 * j1)
    dj2 = int(2.0 * j2)
    dj3 = int(2.0 * j3)
    dj4 = int(2.0 * j4)
    dj5 = int(2.0 * j5)
    dj6 = int(2.0 * j6)
    dj7 = int(2.0 * j7)
    dj8 = int(2.0 * j8)
    dj9 = int(2.0 * j9)
    result = ws.f9j(dj1, dj2, dj3, dj4, dj5, dj6, dj7, dj8, dj9)
    return result


@lru_cache(maxsize=None)
def moshinsky(
    N: int,
    L: int,
    n: int,
    l: int,
    n1: int,
    l1: int,
    n2: int,
    l2: int,
    Lambda: int,
    tan_beta: float,
):
    result = ws.Moshinsky(N, L, n, l, n1, l1, n2, l2, Lambda, tan_beta)
    return result


@lru_cache(maxsize=None)
def hat(j):
    two_j = int(2 * j)
    return np.sqrt(two_j + 1)


na, la, sa, ja = 0, 1, 0.5, 1.5
nb, lb, sb, jb = 0, 1, 0.5, 1.5
J, M = 1, 1

tan_beta = 1.0
temp = 0
for S in [0, 1]:
    L_min = np.abs(la - lb)
    L_max = la + lb
    for L in range(L_min, L_max + 1, 1):
        E_max = 2 * na + la + 2 * nb + lb
        for Lambda in range(0, E_max + 1, 1): 
            twoN_max = np.abs(E_max - Lambda)
            for twoN in range(0, twoN_max + 1, 2):
                N = int(twoN / 2)
                lambda_max = E_max - 2 * N - Lambda
                if lambda_max < 0:
                    continue
                for lambdaa in range(0, lambda_max + 1, 1):
                    twon = E_max - 2 * N - Lambda - lambdaa
                    n = int(twon / 2)
                    j_min = abs(lambdaa - S)
                    j_max = lambdaa + S
                    for j in range(j_min, j_max + 1, 1):
                        for mLambda in range(-Lambda, Lambda + 1):
                            for mj in range(-j, j + 1):
                                fac_hat = hat(ja) * hat(jb) * hat(L) * hat(S)
                                factor_ninej = f9j(la, sa, ja, lb, sb, jb, L, S, J)
                                factor_morsh = moshinsky(N, Lambda, n, lambdaa, na, la, nb, lb, L, tan_beta)
                                factor_pow = (-1) ** (Lambda + lambdaa + S + J) * hat(L) * hat(j) * f6j(Lambda, lambdaa, L, S, J, j)
                                factor_3j = cg(Lambda, mLambda, j, mj, J, M)
                                factor = fac_hat * factor_ninej * factor_morsh * factor_pow * factor_3j
                                temp = temp + factor**2
                                if abs(factor) > 1e-8:
                                    print(f"N{N}, Lambda{Lambda}, factor:{factor}")

print(temp)
