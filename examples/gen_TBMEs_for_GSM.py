import sys

sys.path.append("./lib")
import chiral_potential as chiral_potential
import numpy as np
import os

# * this is a writter of TBMEs for GSM input files.
# * interaction matrix elements are written under LSJ basis in relative space.

potential_type = "n3loemn500"
k_max = 8.0  # in fm^(-1)
N = 100
JMAX = 8
hc = 197.32705

potential = chiral_potential.two_nucleon_potential(potential_type)
print(f"potential type : {potential_type}\n")


def gauss_legendre_line_mesh(a, b, Np):
    x, w = np.polynomial.legendre.leggauss(Np)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5 * (x + 1) * (b - a) + a
    u = w * 0.5 * (b - a)
    return t, u


# ( l, lp, J, S, Tz )
def gen_mtx_channels(N, Jmax):
    assert Jmax >= 0, "Jmax must be nonegative"

    # pp channels first
    tz = -1
    channels_pp = []
    channels_pp.append([0, 0, 0, 0, tz])
    channels_pp.append([1, 1, 0, 1, tz])
    if Jmax > 0:
        for j in range(1, Jmax + 1, 1):
            if j % 2 == 1:
                channels_pp.append([j, j, j, 1, tz])
            else:
                channels_pp.append([j, j, j, 0, tz])
                channels_pp.append([j - 1, j - 1, j, 1, tz])
                channels_pp.append([j - 1, j + 1, j, 1, tz])
                channels_pp.append([j + 1, j - 1, j, 1, tz])
                channels_pp.append([j + 1, j + 1, j, 1, tz])

    # pn channels second
    tz = 0
    channels_pn = []
    channels_pn.append([0, 0, 0, 0, tz])
    channels_pn.append([1, 1, 0, 1, tz])
    if Jmax > 0:
        for j in range(1, Jmax + 1, 1):
            channels_pn.append([j, j, j, 0, tz])
            channels_pn.append([j, j, j, 1, tz])
            channels_pn.append([j - 1, j - 1, j, 1, tz])
            channels_pn.append([j - 1, j + 1, j, 1, tz])
            channels_pn.append([j + 1, j - 1, j, 1, tz])
            channels_pn.append([j + 1, j + 1, j, 1, tz])

    # nn channels last
    tz = 1
    channels_nn = []
    channels_nn.append([0, 0, 0, 0, tz])
    channels_nn.append([1, 1, 0, 1, tz])
    if Jmax > 0:
        for j in range(1, Jmax + 1, 1):
            if j % 2 == 1:
                channels_nn.append([j, j, j, 1, tz])
            else:
                channels_nn.append([j, j, j, 0, tz])
                channels_nn.append([j - 1, j - 1, j, 1, tz])
                channels_nn.append([j - 1, j + 1, j, 1, tz])
                channels_nn.append([j + 1, j - 1, j, 1, tz])
                channels_nn.append([j + 1, j + 1, j, 1, tz])
    return channels_pp + channels_pn + channels_nn


channels = gen_mtx_channels(N, JMAX)
MeshPoints, MeshWeights = gauss_legendre_line_mesh(0, k_max, N)
hccubic = hc**3

dir = "result"  # files are generated in this dir.
if not os.path.exists(dir):
    os.makedirs(dir)

# write mesh points and weights in "mesh_Bessel_rel.dat" first.
file_mesh_name = (
    "./" + dir + "/" + "mesh_Bessel_rel_" + f"{potential_type}_kmax{k_max}_N{N}.dat"
)
with open(file_mesh_name, "w") as fp:
    combined_data = np.column_stack((MeshPoints, MeshWeights))
    np.savetxt(fp, combined_data, fmt="%.17f       %.17f")

# write interaction matrix elements in "v2body_Bessel_rel.dat".
file_mtx_name = (
    "./"
    + dir
    + "/"
    + "v2body_Bessel_rel_"
    + f"{potential_type}_kmax{k_max}_N{N}_Jmax{JMAX}.dat"
)
mtx_to_write = []
fmt_str = ["%d"] * 7 + ["%.18e"]
with open(file_mtx_name, "w") as fp:
    MeshPoints = MeshPoints * hc
    MeshWeights = MeshWeights * hc
    for chan in channels:
        l, lp, J, S, Tz = chan
        for index_k, k in enumerate(MeshPoints):
            for index_kp, kp in enumerate(MeshPoints):
                mtx = potential.potential(lp, l, kp, k, J, S, Tz) * hccubic
                mtx_to_write.append([l, lp, J, S, Tz, index_k, index_kp, mtx])
    np.savetxt(fp, mtx_to_write, delimiter="      ", fmt=fmt_str)

print("complete !")
