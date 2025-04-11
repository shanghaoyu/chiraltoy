import sys

sys.path.append("./lib")

import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt

# initialize an object for computing T-matrices, phase shifts,
nn = nn_studio.nn_studio(
    jmin=0, jmax=1, tzmin=0, tzmax=0, Np=128, mesh_type="gauleg_finite"
)

potential_type = "n3loem"

# initialize an object for the chiral interaction
potential = chiral_potential.two_nucleon_potential(potential_type)

# give the potential to the nn-analyzer
nn.V = potential

idx, selected_channel = nn.lookup_channel_idx(l=2, ll=0, s=1, j=1)
_, potential_matrix = nn.setup_Vmtx(selected_channel[0])

# for plotting potential
mtx = potential_matrix[1]

pp, p = np.meshgrid(nn.pmesh, nn.pmesh)


z_min, z_max = -np.abs(mtx).max(), np.abs(mtx).max()
fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
c = ax.pcolormesh(p, pp, mtx, cmap="RdBu", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
ax.set_xlabel(r"$p$ (MeV)")
ax.set_ylabel(r"$p'$ (MeV)")
plt.savefig(
    f"plot_potential_nonlocal_{potential_type}.png", bbox_inches="tight", dpi=600
)
plt.show()
