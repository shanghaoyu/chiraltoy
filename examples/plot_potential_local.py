import sys

sys.path.append("../lib")

import numpy as np
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt


# initialize an object for the chiral interaction


r = np.linspace(1e-6, 4, 200)

ll, l, s, j, tz = 0, 0, 0, 0, 0


fig = plt.figure(figsize=(6, 6), dpi=160)
plt.xlim(0, 4)
# plt.ylim(-500, 500)
plt.xlabel(r"$r (fm)$")
plt.ylabel(r"$V(r) (\mathrm{MeV})$")
# potential1 = chiral_potential.two_nucleon_potential("av18")
# mtx1 = [potential1.potential_local(ll, l, s, j, tz, rr) for rr in r]
# plt.plot(r, mtx1, color="C1", linestyle="-", linewidth=1.2,label='av18')
# potential1 = chiral_potential.two_nucleon_potential("idaholocal1")
# mtx1 = [potential1.potential_local(ll, l, s, j, tz, rr) for rr in r]
# plt.plot(r, mtx1, color="C2", linestyle="-", linewidth=1.2,label='idaho23')
# potential1 = chiral_potential.two_nucleon_potential("n3loem")
# mtx1 = [potential1.potential_local(ll, l, s, j, tz, rr) for rr in r]
# plt.plot(r, mtx1, color="C3", linestyle="-", linewidth=1.2,label='n3loem')
potential1 = chiral_potential.two_nucleon_potential("minnesoda")
mtx1 = [potential1.potential_local(ll, l, s, j, tz, rr) for rr in r]
plt.plot(r, mtx1, color="C4", linestyle="-", linewidth=1.2,label='minnesoda')




# plt.text(1.8,2000,r"$^3S_1$")
plt.legend()
plt.savefig("plot_potential_local.png", bbox_inches="tight", dpi=600)
plt.show()
