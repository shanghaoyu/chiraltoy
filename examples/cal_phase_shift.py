import sys

sys.path.append("../lib")
import utility
import profiler
import time
import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential


utility.header_message()


################################################################################################################
utility.section_message("Initialization")

t1 = time.time()

nn = nn_studio.nn_studio(jmin=0, jmax=2, tzmin=0, tzmax=0, Np=100)

# define the lab neutron-proton kinetic energies that you want to analyze (denser for low T in this case)
# nn.Tlabs = (
#     [1e-3]
#     + [x / 10 for x in np.arange(1, 11, 0.1)]
#     + [x for x in np.arange(2, 31, 1)]
#     + [x for x in np.arange(40, 500, 10)]
# )
nn.Tlabs = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300]

# initialize an object for the chiral interaction
potential = chiral_potential.two_nucleon_potential("smsn4lo+450")

# give the potential to the nn-analyzer
nn.V = potential

t2 = time.time()
profiler.add_timing("Initialization", t2 - t1)
################################################################################################################


################################################################################################################
utility.section_message("Solving LS")

nn.compute_Tmtx(nn.channels, verbose=True)

t3 = time.time()

nn.store_phase_shifts()

t4 = time.time()
profiler.add_timing("Storing Phase Shifts", t4 - t3)
################################################################################################################


################################################################################################################

utility.section_message("Timings")

profiler.print_timings()

################################################################################################################

utility.footer_message()
