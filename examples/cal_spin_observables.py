import sys

sys.path.append("./lib")
import utility
import profiler
import time
import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt

utility.header_message()


################################################################################################################
utility.section_message("Initialization")

t1 = time.time()

nn = nn_studio.nn_studio(jmin=0, jmax=8, tzmin=0, tzmax=0, Np=32)

nn.Tlabs = [143]

potential_n3loemn500 = chiral_potential.two_nucleon_potential("n4loemn500")

nn.V = potential_n3loemn500

t2 = time.time()
profiler.add_timing("Initialization", t2 - t1)
################################################################################################################


################################################################################################################
utility.section_message("Solving LS")

nn.compute_Tmtx(nn.channels, verbose=False)

t3 = time.time()
profiler.add_timing("Solving LS", t3 - t2)
################################################################################################################


################################################################################################################
utility.section_message("Solving Spin Matrix")

nn.build_m_matrix()

t4 = time.time()
profiler.add_timing("Solving Spin Matrix", t4 - t3)
################################################################################################################


################################################################################################################
utility.section_message("Solving Spin Observables")

nn.cal_spin_observables()
t5 = time.time()
profiler.add_timing("Solving Spin Observables", t5 - t4)

nn.store_observables()
t6 = time.time()
profiler.add_timing("Storing Spin Observables", t6 - t5)
################################################################################################################


################################################################################################################

utility.section_message("Timings")

profiler.print_timings()

################################################################################################################

utility.footer_message()
