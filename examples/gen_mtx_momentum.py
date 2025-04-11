import sys

sys.path.append("./lib")

import nn_studio as nn_studio
import chiral_potential as chiral_potential
import utility
import profiler
import time

utility.header_message()

################################################################################################################

utility.section_message("Interaction initialization")

# initialize an object for the chiral interaction
potential = chiral_potential.two_nucleon_potential("n3loemn500")

################################################################################################################

utility.section_message("Writing momentum matrix elements")

potential.write_mtx_momentum(kmax=8, N=100, Jmax=8)

################################################################################################################

################################################################################################################

utility.section_message("Timings")

profiler.print_timings()

################################################################################################################

utility.footer_message()
