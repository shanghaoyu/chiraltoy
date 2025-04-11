import sys

sys.path.append("../lib")
import utility
import profiler
import time
# import numpy as np
import phaseshiftuncoupled as ps
# import chiral_potential as chiral_potential



def compute_pp_singlechannel_phase_shifts(potential, Tlabs,j,s):

    utility.header_message()


    ################################################################################################################
    utility.section_message("Initialization")

    t1 = time.time()

    phaseshift1s0 = ps.phaseshift_uncouple(potential, j, s, -1, Np=100)

    # define the lab kinetic energies that you want to analyze (denser for low T in this case)
    # nn.Tlabs = (
    #     [1e-6]
    #     + [x / 10 for x in np.arange(1, 11, 1)]
    #     + [x for x in np.arange(2, 31, 1)]
    #     + [x for x in np.arange(40, 360, 10)]
    # )
    phaseshift1s0 .Tlabs = Tlabs

    t2 = time.time()
    profiler.add_timing("Initialization", t2 - t1)
    ################################################################################################################


    ################################################################################################################
    utility.section_message("Solving LS")

    phaseshift1s0.compute_Tmtx_phaseshifts(verbose=False)

    t3 = time.time()
    profiler.add_timing("Solving LS", t3 - t2)

    for num in phaseshift1s0.phase_shifts:
        print(str(num)+',')
    ################################################################################################################


    ################################################################################################################

    utility.section_message("Timings")

    profiler.print_timings()

    ################################################################################################################

    utility.footer_message()

    return phaseshift1s0.phase_shifts