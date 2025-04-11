import singlechannel as phase
import subprocess
import pandas as pd
import sys
Tlabs0=[1,5,10,25,50,100,150,200,250,300,350]
PWA93=[32.68429,54.83247,55.21901,48.67177,38.92972,24.99395,14.75562,6.55239,-0.31338,-6.15473,-11.13174]
expt={"Tlabs":Tlabs0,"PWA93":PWA93}
Tlabs=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500]
# potentials=["av18","n3loemn500","n3loem","cdbonn","n2losat","my","n2loopt"]
potentials=["scsn3lo1fm"]
# potentials=["my"]
phaseshifts={}
phaseshifts["Tlabs"]=Tlabs
for potential in potentials:
    
    phase_shifts=phase.compute_pp_singlechannel_phase_shifts(potential,Tlabs,0,1)
    
    # Process the captured output to extract phase shifts
    
    phaseshifts[potential] = phase_shifts

df= pd.DataFrame(phaseshifts)

df.to_csv('phaseshifts1.csv', index=False)
df= pd.DataFrame(expt)

df.to_csv('expt1.csv', index=False)