import singlechannel as phase
import pandas as pd


def make_pp_partial_wave_name(jmax):
    orbit_table = "SPDFGHIJKLMNOQRTUVWXYZ"
    name = []
    name.append("1S0")
    name.append("3P0")
    for jtemp in range(1, jmax + 1, 1):
        if jtemp % 2 == 0:
            name.append("1" + orbit_table[jtemp] + str(jtemp))
        else:
            name.append("3" + orbit_table[jtemp] + str(jtemp))
    # for jtemp in range(1, jmax + 1, 1):
    #     name.append("3" + orbit_table[jtemp - 1] + str(jtemp))
    #     name.append("3" + orbit_table[jtemp + 1] + str(jtemp))
    #     name.append("E" + str(jtemp))
    return name

def writefile(name,df,width):
    col_width = width  

    format_str = "  ".join([f"{{:<{col_width}.2f}}" for _ in df.columns])

    with open(f'{name}.txt', 'w') as f:
        col_format_str = "  ".join([f"{{:<{col_width}}}" for _ in df.columns])
        f.write(col_format_str.format(*df.columns) + '\n')
        for row in df.itertuples(index=False):
            f.write(format_str.format(*row) + '\n')

########################################################################
# define the parameters
Tlabs=[1,5,10,25,50,100,150,200,250,300]
potential="smsn4lo+450"
jmin=0
jmax=2
########################################################################

# the calculation
phaseshifts={}
phaseshifts["Tlabs"]=Tlabs
names=make_pp_partial_wave_name(jmax)
for jtemp in range(jmin, jmax + 1, 1):
    if jtemp==0:
        phase_shifts=phase.compute_pp_singlechannel_phase_shifts(potential,Tlabs,0,0)
        phaseshifts[names[0]] = phase_shifts
        phase_shifts=phase.compute_pp_singlechannel_phase_shifts(potential,Tlabs,0,1)
        phaseshifts[names[1]] = phase_shifts
    else:
        if jtemp%2==0:
            phase_shifts=phase.compute_pp_singlechannel_phase_shifts(potential,Tlabs,jtemp,0)
            phaseshifts[names[jtemp+1]] = phase_shifts
        else:
            phase_shifts=phase.compute_pp_singlechannel_phase_shifts(potential,Tlabs,jtemp,1)
            phaseshifts[names[jtemp+1]] = phase_shifts


# writeout
df= pd.DataFrame(phaseshifts)
name=f'ppphaseshifts_{potential}_jmin{jmin}_jmax{jmax}'
df.to_csv(f'{name}.csv', index=False)
writefile(name,df,12)