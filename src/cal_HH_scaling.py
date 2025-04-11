import sys
sys.path.append("../lib")
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import density_studio_ave as dense
import pandas as pd
import integration as inte
from scipy.interpolate import make_interp_spline
place='momentum'
namen='neutron_momentum_density_Z2_N1_1I2+_0.dat'
namep='proton_momentum_density_Z2_N1_1I2+_0.dat'
namesum='momentum_density_Z2_N1_1I2+_0.dat'
nucleus='He'
A=3
force='n4loemn500'
def integrate(density,kmin,kmax=4.89):
    k_points=np.linspace(kmin,kmax,300)
    weight=(kmax-kmin)/300
    sumdense=0
    for i,ki in enumerate(k_points):
        sumdense=sumdense+weight*4*np.pi*ki**2*density(ki)
    return sumdense

df1 = pd.read_csv('proton-He3n4loemn500.csv')
# draw the pictures
proton_rhohe3={}
neutron_rhohe3={}
for column in df1.columns:
    proton_rhohe3[column] = df1[column].to_numpy()
rhop_HH_emn500=make_interp_spline(proton_rhohe3['k_p'], proton_rhohe3['rho_p'], k=3)
df1 = pd.read_csv('neutron-He3n4loemn500.csv')
# draw the pictures
for column in df1.columns:
    neutron_rhohe3[column] = df1[column].to_numpy()
rhon_HH_emn500=make_interp_spline(neutron_rhohe3['k_n'], neutron_rhohe3['rho_n'], k=3)


ks=np.linspace(1.5,2.5,40)
ratios=np.zeros((40))
studio=dense.one_nucleon_density(nucleus,A,place,force,32,16,'n4loemn500',namen,namep,namesum)
print(studio.scaling_factor(0,5))
for i,ki in enumerate(ks):
    sum1=integrate(rhop_HH_emn500,ki)*2
    sum2=sum2=integrate(rhon_HH_emn500,ki)
    B=sum1+sum2
    _,C=studio.scaling_factor(ki,5)
    ratio=B/C/A
    ratios[i]=ratio

df = pd.DataFrame({'ks': ks, 'ratios': ratios})

# 将 DataFrame 保存为 CSV 文件
df.to_csv('HH_scaling_He3_n4lo.csv', index=False)