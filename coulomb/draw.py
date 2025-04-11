import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline 

font_family = 'Arial'
font_weight = 'bold'
font_size = 14
font_ticks = {'family': font_family, 'weight': font_weight, 'size': font_size}
font_text_epsilon ={'family': font_family, 'weight': 600, 'size': 16}
font_text=  {'family': font_family, 'weight': font_weight, 'size': 22}
font_label=  {'family': font_family, 'weight': 500, 'size': 12}
phaseshifts= pd.read_csv('phaseshifts1.csv')
expt=pd.read_csv('expt.csv')
# 将每一列数据转换为数组
phase={}
for column in phaseshifts.columns:
    phase[column] = phaseshifts[column].to_numpy()
exp={}
for column in expt.columns:
    exp[column] = expt[column].to_numpy()
# 输出查看结果
fig=plt.figure()

ax1=fig.add_subplot(1,1,1)
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['av18'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='b',linestyle='-',linewidth=1,label='av18')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['n3loemn500'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='r',linestyle='-',linewidth=1,label='n3loemn500')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['n3loem'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='r',linestyle=':',linewidth=1,label='n3loem')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['cdbonn'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='g',linestyle='-',linewidth=1,label='cdbonn')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['n2losat'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='y',linestyle='-',linewidth=1,label='sat')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['my'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='black',linestyle='-',linewidth=1,label='my')
x_smooth = np.linspace(phase['Tlabs'].min(), phase['Tlabs'].max(), 100)
spl = make_interp_spline(phase['Tlabs'], phase['n2loopt'])
y_smooth = spl(x_smooth)
ax1.plot(x_smooth,y_smooth,c='#FF00FF',linestyle='-',linewidth=1,label='n2loopt')
ax1.scatter(exp['Tlabs'],exp['PWA93'],c='black',s=35,zorder=3,edgecolor='white', linewidth=1.5)
ax1.set_xlim(0,550)
ax1.set_ylim(-40,50)
ax1.set_xlabel('Lab. Energy (MeV)', fontdict=font_ticks)
ax1.set_ylabel('Phase Phift (deg)', fontdict=font_ticks)
ax1.set_xticks([0, 100,200,300,400,500])

ax1.set_xticklabels( ['0','100', '200','300','400','500'],
               fontfamily=font_family, fontweight=font_weight, fontsize=font_size)
ax1.set_yticks([-40,-20,0,20,40,60])
ax1.set_yticklabels(['-40','-20','0','20','40','60'],
               fontfamily=font_family, fontweight=font_weight, fontsize=font_size)
ax1.set_yticks([-10,10,30,50], minor=True)
ax1.tick_params(direction='in', which='major', length=6)
ax1.tick_params(direction='in', which='minor', length=3)
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
ax1.tick_params(width=0.5)
ax1.text(200,30,'$\mathregular{^1S_0}$',font_text)


fig.legend(bbox_to_anchor=(0.15, 0.12), loc='lower left',prop=font_label,frameon=False)       
# fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.22, hspace=0.4)
plt.savefig('phaseshift1',dpi=512)
# plt.savefig('phaseshift1P1.eps',format='eps')