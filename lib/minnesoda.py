import numpy as np
def minnesoda(lp,l,s,j,tz,r):
    '''
    v(6): means 
    '''

    vR=200.0*np.exp(-1.487*r**2)
    vT=178.0*np.exp(0.639*r**2)
    vS=91.85*np.exp(0.465*r**2)
    A=1/2.0*(vR+(vT+vS)/2.0)
    B=(vT-vS)/4.0
    C=-1/2.0*(vR+(vT+vS)/2.0)
    D=-1/4.0*(vT-vS)
    vv=np.zeros((4))
    vv[0]=A+B/2.0+C/2.0+D/4.0
    vv[1]=C/2.0+D/4.0
    vv[2]=B/2.0+D/4.0
    vv[3]=D/4.0
    t = (l+s+1)%2
    s1ds2=4*s-3
    t1dt2=4*t-3
    vc=vv[0]+t1dt2*vv[1]+s1ds2*vv[2]+t1dt2*s1ds2*vv[3]
    if tz == 0: # np case
        if lp == l:
            v=vc
        else:
            v=0
    if np.abs(tz) == 1: #pp or nn case
        if t == 1:
            if lp == l:
                v=vc
            else:
                v=0
        else:
            print("this channel is forbidden!")
            return
    return v

    # change to lsj