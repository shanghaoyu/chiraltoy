import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import make_interp_spline
import genfunc_studio as gf
def gauss_legendre_line_mesh(Np,a,b):
    x, w = np.polynomial.legendre.leggauss(Np)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    u = w * 0.5*(b - a)

    return t,u

def fourier_mesh_weight(kmax):
    n0 = 2
    mesh1r, mesh1w = gauss_legendre_line_mesh(64 * n0, 0.0, 1.0)
    mesh2r, mesh2w = gauss_legendre_line_mesh(32 * n0, 1.0, 2.0)
    mesh3r, mesh3w = gauss_legendre_line_mesh(16 * n0, 2.0, 4.0)
    mesh4r, mesh4w = gauss_legendre_line_mesh(8 * n0, 4.0, 8.0)
    mesh5r, mesh5w = gauss_legendre_line_mesh(4 * n0, 8.0, 16.0)
    mesh6r, mesh6w = gauss_legendre_line_mesh(2 * n0, 16.0, 32.0)
    mesh7r, mesh7w = gauss_legendre_line_mesh(n0, 32.0, 40.0)
    meshr = np.concatenate((mesh1r, mesh2r, mesh3r, mesh4r, mesh5r, mesh6r, mesh7r), axis=0)
    meshw = np.concatenate((mesh1w, mesh2w, mesh3w, mesh4w, mesh5w, mesh6w, mesh7w), axis=0)
    return meshr/40.0*kmax,meshw/40.0*kmax



class one_nucleon_density:
    def __init__(self,nucleusname:str,Anum:int,space:str,forcename:str,hw:int,Nmax:int,namen=None,namep=None,namesum=None):
        self.__filepath=f'../src/data/{nucleusname}{Anum}/density/{forcename}'
        self.__foldername=f'hw{hw}-N{Nmax}'
        self.__nucleus=f'{Anum}{nucleusname}'
        self.__forcename=forcename
        self.__filen=namen
        self.__filep=namep
        self.__filesum=namesum
        self.__A=Anum
        self.__hw=hw
        if space=='radial':
            self.__radial=True
        elif space=='momentum':
            self.__radial=False
        else:
            print("wrong type of density!")
            return

        self.__ndata=None
        self.__pdata=None
        self.__sumdata=None
        self.load_data()
        #############################
        self.__rweight=None
        self.rp_sf=None
        self.rhop_sf=None
        self.rn_sf=None
        self.rhon_sf=None
        self.rsum_sf=None
        self.rhosum_sf=None
        ##############################
        self.spe_weight=None
        self.__pweight=None
        self.pp_sf=None
        self.pn_sf=None
        self.psum_sf=None

        ##############################
        self.B_spline_data()
        ##############################
        self.qp_sf=None
        self.tilderhopsf=None
        self.qn_sf=None
        self.tilderhonsf=None
        self.qsum_sf=None
        self.tilderhosumsf=None
        self.__tilderhocal=False
        self.xp_sf=None
        self.spe_x_weight=None
        self.xn_sf=None
        self.xsum_sf=None


        ###############################
        self.__cal_gfunpn1=False
        self.gfuncpn1=None
        self.linkratiosum=None
        self.linkrationut=None
        self.linkratiopro=None
        ###############################

        self.__mu=938.91852
        self.__hbarc=197.32705

        


    def load_data(self):
        '''
        load the data 
        '''
        # get all the data file(folder)
        filenamen=f'{self.__filepath}/{self.__foldername}/{self.__filen}'
        filenamep=f'{self.__filepath}/{self.__foldername}/{self.__filep}'
        filenamesum=f'{self.__filepath}/{self.__foldername}/{self.__filesum}'
        if self.__radial == True:
            self.__ndata = np.loadtxt(filenamen, dtype=[("r", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
            self.__pdata = np.loadtxt(filenamep, dtype=[("r", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
            self.__sumdata = np.loadtxt(filenamesum, dtype=[("r", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
        else:
            self.__ndata = np.loadtxt(filenamen, dtype=[("p", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
            self.__pdata = np.loadtxt(filenamep, dtype=[("p", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
            self.__sumdata = np.loadtxt(filenamesum, dtype=[("p", "<f8"), ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), ("r2rho_im", "<f8")])
    # self.__nndata = np.loadtxt(self.__filenn,dtype=[("r","<f8"),("theta","<f8"),("theta_degree","<f8"),("rho_re","<f8"),("rho_im","<f8"),("r2rho_re","<f8"),("r2rho_im","<f8")])     
    def B_spline_data(self):
        '''
        spline the input data
        '''
        if self.__radial==True:
            self.__rweight=0.04
            bspline_rhop_sf = make_interp_spline(self.__pdata["r"], self.__pdata["rho_re"], k=3)
            bspline_rhon_sf = make_interp_spline(self.__ndata["r"], self.__ndata["rho_re"], k=3)
            bspline_rhosum_sf = make_interp_spline(self.__sumdata["r"], self.__sumdata["rho_re"], k=3)
            # find the max rp of the 
            index = np.where(self.__pdata["rho_re"] < self.__pdata["rho_re"][0]*1e-3)[0][0]
            self.rp_sf= np.arange(0, self.__pdata["r"][index] +self.__rweight, self.__rweight) 
            self.rhop_sf=bspline_rhop_sf(self.rp_sf)
            index = np.where(self.__ndata["rho_re"] < self.__ndata["rho_re"][0]*1e-3)[0][0]
            self.rn_sf= np.arange(0, self.__ndata["r"][index] +self.__rweight, self.__rweight) 
            self.rhon_sf=bspline_rhon_sf(self.rn_sf)
            index = np.where(self.__sumdata["rho_re"] < self.__sumdata["rho_re"][0]*1e-3)[0][0]
            self.rsum_sf= np.arange(0, self.__sumdata["r"][index] + self.__rweight, self.__rweight) 
            self.rhosum_sf=bspline_rhosum_sf(self.rsum_sf)
        else:
            self.__pweight=0.01
            bspline_rhop_sf = make_interp_spline(self.__pdata["p"], self.__pdata["rho_re"], k=3)
            bspline_rhon_sf = make_interp_spline(self.__ndata["p"], self.__ndata["rho_re"], k=3)
            bspline_rhosum_sf = make_interp_spline(self.__sumdata["p"], self.__sumdata["rho_re"], k=3)
            # find the max rp of the 
            # index = np.where(self.__pdata["rho_re"] < 1e-9)[0][0]
            self.pp_sf= np.arange(0, self.__pdata["p"][-2] +self.__pweight, self.__pweight) 
            self.rhop_sf=bspline_rhop_sf(self.pp_sf)
            # index = np.where(self.__ndata["rho_re"] < 1e-9)[0][0]
            self.pn_sf= np.arange(0, self.__ndata["p"][-2] +self.__pweight, self.__pweight) 
            self.rhon_sf=bspline_rhon_sf(self.pn_sf)
            # index = np.where(self.__sumdata["rho_re"] < 1e-9)[0][0]
            # self.psum_sf= np.arange(0, self.__sumdata["p"][index] + self.__pweight, self.__pweight) 
            # self.rhosum_sf=bspline_rhosum_sf(self.psum_sf)
            self.psum_sf,self.spe_weight= fourier_mesh_weight(self.__sumdata["p"][-2]) 
            self.rhosum_sf=bspline_rhosum_sf(self.psum_sf)
        
    def tilderho_CM_q(self,q):
        '''
        q in fm^-1
        '''
        alpha=(self.__A*self.__mu*self.__hw)/(2*self.__hbarc**2)
        return np.exp(-q*q/(8.0*alpha))
    
    def tilderho_CM_r(self,r):
        alpha=(self.__A*self.__mu*self.__hw)/(2*self.__hbarc**2)
        return np.exp(-alpha*r*r/(2.0*self.__A**2))/(2.0*np.pi)**3
        # return 1/(2.0*np.pi)**3
    
    @staticmethod
    def rtoq(rhor,rmesh,wmesh,K,q):
        '''
        Parameters:
        ---------------------------------------
        rmesh: in unit fm
        q: in unit fm^-1
        '''
        tilderhoq=0
        for i,ri in enumerate(rmesh):
            tilderhoq=tilderhoq+spherical_jn(K,q*rmesh[i])*ri**2*wmesh[i]*rhor[i]*(4*np.pi)
        return tilderhoq
    @staticmethod
    def qtor(rhoq,qmesh,wmesh,K,r):
        tilderhor=0
        for i,qi in enumerate(qmesh):
            tilderhor=tilderhor+spherical_jn(K,qmesh[i]*r)*qi**2*wmesh[i]*rhoq[i]/(2*np.pi**2)
        return tilderhor
    def cal_tilderho_sf_q(self):
        wmesh = self.__rweight*np.ones_like(self.rp_sf, dtype=float)
        # get_qmax
        qmesh=np.arange(0,10,0.1)
        tempt=np.zeros(100)
        for i,q in enumerate(qmesh):
            tempt[i]=self.rtoq(self.rhop_sf,self.rp_sf,wmesh,0,q)
        ratio=tempt/self.tilderho_CM_q(qmesh)
        index=np.where(ratio < ratio[0]*1e-2)[0][0]
        # calculate the tilderho
        self.qp_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhopsf=np.zeros((len(self.qp_sf)))
        for i,q in enumerate(self.qp_sf):
            self.tilderhopsf[i]=self.rtoq(self.rhop_sf,self.rp_sf,wmesh,0,q)
        
        wmesh = self.__rweight*np.ones_like(self.rn_sf, dtype=float)
        for i,q in enumerate(qmesh):
            tempt[i]=self.rtoq(self.rhon_sf,self.rn_sf,wmesh,0,q)
        ratio=tempt/self.tilderho_CM_q(qmesh)
        index=np.where(ratio < ratio[0]*1e-2)[0][0]
        # calculate the tilderho
        self.qn_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhonsf=np.zeros((len(self.qn_sf)))
        for i,q in enumerate(self.qn_sf):
            self.tilderhonsf[i]=self.rtoq(self.rhon_sf,self.rn_sf,wmesh,0,q)

        wmesh = self.__rweight*np.ones_like(self.rsum_sf, dtype=float)
        for i,q in enumerate(qmesh):
            tempt[i]=self.rtoq(self.rhosum_sf,self.rsum_sf,wmesh,0,q)
        ratio=tempt/self.tilderho_CM_q(qmesh)
        index=np.where(ratio < ratio[0]*1e-2)[0][0]
        # calculate the tilderho
        self.qsum_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhosumsf=np.zeros((len(self.qsum_sf)))
        for i,q in enumerate(self.qsum_sf):
            self.tilderhosumsf[i]=self.rtoq(self.rhosum_sf,self.rsum_sf,wmesh,0,q)

    def cal_tilderho_sf_r(self):
        wmesh = self.__pweight*np.ones_like(self.pp_sf, dtype=float)
        # get_qmax
        rmesh=np.arange(0,20,0.1)
        tempt=np.zeros(200)
        for i,r in enumerate(rmesh):
            tempt[i]=self.qtor(self.rhop_sf,self.pp_sf,wmesh,0,r)
        ratio=tempt/(self.tilderho_CM_r(rmesh)*(2*np.pi)**3)
        index=np.where(ratio < ratio[0]*1e-4)[0][0]
        # calculate the tilderho
        self.xp_sf=np.arange(0,rmesh[index],self.__pweight)
        self.tilderhopsf=np.zeros((len(self.xp_sf)))
        for i,x in enumerate(self.xp_sf):
            self.tilderhopsf[i]=self.qtor(self.rhop_sf,self.pp_sf,wmesh,0,x)
        
        
        wmesh = self.__pweight*np.ones_like(self.pn_sf, dtype=float)
        # get_qmax
        rmesh=np.arange(0,20,0.1)
        tempt=np.zeros(200)
        for i,r in enumerate(rmesh):
            tempt[i]=self.qtor(self.rhon_sf,self.pn_sf,wmesh,0,r)
        ratio=tempt/(self.tilderho_CM_r(rmesh)*(2*np.pi)**3)
        index=np.where(ratio < ratio[0]*1e-4)[0][0]
        # calculate the tilderho
        self.xn_sf=np.arange(0,rmesh[index],self.__pweight)
        self.tilderhonsf=np.zeros((len(self.xn_sf)))
        for i,x in enumerate(self.xn_sf):
            self.tilderhonsf[i]=self.qtor(self.rhon_sf,self.pn_sf,wmesh,0,x)

        # wmesh = self.__pweight*np.ones_like(self.psum_sf, dtype=float)
        # get_qmax
        rmesh=np.arange(0,20,0.1)
        tempt=np.zeros(200)
        for i,r in enumerate(rmesh):
            tempt[i]=self.qtor(self.rhosum_sf,self.psum_sf,self.spe_weight,0,r)
        ratio=tempt/(self.tilderho_CM_r(rmesh)*(2*np.pi)**3)
        index=np.where(ratio < ratio[0]*1e-4)[0][0]
        print(index)
        # calculate the tilderho
        # use the special points

        self.xsum_sf,self.spe_x_weight=fourier_mesh_weight(rmesh[index])
        self.tilderhosumsf=np.zeros((len(self.xsum_sf)))
        for i,x in enumerate(self.xsum_sf):
            self.tilderhosumsf[i]=self.qtor(self.rhosum_sf,self.psum_sf,self.spe_weight,0,x)

    def rhopro_ti_r(self,r):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_q()
            self.__tilderhocal=True
        wmesh = self.__rweight*np.ones_like(self.qp_sf, dtype=float)
        # calculate rhosf/rhocm
        ratioq=self.tilderhopsf/self.tilderho_CM_q(self.qp_sf)
        # print(ratioq)
        return self.qtor(ratioq,self.qp_sf,wmesh,0,r)
    
    def rhopro_ti_p(self,p):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_r()
            self.__tilderhocal=True
        wmesh = self.__pweight*np.ones_like(self.xp_sf, dtype=float)
        # calculate rhosf/rhocm
        ratior=self.tilderhopsf/(self.tilderho_CM_r(self.xp_sf)*(2*np.pi)**3)
        # print(ratioq)
        return self.rtoq(ratior,self.xp_sf,wmesh,0,p)
    
    def rhonut_ti_r(self,r):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_q()
            self.__tilderhocal=True
        wmesh = self.__rweight*np.ones_like(self.qn_sf, dtype=float)
        # calculate rhosf/rhocm
        ratioq=self.tilderhonsf/self.tilderho_CM_q(self.qn_sf)
        # print(ratioq)
        return self.qtor(ratioq,self.qn_sf,wmesh,0,r)
    
    def rhonut_ti_p(self,p):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_r()
            self.__tilderhocal=True
        wmesh = self.__pweight*np.ones_like(self.xn_sf, dtype=float)
        # calculate rhosf/rhocm
        ratior=self.tilderhonsf/(self.tilderho_CM_r(self.xn_sf)*(2*np.pi)**3)
        # print(ratioq)
        return self.rtoq(ratior,self.xn_sf,wmesh,0,p)
    
    def rhosum_ti_r(self,r):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_q()
            self.__tilderhocal=True
        wmesh = self.__rweight*np.ones_like(self.qsum_sf, dtype=float)
        # calculate rhosf/rhocm
        ratioq=self.tilderhosumsf/self.tilderho_CM_q(self.qsum_sf)
        # print(ratioq)
        return self.qtor(ratioq,self.qsum_sf,wmesh,0,r)
    
    def rhosum_ti_p(self,p):
        if self.__tilderhocal==False:
            # print(self.__tilderhocal)
            self.cal_tilderho_sf_r()
            self.__tilderhocal=True
        # wmesh = self.__pweight*np.ones_like(self.xsum_sf, dtype=float)
        # calculate rhosf/rhocm
        ratior=self.tilderhosumsf/(self.tilderho_CM_r(self.xsum_sf)*(2*np.pi)**3)
        # print(ratioq)
        return self.rtoq(ratior,self.xsum_sf,self.spe_x_weight,0,p)
    
    def cal_genf_pn1(self):
        studio=gf.genfunc_studio(0,2,15,80,200,self.__forcename)
        pmesh,u,w=studio.phipn1(0)
        self.gfuncpn1=make_interp_spline(pmesh,(u+w)/(2.0*np.pi**2), k=3)


    def rhosum_ti_p_link(self,p,klink=2):
        # get the general function np
        if self.__cal_gfunpn1==False:
            self.cal_genf_pn1()
            self.__cal_gfunpn1=True
        if self.linkratiosum is None:
            self.linkratiosum=self.rhosum_ti_p(klink)/self.gfuncpn1(klink)
            # print(self.linkratiosum)
        if p<klink:
            return self.rhosum_ti_p(p)
        else:
            return self.gfuncpn1(p)*self.linkratiosum
        
    def rhonut_ti_p_link(self,p,klink=2):
        # get the general function np
        if self.__cal_gfunpn1==False:
            self.cal_genf_pn1()
            self.__cal_gfunpn1=True
        if self.linkrationut is None:
            self.linkrationut=self.rhonut_ti_p(klink)/self.gfuncpn1(klink)
            # print(f"ratio is{self.linkrationut}")
        if p<klink:
            return self.rhonut_ti_p(p)
        else:
            return self.gfuncpn1(p)*self.linkrationut
        
    def scaling_factor(self,kfermi=1.35):
        # the integral kmax and number
        kmax=6
        number=300
        # calculate the norm of rho_ti
        p=np.linspace(0,kmax,number)
        rho_ti=np.zeros((number))
        for i,pi in enumerate(p):
            rho_ti[i]=self.rhosum_ti_p_link(pi)
        norm=np.sum(rho_ti*p**2*kmax/number*4*np.pi)
        print(f"the norm of {self.__nucleus} is {norm}")
        norm=np.sum(self.gfuncpn1(p)*p**2*kmax/number)
        print(f"the norm of 2H is {norm}")
        sum1=0
        sum2=0
        for i,pi in enumerate(p):
            if pi>kfermi:
                sum2=sum2+self.gfuncpn1(pi)*pi**2*kmax/number
            elif pi<kfermi:
                sum1=sum1+rho_ti[i]*pi**2*kmax/number*4*np.pi
        sum1=self.__A-sum1
        print(f"the sum 3He is {sum1}")
        return sum1/sum2/self.__A
            


class two_nucleon_density:
    def __init__(self,N:int,Z:int,rho:callable=None):
        self.one_body_density=rho
        self.ave_proton_fermi_mom=None
        self.normfermden=None
        self.N=N
        self.Z=Z
        self.A=N+Z
        if self.one_body_density:
            self.cal_ave_fermin_momentum()
            print(f"average proton fermi mom is: {self.ave_proton_fermi_mom}")
            # self.ave_proton_fermi_mom=1.05
            self.cal_norm_fermidense()
    def long_range_density(self,r):
        # the Rmesh and wR
        # Rmesh,wR=gauss_legendre_inf_mesh(100)
        Rmesh=np.arange(0,20,0.1)
        wR=0.1*np.ones_like(Rmesh, dtype=float)
        # the zmesh and wz
        zmesh=np.arange(-1,1,0.05)
        wz=0.05*np.ones_like(zmesh, dtype=float)
        density=0
        for i,Ri in enumerate(Rmesh):
            for j,zi in enumerate(zmesh):
                rplus=np.sqrt(Ri**2+r**2/4.0+Ri*r*zi)
                rminus=np.sqrt(Ri**2+r**2/4.0-Ri*r*zi)
                density=density+Ri**2*wR[i]*wz[j]*self.one_body_density(rplus)*self.one_body_density(rminus)
        return density
    def fermin_momentum(self,R):
        if self.one_body_density(R)>0:
            fermi=(3.0*np.pi**2*self.one_body_density(R))**(1/3)
        else:
            fermi=0
        return fermi
    
    def cal_ave_fermin_momentum(self):
        Rmesh=np.arange(0,10,0.05)
        wR=0.05*np.ones_like(Rmesh, dtype=float)
        A=0
        B=0
        for i,Ri in enumerate(Rmesh):
            A=A+self.fermin_momentum(Ri)*self.one_body_density(Ri)*Ri**2*wR[i]
            B=B+self.one_body_density(Ri)*Ri**2*wR[i]

        self.ave_proton_fermi_mom=A/B
    
    def rho_fermi(self,r):
        dense=self.long_range_density(r)*(1-1/2.0*(3*spherical_jn(1,self.ave_proton_fermi_mom*r)/(self.ave_proton_fermi_mom*r))**2)
        return dense
        
    def cal_norm_fermidense(self):
    
        norm=0
        Rmesh=np.arange(0.02,10,0.05)
        wR=0.05*np.ones_like(Rmesh, dtype=float)
        for i,Ri in enumerate(Rmesh):
            norm=norm+4*np.pi*Ri**2*wR[i]*self.rho_fermi(Ri)
        
        self.normfermden=self.Z*(self.Z-1)/(2.0*norm)
        print("norm for rho_fermi calculated.")
