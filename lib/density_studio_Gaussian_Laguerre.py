import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import make_interp_spline
import genfunc_studio as gf
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import roots_laguerre


def fit_exponential(x, y):
    """
    根据 f(x) = g(x)e^(-ax) 的形式拟合出参数 a。

    参数：
    x: numpy.ndarray
        输入的 x 数据数组。
    y: numpy.ndarray
        输入的 y 数据数组。

    返回：
    a: float
        拟合出的参数 a。
    """
    # 定义目标函数，与 f(x) 的形式相匹配。
    def model(x, a):
        return np.exp(-a * x)  # 假设 g(x)=1，可以根据需要调整

    # 对数据进行拟合，找到 a
    popt, _ = curve_fit(model, x, y)
    
    # 提取拟合参数 a
    a = popt[0]
    return a


def spine_for_glg(x,y,q):
    f=make_interp_spline(x, y, k=3)
    if q<x[-1]:
        return f(q)
    else:
        return y[-1]




class one_nucleon_density:
    def __init__(self,nucleusname:str,Anum:int,space:str,forcename:str,hw:int,Nmax:int,genfuncforce='n3loemn500',namen=None,namep=None,namesum=None,Ngauss=40):
        self.__filepath=f'../src/data/{nucleusname}{Anum}/density/{forcename}'
        self.__foldername=f'hw{hw}-N{Nmax}'
        self.__nucleus=f'{Anum}{nucleusname}'
        self.__forcename=genfuncforce
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
        # 设置阶数 n
        n = Ngauss
        # 获取零点和权重
        self.__glpoints,self.__glweights = roots_laguerre(n)
        self.__rweight=None
        self.rp_sf=None
        self.rhop_sf=np.zeros((n))
        self.rn_sf=None
        self.rhon_sf=np.zeros((n))
        self.rsum_sf=None
        self.rhosum_sf=np.zeros((n))
        ##############################
        self.__pweight=None
        self.pp_sf=self.__glpoints
        self.pn_sf=self.__glpoints
        self.psum_sf=self.__glpoints

        ##############################
        self.spline_data()
        ##############################
        self.qp_sf=None
        self.tilderhopsf=None
        self.qn_sf=None
        self.tilderhonsf=None
        self.qsum_sf=None
        self.tilderhosumsf=None
        self.__tilderhocal=False
        self.xp_sf=None
        self.xn_sf=None
        self.xsum_sf=None


        ###############################
        self.__cal_gfunpn1=False
        self.gfuncpn1=None
        self.__cal_gfunpn0=False
        self.gfuncpn0=None
        self.__normgenfunc=None
        self.linkcontactsum=False
        self.linkcontactnut=False
        self.linkcontactpro=False
        self.contacts=None
        ###############################

        self.__print=False

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
    @staticmethod
    def find_last_decreasing_index(arr):
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                return i
        return len(arr) - 1  
   

    def spline_data(self):
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
            ap=fit_exponential(self.__pdata["p"][-70:-1], self.__pdata["rho_re"][-70:-1])
            an=fit_exponential(self.__ndata["p"][-70:-1], self.__ndata["rho_re"][-70:-1])
            asum=fit_exponential(self.__sumdata["p"][-70:-1], self.__sumdata["rho_re"][-70:-1])
            for i,x in enumerate(self.__glpoints):
                self.rhop_sf[i]=spine_for_glg(self.__pdata["p"],self.__pdata["rho_re"]*np.exp(ap*self.__pdata["p"]),x/ap)/ap
                self.rhon_sf[i]=spine_for_glg(self.__ndata["p"],self.__ndata["rho_re"]*np.exp(an*self.__ndata["p"]),x/an)/an
                self.rhosum_sf[i]=spine_for_glg(self.__sumdata["p"],self.__sumdata["rho_re"]*np.exp(asum*self.__sumdata["p"]),x/asum)/asum                
            print(self.__glpoints,self.__glweights)
            print(self.rhop_sf)
            print(self.rhon_sf)
            print(self.rhosum_sf)
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
        index=np.where(ratio < ratio[0]*1e-3)[0][0]
        # calculate the tilderho
        self.qp_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhopsf=np.zeros((len(self.qp_sf)))
        for i,q in enumerate(self.qp_sf):
            self.tilderhopsf[i]=self.rtoq(self.rhop_sf,self.rp_sf,wmesh,0,q)
        
        wmesh = self.__rweight*np.ones_like(self.rn_sf, dtype=float)
        for i,q in enumerate(qmesh):
            tempt[i]=self.rtoq(self.rhon_sf,self.rn_sf,wmesh,0,q)
        ratio=tempt/self.tilderho_CM_q(qmesh)
        index=np.where(ratio < ratio[0]*1e-3)[0][0]
        # calculate the tilderho
        self.qn_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhonsf=np.zeros((len(self.qn_sf)))
        for i,q in enumerate(self.qn_sf):
            self.tilderhonsf[i]=self.rtoq(self.rhon_sf,self.rn_sf,wmesh,0,q)

        wmesh = self.__rweight*np.ones_like(self.rsum_sf, dtype=float)
        for i,q in enumerate(qmesh):
            tempt[i]=self.rtoq(self.rhosum_sf,self.rsum_sf,wmesh,0,q)
        ratio=tempt/self.tilderho_CM_q(qmesh)
        index=np.where(ratio < ratio[0]*1e-3)[0][0]
        # calculate the tilderho
        self.qsum_sf=np.arange(0,qmesh[index],self.__rweight)
        self.tilderhosumsf=np.zeros((len(self.qsum_sf)))
        for i,q in enumerate(self.qsum_sf):
            self.tilderhosumsf[i]=self.rtoq(self.rhosum_sf,self.rsum_sf,wmesh,0,q)

    def cal_tilderho_sf_r(self):
        # get_qmax
        tempt=np.zeros((len(self.__glpoints)))
        for i,r in enumerate(self.__glpoints):
            tempt[i]=self.qtor(self.rhop_sf,self.pp_sf,self.__glweights,0,r)
        print(tempt)
        # print(index)
        # calculate the tilderho
        # self.xp_sf=self.__glpoints
        # self.tilderhopsf=np.zeros((len(self.xp_sf)))
        # for i,x in enumerate(self.xp_sf):
        #     self.tilderhopsf[i]=self.qtor(self.rhop_sf,self.pp_sf,wmesh,0,x)
        # # self.xp_sf=np.arange(0,20,self.__pweight)
        
        # wmesh = self.__pweight*np.ones_like(self.pn_sf, dtype=float)
        # # get_qmax
        # rmesh=np.arange(0,30,0.1)
        # tempt=np.zeros(300)
        # for i,r in enumerate(rmesh):
        #     tempt[i]=self.qtor(self.rhon_sf,self.pn_sf,wmesh,0,r)
        # ratio=tempt/(self.tilderho_CM_r(rmesh)*(2*np.pi)**3)
        # index=self.find_last_decreasing_index(ratio)
        # # calculate the tilderho
        # self.xn_sf=np.arange(0,rmesh[-1],self.__pweight)
        # self.tilderhonsf=np.zeros((len(self.xn_sf)))
        # for i,x in enumerate(self.xn_sf):
        #     self.tilderhonsf[i]=self.qtor(self.rhon_sf,self.pn_sf,wmesh,0,x)

        # wmesh = self.__pweight*np.ones_like(self.psum_sf, dtype=float)
        # # get_qmax
        # rmesh=np.arange(0,25,0.1)
        # tempt=np.zeros(250)
        # for i,r in enumerate(rmesh):
        #     tempt[i]=self.qtor(self.rhosum_sf,self.psum_sf,wmesh,0,r)
        # ratio=tempt/(self.tilderho_CM_r(rmesh)*(2*np.pi)**3)
        # index=self.find_last_decreasing_index(ratio)
        # print(index)
        # # calculate the tilderho
        # self.xsum_sf=np.arange(0,rmesh[-1],self.__pweight)
        # self.tilderhosumsf=np.zeros((len(self.xsum_sf)))
        # for i,x in enumerate(self.xsum_sf):
        #     self.tilderhosumsf[i]=self.qtor(self.rhosum_sf,self.psum_sf,wmesh,0,x)

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
        wmesh = self.__pweight*np.ones_like(self.xsum_sf, dtype=float)
        # calculate rhosf/rhocm
        ratior=self.tilderhosumsf/(self.tilderho_CM_r(self.xsum_sf)*(2*np.pi)**3)
        # print(ratioq)
        return self.rtoq(ratior,self.xsum_sf,wmesh,0,p)
    
    def cal_genf_pn1(self):
        studio=gf.genfunc_studio(0,2,15,80,200,self.__forcename)
        pmesh,u,w=studio.phipn1(255)
        self.gfuncpn1=make_interp_spline(pmesh,(u+w)/(2.0*np.pi**2), k=3)

    def cal_genf_pn0(self):
        studio=gf.genfunc_studio(0,2,15,80,200,self.__forcename)
        pmesh,phi=studio.phipn0(255)
        self.gfuncpn0=make_interp_spline(pmesh,phi/(2.0*np.pi**2), k=3)
        
    
    @staticmethod
    def fit_functions_least_squares(f1, f2, f3, a, b, n_points=20):
        """
        用最小二乘法拟合 f1(x) = c1 * f2(x) + c2 * f3(x)。

        参数：
            f1: callable, 目标函数 f1(x)
            f2: callable, 基函数 f2(x)
            f3: callable, 基函数 f3(x)
            a: float, 拟合区间的起点
            b: float, 拟合区间的终点
            n_points: int, 在区间 [a, b] 上的采样点数 (默认 100)

        输出：
            c1: float, f2(x) 的系数
            c2: float, f3(x) 的系数
        """
        # 在区间 (a, b) 上生成采样点
        x = np.linspace(a, b, n_points)

        # 计算 f1, f2, f3 在采样点上的值
        y1 = f1(x)  # 目标函数值
        y2 = f2(x)  # f2 的值
        y3 = f3(x)  # f3 的值

        # 构造设计矩阵 F 和目标值 y
        F = np.vstack([y2, y3]).T  # 设计矩阵 (n_points x 2)
        y = y1  # 目标值

        # 使用最小二乘法求解 c1, c2
        C, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        c1, c2 = C

        return c1, c2

    def rhosum_ti_p_link(self,p,klink=[1.9,2.1]):
        # get the general function np
        if self.__cal_gfunpn1==False:
            self.cal_genf_pn1()
            self.__cal_gfunpn1=True
        if self.__cal_gfunpn0==False:
            self.cal_genf_pn0()
            self.__cal_gfunpn0=True
        if self.linkcontactsum==False:
            c1,c0=self.fit_functions_least_squares(self.rhosum_ti_p,self.gfuncpn1,self.gfuncpn0,klink[0],klink[1])
            print(f"nuclear contacts(sum): c1{c1},c0{c0}")
            self.linkcontactsum=True
            self.contacts=[c1,c0]
            self.linkcontactnut=True
            # print(self.linkratiosum)
        if p<klink[0]:
            return self.rhosum_ti_p(p)
        else:
            return self.gfuncpn1(p)*self.contacts[0]+self.gfuncpn0(p)*self.contacts[1]
        
    def rhonut_ti_p_link(self,p,klink=[1.8,2.0]):
        # get the general function np
        if self.__cal_gfunpn1==False:
            self.cal_genf_pn1()
            self.__cal_gfunpn1=True
        if self.__cal_gfunpn0==False:
            self.cal_genf_pn0()
            self.__cal_gfunpn0=True
        if self.linkcontactnut==False:
            c1,c0=self.fit_functions_least_squares(self.rhonut_ti_p,self.gfuncpn1,self.gfuncpn0,klink[0],klink[1])
            print(f"nuclear contacts(neutron): c1:{c1},c0:{c0}")
            self.contacts=[c1,c0]
            self.linkcontactnut=True
            # print(f"ratio is{self.linkrationut}")
        if p<klink[0]:
            return self.rhonut_ti_p(p)
        else:
            return self.gfuncpn1(p)*self.contacts[0]+self.gfuncpn0(p)*self.contacts[1]
        
    def rhopro_ti_p_link(self,p,klink=2):
        # get the general function np
        if self.__cal_gfunpn1==False:
            self.cal_genf_pn1()
            self.__cal_gfunpn1=True
        if self.__cal_gfunpn0==False:
            self.cal_genf_pn0()
            self.__cal_gfunpn0=True
        if self.linkcontactpro==False:
            c1,c0=self.fit_functions_least_squares(self.rhopro_ti_p,self.gfuncpn1,self.gfuncpn0,klink[0],klink[1])
            print(f"nuclear contacts(proton): c1:{c1},c0:{c0}")
            self.contacts=[c1,c0]
            self.linkcontactpro=True
            # print(f"ratio is{self.linkrationut}")
        if p<klink[0]:
            return self.rhopro_ti_p(p)
        else:
            return self.gfuncpn1(p)*self.contacts[0]+self.contacts[1]*self.gfuncpn0(p)
        
    def scaling_factor(self,kfermi=1.35,klink=[1.9,2.1],res=True):
        # the integral kmax and number
        kmax=7
        number=300
        # calculate the norm of rho_ti
        p=np.linspace(0,kmax,number)
        rho_ti=np.zeros((number))
        for i,pi in enumerate(p):
            rho_ti[i]=self.rhosum_ti_p_link(pi,klink=klink)       
        sum1=0
        sum2=0
        sum1p=0
        for i,pi in enumerate(p):
            if pi>kfermi:
                sum2=sum2+self.gfuncpn1(pi)*pi**2*kmax/number
                sum1p=sum1p+rho_ti[i]*pi**2*kmax/number*4*np.pi
            elif pi<kfermi:
                sum1=sum1+rho_ti[i]*pi**2*kmax/number*4*np.pi

        sum1=self.__A-sum1
        if self.__print==False:
            norm1=np.sum(rho_ti*p**2*kmax/number*4*np.pi)
            print(f"the norm of {self.__nucleus} in hw{self.__hw} is: {norm1}")
            norm2=np.sum(self.gfuncpn1(p)*p**2*kmax/number)
            print(f"the norm of 2H in force{self.__forcename} is: {norm2}")
            self.__normgenfunc=norm2
            self.__print=True
        print(f"the sum 3He is {sum1}")
        print(f"the sum 2H is {sum2}")
        if res==True:
            return sum1/sum2/self.__A*self.__normgenfunc,sum2/self.__normgenfunc
        else:
            return sum1p/sum2/self.__A*self.__normgenfunc,sum2/self.__normgenfunc


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
