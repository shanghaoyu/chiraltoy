import os
from scipy import interpolate
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import make_interp_spline


def solve_for_p2(epsilon,p1,z,A=4,Binit=0.0282957,Bfinal=0,eps=10e-5):
    """
    solve the delta function in spectral function, the result is p2

    Parameters
    -------------------------------------------
    epsilon: missing energy                                  GeV
    p1: missing momentum                                     GeV/c
    z: cos(theta) theta is the angle between p1 and p2       (-1,1)
    A: the nuclear number                  
    Binit: binding energy of the initial state (A)           GeV
    Bfinal: binding energy of the  final state (A-2)         GeV
    eps: check for the solution (error)
    """
    mass=0.938918
    # mass=0.9395654
    const=2*mass-epsilon-Binit+Bfinal-p1**2/(2*mass*(A-2))
    a=1/(2*mass*(A-2))**2
    b=p1*z/(mass*(A-2))**2
    c=p1**2*z**2/(mass*(A-2))**2-const/(mass*(A-2))-1
    d=-2*p1*z*const/(mass*(A-2))
    e=const**2-mass**2
    coefficients = [a, b, c, d, e]
    # find the roots
    roots = np.roots(coefficients)
    # select the roots
    # temp=2*mass-epsilon-Binit+Bfinal-(p1**2+roots**2+2*p1*roots*z)**2/(2*mass*(A-2))
    result=roots[np.isreal(roots) & (roots > 0)]
    temp=2*mass-epsilon-Binit+Bfinal-(p1**2+result**2+2*p1*result*z)**2/(2*mass*(A-2))
    p2=result[temp>0]

    # check the solution
    f=epsilon+np.sqrt(p2**2+mass**2)-2*mass+Binit-Bfinal+(p1**2+p2**2+2*p1*p2*z)/(2*mass*(A-2))
    epss=np.abs(f)<eps
    if epss.all():
        return p2
    else:
        print("the solution isn't right!")


def find_p2_min_max(epsilon,p1,A=4,Binit=0.0282957,Bfinal=0):
    """
    

    Parameters
    -------------------------------------------
    epsilon: missing energy                                  GeV
    p1: missing momentum                                     GeV/c
    A: the nuclear number                  
    Binit: binding energy of the initial state (A)           GeV
    Bfinal: binding energy of the  final state (A-2)         GeV
    """
    p20=solve_for_p2(epsilon,p1,-1,A,Binit,Bfinal)
    p2s=[]
    if len(p20)==1:
        for i in range(100):
            z=-1+i*2.0/100
            p2=solve_for_p2(epsilon,p1,z,A,Binit,Bfinal)
            p2s.append(p2)
        if all(p2s[i] >= p2s[i + 1] for i in range(len(p2s) - 1)):
            return p2s[99][0],p2s[0][0]
        else:
            print("single p2,but not monotonic!")
    elif len(p20)==2:
        return p20[1],p20[0]
    elif len(p20)==0:
        return 0,0
    else:
        print("don't know this situation")


def interpolatedphi2(datax,datay,x):
    '''
    The phi^2(x)
    
    Parameters
    -------------------------------------------
    datax,datay: the numeral function of phi
    x: relative momentums,np arrays                                 fm^-1
    
    Return
    -------------------------------------------
    y: np arrays, the phi^2
    '''
    f=interpolate.interp1d(datax,datay,kind=2)
    if isinstance(x, np.ndarray):
        length=len(x)
        y=np.zeros((length))
        for k,xi in enumerate(x):
            if xi>255.0/197.32705:
                y[k]=f(xi)
            else:
                y[k]=0
        return y
    else:
        if x>255.0/197.32705:
            y=f(x)
        else:
            y=0
        return y
    
def gauss_legendre_line_mesh(Np,a,b):
    x, w = np.polynomial.legendre.leggauss(Np)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    u = w * 0.5*(b - a)

    return t,u

def spectralfunction(p1,epsilon1,sigmaCM,datax,datay,A=4,Binit=0.0282957,Bfinal=0):
    '''
    Parameters
    -------------------------------------------
    Return
    -------------------------------------------

    '''
    mass= 0.938918


    # get the p2max and p2min
    p2min,p2max=find_p2_min_max(epsilon1,p1,Bfinal=Bfinal)
    
    # prepare the mesh points of p2
    meshnum=150
    p2,p2w=gauss_legendre_line_mesh(meshnum,p2min,p2max)

    # prepare the intrgtated function of p2
    # constC (nd array)
    constC=epsilon1+np.sqrt(p2**2+mass**2)-2*mass+Binit-Bfinal+(p1**2+p2**2)/(2*mass*(A-2))
    # relative momentum (nd array)
    prel=np.sqrt(p1**2+p2**2+2*constC*mass*(A-2))/2.0*1000/197.32705
    # average momentum **2 (nd array)
    pave2=p1**2+p2**2-2*constC*mass*(A-2)
    # coefficient (const)
    coeff=1.0/(64*np.pi**4*np.sqrt(2*np.pi)**3)*10**9/197.32705**3*4*np.pi
    # the integrated function (as a nd array)
    func=p2*mass*(A-2)/p1*interpolatedphi2(datax,datay,prel)/sigmaCM**3*np.exp(-pave2/(2.0*sigmaCM**2))
    # integrate the function
    spectralfunc=np.sum(coeff*func*p2w)
    return spectralfunc



def solve_for_p2_ppcase(epsilon,p1,z,A=4,Binit=0.0282957,Bfinal=0,eps=10e-7):
    """
    solve the delta function in spectral function, the result is p2

    Parameters
    -------------------------------------------
    epsilon: missing energy                                  GeV
    p1: missing momentum                                     GeV/c
    z: cos(theta) theta is the angle between p1 and p2       (-1,1)
    A: the nuclear number                  
    Binit: binding energy of the initial state (A)           GeV
    Bfinal: binding energy of the  final state (A-2)         GeV
    eps: check for the solution (error)
    """
    massp=0.938272
    massn=0.9395654
    mass= 0.938918
    const=2*massp-epsilon-Binit+Bfinal-p1**2/(2*massn*(A-2))
    a=1/(2*massn*(A-2))**2
    b=p1*z/(massn*(A-2))**2
    c=p1**2*z**2/(massn*(A-2))**2-const/(massn*(A-2))-1
    d=-2*p1*z*const/(massn*(A-2))
    e=const**2-massp**2
    coefficients = [a, b, c, d, e]
    # find the roots
    roots = np.roots(coefficients)
    # select the roots
    # temp=2*mass-epsilon-Binit+Bfinal-(p1**2+roots**2+2*p1*roots*z)**2/(2*mass*(A-2))
    result=roots[np.isreal(roots) & (roots > 0)]
    temp=2*massp-epsilon-Binit+Bfinal-(p1**2+result**2+2*p1*result*z)**2/(2*massn*(A-2))
    p2=result[temp>0]

    # check the solution
    f=epsilon+np.sqrt(p2**2+massp**2)-2*massp+Binit-Bfinal+(p1**2+p2**2+2*p1*p2*z)/(2*massn*(A-2))
    epss=np.abs(f)<eps
    if epss.all():
        return p2
    else:
        print("the solution isn't right!")

def find_p2_min_max_ppcase(epsilon,p1,A=4,Binit=0.0282957,Bfinal=0):
    """
    Parameters
    -------------------------------------------
    epsilon: missing energy                                  GeV
    p1: missing momentum                                     GeV/c
    A: the nuclear number                  
    Binit: binding energy of the initial state (A)           GeV
    Bfinal: binding energy of the  final state (A-2)         GeV
    """
    p20=solve_for_p2_ppcase(epsilon,p1,-1,A,Binit,Bfinal)
    p2s=[]
    if len(p20)==1:
        for i in range(100):
            z=-1+i*2.0/100
            p2=solve_for_p2_ppcase(epsilon,p1,z,A,Binit,Bfinal)
            p2s.append(p2)
        if all(p2s[i] >= p2s[i + 1] for i in range(len(p2s) - 1)):
            return p2s[99][0],p2s[0][0]
        else:
            print("single p2,but not monotonic!")
    elif len(p20)==2:
        return p20[1],p20[0]
    elif len(p20)==0:
        return 0,0
    else:
        print("don't know this situation")

def spectralfunction_ppcase(p1,epsilon1,sigmaCM,datax,datay,A=4,Binit=0.0282957,Bfinal=0):
    '''
    Parameters
    -------------------------------------------
    Return
    -------------------------------------------

    '''
    massp=0.938272
    massn=0.9395654


    # get the p2max and p2min
    p2min,p2max=find_p2_min_max_ppcase(epsilon1,p1,Bfinal=Bfinal)
    
    # prepare the mesh points of p2
    meshnum=150
    p2,p2w=gauss_legendre_line_mesh(meshnum,p2min,p2max)

    # prepare the intrgtated function of p2
    # constC (nd array)
    constC=epsilon1+np.sqrt(p2**2+massp**2)-2*massp+Binit-Bfinal+(p1**2+p2**2)/(2*massn*(A-2))
    # relative momentum (nd array)
    prel=np.sqrt(p1**2+p2**2+2*constC*massn*(A-2))/2.0*1000/197.32705
    # average momentum **2 (nd array)
    pave2=p1**2+p2**2-2*constC*massn*(A-2)
    # coefficient (const)
    coeff=1.0/(64*np.pi**4*np.sqrt(2*np.pi)**3)*10**9/197.32705**3*4*np.pi
    # the integrated function (as a nd array)
    func=p2*massn*(A-2)/p1*interpolatedphi2(datax,datay,prel)/sigmaCM**3*np.exp(-pave2/(2.0*sigmaCM**2))
    # integrate the function
    spectralfunc=np.sum(coeff*func*p2w)
    return spectralfunc

class src:
    '''
    the parameters are included in the dictionary
    
    Parameters
    ---------------------------------------------
    config:{\n
    potname: str  \n
    massnum: int (A) \n
    Binit: float \n
    sigCM: float \n
    contactsratio: float(C^1/C^0) \n 

    }
    '''
    def __init__(self,config:dict):
        self.potname=config.get('potname')
        self.A=config.get('massnum')
        self.Binit=config.get('Binit')
        self.sigCM=config.get('sigCM')
        self.ratio=config.get('contactsratio')
        self.pmesh=None
        self.phipn1=None
        self.phipp0=None
        self.phipn0=None
        self.get_genfunc()


        
    def get_genfunc(self):
        df1 = pd.read_csv('../src/results/phipp0.csv')
        df2 = pd.read_csv('../src/results/phipn1.csv')
        df3 = pd.read_csv('../src/results/phipn0.csv')
        self.pmesh=df1["k"].tolist()
        self.phipp0=df1[self.potname].tolist()
        self.phipn1=df2[self.potname].tolist()
        self.phipn0=df3[self.potname].tolist()

    def ratio_ppnp(self,p1,epsilon1):
        ratio=(spectralfunction(p1,epsilon1,self.sigCM,self.pmesh,self.phipp0,self.A,self.Binit,0)) \
        /(self.ratio*spectralfunction(p1,epsilon1,self.sigCM,self.pmesh,self.phipn1,self.A,self.Binit,0.002225) \
          +spectralfunction(p1,epsilon1,self.sigCM,self.pmesh,self.phipn0,self.A,self.Binit,0.002225))
        return ratio

class correlation_density:
    def __init__(self,nucleusname:str,forcename:str,hw:int,Nmax:int,namenn=None,namepn=None,namepp=None):
        self.__filepath=f'../src/data/{nucleusname}/correlation_density/{forcename}'
        self.__foldername=f'hw{hw}-N{Nmax}'
        self.__filenn=namenn
        self.__filepn=namepn
        self.__filepp=namepp
        self.__nndata=None
        self.__pndata=None
        self.__ppdata=None
        self.load_data()
        if hw == 0:
            self.__ifGamow=True
        else:
            self.__ifGamow=False

        #####################################################################################
        # the variable can be got from the outside
        self.nndensity=None
        self.pndensity=None
        self.ppdensity=None
        self.get_density_func()
    
    @staticmethod
    def get_file_list(folder_path:str,folder_name:str,filename:str):
        file_lists = []
        for name in os.listdir(folder_path):
            # find the folders
            if name.startswith(folder_name):
                folder = os.path.join(folder_path,name)
                # find the file
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_lists.append(file_path)
        return file_lists

    
    def load_data(self):
        '''
        load the data 
        Integrate single-angle data and multi-angle data together
        '''
        # get all the data file(folder)
        if self.__filenn is not None:
            file_list=self.get_file_list(self.__filepath,self.__foldername,self.__filenn)
            data_list = []
            # read all the file
            for filename in file_list:
                data = np.loadtxt(filename, dtype=[("r", "<f8"), ("theta", "<f8"), ("theta_degree", "<f8"), 
                                                ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), 
                                                ("r2rho_im", "<f8")])
                data_list.append(data)
            self.__nndata = np.concatenate(data_list)
            # self.__nndata = np.loadtxt(self.__filenn,dtype=[("r","<f8"),("theta","<f8"),("theta_degree","<f8"),("rho_re","<f8"),("rho_im","<f8"),("r2rho_re","<f8"),("r2rho_im","<f8")])
        if self.__filepn is not None:
            file_list=self.get_file_list(self.__filepath,self.__foldername,self.__filepn)
            data_list = []
            # read all the file
            for filename in file_list:
                data = np.loadtxt(filename, dtype=[("r", "<f8"), ("theta", "<f8"), ("theta_degree", "<f8"), 
                                                ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), 
                                                ("r2rho_im", "<f8")])
                data_list.append(data)
            self.__pndata = np.concatenate(data_list)
            # self.__pndata = np.loadtxt(self.__filepn,dtype=[("r","<f8"),("theta","<f8"),("theta_degree","<f8"),("rho_re","<f8"),("rho_im","<f8"),("r2rho_re","<f8"),("r2rho_im","<f8")])
        if self.__filepp is not None:
            file_list=self.get_file_list(self.__filepath,self.__foldername,self.__filepp)
            data_list = []
            # read all the file
            for filename in file_list:
                data = np.loadtxt(filename, dtype=[("r", "<f8"), ("theta", "<f8"), ("theta_degree", "<f8"), 
                                                ("rho_re", "<f8"), ("rho_im", "<f8"), ("r2rho_re", "<f8"), 
                                                ("r2rho_im", "<f8")])
                data_list.append(data)
            self.__ppdata = np.concatenate(data_list)
            # self.__ppdata = np.loadtxt(self.__filepp,dtype=[("r","<f8"),("theta","<f8"),("theta_degree","<f8"),("rho_re","<f8"),("rho_im","<f8"),("r2rho_re","<f8"),("r2rho_im","<f8")])
    @staticmethod
    def data_transform(x,y,z):
        '''
        translate the 1-D arrays to 
        1-D(X) 1-D(Y)
        2-D(Z)
        '''
        X = np.unique(x)
        Y = np.unique(y)
        print(Y)
        # notice here the sequence: Y,X
        Z = np.empty((len(X), len(Y)))
        for xi, yi, zi in zip(x,y,z):
            # find the indexes
            x_idx = np.where(X == xi)[0][0]  
            y_idx = np.where(Y == yi)[0][0]
            Z[x_idx, y_idx] = zi
        return X,Y,Z
    
    def get_density_func(self):
        if self.__nndata is not None:
            if self.__ifGamow == False:
                X,Y,Z=self.data_transform(self.__nndata['r'],self.__nndata['theta_degree'] , self.__nndata['rho_re'])
                self.nndensity =RegularGridInterpolator((X,Y),Z, method='cubic') 
            else:
                rho_re =self.__nndata['rho_re']
                rho_im =self.__nndata['rho_im']
                rho = rho_re + 1j * rho_im
                rho2 = np.abs(rho)
                X,Y,Z=self.data_transform(self.__nndata['r'],self.__nndata['theta_degree'] , rho2)
                self.nndensity =RegularGridInterpolator((X,Y),Z, method='cubic') 
        if self.__pndata is not None:
            if self.__ifGamow == False:
                X,Y,Z=self.data_transform(self.__pndata['r'],self.__pndata['theta_degree'] , self.__pndata['rho_re'])
                self.pndensity =RegularGridInterpolator((X,Y),Z, method='cubic')
            else:
                rho_re =self.__pndata['rho_re']
                rho_im =self.__pndata['rho_im']
                rho = rho_re + 1j * rho_im
                rho2 = np.abs(rho)
                X,Y,Z=self.data_transform(self.__nndata['r'],self.__pndata['theta_degree'] , rho2)
                self.pndensity =RegularGridInterpolator((X,Y),Z, method='cubic') 
        if self.__ppdata is not None:
            X,Y,Z=self.data_transform(self.__ppdata['r'],self.__ppdata['theta_degree'] , self.__ppdata['rho_re'])
            self.ppdensity =RegularGridInterpolator((X,Y),Z, method='cubic')

    def two_nucleon_density_pn(self,R,r):
        r0=np.sqrt(R**2+r**2/4)
        theta=2*np.arctan(r/(2*R))*180/np.pi
        density=(R**2+r**2/4)/(R*r)*self.pndensity((r0,theta))
        return density
    
    def two_nucleon_density_nn(self,R,r):
        r0=np.sqrt(R**2+r**2/4)
        theta=2*np.arctan(r/(2*R))*180/np.pi
        density=(R**2+r**2/4)/(R*r)*self.nndensity((r0,theta))/r0**2
        return density
    
class src_contacts:
    def __init__(self,potname,genfunrfile:dict):
        '''
        Parameters
        -------------------------------------
        - genfunrfile: this varable has the structure {'pn0': namepn0, 'pn1': namepn1}
        '''
        self.__genfunrfile=genfunrfile
        self.__potname=potname
        self.phipn1r=None
        self.phipn0r=None
        self.__get_genfun_r()
        
    def __get_genfun_r(self):
        df1 = pd.read_csv(self.__genfunrfile['pn0'])
        pn0s={}
        for column in df1.columns:
            pn0s[column] = df1[column].to_numpy()
        r=pn0s['r']
        phipn0=pn0s[self.__potname]
        self.phipn0r=make_interp_spline(r, phipn0, k=3)
        df1 = pd.read_csv(self.__genfunrfile['pn1'])
        pn1s={}
        for column in df1.columns:
            pn1s[column] = df1[column].to_numpy()
        r=pn1s['r']
        phipn1=pn1s[self.__potname]
        self.phipn1r=make_interp_spline(r, phipn1, k=3)
    
    def get_ratio_abinitio_r(self,):

        pass
        ''''''
           