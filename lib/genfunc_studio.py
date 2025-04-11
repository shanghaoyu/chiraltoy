import integration as inte
import numpy as np
import chiral_potential as chiral_potential
from scipy.special import spherical_jn


class genfunc_studio:
    def __init__(self,pmin,pmid,pmax,Nmid,Np,name_pot):
        '''
        Parameters:
        --------------------------------------------------
        - pmin,pmid,pmax,Nmid,Np: sets the parameters to generate two part guassian mesh
        - potential : the name of the potential
        '''
        self.Np=Np
        self.name_pot=name_pot
        self.pmesh,self.wmesh=inte.gaussian_quadrature_mesh(pmax,Np,xmin=pmin,xmid=pmid,nmod=Nmid)
        pot=chiral_potential.two_nucleon_potential(name_pot)
        self.potential=pot.potential
        self.coulombrel=pot.potential_cutoff_relcoulomb
        self.mu=938.91852/2
        self.hbarc=197.32705
        ##############################################
        # deuteron part
        self.energy=None
        self.u=None
        self.w=None
        self.deuteron_solved=False
        # phipn0_r part
        self.phipn0k=None
        self.phipn0_solved=False
        # phipn1_r part
        self.phipn1_solved=False


    
    def get_wavefunc(self,wave):
        '''
        get real wave function
        '''
        wavefunc=np.zeros((self.Np))
        for i in range(self.Np):
            wavefunc[i]=wave[i]/(self.pmesh[i]*np.sqrt(self.wmesh[i]))
        return wavefunc
    
    def solve_for_pn0(self,normk=255.0):
        '''
        solve for the phi0(k)
        normed with \int_0^{\infty}phi0(k)**2=\pi/2
        '''
        mu=938.91852/2
        Np=self.Np
        delta=np.eye(Np)
        Tmtx=np.zeros((Np,Np))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*mu)*delta[i,i]*self.hbarc**2
        # V matrix
        Vmtx=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtx[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 0, 0, 0)*np.sqrt(self.wmesh[j])*self.hbarc**3*self.pmesh[j]    
        # Hamitlon
        Hmtx=np.zeros((Np,Np))
        Hmtx=Vmtx+Tmtx
        eigvals, eigvecs =np.linalg.eigh(Hmtx)

        # select the first state
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]

        min_eigenvector = eigvecs[:, min_eigenvalue_index]
        
        # get the positive value eigenvector
        normal=np.sum(min_eigenvector)
        if normal < 0:
            min_eigenvector=-min_eigenvector

        # check noremalization
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >= normk/197.32705:
                norm=norm+min_eigenvector[i]**2
        self.phipn0k=self.get_wavefunc(min_eigenvector/np.sqrt(norm))


        

    def solve_for_deuteron(self,normk=0):
        '''
        solve for the u(k) and w(k)

        - min_eigenvalue: eigenvalue of deuteron Mev
        - u: u(k)  
        - w: w(k)
        normed with \int_0^{\infty}u**2+w**2=\pi/2
        '''
        Np=self.Np
        delta=np.eye(Np)
        # T matrix
        Tmtx=np.zeros((Np*2,Np*2))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*self.mu)*delta[i,i]*self.hbarc**2
            Tmtx[i+Np,i+Np]=self.pmesh[i]**2/(2*self.mu)*delta[i,i]*self.hbarc**2
        Vmtxpp=np.zeros((Np,Np))
        Vmtxpm=np.zeros((Np,Np))
        Vmtxmp=np.zeros((Np,Np))
        Vmtxmm=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtxpp[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(2, 2, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 1, 1, 0)*np.sqrt(self.wmesh[j])*self.pmesh[j]*self.hbarc**3
                Vmtxpm[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(2, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 1, 1, 0)*np.sqrt(self.wmesh[j])*self.pmesh[j]*self.hbarc**3
                Vmtxmp[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 2, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 1, 1, 0)*np.sqrt(self.wmesh[j])*self.pmesh[j]*self.hbarc**3
                Vmtxmm[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 1, 1, 0)*np.sqrt(self.wmesh[j])*self.pmesh[j]*self.hbarc**3      
        # Hamitlon
        Vmtx=np.block([[Vmtxmm,Vmtxmp],[Vmtxpm,Vmtxpp]])
        Hmtx=Vmtx+Tmtx

        eigvals, eigvecs =np.linalg.eigh(Hmtx)

        # select the first state
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]
        min_eigenvector = eigvecs[:, min_eigenvalue_index]


        # get the positive value eigenvector
        normal=np.sum(min_eigenvector)
        if normal < 0:
            min_eigenvector=-min_eigenvector
        # check noremalization
        
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >= normk/197.32705:
                norm=norm+(min_eigenvector[i]**2+min_eigenvector[i+Np]**2)

        u=min_eigenvector[0:Np]/np.sqrt(norm)
        w=min_eigenvector[Np:2*Np]/np.sqrt(norm)
        self.energy=min_eigenvalue
        self.u=self.get_wavefunc(u)
        self.w=self.get_wavefunc(w)

    def phinn0(self,normk=255.0):
        '''
        Parameters:
        ----------------------------------------------
        - normk: \int_{normk}^{\infty} phipn1 k^2 dk/(2\pi^2) =1  MeV

        Return:
        --------------------------------------------------
        - pmesh: p points
        - phi^0_nn: general function
        '''
        mu=939.5654/2.0
        Np=self.Np
        delta=np.eye(Np)
        Tmtx=np.zeros((Np,Np))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*mu)*delta[i,i]*self.hbarc**2
        # V matrix
        Vmtx=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtx[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 0, 0, 1)*np.sqrt(self.wmesh[j])*self.hbarc**3*self.pmesh[j]    
        # Hamitlon
        Hmtx=np.zeros((Np,Np))
        Hmtx=Vmtx+Tmtx

        eigvals, eigvecs =np.linalg.eigh(Hmtx)

        # select the first state
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]

        min_eigenvector = eigvecs[:, min_eigenvalue_index]

        # check renoremalization
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >=normk/197.32705:# 255.0/197.32705:
                norm=norm+min_eigenvector[i]**2

        # wave function 
        vectornor=np.zeros((Np))
        for i in range(Np):
            vectornor[i]=min_eigenvector[i]/(self.pmesh[i]*np.sqrt(2/np.pi*self.wmesh[i]))
        n_s_total_array = (vectornor**2)*4*np.pi/(norm)

        return self.pmesh,n_s_total_array



    def phipn0(self,normk=255.0):
        '''
        Parameters:
        ----------------------------------------------
        - normk: \int_{normk}^{\infty} phipn1 k^2 dk/(2\pi^2) =1  MeV

        Return:
        --------------------------------------------------
        - pmesh: p points
        - phi^0_pn: general function
        '''
        #central mass system mass
        mu=938.91852/2
        Np=self.Np
        delta=np.eye(Np)
        Tmtx=np.zeros((Np,Np))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*mu)*delta[i,i]*self.hbarc**2
        # V matrix
        Vmtx=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtx[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 0, 0, 0)*np.sqrt(self.wmesh[j])*self.hbarc**3*self.pmesh[j]    
        # Hamitlon
        Hmtx=np.zeros((Np,Np))
        Hmtx=Vmtx+Tmtx
        eigvals, eigvecs =np.linalg.eigh(Hmtx)

        # select the first state
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]

        min_eigenvector = eigvecs[:, min_eigenvalue_index]

        # check renoremalization
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >= normk/197.32705:
                norm=norm+min_eigenvector[i]**2

        # wave function 
        vectornor=np.zeros((Np))
        for i in range(Np):
            vectornor[i]=min_eigenvector[i]/(self.pmesh[i]*np.sqrt(2/np.pi*self.wmesh[i]))
        n_s_total_array = (vectornor**2)*4*np.pi/(norm)
        return self.pmesh,n_s_total_array
    

    def phipn1P1(self,normk=255.0):
        '''
        Parameters:
        ----------------------------------------------
        - normk: \int_{normk}^{\infty} phipn1 k^2 dk/(2\pi^2) =1  MeV

        Return:
        --------------------------------------------------
        - pmesh: p points
        - phi^0_pn: general function
        '''
        #central mass system mass
        mu=938.91852/2
        Np=self.Np
        delta=np.eye(Np)
        Tmtx=np.zeros((Np,Np))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*mu)*delta[i,i]*self.hbarc**2
        # V matrix
        Vmtx=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtx[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(1, 1, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 1, 0, 0)*np.sqrt(self.wmesh[j])*self.hbarc**3*self.pmesh[j]    
        # Hamitlon
        Hmtx=np.zeros((Np,Np))
        Hmtx=Vmtx+Tmtx
        eigvals, eigvecs =np.linalg.eigh(Hmtx)

        # select the first state
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]

        min_eigenvector = eigvecs[:, min_eigenvalue_index]

        # check renoremalization
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >= normk/197.32705:
                norm=norm+min_eigenvector[i]**2

        # wave function 
        vectornor=np.zeros((Np))
        for i in range(Np):
            vectornor[i]=min_eigenvector[i]/(self.pmesh[i]*np.sqrt(2/np.pi*self.wmesh[i]))
        n_s_total_array = (vectornor**2)*4*np.pi/(norm)
        return self.pmesh,n_s_total_array
    
    def phipp0(self,normk=255.0,coulomb=False):
        '''
        Parameters:
        ----------------------------------------------
        - normk: \int_{normk}^{\infty} phipn1 k^2 dk/(2\pi^2) =1  MeV
        - coulomb : True (with coulomb force)

        Return:
        --------------------------------------------------
        - pmesh: p points
        - phi^0_pp: general function
        '''
        mu=938.2720/2
        Np=self.Np
        delta=np.eye(Np)
        Tmtx=np.zeros((Np,Np))
        for i in range(Np):
            Tmtx[i,i]=self.pmesh[i]**2/(2*mu)*delta[i,i]*self.hbarc**2
         # V matrix
        Vmtx=np.zeros((Np,Np))
        for i in range(Np):
            for j in range(Np):
                Vmtx[i,j]=np.sqrt(self.wmesh[i])*self.pmesh[i]*self.potential(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 0, 0, -1)*np.sqrt(self.wmesh[j])*self.hbarc**3*self.pmesh[j]
                if coulomb==True:
                    Vmtx[i,j]=Vmtx[i,j]+np.sqrt(self.wmesh[i])*self.pmesh[i]*self.coulombrel(0, 0, self.pmesh[i]*self.hbarc, self.pmesh[j]*self.hbarc, 0, 0, -1,0)*np.sqrt(self.wmesh[j])*self.pmesh[j]*self.hbarc**3  
        # Hamitlon
        Hmtx=np.zeros((Np,Np))
        Hmtx=Vmtx+Tmtx
        eigvals, eigvecs =np.linalg.eigh(Hmtx)
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvalue = eigvals[min_eigenvalue_index]
        min_eigenvector = eigvecs[:, min_eigenvalue_index]

        # check renoremalization
        normal=np.sum(min_eigenvector**2)
        if np.abs(normal-1)<1.0e-6:
            print("normalized to one!")
        else:
            print("error!")
            return
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >=normk/197.32705:
                norm=norm+min_eigenvector[i]**2
            # wave function 
        vectornor=np.zeros((Np))
        for i in range(Np):
            vectornor[i]=min_eigenvector[i]/(self.pmesh[i]*np.sqrt(2/np.pi*self.wmesh[i]))
        n_s_total_array = (vectornor**2)*4*np.pi/(norm)

        return self.pmesh,n_s_total_array
    
    
    def phipn1(self,normk):
        '''
        Parameters
        ----------------------------------------------
        - normk:  \int_{normk}^{\infty} phipn1 k^2 dk/(2\pi^2)=1  MeV
        Return
        --------------------------------------------------
        - pmesh: p points
        - phi1_u: u(k)^2
        - phi1_w: w(k)^2
        '''
        if self.deuteron_solved == False:
            self.solve_for_deuteron()
            self.deuteron_solved = True
        norm=0
        for i,num in enumerate(self.pmesh):
            if self.pmesh[i] >= normk/self.hbarc:
                norm=norm+(self.u[i]**2+self.w[i]**2)*(self.pmesh[i]**2*self.wmesh[i])
        # wave function
        
        phipn1_u=np.zeros((self.Np))
        phipn1_w=np.zeros((self.Np))
        for i in range(self.Np):
            phipn1_u[i] = (self.u[i]**2)*2*np.pi**2/(norm)
            phipn1_w[i] = (self.w[i]**2)*2*np.pi**2/(norm)
        return self.pmesh,phipn1_u,phipn1_w
    
    def deuteron_r(self,r):
        '''
        Parameters
        -------------------------------------
        - r :  fm

        Returns
        -------------------------------------
        - uror : u(r)/r  fm^-3/2  
        - wror : w(r)/r  fm^-3/2  

        Notes
        ---------------------------------------
        if calculating deuteron_r ,you need to choose a larger pmax
        for example gf=genfunc_studio(0,2,30,80,300,name)
        this setting can calculate r to r>=0.1fm
        '''
        # wave function
        if self.deuteron_solved == False:
            self.solve_for_deuteron()
            self.deuteron_solved = True
        uror=0
        wror=0
        for i,k in enumerate(self.pmesh):
            uror=uror+np.sqrt(2.0/np.pi)*k**2*spherical_jn(0,k*r)*self.u[i]*self.wmesh[i]
            wror=wror+np.sqrt(2.0/np.pi)*k**2*spherical_jn(2,k*r)*self.w[i]*self.wmesh[i]
        return uror,wror
    
    def phipn0_r(self,r,normk=255):
        '''
        Parameters
        -------------------------------------
        - r :  fm

        Returns
        -------------------------------------
        - phipn0r :  fm^-3/2  
        Notes
        ---------------------------------------
        if calculating deuteron_r ,you need to choose a larger pmax
        for example gf=genfunc_studio(0,2,30,80,300,name)
        this setting can calculate r to r>=0.1fm
        '''
        # wave function
        if self.phipn0_solved == False:
            self.solve_for_pn0(normk)
            self.phipn0_solved  = True
        phipn0r=0
        for i,k in enumerate(self.pmesh):
            phipn0r=phipn0r+k**2*spherical_jn(0,k*r)*self.phipn0k[i]*self.wmesh[i]/(2*np.pi**2)
        return phipn0r**2*(2*np.pi**2)
    
    def phipn1_r(self,r,normk=255):
        '''
        Parameters
        -------------------------------------
        - r :  fm

        Returns
        -------------------------------------
        - phipn1r :  fm^-3/2  
        Notes
        ---------------------------------------
        if calculating deuteron_r ,you need to choose a larger pmax
        for example gf=genfunc_studio(0,2,30,80,300,name)
        this setting can calculate r to r>=0.1fm
        '''
        # wave function
        if self.phipn1_solved == False:
            self.solve_for_deuteron(normk)
            self.phipn1_solved = True
        uror=0
        wror=0
        for i,k in enumerate(self.pmesh):
            uror=uror+k**2*spherical_jn(0,k*r)*self.u[i]*self.wmesh[i]/(2*np.pi**2)
            wror=wror+k**2*spherical_jn(2,k*r)*self.w[i]*self.wmesh[i]/(2*np.pi**2)
        return (uror**2+wror**2)*(2*np.pi**2)

    def check_norm1(self):
        norm=0
        for i in range(self.Np):
            norm=norm+(self.u[i]**2+self.w[i]**2)*self.pmesh[i]**2*self.wmesh[i]
        return norm
    
    def check_norm2(self):
        norm=0
        r,rw=inte.gaussian_quadrature_mesh(30,300,xmin=0,xmid=3,nmod=40)
        for i,ri in enumerate(r):
            uror,wror=self.deuteron_r(ri)
            norm=norm+(uror**2+wror**2)*ri**2*rw[i]

        return norm

    # def check_norm_phipn0r(self):
    #     norm=0
    #     r,rw=inte.gaussian_quadrature_mesh(30,300,xmin=0,xmid=2,nmod=40)
        
    #     for i,ri in enumerate(r):
    #         phipn0r=self.phipn0_r(ri)
    #         norm=norm+(phipn0r**2)*ri**2*rw[i]

    #     return norm
    def check_norm_phipn1r(self):
        norm=0
        r,rw=inte.gaussian_quadrature_mesh(30,300,xmin=0,xmid=2,nmod=40)
        
        for i,ri in enumerate(r):
            phipn1r=self.phipn1_r(ri,0)

            norm=norm+phipn1r*ri**2*rw[i]

        return norm



