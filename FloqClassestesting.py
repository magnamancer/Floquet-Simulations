# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:33:00 2022

@author: FentonClawson
"""
import numpy as np
from qutip import *
import emisspecmoduleF as esm 
import scipy as scp
import FloqClassestesting as FC

import itertools
import math

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#Defining evberything to reproduce my exact results first, will generalize later when I figure it out!

class Laser:
    def __init__(self,magnitude,pol,freq):
        self.magnitude = magnitude
        self.pol = pol
        self.freq = freq
        
        self.E = self.magnitude*self.pol

class Bfield:
    def __init__(self,Bvec):
        self.Bvec = Bvec #A vector [Bx,By,Bz] of the magnetic field magnitude in each direction [x,y,z]
        
class LowOp:
    def __init__(self,mat,mag):
        self.mat = mat #The matrix form of the lowering operator
        self.mag = mag

class QD:
    def __init__(self,Hdim,states,dipoles,gfactors = None):
        self.Hdim = Hdim #Dimension of the system
        self.states = states #List of energies of the system, with the ASSUMPTION THAT THERE ARE ONLY TWO ENERGETIC STATES
        self.dipoles = dipoles #dictionary of dipole moment states linkages (as tuples) as keys and their associated polarization(s) in the dot (as tuplevalues)
        self.gfactors = gfactors #An ENERGETICALLY ORDERED list of gfactors of each degnerate (spin-distinguishable) state as [g_parallel,g_perp].
                                 #E.g. for the system in Ned's 2015 paper, the gfactor list would be [[g_(electron,para),g_(electron,perp)],[g_(hole,para),g_(hole,perp)]]
        self.fss = None #Optional propery "fss," the fine structure splitting of states. Given as a dictionary with the bare energies as keys and the fss amount as values.
        

class QSys:
    def __init__(self,QD,Las1,Las2 = None, Bfield = None, LowOp = None):
        self.Las1 = Las1
        self.Las2 = Las2
        self.Bfield = Bfield
        self.QD = QD
        self.LowOp = LowOp
        
        '''
        Defining Useful Constants
        '''
        #WILL NEED TO FIND A WAY TO GENERALIZE DELTA LATER!!!!
        if self.Las2 == None:
            self.Las2 = FC.Laser(0,self.Las1.pol,self.Las1.freq*0.999974 )
            
        #This loop doesn't do anything but reorder the lasers to enforce some assumptions I make about the RWA form of the Hamiltonian
        if self.Las1.freq > self.Las2.freq:
            temp = self.Las1
            self.Las1 = self.Las2
            self.Las2 = temp
        
        
        
        
        self.Lavg = (1/2)*(self.Las1.freq+self.Las2.freq) 
        self.beat  = abs(self.Las2.freq-self.Las1.freq )                             #beat frequency, difference between offresonant laser (L2) and resonant laser (L1)
        self.T     = (2*np.pi/abs(self.beat/2))
        self.Delta = 2*np.pi*self.QD.states[-1]-self.Lavg                             #Average Detuning of the lasers from the excited state

        self.Hargs = {'w': (self.beat/2)}                                             #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.
       
                    
    ############## Hamiltonian parameters ########################################
    
    '''
    Total Hamiltonian Definition, for use in calculating pretty much everything
    '''
    def Ham(self):
        if self.Bfield == None:
            w,v = np.linalg.eig(np.identity(self.QD.Hdim))
        if self.Bfield != None:
            self.AZHam = self.AtomHam()+self.ZHam()
            w,v = np.linalg.eig(self.AZHam)
            self.w = w
        '''
        The function below this is used to reorder the matrix into the expected
            block-diagonal form. It will leave the matrix unchanged if it's
            already in this form
        '''
        
            
        
        v,CorrPerms = esm.reorder(v)
        self.v = Qobj(v)
        
        # test = (self.v.dag()*(self.AtomHam()+self.ZHam())*self.v).full()
        # test[0,0] *= 1
        # test[1,1] *= 1
        
        # testy = Qobj(test)
       
            
        H = [self.v.dag()*(self.AtomHam()+self.ZHam())*self.v,                                                    \
            [self.v.dag()*self.LasHam()[0]*self.v,'exp(1j * w * t )'],                    \
            [self.v.dag()*self.LasHam()[1]*self.v, 'exp(-1j * w * t )']]                  #Full Hamiltonian in string format, a form acceptable to QuTiP
        
        
        # H = [(self.AtomHam()+self.ZHam()),                                                    
            # [self.LasHam()[0],'exp(1j * w * t )'],                    
            # [self.LasHam()[1], 'exp(-1j * w * t )']]   
            
         
        return H
  
    
    '''
    Creating a method to define the laser Hamiltonian(s) in the RWA
    '''
    def LasHam(self):

        '''
        Creating the Rabi Frequencies for use in the Laser Hamiltonians
        '''
        self.Om1 = {}
        self.Om2 =  {}
        self.Om1s = {} 
        self.Om2s = {}
        for idx,element in enumerate(self.QD.dipoles): 
            d = basis(self.QD.Hdim,element[0])*basis(self.QD.Hdim,element[1]).dag()
            self.Om1 [idx]   = np.dot(       self.QD.dipoles[element] ,self.Las1.E) * d                             #Rabi Frequency Omega_1
            self.Om1s[idx]  = np.dot(np.conj(self.QD.dipoles[element]),np.conj(self.Las1.E)) * d.dag()                        #Rabi Frequency Omega_1 star
            self.Om2 [idx]   = np.dot(       self.QD.dipoles[element] ,self.Las2.E) * d                            #Rabi Frequency Omega_2
            self.Om2s[idx]  = np.dot(np.conj(self.QD.dipoles[element]),np.conj(self.Las2.E)) * d.dag()                       #Rabi Frequency Omega_2 star
        
        '''
        Next, I define the forward and backward rotating terms
        '''
        self.forwardrot  = (-1/2)*sum([x+y for (x,y) in zip(self.Om2.values(),self.Om1s.values())])
        self.backwardrot = (-1/2)*sum([x+y for (x,y) in zip(self.Om1.values(),self.Om2s.values())])
        
        return self.forwardrot, self.backwardrot

    
    '''
    Defining the Atomic Hamiltonian in the RWA
    '''
    def AtomHam(self):
        self.AtomHammy = np.zeros((self.QD.Hdim,self.QD.Hdim),dtype='complex')
        N = len(self.QD.states)
        eta = np.zeros(self.QD.Hdim)
        eta[0] = 0#self.Delta/2
        C = sum([basis(self.QD.Hdim,list(i)[0])*basis(self.QD.Hdim,list(i)[1]).dag() for i in self.QD.dipoles])
        for n in range(N):
            for m in range(n+1,N):
                if C[n,m] != 0:
                    eta[m] = eta[n]+self.Lavg
            self.AtomHammy[n,n] = self.QD.states[n]*(2*np.pi)-eta[n]
        
        return Qobj(self.AtomHammy)
    
    
    '''
    Defining the Zeeman Hamiltonian
    '''
    def ZHam(self):
        self.ZHammy = np.zeros((self.QD.Hdim,self.QD.Hdim),dtype='complex')
        if self.Bfield != None:
            Count = 0 #initializing a counter to iterate through the g factor list
            for i in list(range(self.QD.Hdim)):
                for j in list(range(i+1,self.QD.Hdim)):
                    '''
                    Right now, I'm adding the rounding in the line directly below as a patch
                    
                    Because f0 function can't take degenerate manifolds (with or without a magnetic field!), I add something like ~1/1000 of a GHz 
                    detuning to the second manifold.
                    
                    Empirically, it lookds like that doesn't do anything negative at all to the resulting spectra, though that nondegenaracy then
                    fucks up the degeneracy condition I use to build the Zeeman Hamiltonian. By rounding the input, I'm rounding out that small
                    nondegeneracy. WHICH MEANS YOU NEED TO MAKE A BETTER FIX LATER, FENTON.
                    
                    I'm no longer rounding, but now checking absolute percent deviation. If the first entry is zero, check to see if the second entry is zero.
                    If they are, congrats, stick a magnetic field in that bitch. If the first entry is not zero, check to see that its absolute percent deviation
                    from the second entry is less than 1%. If it is, magnetic field. Otherwise ignore it. THIS IS STILL NOT A FULL FIX FENTON. YOU NEED TO FIX 
                    FMODES SO IT DOESN'T FUCK UP WITH DEGENERATE MANIFOLDS. THIS IS JUST A BETTER STOPGAP
                    '''
                    if self.AtomHam()[i,i] == 0:
                        if self.AtomHam()[j,j] == 0:
                            self.ZHammy[i,i] = -1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                            self.ZHammy[j,j] =  1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                            self.ZHammy[i,j] = self.ZHammy[j,i] = (1/2)*self.Bfield.Bvec[0]*self.QD.gfactors[Count][0]
                            Count += 1
                    elif 100*(abs(self.AtomHam()[i,i]-self.AtomHam()[j,j])/abs(self.AtomHam()[i,i])) <= 1 :
                        self.ZHammy[i,i] = -1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                        self.ZHammy[j,j] =  1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                        self.ZHammy[i,j] = self.ZHammy[j,i] = (1/2)*self.Bfield.Bvec[0]*self.QD.gfactors[Count][0]
                        Count += 1
        return Qobj(self.ZHammy)
        
        
        
        
        
        
        
    '''
    Various time evolution methods below
    '''
    
    def ExciteSpec(self,Nt,tau,rho0,time_sense = 0,PDM = 1,detpols = np.array([None,None,None]),opts = None):       
            ############### Time evolving rho0 with solve_ivp#########################
        
        
        
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        
        
        
        Nt2 = Nt*PDM                                        #Number of Points
        timet = self.T                                      #Length of time of tlist defined to be one period of the system
        dt = timet/Nt2                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt2)               #Combining everything to make tlist
        tlistprime = np.linspace(0,timet-(dt*PDM),Nt)
                                                            #Taulist Definition
        Ntau = (Nt)*(tau)*PDM                                    
        taume = ((Ntau/PDM)/Nt)*self.T                          
        dtau = taume/Ntau                                 
        taulist = np.linspace(0, taume-dtau, Ntau)       
        
        
        f0,qe,f_modes_table_t,fstates,fstatesct= esm.PrepWork(self.Ham(),self.T,self.Hargs,tlist,taulist, opts = opts)
        print('found f0, qe, fstates')
        
        '''
        '''

        '''
        '''

        # Also transforming the lowering operator into the new basis
        if self.Bfield != None:
            self.LowOp.matT = self.v.dag()*(self.LowOp.mat)*self.v
            
        amps, lmax = esm.LTrunc(PDM,Nt,tlistprime,taulist,self.LowOp.matT ,f_modes_table_t = f_modes_table_t, opts = opts)
        print('found lmax =', lmax)
        
        Rdic = esm.Rt(qe,amps,lmax,self.LowOp.mag,self.beat,time_sense )
        self.Rdic = Rdic
        
        
        '''
        '''
        print('Built R(t)')

        LowOpDetList =  esm.lowop_detpol_modifier(self.LowOp.mat,self.QD.dipoles,detpols)
        print('set detection polarization')
        

        
        '''
        Looping over detection polarizations, to hopefully make things faster
        '''
        excitevals = {}
        self.test = {}
        for Ldx, Lowp in enumerate(LowOpDetList):
           
            '''
            Doing the raising and lowering operator transformations, to move them
                into the Floquet basis for every t_inf+t
            '''
            # Also transforming the lowering operator into the new basis
            if self.Bfield != None:
                Lowp = self.v.dag()*Lowp*self.v
                
            
            lofloq = fstatesct @ (Lowp).full() @ fstates
            hifloq = np.transpose(lofloq.conj(),axes=(0,2,1)) 
            
            pop_op = hifloq @ lofloq
            
           
            
            # print('finished operator state conversions')
            
            
            '''
                 This loop takes each array output by the solver,
               AOPWSDJMB[AQWOPEIBMQ[Wp(eiv{poqI})]]
            '''
            # print('attempting to solve the IVP')   
            
            rho01 = operator_to_vector(rho0.transform(f0,False)).full()[:,0]
            
            
            t0 = taulist[-1]+taulist[1]
            rho00 = scp.integrate.solve_ivp(esm.rhodot,
                                                   t_span = (0,t0),
                                                   y0=rho01                 ,
                                                   args=(Rdic,self.beat)                 ,
                                                   # method='DOP853'         ,
                                                   t_eval=np.append(taulist,t0)          ,
                                                   rtol=opts.rtol               ,
                                                   atol=opts.atol).y[:,-1]                                                                 
            # print('finished solving the IVP/time evolving rho0')
            '''
            Next step is to iterate this steady state rho_s forward in time. I'll choose the times
            to be evenly spread out within T, the time scale of the Hamiltonian
            
            In this step I also multiply in the lowering operator in the Floquet STATE basis at the correct time
            '''
            
        
       
            
            # self.rhoss = np.reshape(rho00, (self.QD.Hdim,self.QD.Hdim),order='F')
            Pop_t = [ (hifloq[i*PDM] @ lofloq[i*PDM])                       \
                          @ np.reshape(
                                scp.integrate.solve_ivp(esm.rhodot,
                                                        t_span = (t0,t0+tlistprime[-1])  ,
                                                        y0=rho00                ,
                                                        args=(Rdic,self.beat)               ,
                                                        # method='DOP853'         ,
                                                        t_eval=(t0+tlistprime)            ,
                                                        rtol=opts.rtol               ,
                                                        atol=opts.atol).y[:,i]       ,
                                        (self.QD.Hdim,self.QD.Hdim),order='F')  
                            for i in list(range(0,Nt))]
                
            
            
            PopAvg = np.average(Pop_t,axis=0)
            
            excitevals[detpols[Ldx]] = np.trace(PopAvg,axis1=0,axis2=1)
            
            print('Finished Detpol',detpols[Ldx])
            
        return  excitevals
    
    def EmisSpec(self,Nt,tau,rho0,time_sense = 0,PDM = 1,detpols = np.array([None,None,None]), retg1 = 'False', opts = None):       
        ############### Time evolving rho0 with solve_ivp#########################
        
        
        
        
        Nt2 = Nt*PDM                                        #Number of Points
        timet = self.T                                      #Length of time of tlist defined to be one period of the system
        dt = timet/Nt2                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt2)               #Combining everything to make tlist
        tlistprime = np.linspace(0,timet-(dt*PDM),Nt)
                                                            #Taulist Definition
        Ntau = (Nt)*(tau)*PDM                                    
        taume = ((Ntau/PDM)/Nt)*self.T                          
        dtau = taume/Ntau                                 
        taulist = np.linspace(0, taume-dtau, Ntau)       
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-12                                #Absolute tolerance
            opts.rtol = 1e-12                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        

   
        f0,qe,f_modes_table_t,fstates,fstatesct= esm.PrepWork(self.Ham(),self.T,self.Hargs,tlist,taulist)
        print('found f0, qe, fstates')
        

        
        
        
        amps, lmax = esm.LTrunc(PDM,Nt,tlistprime,taulist,self.LowOp.mat ,f_modes_table_t = f_modes_table_t)
        print('found lmax =', lmax)
        
        Rdic = esm.Rt(qe,amps,lmax,self.LowOp.mag,self.beat,time_sense )
        print('Built R(t)')

        LowOpDetList =  esm.lowop_detpol_modifier(self.LowOp.mat,self.QD.dipoles,detpols)
        print('set detection polarization')
        
        
        
        '''
        Looping over detection polarizations, to hopefully make things faster
        '''
        Z = {}
        g1dic = {}
        
        for Ldx, Lowp in enumerate(LowOpDetList):
            '''
            Doing the raising and lowering operator transformations, to move them
                into the Floquet basis for every t+tau
            '''
           
            
            lofloq = fstatesct @ (Lowp).full() @ fstates
            hifloq = np.transpose(lofloq.conj(),axes=(0,2,1)) 
           
            '''
                 This loop takes each array output by the solver,
               AOPWSDJMB[AQWOPEIBMQ[Wp(eiv{poqI})]]
            '''
            # print('attempting to solve the IVP')   
            
            rho01 = operator_to_vector(rho0.transform(f0,False)).full()[:,0]
            
            t0 = taulist[-1]+taulist[1]
            rho00 = scp.integrate.solve_ivp(esm.rhodot,
                                                   t_span = (0,t0),
                                                   y0=rho01                 ,
                                                   args=(Rdic,self.beat)                 ,
                                                   method='DOP853'         ,
                                                   t_eval=np.append(taulist,t0)          ,
                                                   rtol=opts.rtol               ,
                                                   atol=opts.atol).y[:,-1]                                                                 
            # print('finished solving the IVP/time evolving rho0')
            '''
            Next step is to iterate this steady state rho_s forward in time. I'll choose the times
            to be evenly spread out within T, the time scale of the Hamiltonian
            
            In this step I also multiply in the lowering operator in the Floquet STATE basis at the correct time
            '''
            
        
            Bstates = [ lofloq[i*PDM]                       \
                          @ np.reshape(
                                scp.integrate.solve_ivp(esm.rhodot,
                                                        t_span = (t0,t0+tlistprime[-1])  ,
                                                        y0=rho00                ,
                                                        args=(Rdic,self.beat)               ,
                                                        method='DOP853'         ,
                                                        t_eval=(t0+tlistprime)            ,
                                                        rtol=opts.rtol              ,
                                                        atol=opts.atol).y[:,i]       ,
                                        (self.QD.Hdim,self.QD.Hdim),order='F')  
                            for i in list(range(0,Nt))]
                

            
            '''
            Setting up a matrix to have rows equal to the number of tau values and columns equal to the number of t values
            At the end I'll average over each row to get a vector where each entry is a tau value and an averaged t value
            '''
        
            # print('Finished B-States')
            
            AstatesUnAv = np.zeros( (len(Bstates), len(taulist), self.QD.Hdim,self.QD.Hdim), dtype='complex_' ) 
            # print('Starting A States')
            for tdx, Bstate1 in enumerate(Bstates): #First for loop to find the tau outputs for each t value
                #New "starting time"
                t1 = t0+tlistprime[tdx]
                
                # if not (tdx+1)%10 or not (tdx+1)%len(Bstates):
                # print('Filling column',tdx+1,'of',len(Bstates))
                TauBSEvol = np.moveaxis(np.dstack(np.split(scp.integrate.solve_ivp(
                                                    esm.rhodot,
                                                    t_span = (t1,t1+taulist[-1]), 
                                                    y0=np.reshape(Bstate1,(self.QD.Hdim**2,),order='F'),
                                                    args=(Rdic,self.beat),
                                                    method='DOP853',
                                                    t_eval=(t1+taulist),
                                                    rtol=opts.rtol,
                                                    atol=opts.atol).y,
                                        self.QD.Hdim,axis=0)),(0,1,2),(1,0,2))
                '''
                STOP CHANGING THE ORDER OF THE TRANSPOSE FENTON. IT ISN'T GOING TO FIX IT
                TIMES I WAS WEAK: 13
                '''
                AstatesUnAv[tdx,...] = hifloq[(PDM*tdx):(len(taulist)+PDM*tdx)] @ TauBSEvol
            # print('found unaveraged A-States')   
            '''
            Okay so the output matrix from above is a bunch of 2x2 density matrices
            where the value idx1 refers to the tau value and the value idx refers to the t value

            Going forward I should now average over each "row" of t values, i.e. average over idx
            '''
            
            AStatesAvg = np.mean(AstatesUnAv,axis=0)
            
            
            g1 = np.trace(AStatesAvg,axis1=1,axis2=2)
            
            if retg1 == 'True':
                g1dic[detpols[Ldx]] = g1
            
            spec = np.fft.fftshift(np.fft.fft(g1,axis=0))
    
            Z[detpols[Ldx]] = np.real(spec)/(len(g1))
        
            print('Finished Detpol',detpols[Ldx])
        if retg1 == 'False':
            return Z
        elif retg1 == 'True':
            return Z,g1dic
 
    def loFFTplot(self,low_op,PDM,Nt):  
        
        timet = self.T                                     #Length of time of tlist defined to be one period of the system
        dt = timet/Nt                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt)  
        
        f0,qe,f_modes_table_t = self.PrepWork(tlist)
    
    
        lowfloq = []                           #Defining an empty matrix to store the lowering operator in the Floquet mode basis for every time t in tlist
        for idx in range(Nt):
            lowfloq.append(low_op.transform( \
                                            f_modes_table_t[idx*PDM])) #For every time t in tlist, transform the lowering operator to the Floquet mode basis using the Floquet mode table
                

        '''
        Recasting lowfloq as an array because QuTiP stores arrays in a very weird way
        '''
        lowfloqarray = np.zeros((self.QD.Hdim,self.QD.Hdim,Nt),dtype = complex) #Creating an empty array to hold the lowering operator as an array instead of a list of QObjs
        for i in range(self.QD.Hdim):
            for j in range(self.QD.Hdim):
                for k in range(Nt):
                    lowfloqarray[i,j,k] =               \
                                      lowfloq[k][i][0][j] #This loop takes each index of each of the lowering operator QObjects in the list and stores them in an array
        
        # amps=np.zeros_like(lowfloqarray, dtype = complex) #Creating the empty array to hold the Fourier Amplitudes of each index at every harmonic
        amps = scp.fft.fft(lowfloqarray,axis=2) #This loop performs the FFT of each index of the Floquet mode basis lowering operator to find their harmonic amplitudes.
        amps = amps/len(tlist)
            
        fig, ax = plt.subplots(1,1)
        ax.plot(amps[0,0], 'b')
        ax.plot(amps[0,1], 'r')
        ax.plot(amps[1,0], 'r--')
        ax.plot(amps[1,1], 'b--')
        plt.xlabel('frequency (Unsure of units)')
        plt.ylabel('amp (arbitrary)')
        ax.set_title('FFT of Time dependent Lowering operator in Floquet Basis')   
    
    
    

            
    

