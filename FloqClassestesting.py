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
    def lowering_operator(self):
        '''
        Creates the lowering operator of the QD based on the 
        system dimensionality and the states linked by the dipole moments
        '''
        
        return Qobj(sum([np.eye(1, self.Hdim * self.Hdim, k=(i) * self.Hdim + j).reshape(self.Hdim, self.Hdim) for i,j in self.dipoles]))

class QSys:
    def __init__(self,QD,LasList =[], Bfield = None, c_op_list = None):
        self.LasList = LasList
        self.Bfield = Bfield
        self.QD = QD
        self.c_op_list = c_op_list


        if len(self.LasList) == 1:
            freqspoof = self.LasList[0].freq*0.99997
            self.Lavg = (1/2)*(freqspoof+self.LasList[0].freq)
            self.beat  = abs(freqspoof-self.LasList[0].freq )
            
        elif len(self.LasList) == 2:
            self.Lavg = (1/2)*(self.LasList[0].freq+self.LasList[1].freq)
            self.beat  = abs(self.LasList[1].freq-self.LasList[0].freq )
            
        
        
        
         
                                    
        self.T     = (2*np.pi/(self.beat/2))
        self.Delta = self.QD.states[-1]-self.Lavg                             #Average Detuning of the lasers from the excited state

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
        
     
        
      
        '''
        Defining the time dependance as functions for use in the Hamiltonian
        '''
        def H1_coeff(t, args):
            return  np.exp(1j*(self.beat/2)*t) 

        def H2_coeff(t, args):
            return  np.exp(-1j*(self.beat/2)*t) 
        
        H = [(self.AtomHam()+self.ZHam()),                                                    
            [self.LasHam()[0],H1_coeff],                    
            [self.LasHam()[1],H2_coeff]]   
            
        
        
        
        
        return H
  
    
    '''
    Creating a method to define the laser Hamiltonian(s) in the RWA
    '''
   
   
    def LasHam(self):
        
        '''
        The first step is to define all of the Rabi frequencies
        '''
        rabi_freq = {}
        rabi_freq_tilde = {}
        for lasdx, Laser in enumerate(self.LasList):
            for edx, dipole in enumerate(self.QD.dipoles): 
                rabi_freq[edx,lasdx] = np.dot(self.QD.dipoles[dipole],Laser.E)
                rabi_freq_tilde[edx,lasdx] = np.dot(self.QD.dipoles[dipole],np.conj(Laser.E))
        
        '''
        Next I want to define the time-dependance of each Rabi frequency. Keep in mind that tilde terms have 
            negative time dependance comapared to their un-tilde'd counterparts, Fenton
        '''
        laser_beat = {}  
        laser_beat_tilde = {}          
        for lasdx, Laser in enumerate(self.LasList):
            laser_beat[lasdx] = Laser.freq-self.Lavg
            laser_beat_tilde[lasdx] = -Laser.freq-self.Lavg
        
        
        '''
        Finding forward rotating terms by subtracting the beat frequency.
            Anything that ends up at zero is a forward rotating term
        '''
        forward = []
        for lasdx, key in enumerate(laser_beat):
            for edx, moment in enumerate(self.QD.dipoles.keys()):
                dipole_moment = moment
                dip_mat = basis(self.QD.Hdim,dipole_moment[0])*basis(self.QD.Hdim,dipole_moment[1]).dag()
                
                if math.isclose(laser_beat[key] , self.beat/2,abs_tol = 1e-4):
                   
                    forward.append(((-1/2)*rabi_freq[edx,lasdx]*dip_mat).full())
                 
                if math.isclose(laser_beat_tilde[key] , self.beat/2,abs_tol = 1e-4):
                    forward.append((-1/2)*rabi_freq_tilde[edx,lasdx]*dip_mat)
        
        
        '''
        Finding backward rotating terms by subtracting the -beat frequency.
            Anything that ends up at zero is a backward rotating term
        '''
        backward = []
        for lasdx, key in enumerate(laser_beat):
            for edx, moment in enumerate(self.QD.dipoles.keys()):
                dipole_moment = moment
                dip_mat = basis(self.QD.Hdim,dipole_moment[0])*basis(self.QD.Hdim,dipole_moment[1]).dag()
                
                if math.isclose(laser_beat[key] , -self.beat/2,abs_tol = 1e-4):
                   
                    backward.append(((-1/2)*rabi_freq[edx,lasdx]*dip_mat).full())
                 
                if math.isclose(laser_beat_tilde[key] , -self.beat/2,abs_tol = 1e-4):
                    backward.append((-1/2)*rabi_freq_tilde[edx,lasdx]*dip_mat)
        
        '''
        Adding the conjugates to one another
        '''
        backward_appended = backward.copy()  
        forward_appended  = forward.copy()  
        for back_mat in backward:
            forward_appended.append(back_mat.T.conj())
        
        for for_mat in forward:
            backward_appended.append(for_mat.T.conj())
        
        
        self.forwardrot  = sum(forward_appended )
        self.backwardrot = sum(backward_appended)
        
        if not self.forwardrot.any():
            self.forwardrot = np.zeros((self.QD.Hdim,self.QD.Hdim))
        
        if not self.backwardrot.any():
            self.backwardrot = np.zeros((self.QD.Hdim,self.QD.Hdim))
            
       
        
        return Qobj(np.round(self.forwardrot,10)), Qobj(np.round(self.backwardrot,10))

    
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
            self.AtomHammy[n,n] = self.QD.states[n]-eta[n]
        
        return Qobj(np.round(self.AtomHammy,10))
    
    
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
                    
                    Because f0 function can't take degenerate manifolds (without a magnetic field!), I add something like ~1/1000 of a GHz 
                    detuning to the second manifold.
                    
                    Empirically, it lookds like that doesn't do anything negative at all to the resulting spectra, though that nondegenaracy then
                    fucks up the degeneracy condition I use to build the Zeeman Hamiltonian. By rounding the input, I'm rounding out that small
                    nondegeneracy. WHICH MEANS YOU NEED TO MAKE A BETTER FIX LATER, FENTON.
                    
                    I'm no longer rounding, but now checking absolute percent deviation. If the first entry is zero, check to see if the second entry is zero.
                    If they are, congrats, stick a magnetic field in that bitch. If the first entry is not zero, check to see that its absolute percent deviation
                    from the second entry is less than 1%. If it is, magnetic field. Otherwise ignore it. THIS IS STILL NOT A FULL FIX FENTON. YOU NEED TO FIX 
                    FMODES SO IT DOESN'T FUCK UP WITH DEGENERATE MANIFOLDS. THIS IS JUST A BETTER BANDAID
                    '''
                    if abs(self.QD.states[i]-self.QD.states[j]) <= 1e-6 or abs(self.QD.states[i]) != 0 and 100*(abs(self.QD.states[i]-self.QD.states[j])/abs(self.QD.states[i])) <= 1 :
                        self.ZHammy[i,i] = -1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                        self.ZHammy[j,j] =  1*self.Bfield.Bvec[1]*self.QD.gfactors[Count][1]
                        self.ZHammy[i,j] = self.ZHammy[j,i] = (1/2)*self.Bfield.Bvec[0]*self.QD.gfactors[Count][0]
                       
                        Count += 1
        return Qobj(np.round(self.ZHammy,10))
        
        
        
        
        
        
        
    '''
    Various time evolution methods below
    '''
        
    def expect_1op_1t(oper,rho0,tlist,Rdic,omega, taulist = None, opts = None):
        '''
        for finding <op1(t)>
    
        Parameters
        ----------
        op1 : Square matrix operator
            the operator, to be evaluated at t, in the expectation value
        rho0 : vector density supermatrix
           initial state in the FLOQUET basis
        tlist : linspace
            time list over one period of the Hamiltonian, with evenly distributed t-points
        taulist : linspace
            times tau over which to evaluate the two-time expectation value.
        Rdic : Dic
            Hdim**2 by Hdim**2 matrix values, with variable number of keys depending on selected time-dependance.
            Dictionary of time-dependances, built from the system collapse operators in the Floquet basis
        omega : float??
            time period of the Hamiltonian
        opts : TYPE, optional
            optional arguments for solve_ivp solvers. The default is None.
    
        Returns
        -------
        one_op_one_time_expect : list of values
           calculated 1 operator 1 time expectation value.
    
        
        '''
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+6                                #Maximum number of Steps                              #Maximum number of Steps
        
            
        
        op_dims = np.shape(oper)[-1] #operator is square so taking the last dimension should be fine.
                                                                     
        # print('finished solving the IVP/time evolving rho0')
        '''
        Next step is to iterate this steady state rho_s forward in time by one period of the Hamiltonian for the purpose of time averaging the result. I'll choose the times
        to be evenly spread out within T, the time scale of the Hamiltonian.
        
        In this step I also multiply in the population operator in the Floquet STATE basis at the correct time, to get the un-time-averaged excitation spectra. Then I average the result over
        the time axis and take the trace to get the population value (I think?) in the steady state.
        '''
        
        op_rho_ss_unavg = [ (oper[i])                       \
                      @ np.reshape(
                            scp.integrate.solve_ivp(rhodot                       ,
                                                    t_span = (0,tlist[-1])   ,
                                                    y0=rho0                 ,
                                                    args=(Rdic,omega)           ,
                                                    method='DOP853'                  ,
                                                    t_eval=(tlist)            ,
                                                    rtol=opts.rtol                   ,
                                                    atol=opts.atol).y[:,i]           ,
                                    (op_dims,op_dims),order='F')  
                        for i in list(range(0,len(tlist)))]
        
        if taulist is not None:
            op_rho_ss_unavg_many_t  = np.zeros( (len( op_rho_ss_unavg)         , len(taulist), op_dims,op_dims), dtype='complex_' ) 
            for tdx in range(len(tlist)): #First for loop to find the tau outputs for each t value
                #New "starting time"
                initial_tau = tlist[tdx]
                
               
               
                rho_ss_tau_evolve = np.moveaxis(np.dstack(np.split(scp.integrate.solve_ivp(
                                                    rhodot,
                                                    t_span = (initial_tau,initial_tau+taulist[-1]), 
                                                    y0=np.reshape(op_rho_ss_unavg[tdx],(op_dims**2,),order='F'),
                                                    args=(Rdic,omega),
                                                    method='DOP853',
                                                    t_eval=(initial_tau+taulist),
                                                    rtol=opts.rtol,
                                                    atol=opts.atol).y,
                                        op_dims,axis=0)),(0,1,2),(1,0,2))
                        
                op_rho_ss_unavg_many_t[tdx,...] = rho_ss_tau_evolve    
            
                op_rho_ss_avg = np.mean(op_rho_ss_unavg_many_t,axis=0)
            one_op_one_time_expect = np.trace(op_rho_ss_avg,axis1=1,axis2=2)
        
            
        
        
        
        else:
            op_rho_ss_avg = np.average(op_rho_ss_unavg,axis=0)
            one_op_one_time_expect = np.trace(op_rho_ss_avg,axis1=0,axis2=1)
        
        
        
        
        return one_op_one_time_expect
    
     
    def expect_2op_2t(op1,op2,rho0,tlist,taulist,Rdic,omega,opts = None):  
        '''
        for finding <op1(t) op2(t+tau)>
    
        Parameters
        ----------
        op1 : Square matrix operator
            the first operator, to be evaluated at t, in the expectation value
        op2 : Square matrix operator
            the second operator, to be evaluated at t+tau, in the expectation value
        rho0 : vector density supermatrix
           initial state in the FLOQUET basis
        tlist : linspace
            time list over one period of the Hamiltonian, with evenly distributed t-points
        taulist : linspace
            times tau over which to evaluate the two-time expectation value.
        Rdic : Dic
            Hdim**2 by Hdim**2 matrix values, with variable number of keys depending on selected time-dependance.
            Dictionary of time-dependances, built from the system collapse operators in the Floquet basis
        omega : float??
            time period of the Hamiltonian
        opts : TYPE, optional
            optional arguments for solve_ivp solvers. The default is None.
    
        Returns
        -------
        two_op_two_time_expect : list of values
           calculated 2 operator 2 time expectation value.
    
        '''
        
        
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+6   
      
        
      
        op_dims = np.shape(op1)[-1]  
      
        
      
       
     
        
       
        
       
        '''
        Next step is to iterate this steady state rho_s forward in time. I'll choose the times
        to be evenly spread out within T, the time scale of the Hamiltonian
        
        In this step I also multiply in the lowering operator in the Floquet STATE basis at the correct time,
        from (taulist[-1]+dt) to (taulist[-1]+dt+tlist[-1])
        '''
        
    
        one_op_rhoss_prod = [ op1[i]                       \
                              @ np.reshape(
                                    scp.integrate.solve_ivp(rhodot,
                                                            t_span = (0,tlist[-1])  ,
                                                            y0=rho0               ,
                                                            args=(Rdic,omega)               ,
                                                            method='DOP853'         ,
                                                            t_eval=(tlist)            ,
                                                            rtol=opts.rtol              ,
                                                            atol=opts.atol).y[:,i]       ,
                                            (op_dims,op_dims),order='F')  
                                for i in range(len(tlist))]
            
    
        
        '''
        Setting up a matrix to have rows equal to the number of tau values and columns equal to the number of t values
        At the end I'll average over each row to get a vector where each entry is a tau value and an averaged t value
        '''
    
        # print('Finished B-States')
        
        two_op_rho_ss_unavg = np.zeros( (len(one_op_rhoss_prod), len(taulist), op_dims,op_dims), dtype='complex_' ) 
        # print('Starting A States')
        for tdx, one_op_rhoss_single_t in enumerate(one_op_rhoss_prod): #First for loop to find the tau outputs for each t value
            #New "starting time"
            initial_tau = tlist[tdx]
            
            # print('Filling column',tdx+1,'of',len(Bstates))
            one_op_rho_ss_unavg_tau_evolution = np.moveaxis(np.dstack(np.split(scp.integrate.solve_ivp(
                                                rhodot,
                                                t_span = (initial_tau,initial_tau+taulist[-1]), 
                                                y0=np.reshape(one_op_rhoss_single_t,(op_dims**2,),order='F'),
                                                args=(Rdic,omega),
                                                method='DOP853',
                                                t_eval=(initial_tau+taulist),
                                                rtol=opts.rtol,
                                                atol=opts.atol).y,
                                   op_dims,axis=0)),(0,1,2),(1,0,2))
            
            '''
            STOP CHANGING THE ORDER OF THE TRANSPOSE FENTON. IT ISN'T GOING TO FIX IT
            TIMES I WAS WEAK: 14
            '''
            
            two_op_rho_ss_unavg[tdx,...] = op2[(tdx):(len(taulist)+tdx)]@ one_op_rho_ss_unavg_tau_evolution
        # print('found unaveraged A-States')   
        '''
        Okay so the output matrix from above is a bunch of 2x2 density matrices
        where the value idx1 refers to the tau value and the value idx refers to the t value
    
        Going forward I should now average over each "row" of t values, i.e. average over idx
        '''
        
        two_op_rho_ss_avg = np.mean( two_op_rho_ss_unavg,axis=0)
        
        
        two_op_two_time_expect = np.trace(two_op_rho_ss_avg,axis1=1,axis2=2)
        
        return two_op_two_time_expect
    
    def expect_4op_2t(op1,op2,op3,op4,rho0,tlist,taulist,Rdic,omega,opts = None):  
        '''
        for finding <A(t0)*B(t1)*C(t1)*D(t0)>
    
        Parameters
        ----------
        op1 : Square matrix operator
            the first operator, to be evaluated at t, in the expectation value
        op2 : Square matrix operator
            the second operator, to be evaluated at t+tau, in the expectation value
        op3 : Square matrix operator
            the third operator, to be evaluated at t+tau, in the expectation value
        op4 : Square matrix operator
            the fourth operator, to be evaluated at t+tau, in the expectation value
        rho0 : vector density supermatrix
           initial state in the FLOQUET basis
        tlist : linspace
            time list over one period of the Hamiltonian, with evenly distributed t-points
        taulist : linspace
            times tau over which to evaluate the two-time expectation value.
        Rdic : Dic
            Hdim**2 by Hdim**2 matrix values, with variable number of keys depending on selected time-dependance.
            Dictionary of time-dependances, built from the system collapse operators in the Floquet basis
        omega : float??
            time period of the Hamiltonian
        opts : TYPE, optional
            optional arguments for solve_ivp solvers. The default is None.
    
        Returns
        -------
        two_op_two_time_expect : list of values
           calculated 2 operator 2 time expectation value.
    
        '''
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+6   
      
        op_dims = np.shape(op1)[-1]                                                        
       
    
        
       
        
       
        '''
        Evolving one more period forward for averaging purposes
    
        '''
        D_rhoss_A = [ op4[i] @
                    np.reshape(
                        scp.integrate.solve_ivp(rhodot,
                                                t_span = (0,tlist[-1])  ,
                                                y0=rho0                ,
                                                args=(Rdic,omega)               ,
                                                method='DOP853'         ,
                                                t_eval=(tlist)            ,
                                                rtol=opts.rtol              ,
                                                atol=opts.atol).y[:,i]       ,
                                (op_dims,op_dims),order='F')   
                    @ op1[i]
                    for i in range(len(tlist))]
        
        '''
        Forming the innermost t (as opposed to t+tau) portion of the g2 numerator
        '''
        # D_rhoss_A = [sys_f_low[len(taulist)+idx] @ rhoss_floquet_t[idx] @ sys_f_raise[len(taulist)+idx] for idx in range(Nt)]
        
        
    
        unavgd_4op_rhoss   = np.zeros( (len( D_rhoss_A)         , len(taulist), op_dims,op_dims), dtype='complex_' ) 
        for tdx in range(len(tlist)): #First for loop to find the tau outputs for each t value
            #New "starting time"
            initial_tau = tlist[tdx]
            
            
            
            oper_state_tau_evolution = np.moveaxis(np.dstack(np.split(scp.integrate.solve_ivp(
                                                rhodot,
                                                t_span = (initial_tau,initial_tau+taulist[-1]), 
                                                y0=np.reshape(D_rhoss_A[tdx],(op_dims**2,),order='F'),
                                                args=(Rdic,omega),
                                                method='DOP853',
                                                t_eval=(initial_tau+taulist),
                                                rtol=opts.rtol,
                                                atol=opts.atol).y,
                                    op_dims,axis=0)),(0,1,2),(1,0,2))
            
            
            
            
            unavgd_4op_rhoss[tdx,...]   = (op2 @ op3)[(tdx):(len(taulist)+tdx)] @ oper_state_tau_evolution
           
        
        avgd_4op_rhoss   = np.mean(unavgd_4op_rhoss,axis=0)
        
        expect4op2t = np.trace( avgd_4op_rhoss  , axis1=1, axis2=2)
       
        return expect4op2t
    
    def ExciteSpec(self,Nt,tau,rho0,time_sensitivity = 0,detpols = np.array([None,None,None]),opts = None):       
            ############### Time evolving rho0 with solve_ivp#########################
        
        
        
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        
        
        
        Nt = Nt                                        #Number of Points
        timet = self.T                                      #Length of time of tlist defined to be one period of the system
        dt = timet/Nt                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt)               #Combining everything to make tlist
        
        #Taulist Definition
        Ntau = (Nt)*(tau)                                    
        taume = tau*self.T                          
        dtau = dt                                 
        taulist = np.linspace(0, taume-dtau, Ntau)       
        
        ss_time = esm.steadystate_time(np.amin(np.array([c_op.mag for c_op in self.c_op_list])),self.T)
        f0,qe,f_modes_list_one_period,f_states_all_periods,f_states_all_periods_conjt= esm.prepWork(self.Ham(),self.T,self.Hargs,tlist,taulist,ss_time, opts = opts) 
        # print('found f0, qe, f_states_all_periods')
        
        
        
        lowop_floquet_fourier_amps_list = esm.floquet_fourier_amps(Nt,tlist,taulist, [c_op.mat for c_op in self.c_op_list], f_modes_list_one_period, opts = opts)       
        Rdic = esm.floquet_rate_matrix(qe,lowop_floquet_fourier_amps_list,[c_op.mag for c_op in self.c_op_list],self.beat,time_sensitivity )
        # print('Built R(t)')


        lowop_detection_polarization_modified_list =  esm.lowop_detpol_modifier( self.QD.lowering_operator(),self.QD.dipoles,detpols)
        # print('set detection polarization')
        

        
        '''
        Looping over detection polarizations, to hopefully make things faster
        '''
        excitevals = {}
        for Ldx, detpol_low_op in enumerate(lowop_detection_polarization_modified_list):
           
            '''
            Doing the raising and lowering operator transformations, to move them
                into the Floquet basis for every t_inf+t
            '''
            
            pop_op = f_states_all_periods_conjt @ (detpol_low_op.dag()*detpol_low_op).full() @ f_states_all_periods
            
            rho0_floquet = operator_to_vector(rho0.transform(f0,False)).full()[:,0]
            
            rho_steadystate = esm.steadystate_rho(rho0_floquet,ss_time,Nt,Rdic,self.beat/2,self.T)
            excitevals[detpols[Ldx]] = esm.expect_1op_1t(pop_op,rho_steadystate,tlist,Rdic,self.beat/2,opts = opts)
            
            # print('Finished Detpol',detpols[Ldx])
        print('Finished excitation spectrum')  
        return  excitevals
    
    def EmisSpec(self,Nt,tau,rho0,time_sensitivity = 0,detpols = np.array([None,None,None]), retg1 = 'False', opts = None):       
        ############### Time evolving rho0 with solve_ivp#########################
        
             
        
        
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-6                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        
        
        
        Nt = Nt                                        #Number of Points
        timet = self.T                                      #Length of time of tlist defined to be one period of the system
        dt = timet/Nt                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt)               #Combining everything to make tlist
        
        #Taulist Definition
        Ntau = (Nt)*(tau)                                    
        taume = tau*self.T                          
        dtau = dt                                 
        taulist = np.linspace(0, taume-dtau, Ntau)       
        
        '''
        Defining the total time of evolution for use in calculating the floquet states
        '''
        #Taulist Definition
        NtauF = (Nt)*2*(tau)                                    
        taumeF = 2*tau*self.T                          
        dtauF = dt                                 
        taulistF = np.linspace(0, taumeF-dtauF, NtauF)       

        
        ss_time = esm.steadystate_time(np.amin(np.array([c_op.mag for c_op in self.c_op_list])),self.T)
        f0,qe,f_modes_list_one_period,f_states_all_periods,f_states_all_periods_conjt= esm.prepWork(self.Ham(),self.T,self.Hargs,tlist,taulist,ss_time, opts = opts) 
        

               
        lowop_floquet_fourier_amps_list = esm.floquet_fourier_amps(Nt,tlist,taulist, [c_op.mat for c_op in self.c_op_list], f_modes_list_one_period, opts = opts)

        Rdic = esm.floquet_rate_matrix(qe,lowop_floquet_fourier_amps_list,[c_op.mag for c_op in self.c_op_list],self.beat,time_sensitivity )

        
        
        '''
        Looping over detection polarizations, to hopefully make things faster
        '''
        lowop_detection_polarization_modified_list =  esm.lowop_detpol_modifier( self.QD.lowering_operator(),self.QD.dipoles,detpols)
           
        
        
        
        Z = {}
        g1dic = {}
        for Ldx, detpol_low_op in enumerate(lowop_detection_polarization_modified_list):

            sys_f_low = f_states_all_periods_conjt @ (detpol_low_op).full() @ f_states_all_periods
            sys_f_raise = np.transpose(sys_f_low,axes=(0,2,1)).conj()
            
            
            rho0_floquet = operator_to_vector(rho0.transform(f0,False)).full()[:,0]
            
            
            
            rho_steadystate = esm.steadystate_rho(rho0_floquet,ss_time,Nt,Rdic,self.beat/2,self.T)
            g1 = esm.expect_2op_2t(sys_f_low, sys_f_raise, rho_steadystate, tlist, taulist, Rdic, self.beat/2)
            
            if retg1 == 'True':
                g1dic[detpols[Ldx]] = g1
            
            spec = np.fft.fftshift(np.fft.fft(g1,axis=0))
    
            Z[detpols[Ldx]] = np.real(spec)/(len(g1))
        
            print('Finished Detpol',detpols[Ldx])
        
        if retg1 == 'False':
            return Z
        elif retg1 == 'True':
            return Z,g1dic
        
            
    def g2_tau(self,Nt,tau,rho0,time_sensitivity = 0,detpols = np.array([None,None,None]), opts = None):
        ############### Time evolving rho0 with solve_ivp#########################
        
        
        
        
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-8                                  #Absolute tolerance
            opts.rtol = 1e-10                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        
        
        
        Nt = Nt                                        #Number of Points
        timet = self.T                                      #Length of time of tlist defined to be one period of the system
        dt = timet/Nt                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt)               #Combining everything to make tlist
        
        #Taulist Definition
        Ntau = (Nt)*(tau)                                    
        taume = tau*self.T                          
        dtau = dt                                 
        taulist = np.linspace(0, taume-dtau, Ntau)       
        
        '''
        Defining the total time of evolution for use in calculating the floquet states
        '''
        #Taulist Definition
        NtauF = (Nt)*2*(tau)                                    
        taumeF = 2*tau*self.T                          
        dtauF = dt                                 
        taulistF = np.linspace(0, taumeF-dtauF, NtauF)       

        
        ss_time = esm.steadystate_time(np.amin(np.array([c_op.mag for c_op in self.c_op_list])),self.T)
        f0,qe,f_modes_list_one_period,f_states_all_periods,f_states_all_periods_conjt= esm.prepWork(self.Ham(),self.T,self.Hargs,tlist,taulist,ss_time, opts = opts) 
        
        print('found f0, qe, f_states_all_periods')
               
        '''
        Doing the raising and lowering operator transformations, to move them
            into the Floquet basis for every t_inf+t

        Calling the raising and lowering operators for use below
        
        '''      

        lowop_floquet_fourier_amps_list = esm.floquet_fourier_amps(Nt,tlist,taulist, [c_op.mat for c_op in self.c_op_list], f_modes_list_one_period, opts = opts)

        Rdic = esm.floquet_rate_matrix(qe,lowop_floquet_fourier_amps_list,[c_op.mag for c_op in self.c_op_list],self.beat,time_sensitivity )
        print('Built R(t)')
        
        
        '''
        Looping over detection polarizations, to hopefully make things faster
        '''
        lowop_detection_polarization_modified_list =  esm.lowop_detpol_modifier( self.QD.lowering_operator(),self.QD.dipoles,detpols)
           
    
        g2func = {}
        for Ldx, detpol_low_op in enumerate(lowop_detection_polarization_modified_list):

            sys_f_low = f_states_all_periods_conjt @ (detpol_low_op).full() @ f_states_all_periods
            sys_f_raise = np.transpose(sys_f_low,axes=(0,2,1)).conj()
            pop_op = f_states_all_periods_conjt @ (detpol_low_op.dag()*detpol_low_op).full() @ f_states_all_periods

            rho0_floquet = operator_to_vector(rho0.transform(f0,False)).full()[:,0]
            rho_steadystate = esm.steadystate_rho(rho0_floquet,ss_time,Nt,Rdic,self.beat/2,self.T)
            

            g2_denom_expect = esm.expect_1op_1t(pop_op, rho_steadystate, tlist,  Rdic, self.beat/2,taulist=taulist)
            g2_numer_expect = esm.expect_4op_2t(sys_f_raise, sys_f_raise, sys_f_low, sys_f_low, rho_steadystate, tlist, taulist, Rdic, self.beat/2)
                
            '''
            Taking t = 0 as "statistically stationary" or w/e it's called means 
            the t value I use doesn't really matter I think, so I can just use
            the t=0 as my t value for all tau values
            ''' 
            g2func[detpols[Ldx]] = np.array([g2_numer_expect[taudx]/(g2_denom_expect[0] * g2_denom_expect[taudx]) for taudx in range(len(taulist))])
            print('finished',detpols[Ldx])
        return g2func, taulist      
            
            
    def loFFTplot(self,low_op,Nt):  
        
        timet = self.T                                     #Length of time of tlist defined to be one period of the system
        dt = timet/Nt                                      #Time point spacing in tlist
        tlist = np.linspace(0, timet-dt, Nt)  
        
        f0,qe,f_modes_table_t = self.PrepWork(tlist)
    
    
        lowfloq = []                           #Defining an empty matrix to store the lowering operator in the Floquet mode basis for every time t in tlist
        for idx in range(Nt):
            lowfloq.append(low_op.transform( \
                                            f_modes_table_t[idx])) #For every time t in tlist, transform the lowering operator to the Floquet mode basis using the Floquet mode table
                

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
    
    
    def TransitionEnergies(self):  
        aa = self.Ham()
        
        
        TransHam = np.zeros((self.QD.Hdim,self.QD.Hdim),dtype='complex')
        N = len(self.QD.states)
        for n in range(N):
            TransHam[n,n] = self.QD.states[n]
        
        TransMat = self.v.dag()*(Qobj(TransHam)+self.ZHam())*self.v
        
        
        return np.array([TransMat[0,0],TransMat[1,1],TransMat[2,2],TransMat[3,3]])/(2*np.pi)

            
    

