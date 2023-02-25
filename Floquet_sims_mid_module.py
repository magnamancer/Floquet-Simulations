# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:20:29 2023

@author: Fenton
"""
import numpy.ma as ma
import numpy as np
from qutip import *
import Floquet_sims_lowest_module as flm
import Floquet_sims_mid_module as fmm
import itertools
import math
import scipy as scp

def floquet_modes2(H, T, args=None, sort=False, U=None, options=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.

    Returns a list of :class:`qutip.qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]. The optional parameter `sort`
    decides if the output is to be sorted in increasing quasienergies or not.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    args : dictionary
        dictionary with variables required to evaluate H

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    U : :class:`qutip.qobj`
        The propagator for the time-dependent Hamiltonian with period `T`.
        If U is `None` (default), it will be calculated from the Hamiltonian
        `H` using :func:`qutip.propagator.propagator`.
    
    options : :class:`qutip.Options`
        with options for the ODE solver.

    Returns
    -------

    output : list of kets, list of quasi energies

        Two lists: the Floquet modes as kets and the quasi energies.

    """

    if options is None:
        options = Options()
        options.atol = 1e-8   # Absolute tolerance
        options.rtol = 1e-6   # Relative tolerance
        options.nsteps= 1e+8
        
    
    options.rhs_reuse = True
    rhs_clear()
    #print(options)
    
    if U is None:
        # get the unitary propagator
        U = propagator(H, T, [], args, options)

    #print(options)
    # find the eigenstates for the propagator
    evals, evecs = np.linalg.eig(U.full())

    eargs = np.angle(evals)

    # make sure that the phase is in the interval [-pi, pi], so that
    # the quasi energy is in the interval [-pi/T, pi/T] where T is the
    # period of the driving.  eargs += (eargs <= -2*pi) * (2*pi) +
    # (eargs > 0) * (-2*pi)
    eargs += (eargs <= -np.pi) * (2 * np.pi) + (eargs > np.pi) * (-2 * np.pi)
    e_quasi = -eargs / T

    # sort by the quasi energy
    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    # prepare a list of kets for the floquet states
    new_dims = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(np.matrix(evecs[:, o]).T,
                       dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]

def floquet_modes_table2(f_modes_0, f_energies, tlist, H, T, args=None, options=None):
    """
    Pre-calculate the Floquet modes for a range of times spanning the floquet
    period. Can later be used as a table to look up the floquet modes for
    any time.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at :math:`t`

    f_energies : list
        Floquet energies.

    tlist : array
        The list of times at which to evaluate the floquet modes.

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        dictionary with variables required to evaluate H
    
    options : :class:`qutip.Options`
        with options for the ODE solver.

    Returns
    -------

    output : nested list

        A nested list of Floquet modes as kets for each time in `tlist`

    """
    # truncate tlist to the driving period
    #tlist_period = tlist[np.where(tlist <= T)]
    tlist_period = tlist

    f_modes_table_t = [[] for t in tlist_period]

    if options is None:
        options = Options()
    
    options.rhs_reuse = True
    rhs_clear()
    #print(options)

    for n, f_mode in enumerate(f_modes_0):
        output = sesolve(H, f_mode, tlist_period, [], args, options)
        for t_idx, f_state_t in enumerate(output.states):
            f_modes_table_t[t_idx].append(
                f_state_t * np.exp(1j * f_energies[n] * tlist_period[t_idx]))
    return f_modes_table_t

def prepWork(H,T,args,tlist,taulist, start_time = 0, opts = None):
    '''
    Internal Function for use in calculating time evolution with FLoquet Theory

    Parameters
    ----------
    H : List of [[Qobj_H0],[Qobj_H1,func1],[Qobj_H2,func2]...]
        System Hamiltonian, time-dependant in functional form.
    T : float
        System period.
    args : dictionary
        dictionary with variables required to evaluate H
    tlist : np.linspace
        List of Nt time points evenly distributed within one period T of the Hamiltonian
    taulist : np.linspace
        List of Ntau time points evenly distributed tau periods T of the Hamiltonian over which to evolve the system. 
    start_time : float, optional
        The WHOLE NUMBER OF PERIODS forward time at which to begin creating
            the state tables. Called ss_time because it's usually used to pass
            the steadystate time to prepWork for steadystate calculations. 
            The default is None, i.e. create f_states from t=0 onward.
    opts : ??, optional
        options for the ODE solver.
        The default is None.

    Returns
    -------
    f0 : list
        list of t=0 floquet mode/state eigenvectors.
    qe : list
        List of Floquet quasienergies of the system.
    f_modes_list_one_period : numpy array
        List of Floquet modes of the system for one period (you will only
            ever need one period as the modes are cyclical with a period
            of one ssytem period).
    fstates : numpy array
       3D numpy array of Floquet states of the system for all times 
           ss_time+taulist
    fstatesct : numpy array
        Conjugate transpose of fstates, to save space when constantly calling
            it elsewhere.

    '''
    if opts == None:
        opts = Options()                                  
        opts.atol = 1e-4                                
        opts.rtol = 1e-6                                  
        opts.nsteps= 10e+4                                
    
    #Calling the dimensionality of the system, for use later
    Hdim = np.shape(H[0])[0]
    
    #Solving the times over which I'll need to solve the lowering and raising 
    #   operators. 
    end_time = start_time+taulist[-1]+taulist[1]  
    num_periods = int(np.ceil((end_time)/T))
    time_evolution_t_points = np.concatenate((start_time+taulist, 
                                             end_time+tlist))     
    
    #Solving for the initial modes and quasienergies given by f0,qe respectively
    f0, qe = floquet_modes2( H, T,  args, sort = False, options = opts)

    #Solving the mode table for a single period of driving, for use in transformations of the lowering operator and initial state
    f_modes_table_t = floquet_modes_table2(             
                                           f0,  qe, tlist, H,  T,    
                                           args, options = opts) 
    
    
    '''
    To form the Floquet State Table (in the form of a Numpy array), I:
        1) First solve the Floquet modes for one period, 
            1b) Stacking the list of lists into a list of arrays and then an overall numpy array
        3) Tiling out the mode table for the full number of periods of evolution+1 for some extra room
        3) For each time "t" and mode "mode," the complex exponential given by exp(-1j*qe[mode]*t) is multiplied into the corresponding
            mode at time t to solve for the Floquet State basis over the full time period of evolution given by tau
            
    '''
    
    
    #Modes Table for a single period
    f_modes_list_one_period = np.stack([np.hstack([i.full() for i in modest]) 
                                        for modest in f_modes_table_t])

    #Full time evaluation is tau number of periods +1 period for time averaging, so I need that many periods of modes
    f_modes_list_full_time = np.tile(f_modes_list_one_period, (num_periods+1,1,1)) 
    
    #Transposing to make the list comprehension much easier in f_states_transposed
    f_modes_transpose = np.transpose(f_modes_list_full_time,(0,2,1))
    
    #Creating the array of exponentials for multiplying the modes to create the states.
    mode_exponentials = np.stack([np.stack([np.exp(-1j*qe[mode]*(t)) 
                                            for mode in range(Hdim)]) 
                                  for t in time_evolution_t_points])
    
    #The transposed floquet states are calculated
    f_states_transposed = np.stack([
                              np.stack([
                                  f_modes_transpose[t,mode]*mode_exponentials[t,mode]
                                  for mode in range(Hdim)])
                              for t in range(len(time_evolution_t_points))],axis=0)
    
    #un-transposing the states
    fstates =  np.transpose(f_states_transposed,(0,2,1))
    #Solving the dagger now, to save space when called later
    fstatesct = np.transpose(fstates.conj(),axes=(0,2,1))
    
    return f0,qe,f_modes_list_one_period,fstates,fstatesct

def evol_time(c_op_mag,T,evol_time = None):  
    '''
    Parameters
    ----------
    Parameters
    ----------
    c_op_mag : float
        smallest magnitude of all collapse operators of the system
    T : float
        Period of the Hamiltonian.
    evol_time : INT, optional
        the WHOLE NUMBER OF PERIODS forward to evolve the system. 
        The default is to find the steadystate time.

    Returns
    -------
    periods_required : int
        WHOLE NUMBER of periods foward at which the steady state occurs.

    '''

    #If no time is input, assume steadystate dynamics.
    if evol_time == None:
        time_scale = (2*np.pi)/c_op_mag        
        periods_required = int(np.ceil(time_scale/T))
    #Else, pass the evol_time as a number of periods of evolution
    else:
        periods_required = int(np.ceil(evol_time*T))
    
    return periods_required

#DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED 
#DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED 
#DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED 
#DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED 
#DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED 
'''
def f_states_table(f0, qe, tlist, taulist, H=None, T=None, args=None, fmodest=None):
    """
    Takes an input supervector and turns it into a density matrix in the desired basis

    Parameters
    ----------
    f0: list of QObjs
        Intial Floquet Modes of the System
    qe: list
        Quasienergies of the system
    tlist: list
        Some time list distributed over one period T of the system.
    taulist: list
        the time over which the system will be evolved. 
    H: Nested list of QObjs
        The Hamiltonian of the System
    T: Float
        The characteristic period of the Hamiltonian
    args: Dictionary
        Time dependent arguments of the system
    fmodest: OPTIONAL list of Qobjs
        Optional list of floquet modes precomputed by the function f_modes_table_t. If 
        mo input is specified, computes the floquet mode table internally
   

    Returns
    -------
    fstates_table: a density matrix in the desired basis at the specified time t

    """
    
    if fmodest == None:
        #
        #First we calculate the fmodes
        #
        Nt = 2**10
        time = T
        dt = time/Nt
        tlist = np.linspace(0, time-dt, Nt) 
        
        opts = Options()
        opts.atol = 1e-18   # Absolute tolerance
        opts.rtol = 1e-15   # Relative tolerance
        opts.nsteps= 10000000
        
        fmodest = esm.floquet_modes_table2(f0, qe, tlist, H, T, args, options=opts)
        
    #
    #Defining the dims and shape of the eventual states using fmodest
    #
    dims1 = fmodest[0][0].dims
    shape1 = fmodest[0][0].shape
    
    fstates = []
    resket0 = []
    for idx, tau in enumerate(taulist):
        #
        #First we check to find the nearest point in tlist
        #
        #Subtracting tau from every t point
        array = tlist-(tau%T)
        
        #taking the absolute value of the above array and finding the minimum, which finds the closest t point
        miniabs = min(np.abs(array))
        
        #Taking the first index argument
        #This elemenates the need to specify elements later and also takes care of if there are two indices of t equadistant from tau
        index = np.where(np.abs(array) == miniabs)[0][0]
        
        #Finding the sign for use interpolating down below
        sgn = np.sign(tlist[index]-tau)
        
        #
        #Next, I'll use mini to calculate the fstates for the given tau value
        #
        if miniabs == 0:
            #If the time tau falls exactly on a t point, constructing the 
            #state is as easy as multiplying the lookup mode at that time tau
            #by the complex exponential
            #fstates.append([Qobj(floquet_modes_t_lookup(fmodest, tau, T)[i]*np.exp(-1j*qe[i]*tau),dims=dims1,shape=shape1) for i in list(range(fmodest[0][0].shape[0]))])
            t0 = tlist[index]
            if index+1 == len(tlist):
                t1 = T
            elif index+1 != len(tlist):
                t1 = tlist[index+1]
            
        elif miniabs != 0:
            if sgn < 0:
                #sgn <0 means the tlist point occurs before the taulist point!
                #So the two points to interpolate will be tlist[index] and tlist[index+1]
                #The zero arguments in index just flatten it to an integer
                t0 = tlist[index]
                #Writing in an exception for when tlist[i+1] happends to be at the very end of the period
                if index+1 == len(tlist):
                    t1 = T
                elif index+1 != len(tlist):
                    t1 = tlist[index+1]
            elif sgn > 0:
                #sgn >0 means the tlist point occurs after the taulist point!
                #So the two points to interpolate will be tlist[index-1] and tlist[index]
                #The zero arguments in index just flatten it to an integer
                t0 = tlist[index-1]
                t1 = tlist[index]
               
        #The next step is to interpolate the two points
        
        ket0 = floquet_modes_t_lookup(fmodest, t0, T)
        ket1 = floquet_modes_t_lookup(fmodest, t1, T)
        
        #Finding the resultant ket now
        #Interpolating with Ned's suggestion instead of the wikipedia suggestion
        #y = y0 * (x1 - x)/(x1 - x0) + y1 * (x - x0)/(x1 - x0) 
        resket0.append([Qobj(ket0[i]*((t1-(tau%T))/(t1-t0))+ket1[i]*(((tau%T)-t0)/(t1-t0))) for i in list(range(fmodest[0][0].shape[0]))])
        
        
        #Finally I'll make the floquet state from this interpolated state:
        fstates.append([Qobj(resket0[idx][i]*np.exp(-1j*qe[i]*tau),dims=dims1,shape=shape1) for i in list(range(fmodest[0][0].shape[0]))])
        
    return fstates

'''