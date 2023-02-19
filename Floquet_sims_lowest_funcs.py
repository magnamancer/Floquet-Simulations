# -*- coding: utf-8 -*-

import numpy.ma as ma
import numpy as np
from qutip import *
import itertools
import math
import scipy as scp

'''
This script contains 'lowest-level' definition functions that are called
    by many things in the actual module
'''
################### Floquet mode and state solver Functions ############

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

'''
PREPWORK SHOULD BE MOVED TO 2ND LEVEL

'''
def prepWork(H,T,args,tlist,taulist=None, ss_time = None, opts = None):
    """
    Internal Function for use in calculating time evolution with FLoquet Theory

    Parameters
    ----------

    Nt : Number of points per period. More takes longer

    tau : The number of periods through which you'd like to time evolve'
    
    rho0 : Initial state. If none, will initialize in the ground state.

    Returns
    -------

    output : list of kets, list of quasi energies

        Two lists: the Floquet modes as kets and the quasi energies.

    """  
    
    if opts == None:
        opts = Options()                                  #Setting up the options used in the mode and state solvers
        opts.atol = 1e-4                                #Absolute tolerance
        opts.rtol = 1e-6                                  #Relative tolerance
        opts.nsteps= 10e+4                                #Maximum number of Steps                              #Maximum number of Steps
    
    
    #Making the assumption here that Hdim equals the number of rows of the time-independent Hamiltonian. Might not work always. Works now.
    Hdim = np.shape(H[0])[0]
    
    
    if ss_time != None:
        start_time = ss_time
    else:
        start_time = 0
    
    # Setting useful constants to be used in a few lines
    end_time = start_time+taulist[-1]+taulist[1]  #The length of time Tau forward for which the system is being evolved
    num_periods = int(np.ceil((end_time)/T))
    time_evolution_t_points = np.concatenate((start_time+taulist, end_time+tlist)) #The times over which I'll need to solve the lowering and raising operators. Runs from Tau to 2*Tau+T = 2*(N*T)+T=T(2N+1)
    
    
    # print('starting f0')
    #Solving for the initial modes and quasienergies given by f0,qe respectively
    f0, qe = floquet_modes2( H, T,  args, sort = False, options = opts)
    
    # aa,corrperms = reorder(np.hstack([i.full() for i in f0]))
    

    #Solving the mode table for a single period of driving, for use in transformations of the lowering operator and initial state
    f_modes_table_t = floquet_modes_table2(             \
                              f0,  qe, tlist, H,  T,    \
                                  args, options = opts) 
    
    
    '''
    To form the Floquet State Table (in the form of a Numpy array), I:
        1) First solve the Floquet modes for one period, 
            1b) Stacking the list of lists into a list of arrays and then an overall numpy array
        3) Tiling out the mode table for the full number of periods of evolution+1 for some extra room
        3) For each time "t" and mode "mode," the complex exponential given by exp(-1j*qe[mode]*t) is multiplied into the corresponding
            mode at time t to solve for the Floquet State basis over the full time period of evolution given by tau
            
    AS OF NOW THE TAULIST ONLY GOES FROM 0 TO TAU+T. THIS WORKS FOR EXCITESPEC BUT NOT EMISSPEC. DEAL WITH THIS WHEN YOU NEED TO, FENTON
    '''
    
    
    #Modes Table for a single period
    f_modes_list_one_period = np.stack([np.hstack([i.full() for i in modest]) for modest in f_modes_table_t])

    #Full time evaluation is tau number of periods +1 period for time averaging, so I need that many periods of modes
    f_modes_list_full_time = np.tile(f_modes_list_one_period, (num_periods+1,1,1)) 
    
    f_modes_transpose = np.transpose(f_modes_list_full_time,(0,2,1))
    
    mode_exponentials = np.stack([np.stack([np.exp(-1j*qe[mode]*(t)) for mode in range(Hdim)]) for t in time_evolution_t_points])
    
    
    
    f_states_transposed = np.stack([
                            np.stack([
                                        f_modes_transpose[t,mode]* mode_exponentials[t,mode]
                                            for mode in range(Hdim)] )
                                                for t in range(len(time_evolution_t_points))],axis=0)
    
    
    fstates =  np.transpose(f_states_transposed,(0,2,1))
   
    
    
    #Solving for the conjugate transpose of the states here, for ease of use in other areas
    fstatesct = np.transpose(fstates.conj(),axes=(0,2,1))
    
    return f0,qe,f_modes_list_one_period,fstates,fstatesct




def floquet_fourier_amps(Nt,tlist,taulist,bath_operator_list, floquet_modes_array, opts = None):
    amplitude_list = []
    for c_op in bath_operator_list:
        if opts == None:
            opts = Options()                                  #Setting up the options used in the mode and state solvers
            opts.atol = 1e-8                                  #Absolute tolerance
            opts.rtol = 1e-8                                  #Relative tolerance
            opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
        
        
        Hdim = np.shape(c_op)[0]
        
        '''
        I'm going to keep this as its own method for now, as it might be nice
            to be able to pull graphs of the FFT of the lowering operators when
            desired.
            
            
        This function transforms the lowering operator into the Floquet mode basis
        '''
        
        floquet_modes_array_conjT = np.transpose(floquet_modes_array.conj(),(0,2,1) )
        
        c_op_Floquet_basis = (floquet_modes_array_conjT @ c_op @ floquet_modes_array)
                   
    
        
        '''
        Performing the 1-D FFT to find the Fourier amplitudes of this specific
            lowering operator in the Floquet basis
            
        Divided by the length of tlist for normalization
        '''
        
        c_op_Fourier_amplitudes = np.sum(scp.fft.fft(c_op_Floquet_basis,axis=0),axis=0)/(len(tlist))
        amplitude_list.append(c_op_Fourier_amplitudes)
        
     
    return amplitude_list





def floquet_rate_matrix(qe,fourier_amplitude_matrices_list,c_op_rates,beatfreq,time_sense=0):
    ##########################Solving the R(t) tensor#########################
    Rate_matrix_list = []
    for cdx, c_op_amplitude_matrix in enumerate(fourier_amplitude_matrices_list):
        Hdim = len(qe)
        
        '''
        First, divide all quasienergies by omega to get everything in terms of omega. 
            This way, once I've figured out the result of the exponential term addition, 
            I can just multiply it all by w
            
        Defining the exponential sums in terms of w now to save space below
        '''
        def delta(a,ap,b,bp):
            return ((qe[a]-qe[ap]) - (qe[b]-qe[bp]))/(beatfreq/2)
        
        '''
        Next step is to create the R matrix from the "good" values of the indices
        Need to Condense these loop later with the functions Ned sent me in an email
        
        The loops over n,m,p,q are to set the n,m,p,q elements of the R matrix for a 
            given time dependence. The other loops are to do sums within those elements.
            The Full form of the FLime as written here can be found on the "Matrix Form 
            FLiME" OneNote page on my report tab. Too much to explain it here!
        
        '''
        
        '''
        Noteworthy for sum ordering purposes: itertools likes to iterate over the LAST thing first.
            E.g. if you iterated over all combinations of a list of 4 numbers from 0-3, it would go
            [0,0,0,0],[0,0,0,1],[0,0,0,2]...[0,0,1,0],[0,0,1,1],....[3,3,3,3]
        '''
        iterations_test_list = [Hdx for Hdx in itertools.product(range(0,Hdim),repeat = 4)]  #Iterating over all possible iterations of A,AP,B,BP
        
        '''
        From here on out, I'll be using the same indices for everything:
            
        alpha = var_name[0]
        alpha' = var_name[1]
        beta= var_name[2]
        beta' = var_name[3]
        '''
        
        
        #The asterisk is to unpack the tuple from iterations_test_list into arguments for the function
        time_dependence_list = [delta(*test_itx) for test_itx in iterations_test_list]
        
        valid_TDXs = (~ma.masked_where(np.absolute(time_dependence_list)>time_sense,time_dependence_list).mask).nonzero()[0] #Creates a list of the indices of the time dependence coefficient array whose entries are within the time dependence constraint time_sense
        
    
    
        valid_time_dependence_summation_index_values = [tuple(iterations_test_list[valid_index]) for valid_index in valid_TDXs] #creates a list tuples that are the valid (a,b,ap,bp,l,lp) indices to construct R(t) with the given secular constraint
    
        
        c_op_R_tensor = {} #Time to build R(t). Creating an empty dictionary.
        for vdx, vals in enumerate(valid_time_dependence_summation_index_values): #For every entry in the list of tuples, create R(t)
            a  = vals[0]
            ap = vals[1]
            b  = vals[2]
            bp = vals[3]
            
            R_slice = np.zeros((Hdim**2,Hdim**2),dtype = complex)  
            for idx in np.ndindex(Hdim,Hdim,Hdim,Hdim): #iterating over the indices of R_slice to set each value. Should figure out something faster later
                m = idx[0]
                n = idx[1]
                p = idx[2]
                q = idx[3]
                
                R_slice[m+Hdim*n,p+Hdim*q] =                                       \
                    c_op_rates[cdx]*c_op_amplitude_matrix[a,b]*np.conj(c_op_amplitude_matrix[ap,bp])*                          \
                                ( kron(m, a) * kron(n,ap) * kron(p,b) * kron(q,bp) \
                           -(1/2)*kron(a,ap) * kron(m,bp) * kron(p,b) * kron(q, n) \
                           -(1/2)*kron(a,ap) * kron(n, b) * kron(p,m) * kron(q,bp))
            
            try:
                c_op_R_tensor[time_dependence_list[valid_TDXs[vdx]]] += R_slice  #If this time-dependence entry already exists, add this "slice" to it
            except KeyError:
                c_op_R_tensor[time_dependence_list[valid_TDXs[vdx]]]  = R_slice   #If this time-dependence entry doesn't already exist, make it
    
        Rate_matrix_list.append(c_op_R_tensor)
        
        
    total_R_tensor = {}
    for Rdic_idx in Rate_matrix_list:
        for key in Rdic_idx:
            try:
                total_R_tensor[key] += Rdic_idx[key]  #If this time-dependence entry already exists, add this "slice" to it
            except KeyError:
                total_R_tensor[key]  = Rdic_idx[key]   #If this time-dependence entry doesn't already exist, make it
        
    
    return total_R_tensor

'''
Creating a function that takes the Rdictionary, multiples each key by its associated 
    time dependence, then adds everything up to get a single
    time-dependent R tensor for the ODE solver to solve

This function is defined seperately from the above so that the R tensor dictionary
    isn't reconstructed every time the ODE solver below calls the function!
'''
def rhodot(t,p,Rdic,beatfreq,):
    Hdim = np.shape(p)[0]
    R = np.zeros((Hdim,Hdim),dtype=complex)
    for idx, targ in enumerate(Rdic.keys()):
        R += Rdic[targ]*np.exp(-1j*targ* (beatfreq/2)*t)
    R1 = (R)@p
    
    return R1




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

def steadystate_time(c_op_mag,T):  
    time_scale = (2*np.pi)/c_op_mag
    
    periods_required = int(np.ceil(time_scale/T))
    
    return periods_required

def steadystate_rho(rho0,ss_time,Nt,Rdic,omega,T,opts = None):
    
    if opts == None:
        opts = Options()                                  #Setting up the options used in the mode and state solvers
        opts.atol = 1e-6                                  #Absolute tolerance
        opts.rtol = 1e-8                                  #Relative tolerance
        opts.nsteps= 10e+6  
        
    #Taulist Definition
    Ntau = (Nt)*(ss_time)                                    
    taume = ss_time*T                          
    dtau = taume/Ntau                                 
    taulist = np.linspace(0, taume-dtau, Ntau)       
    steadystate_time = taulist[-1]+taulist[1]
    
    
    rho_steadystate= scp.integrate.solve_ivp(rhodot                   ,
                                            t_span = (0,steadystate_time),
                                            y0=rho0              ,
                                            args=(Rdic,omega)        ,
                                            method='DOP853'              ,
                                            t_eval=np.append(taulist,steadystate_time) ,
                                            rtol=opts.rtol               ,
                                            atol=opts.atol).y[:,-1]     
    return rho_steadystate
'''
Misc functions that I don't wanna catagorize rn
'''


def freqarray(T,Nt,tau):       
    ############### Finding the fgrequency array over which to plot time evolution results #########################
    Ntau = int((Nt)*(tau))                                    
    taume = ((Ntau)/Nt)*T                             
    dtau = taume/(Ntau)                                 
                                                                                                
    omega_array = np.fft.fftshift(np.fft.fftfreq(Ntau,dtau))
    
    # omega_array1 = np.fft.fftfreq(Ntau,dtau)
    # omega_array = np.fft.fftshift(omega_array1)
    
    return omega_array


def mat(i,j):
    """
    Creates a matrix operator |i><j|

    Parameters
    ----------
    i : int
        ket state
    i : int
        bra stat

    Returns
    -------
    operator |i><j|

    """    
    return(basis(2,i)*basis(2,j).dag())


'''
Defining a kronecker delta function for later use
This will take the input matrix elements a,b, and return 1 if a=b, 0 else
'''
def kron(a,b):
    """
    Defining a matrix Kronecker Delta function
    This is really only used in the MEtensor function but might be useful later.

    Parameters
    ----------
    a : int
        matrix element a
    b : int
        matrix element b

    Returns
    -------
    0 if a!=b 
    1 if a=b

    """    
    if a == b:
        value = 1
    else:
        value = 0
    return value


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



def reorder(M,state0mat = None):
    Mperms = [M[:,i] for i in list(itertools.permutations(range(0,shape(M)[0]),shape(M)[0]))] #Getting every permutation of the input matrix
    
    #The below line does the column-wise dot product of each permutation of the basis vectors with the identity matrix (the matrix form of the "computational basis" vectors)
    #First each column is multiplied elementwise, then the result of each product is summed to get the dot product for the row
    #Then, the absolute square is taken and this is added for the result of every column dot product in the "matrix column dot product thingy"
    #The permutation with the maximum result is the correct transformation matrix
    if state0mat == None:
        dots = [sum([abs(np.dot(Mperm[:,i],np.identity(np.shape(M)[1])[i]))**2 for i in range(np.shape(M)[1])]) for Mperm in Mperms]
    else:
        dots = [sum([abs(np.dot(Mperm[:,i],state0mat[i])) for i in range(np.shape(M)[1])]) for Mperm in Mperms]
    
    CorrPerm = np.where(dots == np.amax(dots))[0][0]
    v = M[:,list(itertools.permutations(range(0,shape(M)[0]),shape(M)[0]))[CorrPerm]]
    # v = np.sqrt(1/2)*np.array([[1,1,0,0],[1,-1,0,0],[0,0,1,1],[0,0,1,-1]])
        


   
    return v, list(itertools.permutations(range(0,shape(M)[0]),shape(M)[0]))[CorrPerm]

def lowop_detpol_modifier(lowop,dipoles,detpols):
    detpoldic = {
    'X' : np.array([1,0,0]),
    'Y' : np.array([0,1,0]),
    'D' : (1/np.sqrt(2))*np.array([1,1,0]),
    'SP' :(1/np.sqrt(2))*np.array([1,1j,0]),
    'SM' :(1/np.sqrt(2))*np.array([1,-1j,0]),
    'NP' : np.array([None,None,None])}


    pollist = list([detpoldic[i] for i in detpols])


    lowlist = []
    for detpol in pollist:
        if detpol.any() != None:
            Lowp = lowop.full()
            for key in dipoles:
                if Lowp[key] != 0:
                    if np.dot(dipoles[key],detpol) != 0:
                       Lowp[key] = np.dot(dipoles[key],detpol)
                    else:
                        # print('this should not have happened')
                        Lowp[key] *= 0
            Lowp = Qobj(Lowp)  
            lowlist.append(Lowp)
        else:
            Lowp = lowop.full()
            for key in dipoles:
                if lowop[key] != 0:
                    Lowp[key] = np.dot(dipoles[key],detpoldic['X'])+np.dot(dipoles[key],detpoldic['Y'])
            Lowp = Qobj(Lowp) 
            lowlist.append(Lowp)
    return lowlist


