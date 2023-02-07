# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:19:23 2020

@author: Edward

FLOQUET EMISSION SPECTRUM MODULE
This module contains functions necessary for calculating the emission spectrum
of a quantum system. It uses the Floquet theory and codes from the QuTiP library

"""
import numpy.ma as ma
import numpy as np
from qutip import *
import itertools
import math
import scipy as scp
################### Here we redefine versions of some floquet functions #######

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
Below are some meatier functions used for time evolution purposes,
primarily

'''
def prepWork(H,T,args,tlist,taulist=None, opts = None):
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
    
    
    # Setting useful constants to be used in a few lines
    steadystate_time = taulist[-1]+taulist[1]  #The length of time Tau forward for which the system is being evolved
    tau = int(np.ceil((steadystate_time)/T))
    time_evolution_t_points = np.concatenate((taulist, steadystate_time+tlist)) #The times over which I'll need to solve the lowering and raising operators. Runs from Tau to 2*Tau+T = 2*(N*T)+T=T(2N+1)
    
    
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
    f_modes_list_full_time = np.tile(f_modes_list_one_period, (tau+1,1,1)) 
    
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


def ket2super(rho0, basis1, basis2, spec=None, ft=None, qe=0, t=0):
    """
    Takes an input ket and turns it into a supervector in the desired basis

    Parameters
    ----------
    rho0 : Qobj
        The input state in the form of a ket or density matrix
    basis1:
        The basis in which the initial state rho0 is supplied. 
        'comp' for computational basis or 'floq' for Floquet basis
    basis2: 
        The desired output basis of the function. 
        'comp' for computational basis or 'floq' for Floquet basis
    spec:
        If 'floq' is an input, describes whether the Floquet MODE basis or
        Floquet STATE basis is desired.
        'mode' for mode basis, 'state' for state basis
    ft :Qobj
        The floquet modes at the desired time t of transformation. I think I set
        this function up such that using a lookup table as an input should work.
    qe :array
        the quasieneries
    t :float
        the time at which the transformation is to be done

    Returns
    -------
    rho0: a supervector in the desired basis at the specified time t

    """
    
    #
    # check initial state
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()
    
    #
    # Uncapitalizing any inputs
    #
    basis1 = basis1.lower()
    basis2 = basis2.lower()
    if spec != None:
        spec = spec.lower()
    
    #
    # Solving for the density matrix in the desired basis
    #
    if basis1 != basis2:
        if basis2 == 'comp':
            direction = True
        elif basis2 == 'floq':
            direction = False
        else:
            print('not an acceptable basis')
            return None
        
        #assumes the mode is desired if nothing is specified in the 'spec' argument
        if spec == 'mode' or spec == None:
            qe = np.zeros_like(range(ft[0].shape[0]))
        else:
            qe = qe
            
        dims1 = ft[0].dims
        shape1 = ft[0].shape
        #The exponential goes to 1 if the modes are desired, per the above if/else
        fmt = [Qobj(ft[i]*np.exp(-1j*qe[i]*t),dims=dims1,shape=shape1) for i in list(range(ft[0].shape[0]))]

        rho0 = rho0.transform(fmt,direction)


    elif basis1 == basis2:
      rho0=rho0 
      
    #
    #moving from a density matrix to a super vector
    #
    rho0 = operator_to_vector(rho0)
    
    return rho0

def super2ket(rho0, basis1, basis2, spec=None, ft=None, qe=0, t=0):
    """
    Takes an input supervector and turns it into a density matrix in the desired basis

    Parameters
    ----------
    rho0 : Qobj
        The input state in the form of a ket or density matrix
    basis1:
        The basis in which the initial state rho0 is supplied. 
        'comp' for computational basis or 'floq' for Floquet basis
    basis2: 
        The desired output basis of the function. 
        'comp' for computational basis or 'floq' for Floquet basis
    spec:
        If 'floq' is an input, describes whether the Floquet MODE basis or
        Floquet STATE basis is desired.
        'mode' for mode basis, 'state' for state basis
    fmt :Qobj
        The floquet modes at the desired time t of transformation
    qe :array
        the quasieneries
    t :float
        the time at which the transformation is to be done

    Returns
    -------
    rho0: a density matrix in the desired basis at the specified time t

    """
    #
    # Uncapitalizing any inputs
    #
    basis1 = basis1.lower()
    basis2 = basis2.lower()
    if spec != None:
        spec = spec.lower()
    
    #
    #moving from a density matrix to a super vector
    #
    rho0 = vector_to_operator(rho0)
    
    #
    # Solving for the density matrix in the desired basis
    #
    if basis1 != basis2:
        if basis2 == 'comp':
            direction = True
        elif basis2 == 'floq':
            direction = False
        else:
            print('not an acceptable basis')
            return None
        
        #assumes the mode is desired if nothing is specified in the 'spec' argument
        if spec == 'mode' or spec == None:
            qe = np.zeros_like(range(ft[0].shape[0]))
        else:
            qe = qe
            
        dims1 = ft[0].dims
        shape1 = ft[0].shape
        #In here, the exponential goes to 1 if the mode is desired, per the if/else directly above
        fmt = [Qobj(ft[i]*np.exp(-1j*qe[i]*t),dims=dims1,shape=shape1) for i in list(range(ft[0].shape[0]))]
        rho0 = rho0.transform(fmt,direction)
    
        
    elif basis1 == basis2:
      rho0=rho0 
    
    return rho0


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


def transformF(rho, inpt,inverse = False,Sparse = True):
    """
    The purpose of this function is to take the input in one basis and transform
    it to the basis defined by the input basis inpt
    
    Parameters
    ----------
    
    inpt:
        the input basis states
    inverse:
        not useful YET. Leaving it in to make it so I don't have to rewrite things.
        Might use it later to specificy whether I'm going into or out of the basis 
        defined by inpt
    Sparse:
        Same as above, except this option is to use sparse matrices. Again,
        might write this in later if I feel nice


    Returns
    -------
    oper:
        operator in new basis

    """    
       
    #First thing is to figure out how large the input basis state list is
    l = len(inpt)
    #Next I'll create a dictionary with the basis vectors
    d= {}
    for i in range(l):
        d[i] = inpt[i].full()
        
    #Finally, I'll hstack the dictionary to create the basis change matrix V
    V = Qobj(np.hstack(d[i] for i in range(l)).T)
    
    
    #Next, take the input vector or operator and change the basis
    if rho.isket == True:
        oper = V.dag()*rho
    elif rho.isoper == True:
        oper = V.dag()*rho*V
    else:
        oper = []
        print("Something went wrong. Supply a proper ket or operator!")
    
    return oper


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


