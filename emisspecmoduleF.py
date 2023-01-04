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
def PrepWork(H,T,args,tlist,taulist=None,rho0 = None, opts = None):
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
        opts.atol = 1e-6                                #Absolute tolerance
        opts.rtol = 1e-8                                  #Relative tolerance
        opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
    
    
    #Making the assumption here that Hdim equals the number of rows of the time-independent Hamiltonian. Might not work always. Works now.
    Hdim = np.shape(H[0])[0]
    
    
    
    t0 = taulist[-1]+taulist[1]  #The leng of time Tau forward for which the system is being evolved
    tlistnew = np.concatenate((taulist,t0+tlist)) #The times over which I'll need to solve the lowering and raising operators. Runs from Tau to 2*Tau+T = 2*(N*T)+T=T(2N+1)
    
    
    print('starting f0')
    f0, qe = floquet_modes2( H, T,  args, sort = True, options = opts)
    
    # _,CorrPerms = reorder(np.hstack([i.full() for i in f0]))
    
    # f0 = [f0[i] for i in CorrPerms]
    # qe = [qe[i] for i in CorrPerms]
    
    f_modes_table_t = floquet_modes_table2(     \
                              f0,  qe, tlist, H,  T,    \
                                  args, options = opts) #For use in transformations of the lowering operator and initial state
    
    
    '''
    To form the Floquet State Table (in the form of a Numpy array), I:
        1) Tile the mode table N times, where N is the number of whole (maybe partial in the future?) periods forward
                in time defined by the time tau
        2) Create an araay "qtime" of the product of every qe[i]*(Tau+t) to facilitate Numpy array multiplication, which I think is faster than list comprehensions? Not sure on that one.
        3) Finally, every fmode array slice (for each time Tau+t) is multiplied directly (as opposed to Matmul)) by exp(-1j*qtime[i,k]) = exp(-1*qe[i]*)
    '''
    

    
    fmodestile = np.tile(np.stack([np.hstack([f_modes_table_t[idx][i].full() for i in range(Hdim)]) for idx in range(len(tlist))],axis = 2),int(np.ceil(len(taulist)/len(tlist))+1))
   
    
   
    fstates = np.stack([np.stack([np.transpose(np.stack([fmodestile[:,i,k]*np.exp(-1j*qe[i]*(t0+tlistnew[k])) for i in range(Hdim)]))]) for k in range(np.shape(fmodestile)[2])])[:,0,:,:]

    
    # qtime = np.outer(qe,(t0+tlistnew))
    # fstates = np.stack([
    #             np.stack([
    #                 np.hstack([
    #                     fmodestile[:,i,k]*np.exp(-1j*qtime[i,k]) for i in range(Hdim)])]) for k in range(shape(fmodestile)[2])])[:,0,:,:]
    
    fstatesct = np.transpose(fstates.conj(),axes=(0,2,1))
    
    return f0,qe,f_modes_table_t,fstates,fstatesct




def LTrunc(PDM,Nt,tlist,taulist,low_op,f_modes_table_t, opts = None):
    
    if opts == None:
        opts = Options()                                  #Setting up the options used in the mode and state solvers
        opts.atol = 1e-8                                #Absolute tolerance
        opts.rtol = 1e-8                                  #Relative tolerance
        opts.nsteps= 10e+8                                #Maximum number of Steps                              #Maximum number of Steps
    
    
    Hdim = np.shape(low_op)[0]
    
    '''
    I'm going to keep this as its own method for now, as it might be nice
        to be bale to pull graphs of the FFT of the lowering operators when
        desired.'
    '''
 
    lowfloq = []                           #Defining an empty matrix to store the lowering operator in the Floquet mode basis for every time t in tlist
    for idx in range(Nt):
        lowfloq.append(low_op.transform( \
                                        f_modes_table_t[idx*PDM],False)) #For every time t in tlist, transform the lowering operator to the Floquet mode basis using the Floquet mode table
            

    '''
    Recasting lowfloq as an array because QuTiP stores arrays in a very weird way
    '''
    lowfloqarray = np.zeros((Hdim,Hdim,Nt),dtype = complex) #Creating an empty array to hold the lowering operator as an array instead of a list of QObjs
    for i in range(Hdim):
        for j in range(Hdim):
            for k in range(Nt):
                lowfloqarray[i,j,k] =               \
                                  lowfloq[k][i][0][j] #This loop takes each index of each of the lowering operator QObjects in the list and stores them in an array

    '''
    Performing the 1-D FFT
    '''
    amps=np.zeros_like(lowfloqarray, dtype = complex) #Creating the empty array to hold the Fourier Amplitudes of each index at every harmonic
    amps = scp.fft.fft(lowfloqarray,axis=2) #This loop performs the FFT of each index of the Floquet mode basis lowering operator to find their harmonic amplitudes.
    amps = (np.real(amps))/len(tlist)
    
    
    # amps1 = (1/len(tlist))*np.stack([lowfloq[i].dag().full() @ lowfloq[i].full()],axis = 2)
    
    
    '''
    Finding the FFT frequency peaks using np.where
    '''
    
   
    
    indices = {}                                      #Initializing an empty dictionary to story the index of the peak harmonic for each lowering operator index
    for i in range(Hdim):
        for j in range(Hdim):
            indices[i+Hdim*j] = np.where(           \
                              np.round(abs(amps[i,j]),abs(math.floor(math.log10(opts.atol)))-1) >= opts.atol) #This loop stores the index of each peak harmonic in the dictionary, with the key of the dictionary being the operator matrix index.
            indices[i+Hdim*j]=[x-len(tlist) if x>len(tlist)/2 else x for x in indices.get(i+Hdim*j)[0]]    
    
    
    '''
    The next step is to figure out which of these peaks has the largest absolute 
       value, which informs where I truncate my sum over l. The loop below also finds
       the "truer" value of l for each peak and uses that for the comparison. E.g. if
       the FFT has 500 points and a peak has index 499, its actual index is [-2], giving
       |l| = 2, NOT |l| = 499
    '''
   
    lmax = []                                         #Creating an empty array to hold the absolute value of the index of each peak
    for i in range(Hdim):
        for j in range(Hdim):
            try:
                index = np.where(np.abs(indices[i+Hdim*j]) == np.amax(np.abs(indices[i+Hdim*j])))
                lmax.append(abs(indices[i+Hdim*j][index[0][0]]))
            except ValueError:
                lmax.append(0)
    lmax = int((len(tlist)/2) - 1) #np.amax(lmax)
    
    return amps,lmax





def Rt(qe,amps,lmax,Gamma,beatfreq,time_sense=0):
    ##########################Solving the R(t) tensor#########################
    Hdim = len(qe)
    
    '''
    First, divide all quasienergies by omega to get everything in terms of omega. 
        This way, once I've figured out the result of the exponential term addition, 
        I can just multiply it all by w
    '''
    qw = []
    for i in range(Hdim):                                     
        qw.append(qe[i]/(beatfreq/2))                                       #Characteristic Time period of the Hamiltonian - defined with abs(beat/2) as beat/2 is negative. 

        
    
    '''    
    Defining the exponential sums in terms of w now to save space below
    '''
    def delta(x,y,l):
        return (qw[x]-qw[y]+l)
    
    '''
    Next step is to create the R matrix from the "good" values of the indices
    Need to Condense these loop later with the functions Ned sent me in an email
    
    The loops over n,m,p,q are to set the n,m,p,q elements of the R matrix for a 
        given time dependence. The other loops are to do sums within those elements.
        The Full form of the FLime as written here can be found on the "Matrix Form 
        FLiME" OneNote page on my report tab. Too much to explain it here!
    
    '''
      
 
    
    # at = [p for p in itertools.product(range(0,Hdim),repeat = 4)]
    # bt = [p for p in itertools.product(range(-lmax,lmax+1),repeat = 2)]
    testidx = [list(sum(p, ())) for p in itertools.product(                                                                      #Taking outer product of below to find all possible indices for a,b,ap,bp,l,lp
                                                            [Hdx for Hdx in itertools.product(range(0,Hdim),repeat = 4)], #All possible iterations of a,b,ap,bp
                                                            [ldx for ldx in itertools.product(range(-lmax,lmax+1),repeat = 2)])]  #All possible iterations of l,lp
    
    
    TimeDepCoeffArray = [np.round(delta(index[0],index[1],index[4]) - delta(index[2],index[3],index[5]),14) for index in testidx] #time dependence of the Hamiltonian
    
    ValidTimeDepIdx = (~ma.masked_greater(np.absolute(TimeDepCoeffArray),time_sense).mask).nonzero()[0] #Creates a list of the indices of the time dependence coefficient array whose entries are within the time dependence constraint time_sense

    validdx = [tuple(testidx[i]) for i in ValidTimeDepIdx] #creates a list tuples that are the valid (a,b,ap,bp,l,lp) indices to construct R(t) with the given secular constraint
    
    
    
    Rdic = {} #Time to build R(t). Creating an empty dictionary.
    for rdx,row in enumerate(validdx): #For every entry in the list of tuples, create R(t)
        # a = row[0]
        # b = row[1]
        # ap = row[2]
        # bp = row[3]
        # l = row[4]
        # lp = row[5]
        #Below creates Rt[m+Hdim*n,p+Hdim*q] by doing a list comprehension to build the resulting R(t) as a column-stacked matrix, then reshaping it to the proper dimensions
        Rt = np.reshape([
                Gamma*amps[row[0],row[1],row[4]]*np.conj(amps[row[2],row[3],row[5]])   * 
                    (     kron(m     ,row[0])*kron(n,row[2])*kron(p,row[1])*kron(q,row[3]) -        #First Term of the FLiME                        
                    (1/2)*kron(row[0],row[2])*kron(m,row[3])*kron(p,row[1])*kron(q,n     ) - 
                    (1/2)*kron(row[0],row[2])*kron(n,row[1])*kron(p,     m)*kron(q,row[3])) 
                  for (m,n,p,q) in [row for row in itertools.product(range(0,Hdim),repeat = 4)]],                                                          #the order of n,m,p,q here is *very* important for some reason that I don't feel like investigating. Something something transpose
              (Hdim**2,Hdim**2))
                            
        try:
            Rdic[TimeDepCoeffArray[ValidTimeDepIdx[rdx]]] += Rt  #If this time-dependence entry already exists, add this "slice" to it
        except KeyError:
            Rdic[TimeDepCoeffArray[ValidTimeDepIdx[rdx]]] =  Rt   #If this time-dependence entry doesn't already exist, make it
    
    return Rdic
    #a=p[0],b=1,ap=2,bp=3,l=4,lp=5, n=6,m=7,p=8,q=9
    
    
    
    
    # Rdic = {}
    # #Setting the R matrix elements n+Hdim*m, p+Hdim*q
    # for n in range(Hdim):
    #     for m in range(Hdim):
    #         for p in range(Hdim):
    #             for q in range(Hdim):
    #                 #Performing sums within each R matrix element
    #                 for a in range(Hdim):
    #                     for ap in range(Hdim):
    #                         for b in range(Hdim):
    #                             for bp in range(Hdim):
    #                                 for l in range(-lmax,lmax+1):
    #                                     for lp in range(-lmax,lmax+1):
    #                                         if np.round(abs(delta(a,b,l)-delta(ap,bp,lp)),10) <= time_sense:
                                                
    #                                             #First Term of the FLiME
    #                                             Rt = np.zeros((Hdim**2,Hdim**2),dtype=complex)
    #                                             Rt[m+Hdim*n,p+Hdim*q] += Gamma*\
    #                                             amps[a,b,l]*np.conj(amps[ap,bp,lp]) * \
    #                                                     kron(m,a)*kron(n,ap)*kron(p,b)*kron(q,bp)  
                                                
                                                
                                                
    #                                             #Second Term of the FLiME
    #                                             Rt[m+Hdim*n,p+Hdim*q] += Gamma*\
    #                                             amps[a,b,l]*np.conj(amps[ap,bp,lp]) * \
    #                                                     (-1/2)*kron(a,ap)*kron(m,bp)*kron(p,b)*kron(q,n)
                                                
                                                
    #                                             #Third Term of the FLiME
    #                                             Rt[m+Hdim*n,p+Hdim*q] += Gamma*\
    #                                             amps[a,b,l]*np.conj(amps[ap,bp,lp]) * \
    #                                                     (-1/2)*kron(a,ap)*kron(n,b)*kron(p,m)*kron(q,bp)
                                                
                                                
    #                                             '''
    #                                             Finally, For each index, I append the total R(t) of that
    #                                                 index to a dictionary. The key of the dictionary 
    #                                                 entries are actually the (pre omega multiplication)
    #                                                 time dependencies, which is convenient I think. This 
    #                                                 way, each index that satisfies the condition above 
    #                                                 is just appended to its appropriate time dependent
    #                                                 dictionary entry, with time dependence added below!
                                                    
    #                                             '''
    
    #                                             try:
    #                                                 Rdic[delta(a,b,l)-delta(ap,bp,lp)] += Rt
    #                                             except KeyError:
    #                                                 Rdic[delta(a,b,l)-delta(ap,bp,lp)] = Rt   
                                                    
    # return Rdic

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
        R += Rdic[targ]*np.exp(1j*targ* beatfreq/2*t)
    R1 = (R)@p
    
    return R1








'''
Misc functions that I don't wanna catagorize rn
'''


def freqarray(T,Nt,tau,PDM = 1):       
    ############### Finding the fgrequency array over which to plot time evolution results #########################
    Ntau = int((Nt)*(tau)*PDM)                                    
    taume = ((Ntau/PDM)/Nt)*T                             
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
    # v = np.sqrt(1/2)*np.array([[-1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,-1]])
        


   
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


