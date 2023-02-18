# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:35:40 2023

@author: Fenton
"""

'''
all below are emissspecmodule, unless otherwise noted.
'''

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
