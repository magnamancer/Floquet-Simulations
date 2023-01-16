# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:26:17 2023

@author: FentonClawson
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:21:03 2022

@author: FentonClawson
"""

import FloqClassestesting as FC
import numpy as np
import emisspecmoduleF as esm 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from qutip import *
'''
Defining the Dot
'''
norm = np.sqrt(2)
states = [0,0,280,280+1e-7] #Resonance arbitrarily decided
dipole = {(0,2):(1/norm,-1j/norm,0),(1,3):(1/norm,1j/norm,0)}
gfactors = [[0.5,0],[0.24,-0]] #Parallel parts (first one in each tuple) From Ned's 2015 paper
dot = FC.QD(4,states,dipole,gfactors)


'''
Defining the Lowering Operator
'''
lowmat = np.zeros((4,4),dtype=complex)
lowmat[0,2] = 1
lowmat[1,3] = 1
lowop = FC.LowOp(Qobj(lowmat),np.sqrt(2*np.pi*.0000025))
#The lowering operator magnitude is VERY important Fenton. See if you can find a physical value later.
'''
Defining Detection polarization
'''

'''
A bunch of empty arrays, for use testing stuff
'''

EG1 = [] #Empty list to hold the spectra
EG2 = []
EE1 = []
EE2 = []

transX1 = [] #Empty list to hold the spectra
transX2 = []
transY1 = []
transY2 = []




'''
Defining Polarizations
'''
LP = {
'X' : np.array([1,0,0]),
'Y' : np.array([0,1,0]),
'D' : (1/np.sqrt(2))*np.array([1,1,0]),
'SP' :(1/np.sqrt(2))*np.array([1,1j,0]),
'SM' :(1/np.sqrt(2))*np.array([1,-1j,0]),
'NP' : np.array([None,None,None])}






tau = 500 #Length of time to go forward, in units of T, the system Frequency
Nt = 2**8 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator
PDM = 2**0 #If the spectrumm isn't as wide as it needs to be, increase the power of 2 here.
interpols = 2**0 #interpolation, for if the spectra are doing the *thing*


power_range = 100
P_array = np.zeros(power_range)
for i in range(power_range):
    P_array[i]=(((i*1)+0))
    
test = [] 
testqe = []
  
for idz, val in enumerate(P_array):
    print('working on spectra',idz+1,'of',len(P_array))
   
    '''
    Defining the magnetic field power
    '''
    Bpower = val*10**(-3)
    
    '''
    Defining the laser
    ''' 
    Lpower = 2*np.pi*0.001
    
    Lpol = 'D'
    
    
    Lfreq = 2*np.pi*(280)
   
    
    L2 = FC.Laser(Lpower,LP[Lpol],Lfreq) 
    
    
    '''
    Defining the Magnetic Field
    '''
    B = FC.Bfield([Bpower,0])
    
    
    '''
    Defining the initial state
    '''
    rho00 = ((1/2)*(basis(4,0)*basis(4,0).dag()+basis(4,1)*basis(4,1).dag()))
    
   
    
    Exp = FC.QSys(dot,L2,Bfield = B,LowOp = lowop)
    

    if idz == 0:
        omega_array = esm.freqarray(Exp.T,Nt,tau,PDM = PDM)    
    

    # spec1,g1dic = Exp.EmisSpec(Nt,tau,rho0=rho00, PDM = PDM,time_sense=0.0, detpols = ['X','Y','SP','SM'],retg1='True')

    Transitions = Exp.TransitionEnergies()

    transX1.append(abs(Transitions[0]-Transitions[2]))
    transX2.append(abs(Transitions[1]-Transitions[3]))
    transY1.append(abs(Transitions[0]-Transitions[3]))
    transY2.append(abs(Transitions[1]-Transitions[2]))
    
    EG1.append(Transitions[0])
    EG2.append(Transitions[1])
    EE1.append(Transitions[2])
    EE2.append(Transitions[3])
   

    




   
        




# # For plotting Excitation Arrays
fig, ax = plt.subplots(1,1)                                                    #Plotting the results!

#Plotting the eigenergies
ax.plot(P_array*10**(-2),transX1, color = 'r' ) 
ax.plot(P_array*10**(-2),transX2, color = 'y' ) 
ax.plot(P_array*10**(-2),transY1, color = 'b' ) 
ax.plot(P_array*10**(-2),transY2, color = 'c' ) 
ax.legend(['X1','X2','Y1','Y2'])
ax.set_ylabel("Transition Energy") 




fig.suptitle(F"Voigt Config Transition Energies $\Omega_1$ = 1 GHz {Lpol}, B = {Bpower}" )


fig, ax = plt.subplots(1,2)   
#Plotting the eigenergies
ax[0].plot(P_array*10**(-2),EG1, color = 'r' ) 
ax[0].plot(P_array*10**(-2),EG2, color = 'y' ) 
ax[1].plot(P_array*10**(-2),EE1, color = 'b' ) 
ax[1].plot(P_array*10**(-2),EE2, color = 'c' ) 
ax[0].legend(['E1','E2'])
ax[1].legend(['T1','T2'])
# ax.set_ylabel("Energy state splittings") 

