# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:03:59 2023

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
import time
'''
Defining the Dot
'''
norm = np.sqrt(2)
states = [0,0,280,280+1e-7] #Resonance arbitrarily decided
dipole = {(0,2):(1/norm,-1j/norm,0),(1,3):(1/norm,1j/norm,0)}
gfactors = [[0.5,0],[-0.24,-0]] #Parallel parts (first one in each tuple) From Ned's 2015 paper
dot = FC.QD(4,states,dipole,gfactors)


'''
Defining the Lowering Operator
'''
manifolds_collapse_rate = np.sqrt(2*np.pi*.0000025)

collapse_operator_leftcirc= \
                            np.array(
                                  [[0, 0, 1 , 0],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0]])
    
    
collapse_operator_rightcirc = \
                            np.array(
                                  [[0, 0, 0 , 0],
                                  [0, 0, 0 , 1],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0]])

c_op_lc = FC.LowOp(collapse_operator_leftcirc,manifolds_collapse_rate)
c_op_rc = FC.LowOp(collapse_operator_rightcirc,manifolds_collapse_rate)


electron_spin_flip_rate_percent = 50
electron_spin_flip_rate = manifolds_collapse_rate*(electron_spin_flip_rate_percent/100)

collapse_operator_S_plus = \
                            np.array(
                                  [[0, 1, 0 , 0],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0]])

collapse_operator_S_minus =\
                            np.array(
                                  [[0, 0, 0 , 0],
                                  [1, 0, 0 , 0],
                                  [0, 0, 0 , 0],
                                  [0, 0, 0 , 0]])
                            
c_op_sp = FC.LowOp(collapse_operator_S_plus,electron_spin_flip_rate)
c_op_sm = FC.LowOp(collapse_operator_S_minus,electron_spin_flip_rate)


collapse_operator_list = [c_op_lc,
                          c_op_rc,
                          c_op_sp,
                          c_op_sm]


'''
Defining Detection polarization
'''

'''
A bunch of empty arrays, for use testing stuff
'''
Z0L = [] #Empty list to hold the spectra
Z0C = []
ZX = []
ZY = []
ZP = []
ZM = []

Z0Lavg = [] #Empty list to hold the spectra
Z0Cavg = []
ZXavg = []
ZYavg = []
ZPavg = []
ZMavg = []

Z0g1 = []
ZXg1 = []
ZYg1 = []
ZPg1 = []
ZMg1 = []

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






tau = 6000#Length of time to go forward, in units of T, the system Frequency
Nt = 2**5 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator
PDM = 2**0 #If the spectrumm isn't as wide as it needs to be, increase the power of 2 here.
interpols = 2**0 #interpolation, for if the spectra are doing the *thing*


point_spacing = 0.0001
detuning0 = -.000


power_range = 1
P_array = np.zeros(power_range)
for i in range(power_range):
    P_array[i]=(((i*1)+0))
    

start_time = time.time()

Bpower = 6e-2  
for idz, val in enumerate(P_array):
    print('working on spectra',idz+1,'of',len(P_array))
   
      
    '''
    Defining the lasers
    ''' 
    P2 = 0.001
    P1 = 0.0
    L2power = 2*np.pi*P2
    L1power = 2*np.pi*P1
    
    
    L2pol = 'D'
    L1pol = 'SM'
    
    detuning = detuning0+point_spacing*val
    ACdetune = -2
    L2freq = 2*np.pi*(280+detuning)
    L1freq = 2*np.pi*(280+ACdetune)
    
    L1 = FC.Laser(L1power,LP[L1pol],L1freq)
    L2 = FC.Laser(L2power,LP[L2pol],L2freq) 
    
    
    '''
    Defining the Magnetic Field
    '''
    B = FC.Bfield([Bpower,0])
    
    
    '''
    Defining the initial state
    '''
    rho00 = ((1/2)*(basis(4,0)*basis(4,0).dag()+basis(4,0)*basis(4,0).dag()))
    
   
    
    Exp = FC.QSys(dot,L2,L1,Bfield = B,c_op_list = collapse_operator_list)
    

    if idz == 0:
        omega_array = esm.freqarray(Exp.T,Nt,tau)  
        Transitions = Exp.TransitionEnergies()
        transX1=(abs(Transitions[2]-Transitions[0]))
        transX2=(abs(Transitions[3]-Transitions[1]))
        transY1=(abs(Transitions[2]-Transitions[1]))
        transY2=(abs(Transitions[3]-Transitions[0]))
    

    g2,taulist = Exp.g2_tau(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'])



    Z0L.append(g2['X']+g2['Y'])
    Z0C.append(g2['SP']+g2['SM'])
    ZX.append(g2['X'])
    ZY.append(g2['Y'])
    ZP.append(g2['SP'])
    ZM.append(g2['SM'])



total_time = time.time()-start_time
# For plotting individual spectra
idx = 0                                            #Plotting the results!


# # For plotting Excitation Arrays
fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

'''
multiplying taulist by 1e-12 to get it in seconds
''' 
ax[0,0].plot((taulist*1e-12),ZX[idx], color = 'purple')
ax[0,0].set_title(F'detpol = X' )
ax[0,0].set_ylabel(F"$g^{(2)}(\\tau)$")

ax[0,1].plot((taulist*1e-12),ZX[idx], color = 'slateblue')
ax[0,1].set_title(F'detpol = Y' )

ax[1,0].plot((taulist*1e-12),ZP[idx], color = 'green')
ax[1,0].set_xlabel('$\\tau$ [seconds]')
ax[1,0].set_ylabel(F"$g^{(2)}(\\tau)$") 
ax[1,0].set_title(F'detpol = $\sigma_+$' )

ax[1,1].plot((taulist*1e-12),ZM[idx], color = 'orangered',linestyle = 'dashed')
ax[1,1].set_xlabel('$\\tau$ [seconds]')
ax[1,1].set_title(F'detpol = $\sigma_-$' )


fig.suptitle(F"$g^{(2)}$($\\tau$) with $\Omega_1$ = 1 GHz {L2pol}, $\Delta_1$ = {detuning0+point_spacing*idx} GHz,$\Omega_2$ = {P1} THz {L1pol},$\Delta_2$ = {ACdetune} THz B = {Bpower}, $e^-$ rate (% of spont emis rate) = {electron_spin_flip_rate_percent}%" )



# clims = [1e-6,1e-2]
# Om1 = np.dot(       list(dot.dipoles.values())[0] ,Exp.Las2.E)
# # Plot on a colorplot
# fig, ax = plt.subplots(2,2)
# limits = [omega_array[0],\
#           omega_array[-1],\
#           detuning0+P_array[0]*point_spacing,\
#           detuning0+P_array[-1]*point_spacing]
    
# pos = ax[0,0].imshow(ZX,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
# ax[0,0].set_ylabel('$\Delta_1$ [THz]') 
# ax[0,0].set_title(F'detpol = X' )

# pos = ax[0,1].imshow(ZY,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
# ax[0,1].set_ylabel('$\Delta_1$ [THz]') 
# ax[0,1].set_title(F'detpol = Y' )

# pos = ax[1,0].imshow(ZP,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
# ax[1,0].set_xlabel('$\omega$ (THz)')
# ax[1,0].set_ylabel('$\Delta_1$ [THz]') 
# ax[1,0].set_title(F'detpol = SP' )

# pos = ax[1,1].imshow(ZM,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
# ax[1,1].set_xlabel('$\omega$ (THz)')
# ax[1,1].set_ylabel('$\Delta_1$ [THz]') 
# ax[1,1].set_title( 'detpol = SM' )

# fig.suptitle(F"Voigt Config with $\Omega_1$ = 1 GHz, excitation laser = {Lpol}, B = {Bpower}" )
# fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays


# #Plotting just the unpolarized case
# Om1 = np.dot(       list(dot.dipoles.values())[0] ,Exp.Las2.E)
# # Plot on a colorplot
# fig, ax = plt.subplots(1,1)
# limits = [omega_array[0]-(Exp.beat/2)/(2*np.pi),\
#           omega_array[-1]-(Exp.beat/2)/(2*np.pi),\
#           detuning0+P_array[0]*point_spacing,\
#           detuning0+P_array[-1]*point_spacing]
# pos = ax.imshow(Z0L,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
# # ax.axvline(x=(0*Om1/(2*np.pi)), color='y', linestyle = 'solid',linewidth =3)
# # ax.axvline(x=(1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# # ax.axvline(x=(-1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# ax.set_xlabel('$\omega$ (THz)')
# ax.set_ylabel('$\Delta_1$ [THz]') 
# ax.set_title(F'Voigt Config with $\Omega_1$ = 1 GHz, excitation laser = {Lpol}, detpol = None, B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt}' )
# fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays
