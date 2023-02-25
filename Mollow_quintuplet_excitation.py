# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:20:35 2023

@author: Fenton





The aim of this scipt is to show what happens to the excitation spectra
of a 3 level V system as Rabi Frequency of the monochromatic laser is increased

Expectations: Honestly not super sure. I would think the excitation
value would just get bigger as RF is increased? Don't know how useful this
one will be.

NOTE: NO MAGNETIC FIELD. HAVEN'T GENERALIZED THE ZEEMAN HAMILTONIAN
ENOUGH TO TAKE THIS SYSTEM IN YET.'
"""

from qutip import *
import time
import numpy as np

import Floquet_sims_lowest_module as flm
import Floquet_sims_mid_module as fmm
import Floquet_Module_Classes as FC

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

'''
Defining the Dot
'''
norm = np.sqrt(2)
states = [0,2*np.pi*280,2*np.pi*(280+1e-7)] #Resonance arbitrarily decided
dipole = {(0,1):(1,0,0),(0,2):(0,1,0)}
# gfactors = [[-0.24,-0]] #Parallel parts (first one in each tuple) From Ned's 2015 paper
dot = FC.QD(3,states,dipole)


'''
Defining the Lowering Operator
'''
manifolds_collapse_rate = np.sqrt(2*np.pi*.0000025)

collapse_operator_x= \
                            np.array(
                                  [[0, 1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])
    
    
collapse_operator_y = \
                            np.array(
                                  [[0, 0, 1],
                                  [0, 0, 0],
                                  [0, 0, 0]])
    

c_op_x = FC.Collapse_Op(collapse_operator_x,manifolds_collapse_rate)
c_op_y = FC.Collapse_Op(collapse_operator_y,manifolds_collapse_rate)


collapse_operator_list = [c_op_x,
                          c_op_y]



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






tau = 500 #Length of time to go forward, in units of T, the system Frequency
Nt = 2**4 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator

point_spacing = 0.0001
detuning0 = -.005


power_range = 101
P_array = np.zeros(power_range)
for i in range(power_range):
    P_array[i]=(((i*1)))

RF_range = 10
RF_array = np.zeros(RF_range)
for i in range(RF_range):
    RF_array[i]=((((i+1)*1)))

start_time = time.time()

Z0Lavg = np.zeros((RF_range,power_range),dtype=float) #Empty list to hold the spectra
Z0Cavg = np.zeros((RF_range,power_range),dtype=float)
ZXavg = np.zeros((RF_range,power_range),dtype=float)
ZYavg = np.zeros((RF_range,power_range),dtype=float)
ZPavg = np.zeros((RF_range,power_range),dtype=float)
ZMavg = np.zeros((RF_range,power_range),dtype=float)


for idR, RF in enumerate(RF_array): 
    print('working on spectra',idR+1,'of',len(RF_array))
    for idz, val in enumerate(P_array):   
       
       
          
        '''
        Defining the lasers
        ''' 

        P2 = 0.001*RF
        P1 = 0.0
        L2power = 2*np.pi*P2
        
        L2pol = 'D'
        
        detuning = detuning0+point_spacing*val
        L2freq = 2*np.pi*(280+detuning)

        
        L2 = FC.Laser(L2power,LP[L2pol],L2freq) 
        
    
        
        
        '''
        Defining the initial state
        '''
        rho00 = ((1/2)*(basis(3,0)*basis(3,0).dag()+basis(3,0)*basis(3,0).dag()))
        
       
        
        Exp = FC.QSys(dot,LasList = [L2],c_op_list = collapse_operator_list)
        
    
        if idz == 0:
            omega_array = flm.freqarray(Exp.T,Nt,tau)  
           
        
    
       

      
   
    
    
    
        spec1 = Exp.ExciteSpec(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'])
        
        
    
    
        # #For ExciteSpec
        Z0Lavg[idR,idz] = (spec1['X']+spec1['Y'])
        Z0Cavg[idR,idz] = (spec1['SP']+spec1['SM'])
        ZXavg[idR,idz] = (spec1['X'])
        ZYavg[idR,idz] = (spec1['Y'])
        ZPavg[idR,idz] = (spec1['SP'])
        ZMavg[idR,idz] = (spec1['SM'])

    





    
# # For plotting individual spectra
# idx = 0                                            #Plotting the results!


# # # For plotting Excitation Arrays
# fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

# #First plot to see how the linear polarizations works out
# ax[0,0].semilogy( Z0Lavg[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,0].semilogy(ZXavg[idx], color = 'slateblue')
# ax[0,0].semilogy(ZYavg[idx], color = 'navy',linestyle = 'dashed')
# ax[0,0].legend(['NoPol','x','y'])
# ax[0,0].set_xlabel('$\omega$ [THz]')
# ax[0,0].set_ylabel("Amplitude (arb)") 

# #Second plot to look at circular polarizations
# ax[0,1].semilogy(Z0Cavg[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,1].semilogy(ZPavg[idx], color = 'green')
# ax[0,1].semilogy(ZMavg[idx], color = 'orangered',linestyle = 'dashed')
# ax[0,1].legend(['NoPol','SP','SM'])
# ax[0,1].set_xlabel('$\omega$ [THz]')
# ax[0,1].set_ylabel("Amplitude (arb)") 





    






clims = [1e-4,1e+0]

# Plot on a colorplot
fig, ax = plt.subplots(2,2)
limits = [detuning0,\
          detuning0+point_spacing*P_array[-1],\
          RF_array[0],\
          RF_array[-1]]
    
pos = ax[0,0].imshow(ZXavg,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Delta_1$ [THz]') 
ax[0,0].set_title(F'detpol = X' )

pos = ax[0,1].imshow(ZYavg,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,1].set_title(F'detpol = Y' )

pos = ax[1,0].imshow(ZPavg,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\omega$ (THz)')
ax[1,0].set_ylabel('$\Delta_1$ [THz]') 
ax[1,0].set_title(F'detpol = SP' )

pos = ax[1,1].imshow(ZMavg,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,1].set_xlabel('$\omega$ (THz)')

ax[1,1].set_title( 'detpol = SM' )

fig.suptitle(F"3LS Excitation Spectrum with $\Omega_1$ polarization = {P2} THz {L2pol} polarization and swept RF (detuning is from lower resonance)")
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays



# Plot on a colorplot
fig, ax = plt.subplots(1,1)
limits = [detuning0,\
          detuning0+point_spacing*P_array[-1],\
          RF_array[0],\
          RF_array[-1]]
pos = ax.imshow(Z0Lavg,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
# ax.axvline(x=(0*Om1/(2*np.pi)), color='y', linestyle = 'solid',linewidth =3)
# ax.axvline(x=(1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# ax.axvline(x=(-1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
ax.set_xlabel('$\omega$ (THz)')
ax.set_ylabel('$\Delta_1$ [THz]') 
fig.suptitle(F"3LS Excitation Spectrum with $\Omega_1$ polarization = {P2} THz {L2pol} polarization and swept RF (detuning is from lower resonance)")

fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays


