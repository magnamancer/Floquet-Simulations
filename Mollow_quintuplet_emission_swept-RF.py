# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:35:03 2023

@author: Fenton






The aim of this scipt is to show what happens to the emission peaks
of a 3 level system as detuning of the monochromatic excitation laser is kept
at a single detuning and swept through a range of powers

Expectations: If the detuning (detuning0) is exactly between resonances 
    (detuning = 2.5e-3), a quintuplet is expected. Anywhere else, I expect a 
    septuplet. Additionally, the peaks should spread out as the rabi
    frequency increases, according to the generlized Rabi Frequency

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
states = [0,2*np.pi*280,2*np.pi*(280+5e-3)] #Resonance arbitrarily decided
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






tau = 50 #Length of time to go forward, in units of T, the system Frequency
Nt = 2**5 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator



point_spacing = 0.0001
detuning0 = 2.5e-3


RF_range = 50
RF_array = np.zeros(RF_range)
for i in range(RF_range):
    RF_array[i]=((((i+1)*1)))

start_time = time.time()


for idR, RF in enumerate(RF_array): 
    print('working on spectra',idR+1,'of',len(RF_array))

   
   
      
    '''
    Defining the lasers
    ''' 
    Power0 = 0.001
    P2 = Power0*RF
    L2power = 2*np.pi*P2
    
    L2pol = 'D'

    detuning = detuning0
    L2freq = 2*np.pi*(280+detuning)


    L2 = FC.Laser(L2power,LP[L2pol],L2freq) 
    

    
    
    '''
    Defining the initial state
    '''
    rho00 = ((1/2)*(basis(3,0)*basis(3,0).dag()+basis(3,0)*basis(3,0).dag()))
    
   
    
    Exp = FC.QSys(dot,LasList = [L2],c_op_list = collapse_operator_list)
    

    if idR == 0:
        omega_array = flm.freqarray(Exp.T,Nt,tau)  
       
    

    spec1 = Exp.EmisSpec(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'],retg1='True')

    Z0L.append(spec1['X']+spec1['Y'])
    Z0C.append(spec1['SP']+spec1['SM'])
    ZX.append(spec1['X'])
    ZY.append(spec1['Y'])
    ZP.append(spec1['SP'])
    ZM.append(spec1['SM'])

      
    
    




    



freqlims = [omega_array[0]-(Exp.beat/(4*np.pi)),omega_array[-1]-(Exp.beat/(4*np.pi))]#[-0.05,0.05]


frequency_range = (omega_array-(Exp.beat/(4*np.pi)))
idx0 = np.where(abs(frequency_range-freqlims[0]) == np.amin(abs((frequency_range-freqlims[0] ))))[0][0]
idxf = np.where(abs(frequency_range-freqlims[1]) == np.amin(abs((frequency_range-freqlims[1] ))))[0][0]

plot_freq_range = frequency_range[idx0:idxf]
Z0L_truncated = np.stack([Z0Li[idx0:idxf] for Z0Li in Z0L])
ZX_truncated = np.stack([ZXi[idx0:idxf] for ZXi in ZX])
ZY_truncated = np.stack([ZYi[idx0:idxf] for ZYi in ZY])
ZP_truncated = np.stack([ZPi[idx0:idxf] for ZPi in ZP])
ZM_truncated = np.stack([ZMi[idx0:idxf] for ZMi in ZM])




clims = [1e-6,1e-1]

# Plot on a colorplot
fig, ax = plt.subplots(2,2)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          Power0+RF_array[0]*1,\
          Power0+RF_array[-1]*1]
    
pos = ax[0,0].imshow(ZX_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Omega_1$ [THz]') 
ax[0,0].set_title(F'detection polarization = X' )

pos = ax[0,1].imshow(ZY_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,1].set_title(F'detection polarization = Y' )

pos = ax[1,0].imshow(ZP_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\omega$ (THz)')
ax[1,0].set_ylabel('$\Omega_1$ [THz]') 
ax[1,0].set_title(F'detection polarization = SP' )

pos = ax[1,1].imshow(ZM_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,1].set_xlabel('$\omega$ (THz)')

ax[1,1].set_title( 'detection polarization = SM' )

fig.suptitle(F"3LS Emission Spectrum with $\Omega_1$ polarization = {L2pol} detuned {detuning*1000} GHz from lower resonance" )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays



# Plot on a colorplot
fig, ax = plt.subplots(1,1)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          detuning0+RF_array[0]*1,\
          detuning0+RF_array[-1]*1]
pos = ax.imshow(Z0L_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
# ax.axvline(x=(0*Om1/(2*np.pi)), color='y', linestyle = 'solid',linewidth =3)
# ax.axvline(x=(1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# ax.axvline(x=(-1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
ax.set_xlabel('$\omega$ (THz)')
ax.set_ylabel('$\Omega_1$ [THz]') 
ax.set_title(F"3LS Emission Spectrum with $\Omega_1$ polarization = {L2pol} detuned {detuning*1000} GHz from lower resonance ")
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays