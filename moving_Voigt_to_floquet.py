# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:03:03 2023

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



electron_spin_flip_rate = manifolds_collapse_rate*(50/100)

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
Defining Polarizations
'''
LP = {
'X' : np.array([1,0,0]),
'Y' : np.array([0,1,0]),
'D' : (1/np.sqrt(2))*np.array([1,1,0]),
'SP' :(1/np.sqrt(2))*np.array([1,1j,0]),
'SM' :(1/np.sqrt(2))*np.array([1,-1j,0]),
'NP' : np.array([None,None,None])}







tau = 3000 #Length of time to go forward, in units of T, the system Frequency
Nt = 2**2 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator
PDM = 2**0 #If the spectrumm isn't as wide as it needs to be, increase the power of 2 here.
interpols = 2**0 #interpolation, for if the spectra are doing the *thing*


point_spacing = 0.0001
detuning0 = -.005


detune_range = 151
detune_array = np.array([i for i in range(detune_range)])

power_range = 101
power_array = np.array([i for i in range(power_range)])
     
'''
A bunch of empty arrays, for use storing data
'''

Z0L = np.zeros((power_range,detune_range)) #Empty list to hold the spectra
Z0C = np.zeros((power_range,detune_range))
ZX = np.zeros((power_range,detune_range))
ZY = np.zeros((power_range,detune_range))
ZP = np.zeros((power_range,detune_range))
ZM = np.zeros((power_range,detune_range))

start_time = time.time()

Bpower = 4e-2  
for pdz, power in enumerate(power_array):
    print('working on Power',pdz+1,'of',len(power_array))
    for idz, val in enumerate(detune_array):
        if idz%5 == 0:
            print('working on spectra',idz+1,'of',len(detune_array))
       
         
        '''
        Defining the lasers
        ''' 
        P2 = 0.001
        P1 = 0.002*pdz
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
        
    
        spec1 = Exp.ExciteSpec(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'])
        
        
    
    
        # #For ExciteSpec
        Z0L[pdz,idz] = spec1['X']+spec1['Y']
        Z0C[pdz,idz] = spec1['SP']+spec1['SM']
        ZX[pdz,idz] = spec1['X']
        ZY[pdz,idz] = spec1['Y']
        ZP[pdz,idz] = spec1['SP']
        ZM[pdz,idz] = spec1['SM']


total_time = time.time()-start_time

# For plotting individual spectra
idx = 0                                            #Plotting the results!


# # For plotting Excitation Arrays
fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

#First plot to see how the linear polarizations works out
ax[0,0].semilogy( detuning0+detune_array*point_spacing,Z0L[idx], color = 'k' ) #Plokktting the dirint result for comparison
ax[0,0].semilogy(detuning0+detune_array*point_spacing, ZX[idx], color = 'slateblue')
ax[0,0].semilogy(detuning0+detune_array*point_spacing, ZY[idx], color = 'navy',linestyle = 'dashed')
ax[0,0].legend(['NoPol','x','y'])
ax[0,0].set_xlabel('$\omega$ [THz]')
ax[0,0].set_ylabel("Amplitude (arb)") 

#Second plot to look at circular polarizations
ax[0,1].semilogy( detuning0+detune_array*point_spacing,Z0C[idx], color = 'k' ) #Plokktting the dirint result for comparison
ax[0,1].semilogy(detuning0+detune_array*point_spacing, ZP[idx], color = 'green')
ax[0,1].semilogy(detuning0+detune_array*point_spacing, ZM[idx], color = 'orangered',linestyle = 'dashed')
ax[0,1].legend(['NoPol','SP','SM'])
ax[0,1].set_xlabel('$\omega$ [THz]')
ax[0,1].set_ylabel("Amplitude (arb)") 

# #First plot to see how the linear polarizations works out
# ax[1,0].semilogy( omega_array-(Exp.beat/(4*np.pi)),abs(Z0L[idx]-Z0C[idx])*100, color = 'k' ) #Plokktting the dirint result for comparison
# ax[1,0].legend(['Absolute percent deviation of Linear X+Y from circular SP+SM'])
# ax[1,0].set_xlabel('$\omega$ [THz]')
# ax[1,0].set_ylabel("Amplitude (arb)") 

# #Third for linear percent deviation
# ax[1,0].plot(Z0g1[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[1,0].plot(ZXg1[idx], color = 'slateblue', linestyle = 'dashed')
# ax[1,0].plot(ZYg1[idx], color = 'lightsteelblue' )
# ax[1,0].legend(['steps'])
# ax[1,0].set_ylabel("g1") 

# #Fourth for circular percent deviation
# ax[1,1].plot(Z0g1[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[1,1].plot(ZPg1[idx], color = 'green')
# ax[1,1].plot(ZMg1[idx], color = 'orangered' , linestyle = 'dashed')
# ax[1,1].legend(['NoPol','P','M'])
# ax[1,1].set_xlabel('steps')
# ax[1,1].set_ylabel("g1") 


fig.suptitle(F"V->PF Config with $\Omega_1$ = 1 GHz {L2pol}, $\Delta_1$ = {detuning0+point_spacing*idx} GHz, B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt}" )





clims = [1e-3,1e-0]
Om1 = np.dot(       list(dot.dipoles.values())[0] ,Exp.Las2.E)
# Plot on a colorplot
fig, ax = plt.subplots(2,2)
limits = [detuning0,\
          detuning0+detune_array[-1]*point_spacing,\
          power_array[0]*.002,\
          power_array[-1]*.002]
    
pos = ax[0,0].imshow(ZX,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Omega_2$ [THz]') 


pos = ax[0,1].imshow(ZY,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)

pos = ax[1,0].imshow(ZP,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\Delta_1$ (THz)')
ax[1,0].set_ylabel('$\Omega_2$ [THz]') 
ax[1,0].set_title(F'detpol = SP' )

pos = ax[1,1].imshow(ZM,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,1].set_xlabel('$\Delta_1$ (THz)')
ax[1,1].set_ylabel('$\Omega_2$ [THz]')  
ax[1,1].set_title( 'detpol = SM' )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays
fig.suptitle(F"V->PF Config with $\Omega_1$ = 1 GHz, excitation laser = {L2pol}, B = {Bpower}" )



#Plotting just the unpolarized case
Om1 = np.dot(       list(dot.dipoles.values())[0] ,Exp.Las2.E)
# Plot on a colorplot
fig, ax = plt.subplots(1,1)
limits = [detuning0,\
          detuning0+detune_array[-1]*point_spacing,\
          power_array[0]*.002,\
          power_array[-1]*.002]
    
pos = ax.imshow(Z0L,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
# ax.axvline(x=(0*Om1/(2*np.pi)), color='y', linestyle = 'solid',linewidth =3)
# ax.axvline(x=(1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# ax.axvline(x=(-1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
ax.set_xlabel('$\Delta_1$ (THz)')
ax.set_ylabel('$\Omega_2$ [THz]') 
ax.set_title(F'V->PF Config with $\Omega_1$ = 1 GHz, excitation laser = {L2pol}, detpol = None, B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt}' )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays



# # # For plotting Excitation Arrays
# fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

# #First plot to see how the linear polarizations works out
# ax[0,0].plot(detuning0+P_array*point_spacing,Z0Lavg, color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,0].plot(detuning0+P_array*point_spacing,ZXavg, color = 'slateblue', linestyle = 'dashed')
# ax[0,0].plot(detuning0+P_array*point_spacing,ZYavg, color = 'lightsteelblue' )
# ax[0,0].legend(['NoPol','x','y'])
# ax[0,0].set_ylabel("Average value of Spectrum") 
# ax[0,0].axvline(x=transX1-280,color = 'r')
# ax[0,0].axvline(x=transX2-280,color = 'r')
# ax[0,0].axvline(x=transY1-280,color = 'g')
# ax[0,0].axvline(x=transY2-280,color = 'g')

# # #First plot to see how the linear polarizations works out
# # ax[0,0].plot(Z0Lavg, color = 'k' ) #Plokktting the dirint result for comparison
# # ax[0,0].plot(ZXavg, color = 'slateblue', linestyle = 'dashed')
# # ax[0,0].plot(ZYavg, color = 'lightsteelblue' )
# # ax[0,0].legend(['NoPol','x','y'])
# # ax[0,0].set_ylabel("Average value of Spectrum") 


# #Second plot to look at circular polarizations
# ax[0,1].plot(  detuning0+P_array*point_spacing, Z0Cavg, color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,1].plot(detuning0+P_array*point_spacing, ZPavg, color = 'navajowhite')
# ax[0,1].plot(detuning0+P_array*point_spacing, ZMavg, color = 'orangered' , linestyle = 'dashed')
# ax[0,1].legend(['NoPol','SP','SM'])
# ax[0,1].set_ylabel("Average value of Spectrum") 

# #Third for linear percent deviation
# ax[1,0].plot(  detuning0+P_array*point_spacing, np.array(Z0Lavg)/np.array(Z0Lavg), color = 'k' ) #Plokktting the dirint result for comparison
# ax[1,0].plot(detuning0+P_array*point_spacing, np.array(ZXavg)/np.array(Z0Lavg), color = 'slateblue', linestyle = 'dashed')
# ax[1,0].plot(detuning0+P_array*point_spacing, np.array(ZYavg)/np.array(Z0Lavg), color = 'lightsteelblue' )
# ax[1,0].legend(['NoPol','x','y'])
# ax[1,0].set_xlabel('$\Delta_1$ [THz]')
# ax[1,0].set_ylabel("Polarizations/NoPol") 

# #Fourth for circular percent deviation
# ax[1,1].plot(  detuning0+P_array*point_spacing, np.array(Z0Cavg)/np.array(Z0Cavg), color = 'k' ) #Plokktting the dirint result for comparison
# ax[1,1].plot(detuning0+P_array*point_spacing, np.array(ZPavg)/np.array(Z0Cavg), color = 'navajowhite')
# ax[1,1].plot(detuning0+P_array*point_spacing,np.array(ZMavg)/np.array(Z0Cavg), color = 'orangered' , linestyle = 'dashed')
# ax[1,1].legend(['NoPol','P','M'])
# ax[1,1].set_xlabel('$\Delta_1$ [THz]')
# ax[1,1].set_ylabel("Polarizations/NoPol") 

# # # #Third for linear percent deviation
# # # ax[1,0].plot(detuning0+P_array*point_spacing, ((np.array(Z0avg)-(np.array(ZXavg)+np.array(ZYavg)))/np.array(Z0avg))*100, color = 'k' ) #Plokktting the dirint result for comparison
# # # ax[1,0].legend(['NoPol-(XPol+YPol)'])
# # # ax[1,0].set_xlabel('$\Delta_1$ [THz]')
# # # ax[1,0].set_ylabel("Percent Deviation") 

# # # #Fourth for circular percent deviation
# # # ax[1,1].plot(detuning0+P_array*point_spacing, ((np.array(Z0avg)-(np.array(ZXavg)+np.array(ZYavg)))/np.array(Z0avg))*100, color = 'k' ) #Plokktting the dirint result for comparison
# # # ax[1,1].legend(['NoPol-(SPPol+SMPol)'])
# # # ax[1,1].set_xlabel('$\Delta_1$ [THz]')
# # # ax[1,1].set_ylabel("Percent Deviation") 


# fig.suptitle(F"Voigt Config with $\Omega_1$ = 1 GHz {Lpol}, B = {Bpower}, $\tau$ = {tau}, Nt = {Nt}" )



