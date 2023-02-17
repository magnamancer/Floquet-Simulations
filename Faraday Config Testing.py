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
states = [0,0,2*np.pi*280,2*np.pi*(280+1e-7)] #Resonance arbitrarily decided
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



electron_spin_flip_rate = manifolds_collapse_rate*(100/100)

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






tau = 2000 #Length of time to go forward, in units of T, the system Frequency
Nt = 2**3 #Number of points to solve for in each period of the system. Minimum depends on the lowering operator
PDM = 2**0 #If the spectrumm isn't as wide as it needs to be, increase the power of 2 here.
interpols = 2**0 #interpolation, for if the spectra are doing the *thing*


point_spacing = 0.0001
detuning0 = -.005


power_range = 151
P_array = np.zeros(power_range)
for i in range(power_range):
    P_array[i]=(((i*1)+0))
    
start_time = time.time()
Bpower = 4e-2  
for idz, val in enumerate(P_array):
    print('working on spectra',idz+1,'of',len(P_array))
   
     
    '''
    Defining the lasers
    ''' 
    P2 = 0.001
    P1 = 0.2
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
    
    
    
    Exp = FC.QSys(dot,LasList = [L1,L2], Bfield = B,c_op_list = collapse_operator_list)
    
    
    if idz == 0:
        omega_array = esm.freqarray(Exp.T,Nt,tau)  
        Transitions = Exp.TransitionEnergies()
        transX1=(abs(Transitions[2]-Transitions[0]))
        transX2=(abs(Transitions[3]-Transitions[1]))
        transY1=(abs(Transitions[2]-Transitions[1]))
        transY2=(abs(Transitions[3]-Transitions[0]))
    
    
    spec1,g1dic = Exp.EmisSpec(Nt,tau,rho0=rho00,time_sensitivity=0.0, detpols = ['X','Y','SP','SM'],retg1='True')
    
    # spec1 = Exp.ExciteSpec(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'])
    
    
    
    
    # # #For ExciteSpec
    # Z0Lavg.append(spec1['X']+spec1['Y'])
    # Z0Cavg.append(spec1['SP']+spec1['SM'])
    # ZXavg.append(spec1['X'])
    # ZYavg.append(spec1['Y'])
    # ZPavg.append(spec1['SP'])
    # ZMavg.append(spec1['SM'])
    




    Z0L.append(spec1['X']+spec1['Y'])
    Z0C.append(spec1['SP']+spec1['SM'])
    ZX.append(spec1['X'])
    ZY.append(spec1['Y'])
    ZP.append(spec1['SP'])
    ZM.append(spec1['SM'])

    Z0g1.append(g1dic['X']+g1dic['Y'])
    ZXg1.append(g1dic['X'])
    ZYg1.append(g1dic['Y'])
    ZPg1.append(g1dic['SP'])
    ZMg1.append(g1dic['SM'])
    
    Z0Lavg.append(np.average(Z0L[-1]))
    Z0Cavg.append(np.average(Z0C[-1]))
    ZXavg.append(np.average(ZX[-1]))
    ZYavg.append(np.average(ZY[-1]))
    ZPavg.append(np.average(ZP[-1]))
    ZMavg.append(np.average(ZM[-1]))
    
total_time = time.time()-start_time

Z0Lavg = np.array(Z0Lavg)
ZXavg = np.array(ZXavg)
ZYavg = np.array(ZYavg)
ZPavg = np.array(ZPavg)
ZMavg = np.array(ZMavg)

# # For plotting individual spectra
# idx = 0                                            #Plotting the results!


# # # For plotting Excitation Arrays
# fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

# #First plot to see how the linear polarizations works out
# ax[0,0].semilogy( omega_array-(Exp.beat/(4*np.pi)),Z0L[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,0].semilogy(omega_array-(Exp.beat/(4*np.pi)), ZX[idx], color = 'slateblue')
# ax[0,0].semilogy(omega_array-(Exp.beat/(4*np.pi)), ZY[idx], color = 'navy',linestyle = 'dashed')
# ax[0,0].legend(['NoPol','x','y'])
# ax[0,0].set_xlabel('$\omega$ [THz]')
# ax[0,0].set_ylabel("Amplitude (arb)") 

# #Second plot to look at circular polarizations
# ax[0,1].semilogy( omega_array-(Exp.beat/(4*np.pi)),Z0C[idx], color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,1].semilogy(omega_array-(Exp.beat/(4*np.pi)), ZP[idx], color = 'green')
# ax[0,1].semilogy(omega_array-(Exp.beat/(4*np.pi)), ZM[idx], color = 'orangered',linestyle = 'dashed')
# ax[0,1].legend(['NoPol','SP','SM'])
# ax[0,1].set_xlabel('$\omega$ [THz]')
# ax[0,1].set_ylabel("Amplitude (arb)") 

# # #First plot to see how the linear polarizations works out
# # ax[1,0].semilogy( omega_array-(Exp.beat/(4*np.pi)),abs(Z0L[idx]-Z0C[idx])*100, color = 'k' ) #Plokktting the dirint result for comparison
# # ax[1,0].legend(['Absolute percent deviation of Linear X+Y from circular SP+SM'])
# # ax[1,0].set_xlabel('$\omega$ [THz]')
# # ax[1,0].set_ylabel("Amplitude (arb)") 

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


# fig.suptitle(F"Voigt Config with $\Omega_1$ = 1 GHz {L2pol}, $\Delta_1$ = {detuning0+point_spacing*idx} GHz, B = {Bpower}" )


freqlims = [-0.01,0.02]#[omega_array[0]-(Exp.beat/(4*np.pi)),omega_array[-1]-(Exp.beat/(4*np.pi))]


frequency_range = (omega_array-(Exp.beat/2)/(2*np.pi))
idx0 = np.where(abs(frequency_range-freqlims[0]) == np.amin(abs((frequency_range-freqlims[0] ))))[0][0]
idxf = np.where(abs(frequency_range-freqlims[1]) == np.amin(abs((frequency_range-freqlims[1] ))))[0][0]

plot_freq_range = frequency_range[idx0:idxf]
Z0L_truncated = np.stack([Z0Li[idx0:idxf] for Z0Li in Z0L])
ZX_truncated = np.stack([ZXi[idx0:idxf] for ZXi in ZX])
ZY_truncated = np.stack([ZYi[idx0:idxf] for ZYi in ZY])
ZP_truncated = np.stack([ZPi[idx0:idxf] for ZPi in ZP])
ZM_truncated = np.stack([ZMi[idx0:idxf] for ZMi in ZM])




clims = [1e-7,1e-1]
# Plot on a colorplot
fig, ax = plt.subplots(2,2)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          detuning0+P_array[0]*point_spacing,\
          detuning0+P_array[-1]*point_spacing]
    
pos = ax[0,0].imshow(ZX_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Delta_1$ [THz]') 
ax[0,0].set_title(F'detpol = X' )

pos = ax[0,1].imshow(ZY_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,1].set_ylabel('$\Delta_1$ [THz]') 
ax[0,1].set_title(F'detpol = Y' )

pos = ax[1,0].imshow(ZP_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\omega$ (THz)')
ax[1,0].set_ylabel('$\Delta_1$ [THz]') 
ax[1,0].set_title(F'detpol = SP' )

pos = ax[1,1].imshow(ZM_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,1].set_xlabel('$\omega$ (THz)')
ax[1,1].set_ylabel('$\Delta_1$ [THz]') 
ax[1,1].set_title( 'detpol = SM' )

fig.suptitle(F"Pseudo-Faraday Config with $\Omega_1$ = 1 GHz {L2pol},$\Omega_2$ = 200 GHz {L1pol},$\Delta_2$ = {ACdetune} THz B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt}" )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays






# Plot on a colorplot
fig, ax = plt.subplots(1,1)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          detuning0+P_array[0]*point_spacing,\
          detuning0+P_array[-1]*point_spacing]
pos = ax.imshow(Z0L_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
# ax.axvline(x=(0*Om1/(2*np.pi)), color='y', linestyle = 'solid',linewidth =3)
# ax.axvline(x=(1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
# ax.axvline(x=(-1*Om1/(2*np.pi)), color='y', linestyle = 'dashed',linewidth =3)
ax.set_xlabel('$\omega$ (THz)')
ax.set_ylabel('$\Delta_1$ [THz]') 
fig.suptitle(F"Pseudo-Faraday Config with $\Omega_1$ = 1 GHz {L2pol},$\Omega_2$ = 200 GHz {L1pol},$\Delta_2$ = {ACdetune} THz B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt} No detection polarization" )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays



# # For plotting Excitation Arrays
fig, ax = plt.subplots(2,2)                                                    #Plotting the results!

#First plot to see how the linear polarizations works out
ax[0,0].plot(detuning0+P_array*point_spacing,Z0Lavg, color = 'k' ) #Plokktting the dirint result for comparison
ax[0,0].plot(detuning0+P_array*point_spacing,ZXavg, color = 'slateblue', linestyle = 'dashed')
ax[0,0].plot(detuning0+P_array*point_spacing,ZYavg, color = 'lightsteelblue' )
ax[0,0].legend(['NoPol','x','y'])
ax[0,0].set_ylabel("Average value of Spectrum") 
# ax[0,0].axvline(x=transX1-280)
# ax[0,0].axvline(x=transX2-280)
# ax[0,0].axvline(x=transY1-280)
# ax[0,0].axvline(x=transY2-280)

# #First plot to see how the linear polarizations works out
# ax[0,0].plot(Z0Lavg, color = 'k' ) #Plokktting the dirint result for comparison
# ax[0,0].plot(ZXavg, color = 'slateblue', linestyle = 'dashed')
# ax[0,0].plot(ZYavg, color = 'lightsteelblue' )
# ax[0,0].legend(['NoPol','x','y'])
# ax[0,0].set_ylabel("Average value of Spectrum") 


#Second plot to look at circular polarizations
ax[0,1].plot(  detuning0+P_array*point_spacing, Z0Cavg, color = 'k' ) #Plokktting the dirint result for comparison
ax[0,1].plot(detuning0+P_array*point_spacing, ZPavg, color = 'navajowhite')
ax[0,1].plot(detuning0+P_array*point_spacing, ZMavg, color = 'orangered' , linestyle = 'dashed')
ax[0,1].legend(['NoPol','SP','SM'])
ax[0,1].set_ylabel("Average value of Spectrum") 

#Third for linear percent deviation
ax[1,0].plot(  detuning0+P_array*point_spacing, np.array(Z0Lavg)/np.array(Z0Lavg), color = 'k' ) #Plokktting the dirint result for comparison
ax[1,0].plot(detuning0+P_array*point_spacing, np.array(ZXavg)/np.array(Z0Lavg), color = 'slateblue', linestyle = 'dashed')
ax[1,0].plot(detuning0+P_array*point_spacing, np.array(ZYavg)/np.array(Z0Lavg), color = 'lightsteelblue' )
ax[1,0].legend(['NoPol','x','y'])
ax[1,0].set_xlabel('$\Delta_1$ [THz]')
ax[1,0].set_ylabel("Polarizations/NoPol") 

#Fourth for circular percent deviation
ax[1,1].plot(  detuning0+P_array*point_spacing, np.array(Z0Cavg)/np.array(Z0Cavg), color = 'k' ) #Plokktting the dirint result for comparison
ax[1,1].plot(detuning0+P_array*point_spacing, np.array(ZPavg)/np.array(Z0Cavg), color = 'navajowhite')
ax[1,1].plot(detuning0+P_array*point_spacing,np.array(ZMavg)/np.array(Z0Cavg), color = 'orangered' , linestyle = 'dashed')
ax[1,1].legend(['NoPol','P','M'])
ax[1,1].set_xlabel('$\Delta_1$ [THz]')
ax[1,1].set_ylabel("Polarizations/NoPol") 

# # #Third for linear percent deviation
# # ax[1,0].plot(detuning0+P_array*point_spacing, ((np.array(Z0avg)-(np.array(ZXavg)+np.array(ZYavg)))/np.array(Z0avg))*100, color = 'k' ) #Plokktting the dirint result for comparison
# # ax[1,0].legend(['NoPol-(XPol+YPol)'])
# # ax[1,0].set_xlabel('$\Delta_1$ [THz]')
# # ax[1,0].set_ylabel("Percent Deviation") 

# # #Fourth for circular percent deviation
# # ax[1,1].plot(detuning0+P_array*point_spacing, ((np.array(Z0avg)-(np.array(ZXavg)+np.array(ZYavg)))/np.array(Z0avg))*100, color = 'k' ) #Plokktting the dirint result for comparison
# # ax[1,1].legend(['NoPol-(SPPol+SMPol)'])
# # ax[1,1].set_xlabel('$\Delta_1$ [THz]')
# # ax[1,1].set_ylabel("Percent Deviation") 


fig.suptitle(F"Pseudo-Faraday Config Excitation Spectrum with $\Omega_1$ = 1 GHz {L2pol},$\Omega_2$ = 200 GHz {L1pol},$\Delta_2$ = {ACdetune} THz B = {Bpower}, $\\tau$ = {tau}, Nt = {Nt}" )



