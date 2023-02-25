# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:17:40 2023

@author: Fenton
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:20:35 2023

@author: Fenton





The aim of this scipt is to show a relatively simple resonantly excited
2 level system

Plan: I'm going to construct a 2LS, then produce a colormap of the
emission spectra as near-resonant laser detuning is swept through resonance

I'll try to explain this script to death so that it can be used or at least
modified later to an "intro" script '
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
norm = np.sqrt(2) #This is just for normalization

#Defining the energetic states of the Bare dot (in THz)
states = [0,2*np.pi*280] 

#Dipole dictionary is a dictionary of {states linked:dipole moment polarization} key/value pairs
#In this case, states 0 and 1 are linked by a dipole moment that interacts with sigma+ light
dipole = {(0,1):(1/norm,-1j/norm,0)} 

#The full dot object is constructed from the number of states (this needs to be removed as # of states is easy to find internally),
#   the list of bare states (above),
#   and the dipole moment dictionary (above)
dot = FC.QD(2,states,dipole)


'''
Defining the Collapse Operator

I'm not positive that the terminology I've been using is correct/universal, 
    so I'm going to explain it a little to hopefully clear up any confusion
    
Internally, the lowering operator for the system is found from the dipole moment
    states linkages and system dimensionality. The collapse operators, however,
    are a separate set of operators that define how
    the system interacts with the world. Both are used, but in different ways
    internally. Further, all collapse operators must be defined, even the 
    collapse operator that has the same matrix form as the
    lowering operator (that corresponds to spontaneous emission)
'''

#Setting the rate of the collapse operator
manifolds_collapse_rate = np.sqrt(2*np.pi*.0000025)

#Creating the collapse operator matrix form representation
collapse_operator_SP= \
                            np.array(
                                  [[0, 1],
                                  [0, 0]])
    
#Making the collapse operator object
c_op_SP = FC.Collapse_Op(collapse_operator_SP,manifolds_collapse_rate)


#Creating the list of collapse operator object. 
#Even if there's only one, this needs to be fed in as a list!
collapse_operator_list = [c_op_SP]



'''
These empty arrays store spectra/data to allow me to construct colormaps
out of multiple, e.g., emisison spectra.
'''
#These are for the Emission Spectra
Z0L = [] #For holding "Unpolarized" detection, formed from the sum of X+Y
Z0C = [] #For holding "Unpolarized" detection, formed from the sum of sigma+ and sigma-
ZX = [] #For holding X polarized detection
ZY = [] #For holding Y polarized detection
ZP = [] #For holding sigma+ polarized detection
ZM = [] #For holding sigma- polarized detection

Z0Lavg = [] #For holding "Unpolarized" detection, formed from the sum of X+Y
Z0Cavg = [] #For holding "Unpolarized" detection, formed from the sum of sigma+ and sigma-
ZXavg = [] #For holding X polarized detection
ZYavg = [] #For holding Y polarized detection
ZPavg = [] #For holding sigma+ polarized detection
ZMavg = [] #For holding sigma- polarized detection


'''
Defining Polarizations for use below
'''
LP = {
'X' : np.array([1,0,0]),
'Y' : np.array([0,1,0]),
'D' : (1/np.sqrt(2))*np.array([1,1,0]),
'SP' :(1/np.sqrt(2))*np.array([1,1j,0]),
'SM' :(1/np.sqrt(2))*np.array([1,-1j,0]),
'NP' : np.array([None,None,None])}





'''
Setting time arguments. These work out, "experimentally," to 

tau - controls density of points in final (emission) spectra. 
Nt - controls the width of the final (emission) spectra

That's how I use them, anyways.

NOTE: I haven't actually tried testing this in awhile, but last I checked
Nt NEEDS to be a power of 2. It's something having to do with the harmonics of the
system frequency. Don't remember off the top of my head, but I have it written
down somewhere in my OneNote.'
'''
tau = 50 #Number of periods forward to simulate the system for the two-time operator expectation value
Nt = 2**5 #Number of points to solve for in each period of the system. Haven't checked in awhile



'''
The array below is just to help me set values for looping through multiple spectra 
"experiments." Right now, I'm going to use it to sweep through 101 excitation
laer detunings close to the system resonance and calcualte the emission spectra 
for each
'''
detune_range = 101
D_array = np.zeros(detune_range)
for i in range(detune_range):
    D_array[i]=(i*5)


for idz, val in enumerate(D_array):
   
    print('working on spectra',idz+1,'of',len(D_array))
   
      
    '''
    Defining the lasers
    ''' 
    
    #Defining Excitation Laser power (In THz)
    P0 = 0.03 
    Lpower = 2*np.pi*P0

    #Defining Laser polarization, from the earlier defined polarization list
    Lpol = 'SP'

    #Defining Detuning as the initial detuning+some difference defined by the for loop iteration and D_array
    point_spacing = 0.0001 #Spacing between laser detunings (in THz)
    detuning0 = -.025 #Initial excitation Laser Detuning
    detuning = detuning0+point_spacing*val #Setting the excitation laser detuning for this specific iteration of the for loop
    Lfreq = 2*np.pi*(280+detuning) #Setting the excitation laser frequency from the calculated detuning

    
    #Finally, the laser object is created
    L = FC.Laser(Lpower,LP[Lpol],Lfreq) 
    
    
    

    '''
    Defining the initial state using QuTiP's basis functions'
    '''
    rho00 = basis(2,0)*basis(2,0).dag()
    
   
    '''
    Finally, the "quantum system" object is set up by taking in the previously 
        defined quantum dot object, laser object, and collapse operator list.
    
    There is an optional input for a magnetic field (Lateral or Perpindicular
        to dot growth direction), but that has no meaning for a two-level dot.
    '''
    Exp = FC.QSys(dot,LasList = [L],c_op_list = collapse_operator_list)
    
    
    #If it's the first loop iteration, find the frequency values for the emission spectra 
    if idz == 0:
        omega_array = flm.freqarray(Exp.T,Nt,tau)  
       
    


    '''
    Exp.EmisSpec is used to find the dictionary of detection-polarization-modified
        emission spectra. These are then extracted into their respective empty
        lists that I defined earlier.
        
    Time_sensitivity is the time-dependance allowed in the internal secular 
        approximation (NOT the RWA). Currently only works for 0 and certain
        "choice" other values depending on the system. I think it has to do
        with either a mistake I made in setting up the Rate matrix solver (that 
        somehow doesn't effect the time-independant case?'),or some sort of 
        "symmetry" that requires I increase the time-dependance in discrete 
        chunks. Not sure yet.
    
    detpols is the desired detection polarization for each "experiment." Turns
        out it's pretty cheap to solve them all together for each spectrum.
    '''
    #Finding the emission spectra
    spec1 = Exp.EmisSpec(Nt,tau,rho0=rho00,time_sensitivity=0, detpols = ['X','Y','SP','SM'])
    
    Z0L.append(spec1['X']+spec1['Y'])
    Z0C.append(spec1['SP']+spec1['SM'])
    ZX.append(spec1['X'])
    ZY.append(spec1['Y'])
    ZP.append(spec1['SP'])
    ZM.append(spec1['SM'])

  
    
    Z0Lavg.append(np.average(Z0L[-1]))
    Z0Cavg.append(np.average(Z0C[-1]))
    ZXavg.append(np.average(ZX[-1]))
    ZYavg.append(np.average(ZY[-1]))
    ZPavg.append(np.average(ZP[-1]))
    ZMavg.append(np.average(ZM[-1]))
    



    





'''
Uncomment to create colormaps of emission spectra

Change freqlims to move around the frequency bounds of the plot.
'''
freqlims = [omega_array[0]-(Exp.beat/(4*np.pi)),omega_array[-1]-(Exp.beat/(4*np.pi))]


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

fig, ax = plt.subplots(2,2)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          detuning0+D_array[0]*point_spacing,\
          detuning0+D_array[-1]*point_spacing]

#Plotting Linear Detection Polarization X    
pos = \
ax[0,0].imshow(ZX_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower', extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Delta_1$ [THz]') 
ax[0,0].set_title(F'detpol = X' )

#Plotting Linear Detection Polarization Y    
pos = \
ax[0,1].imshow(ZY_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,1].set_title(F'detpol = Y' )

#Plotting Circular Detection Polarization sigma+   
pos = \
ax[1,0].imshow(ZP_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\omega$ (THz)')
ax[1,0].set_ylabel('$\Delta_1$ [THz]') 
ax[1,0].set_title(F'detpol = $\sigma_+$' )

#Plotting Circular Detection Polarization sigma-
pos = \
ax[1,1].imshow(ZM_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,1].set_xlabel('$\omega$ (THz)')
ax[1,1].set_title( 'detpol = $\sigma_-$' )

fig.suptitle(F"Near-resonant 2 Level System with $\Omega_1$ = {P0*1000} GHz, {Lpol} polarized, " )
fig.colorbar(pos, ax=ax)



# Plotting the no-polarization-detection case, built out of X+Y detection polarization
fig, ax = plt.subplots(1,1)
limits = [plot_freq_range[0],\
          plot_freq_range[-1],\
          detuning0+D_array[0]*point_spacing,\
          detuning0+D_array[-1]*point_spacing]
pos = ax.imshow(Z0L_truncated,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims) 
ax.set_xlabel('$\omega$ (THz)')
ax.set_ylabel('$\Delta_1$ [THz]') 
fig.suptitle(F"Near-resonant 2 Level System with $\Omega_1$ = {P0*1000} GHz, {Lpol} polarized, " )
fig.colorbar(pos, ax=ax)# # For plotting Excitation Arrays


