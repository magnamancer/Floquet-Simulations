# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 00:49:55 2023

@author: FentonClawson
"""
point_spacing = 0.0001
detuning0 = -.005


detune_range = 151
detune_array = np.zeros(detune_range)
for i in range(detune_range):
    detune_array[i]=(((i*1)+0))


plot_detune_range = detuning0+detune_array*point_spacing


clims = [1e-3,1]
# Plot on a colorplot
fig, ax = plt.subplots(2,2)
limits = [plot_detune_range[0],\
          plot_detune_range[-1],\
          P_array[0],\
          P_array[-1]]
    
pos = ax[0,0].imshow(ZX_tot,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,0].set_ylabel('$\Delta_1$ [THz]') 
ax[0,0].set_title(F'detpol = X' )

pos = ax[0,1].imshow(ZY_tot,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[0,1].set_ylabel('$\Delta_1$ [THz]') 
ax[0,1].set_title(F'detpol = Y' )

pos = ax[1,0].imshow(ZSP_tot,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
            extent = limits,  norm=matplotlib.colors.LogNorm(), clim = clims)
ax[1,0].set_xlabel('$\omega$ (THz)')
ax[1,0].set_ylabel('$\Delta_1$ [THz]') 
ax[1,0].set_title(F'detpol = SP' )

pos = ax[1,1].imshow(ZSM_tot,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
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
