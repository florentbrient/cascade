#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:02:30 2025

The goal of the script to analysis the saved data and plot figures

Input:
    - Netcdf data (ex: ../data/spectra/Spectra_FIR1k_V0010_001.nc)

file
@author: fbrient
"""

import numpy as np
import tools as tl
import pylab as plt
import math
import glob
import netCDF4 as nc
from netCDF4 import num2date
import datetime



def nc_dataset_list(file):
    return nc.Dataset(file, 'r')



# Open all netcdf files
pathin   = "../data/spectra/"
prefix   = "FIRZ4" #"IHOP" #"FIR1k"
filein0  = pathin+'Spectra_*'+prefix+'*.nc'
data     = tl.read_netcdfs(filein0, dim='time')
#filein0  = pathin+'Spectra*'+prefix+'*.nc'
#data2     = tl.read_netcdfs(filein0, dim='time')

#filein0F = filein0.replace('Spectra','SpectraFilter')
#dataF    = tl.read_netcdfs(filein0F)


############ Open 1D to find time of sunlight
path0 = '/home/fbrient/MNH/V5-7-0/*/'+prefix+'/'
file0 = path0+'*'+'.1.'+'*'+'.000'+'*'+'.nc'
# List of files with glob
files = sorted(glob.glob(file0))
timing2 = []
datarad = []
for idx,file in enumerate(files):
    # Open data from files
    print('file : ',file)
    data1D = nc_dataset_list(file)
    
    timing = data1D['time_les'][:]  # start : 25300
    units = data1D['time_les'].units

    timing = timing.astype(int)
    timing2.append(timing.data[:])
    
    tmp  = data1D['LES_budgets']['Radiation']['Cartesian']['Not_time_averaged']['Not_normalized']['cart']
    datarad.append(tmp['DTHRADSW'][:])

# Timing
time1D = np.concatenate(timing2)
calendar = "standard"  # could also be 'gregorian'
# Convert to datetime objects
time_datetimes = num2date(time1D, units=units, calendar=calendar)

# Reference date is just the first
ref_date = time_datetimes[0] - datetime.timedelta(seconds=int(time1D[0]))
time_datetimes = [ref_date +
                  datetime.timedelta(seconds=int(s)) for s in time1D[:]]
# Convert to hours since start
time_hours = [(t - time_datetimes[0]).total_seconds() /
              3600 for t in time_datetimes]
time_hours = np.array(time_hours)
nt         = len(time1D)

# Find day and night
DTHRADSW = datarad
DTHRADSW = np.concatenate(DTHRADSW, axis=0)
DTHRADSW = np.max(DTHRADSW,axis=1)
epsilon = 1e-10
day = (DTHRADSW>epsilon)
print(day)




# Name of var 2D: LWP or PRW
nvar = "LWP"
nvar0= 'RCT'
if prefix=="IHOP":
    nvar = "PRW"
    nvar0= 'RVT'
nvarall = ['WT','THL','TKEh','TKEv','v1']+[nvar0]
nv      = len(nvarall)

# Dir for figures
pathout = "../figures/Budget/"
pathout+= prefix+'/'
tl.mkdir(pathout)

# wavelength, altitude, time and PBL height
kv       = data.kv.values
kbins    = data.kbins.values
kcenter  = data.k_center.values
z        = data.z.values
time     = data.time.values #+1 (error)
kvdiff   = np.diff(kv)


timemax  = 100
if prefix == "FIZ4c":
    timemax=24
idxtmax  = min(time.max(),timemax)
cond     = (time >= 0) & (time <= idxtmax)    
indices  = np.where(cond)[0]

# day hours
day_hours = day[[tl.near(time[ij],time_hours) for ij in time]]

# Remove half of points if len(indices) too long
dt = 1
if len(indices)>25:
    indices=indices[1::2]
labels = ['t+'+str(ij+dt).zfill(2) for ij in indices]
labels = np.array(labels)

# Compute altitude differences
dz = np.diff(z)  # Differences between consecutive altitudes
dz = np.insert(dz, 0, dz[0])  # Assume first weight equals first diff

# Initializing
nt,nkv,nz =np.shape(data.PI_E[indices])
nbins,ncenter  = len(kbins),len(kcenter)

PI_E_sum,PI_Ew_sum,PI_Z_sum,Erad_sum =  [np.zeros((nt,nkv))*np.nan for ij in range(4)]
Eradv2_sum = np.zeros((nt,ncenter))*np.nan
var_sum,kvmax,kvmax2,kvmaxLWP,kin = [np.zeros(nt)*np.nan for ij in range(5)]
kvmaxRCTz = np.zeros((nv,nt,nz))*np.nan
Ers,ErLWPs,ErTHLs,ErWTs = [np.zeros((nt,nkv))*np.nan for ij in range(4)]
Erv2s  = np.zeros((nt,ncenter))*np.nan 
varLWP = "E1dr_"+nvar
varRCT = ["E1dr_"+ij for ij in nvarall] 
ErRCT_sum,ErRCTs_sum = [np.zeros((nv,nt,nkv))*np.nan  for ij in range(2)]
kvmaxRCT  = np.zeros((nv,nt))*np.nan

# ['E1dr_WT', 'E1dr_THL', 'E1dr_RCT']

# Caption axis
y1lab,y2lab= r'E $(m^3/s^2)$',r'$\Pi (m^2/s^3)$'


# Plot k -5/3 slope
plotlines=True

print(indices)
for idxt,tc in enumerate(indices):
    tt = time[tc]
    print(idxt,tc,tt)
    # Find PBL height,
    # Remove first layer, and add one layer above the PBL (overshoot)
    PBL     = data.PBL[tc].values
    kPBL    = tl.z2k(PBL) # rad/m
    idxpbl  = tl.near(z,PBL)
    idxall  = np.arange(1,idxpbl+1) 
    weights = dz[idxall] 
    
    print(idxt,data.kperp.shape,data.kpara.shape,data.k3D.shape)
    
    # Averaging between the surface and PBL
#    Erad_sum[idxt,:]   =np.average(data.E1dr_v1[tc,:,idxall], axis=-1, weights=weights)

    Erad_sum[idxt,:]   =nanweighted_average(data.E1dr_v1[tc,:,idxall],weights=weights) 
    Eradv2_sum[idxt,:] =nanweighted_average(data.E1dr_v2[tc,:,idxall],weights=weights)  


#    ErWT_sum[idxt,:]  = np.average(data[varRCT[0]][tc,:,idxall], axis=-1, weights=weights)

    # Index of TKE
    try:
        var_sum[idxt]  = np.average(data.varTKE[tc,idxall], axis=-1,weights=weights) 
    except:
        var_sum[idxt]  = np.average(data.variance[tc,idxall], axis=-1,weights=weights)

    
    # Data to plot
    kPBL = np.tile(kPBL, (1, 1))

    #Er = np.tile(Erad_sum[idxt,:], (1, 1))
    #Erv2 = np.tile(Eradv2_sum[idxt,:], (1, 1))
    #ErLWP = np.tile(data[varLWP][tc,:], (1, 1))      
    
    # Smooth and calculate max
    #Ers[idxt,:]    = tl.smooth(Er,sigma=1)
    #Erv2s[idxt,:]  = tl.smooth(Erv2,sigma=1)
    #ErLWPs[idxt,:] = tl.smooth(ErLWP,sigma=1)

    
    # Maximum PDF for cell quantification
    #kvmax[idxt]    = kv[Ers[idxt,:].argmax()]
    #kvmax2[idxt]   = kbins[Erv2s[idxt,:].argmax()]
    # kvmax2 is not use in the plot so far (Apr 9)
    # Check the best one v1 or v2?
    #kvmaxLWP[idxt] = kv[ErLWPs[idxt,:].argmax()]
    
    
    # New calcul, more simple
    sigma = 1
    kvmax[idxt],Ers[idxt,:],Er          = findkvmax(kv,data.E1dr_v1[tc,:,idxall],weights=weights,sigma=sigma)
    kvmax2[idxt],Erv2s[idxt,:],Erv2     = findkvmax(kbins,data.E1dr_v2[tc,:,idxall],weights=weights,sigma=sigma)
    kvmaxLWP[idxt],ErLWPs[idxt,:],ErLWP = findkvmax(kv,data[varLWP][tc,:],weights=None,sigma=sigma)
    
    
    PI_E_sum[idxt,:] =nanweighted_average(data.PI_E[tc,:,idxall],weights=weights)
    PI_Ew_sum[idxt,:]=nanweighted_average(data.PI_Ew[tc,:,idxall],weights=weights) 
    PI_Z_sum[idxt,:] =nanweighted_average(data.PI_Z[tc,:,idxall],weights=weights)

    PI = np.tile(PI_E_sum[idxt,:], (1, 1))

    
    # Save Spectra THL and WT
    for ij,vartmp in enumerate(varRCT):
        #ErRCT_sum[ij,idxt,:]  = nanweighted_average(data[vartmp][tc, :, idxall],weights=weights) 
        #np.average(data[vartmp][tc,:,idxall], axis=-1, weights=weights)
        #ErRCTs_sum[ij,idxt,:] = tl.smooth(ErRCT_sum[ij,idxt,:],sigma=1) 
        #kvmaxRCT[ij,idxt]     = kv[ErRCTs_sum[ij,idxt,:].argmax()]
        print(ij,vartmp)
        kvmaxRCT[ij,idxt],ErRCTs_sum[ij,idxt,:],ErRCT_sum[ij,idxt,:] = findkvmax(kv,data[vartmp][tc, :, idxall],weights=weights,sigma=sigma)

        
    #    ErTHLs[idxt,:]    = tl.smooth(ErTHL_sum[idxt,:],sigma=1)
    #    ErWTs[idxt,:]     = tl.smooth(ErWT_sum[idxt,:],sigma=1)
    #    kvmaxTHL[idxt] = kv[ErTHLs[idxt,:].argmax()]
    #    kvmaxWT[idxt]  = kv[ErWTs[idxt,:].argmax()]
    
    # Calcul spectra for 3 field (WT, THLM, RCT)
    # Save along the z axis
    for idxz,zz in enumerate(z):
        for idxnn,nn in enumerate(varRCT):
            # ErRCT  = np.tile(data[nn][tc,:,idxz], (1, 1))
            # ErRCTs = tl.smooth(ErRCT,sigma=1) # not saved
            # if not np.isnan(ErRCTs.max()):
            #     kvmaxRCTz[idxnn,idxt,idxz] = kv[ErRCTs.argmax()]
                
            kvmaxRCTz[idxnn,idxt,idxz],_,_ = findkvmax(kv,data[nn][tc,:,idxz],sigma=sigma)
    
    # Find scale of energy injection
    # Based on the gradient of PI_E
    
    #Ecum  = np.cumsum(PI[0,:])/np.sum(PI[07,:])
    #Ecum = integrate.cumulative_trapezoid(PI[0,:],kv,initial=0)/(kv-kv[0])
    #print(kv,Ecum)
    # Wrong: change by np.trapz??? np.trapzcum?
    
    #eps   = 0.02 #1%
    #kin[idxt]   = kv[np.argmax(Ecum>eps)] # First time higher than eps
    
    grad = np.gradient(PI_E_sum[idxt,:],kv)
    grad = tl.smooth(grad,sigma=2)
    # Need a condition here -> not find a good one !
    #if np.mean(grad)>0:
    kin[idxt]  = kv[np.argmax(grad)] # First time higher than eps
    #print('kin ',kin[idxt], tl.z2k(kin[idxt]), idxt)
    #else:
    #    kin[idxt]  = np.nan 
    
    
    # Plot Figures for each time
    tmp = np.tile(Ers[idxt,:], (1, 1))
    namefig=pathout+'E_PI_'+prefix+'_'+"{:02}".format(tt)
    #print(kv)
    tl.plot_flux(kv,Er,PI=PI,
              kPBL=kPBL,#kin=kin[idxt],
              kcell=kvmaxLWP[idxt],smooth=tmp,\
              y1lab=y1lab,y2lab=y2lab,
              plotlines=plotlines,namefig=namefig)
        
    # Plot Pi for different altitude
    namefig=pathout+'PI_z_'+prefix+'_'+"{:02}".format(tt)
    zplot  = [0.1,0.25,0.5,0.75,0.90,1] 
    Erzall = [data.PI_E[tc,:,tl.near(z,idx*PBL)] for idx in zplot]
    ztmp   = [str(idx) for idx in zplot]
    tl.plot_flux(kv,Erzall,
              kPBL=kPBL[0],#kin=kin[idxt],
              kcell=kvmaxLWP[idxt],logy=False,\
              y1lab=y2lab,labels=ztmp,
              plotlines=plotlines,namefig=namefig)    

        
        
        
    # Plot LWP spectra
    tmp = np.tile(ErLWPs[idxt,:], (1, 1))
    y1labNVAR = f"$E_{{{nvar}}}$"
    namefig=pathout+'Er'+nvar+'_'+prefix+'_'+"{:02}".format(tt)
    tl.plot_flux(kv,ErLWP,kPBL=kPBL,
                 kcell=kvmaxLWP[idxt],
                 smooth=tmp,
                 y1lab=y1labNVAR, plotlines=plotlines,
                 namefig=namefig)
        
    # Plot Spectra and flux from  3D fields
    ktmp  = [data.k3D.values,data.kperp.values,data.kpara.values]
    ktmp  = np.array(ktmp)
    
#    Etmp  = [data.E_k.values,data.E_perp.values,data.E_para.values]
#    Pitmp = [data.Pi_k.values,data.Pi_perp.values,data.Pi_para.values]
    chtmp = ['k','perp','para']
    k3D,E3D,Pi3D = [],[],[]
    for ij,ch in enumerate(chtmp):
        namefig=pathout+'E'+ch+'_PI_'+prefix+'_'+"{:02}".format(tt)
        Etmp,Ptmp   = data['E_'+ch][idxt], data['Pi_'+ch][idxt]

        idxtmp = ~np.isnan(Etmp)
        Ehere  = np.tile(Etmp[idxtmp], (1, 1))
        khere=ktmp[ij][idxtmp] 
        
        # FB: TO BE REMOVED AFTER CLEANING THE spectral_analysis_v2 CODE !
 #       if ch != "k":
 #           Ptmp /= (514*514*39)
            
        Pihere = np.tile(Ptmp[idxtmp], (1, 1))
        tl.plot_flux(khere,Ehere,PI=Pihere,
              kPBL=kPBL,#kin=kin[idxt],
              kcell=kvmaxLWP[idxt],
              y1lab='E'+ch,y2lab='PiE',
              plotlines=plotlines,namefig=namefig)
        k3D  += [khere]
        E3D  += [Ehere[0,:]]
        Pi3D += [Pihere[0,:]]
    
    # Plot Ek, Eperp and Epara on the same figure  
#    Ehere  = [data['E_'+ch][idxt].data[] for ch in chtmp]
#    Pihere = [data['Pi_'+ch][idxt].data for ch in chtmp]
    namefig=pathout+'E3D_'+prefix+'_'+"{:02}".format(tt)
    tl.plot_flux(np.array(k3D),np.array(E3D),PI=np.array(Pi3D),
          kPBL=kPBL,#kin=kin[idxt],
          kcell=kvmaxLWP[idxt],
          y1lab=y1lab, 
          y2lab=y2lab,
          labels=chtmp,normalized=True,
          plotlines=2,namefig=namefig)
    
    # Plot separation Pi_hh, Pi_hv...
    chtmp      = ['Pi_k2','Pi_hh','Pi_hv','Pi_vh','Pi_vv']
    chtmplab   = ['Total','hh','hv','vh','vv']
    colors     = ['b','c','m','m','c']
    linestyles = ['-','--','--',':',':']
    
    Pi3D = []
    for ij,ch in enumerate(chtmp):
        Pitmp  = data[ch][idxt]
        idxtmp = ~np.isnan(Pitmp)
        Pi3D  += [Pitmp[idxtmp]]
    #kbins = np.array(data['kbins'])
        
    namefig=pathout+'E3Dhv_'+prefix+'_'+"{:02}".format(tt)
    tl.plot_flux(kbins[idxtmp],Pi3D,
          kPBL=kPBL,
          kcell=kvmaxLWP[idxt],logy=False,
          y1lab=y2lab,
          labels=chtmplab,normalized=True,
          colors=colors,linestyles=linestyles,
          plotlines=plotlines,namefig=namefig)
      
    
    # Plot E1dr_WT, E1dr_THL, E1dr_RCT
    for ij,ch in enumerate(varRCT):
        Ehere  = ErRCT_sum[ij,idxt,:]
        Ehere  = np.tile(Ehere, (1, 1))
        namefig=pathout+ch+'_'+prefix+'_'+"{:02}".format(tt)
        tl.plot_flux(kv,Ehere,
                     kPBL=kPBL,#kin=kin[idxt],
                     kcell=kvmaxLWP[idxt],
                     y1lab=ch,
                     plotlines=plotlines,namefig=namefig)
        
        
    
    
        
# Test contourf
# Plot Pi as function of (k,z) as a contour (log scale in k)
plt.figure(figsize=(8,5))
plt.pcolormesh(kv, z, data.PI_E[0,:,:].T, shading='auto')  # Pi shape (Nz, n_bins)
plt.xscale('log')
plt.colorbar(label=r'$\Pi(k_\perp,z)$')
plt.xlabel(r'$k_\perp$')
plt.ylabel('z index')
plt.title('Spectral flux vs height')
plt.show()

ErLWP  = data[varLWP].data[indices]
timecut = time[indices]


# Plot Spectra decomposition
normalized=True
xline=0
if normalized:
    xline = 1
else:
#    zmax = np.mean(data.PBL[indices].values)
    if kPBL is not None :
        xline = kPBL

idxzi    = tl.near(z,np.max(data.PBL.values))
idxzimax = min(idxzi+15,len(z)-1)
idxmax   = np.arange(1,idxzimax)
zloop    = z[idxmax]

#idx0    = tl.near(zmax,z).values+15
#idxmax  = np.arange(1,idx0+1)
#zloop  = z[idxmax]
k1d_grid, z_grid = np.meshgrid(kv,zloop)

xsize=(14,10);fts=20;lw=2.5
# Pi 2D
minmax = 1e-3
levelsPI = [-minmax,minmax,21]
levelsgrad =  [-minmax*100,minmax*100,21]
for idxt,tt in enumerate(timecut):
    PBL = data.PBL[idxt]
    namefig0=pathout+'NAME_'+prefix+'_'+"{:02}".format(tt)+'.png'
    
    dataplot = {}
    PI_E  = data.PI_E[idxt,:,idxmax].transpose()
    dataplot['PIE2D']=PI_E
    PI_Ew = data.PI_Ew[idxt,:,idxmax].transpose()
    dataplot['PIEw2D']=PI_Ew
    grad,gradw = [np.zeros(PI_E.shape) for ij in range(2)]
    for idxz,zz in enumerate(zloop):
        grad[idxz,:]  = -1*np.gradient(PI_E[idxz,:],kv)
        gradw[idxz,:] = -1*np.gradient(PI_Ew[idxz,:],kv)
    dataplot['GradPi_E']=grad
    dataplot['GradPi_Ew']=gradw
    
    for key in dataplot.keys():
        levels= levelsPI
        if 'Grad' in key:
            levels= levelsgrad
        namefig=namefig0.replace('NAME',key)
        tl.plot_spectraTKE2D(dataplot[key],k1d_grid,z_grid,
                             index=key,levels0=levels,
                             xline=xline,yline=PBL,xline2=kvmaxLWP[idxt],
                             namefig=namefig,lw=lw)
        
    
    # datatmp = data.PI_E[idxt,:,idxmax].transpose()
    # namefig=pathout+'E_PI2D_'+prefix+'_'+"{:02}".format(tt)+'.png'
    # tl.plot_spectraTKE2D(datatmp,k1d_grid,z_grid,
    #               index='PI_E',levels0=levels,
    #               xline=xline,yline=PBL,xline2=kvmaxLWP[idxt],
    #               namefig=namefig,lw=lw)
    
    # namefig=pathout+'GradPiE_'+prefix+'_'+"{:02}".format(tt)+'.png'
    # tl.plot_spectraTKE2D(grad,k1d_grid,z_grid,
    #                       index='GradPi_E',
    #                       xline=xline,yline=PBL,xline2=kvmaxLWP[idxt],
    #                       namefig=namefig,lw=lw)



# Comparing variance and integral spectra
#var_sp = np.trapz(ErLWP,kv)
#print(var_sp/data.var_LWP)
#plt.scatter(var_sp,data.var_LWP);plt.show()


# increasing variance (spectra (t-1) - spectra (t))
#diff_ErLWP=[]
#diff_ErLWP+= [ErLWPs[tt,:]-ErLWPs[tt-1,:] for tt in np.arange(len(indices))[1::]]
#diff_ErLWP = np.array(diff_ErLWP)

#dE/dt
diff_ErLWP = np.diff(ErLWPs,axis=0)/np.diff(timecut)[:, None]*np.ones(nkv)
diff_Er    = np.diff(Ers,axis=0)/np.diff(timecut)[:, None]*np.ones(nkv)


# Plot TKE Figures for all time
kPBLall = tl.z2k(data.PBL).data[indices]

Eall = {}
Eall['v1']=[kv,Erad_sum]
Eall['v2']=[kcenter,Eradv2_sum]

namefig=pathout+'E'+'v1'+'_PI_'+prefix+'_All'
tl.plot_flux(Eall['v1'][0],Eall['v1'][1],PI=PI_E_sum,
          kPBL=kPBLall,
          y1lab='E_'+key,y2lab='PiE',
          labels=labels,
          plotlines=plotlines,namefig=namefig)

for key in Eall:
    namefig=pathout+'E'+key+'_'+prefix+'_All'
    tl.plot_flux(Eall[key][0],Eall[key][1],kPBL=kPBLall,
             labels=labels,
          y1lab='E_'+key,plotlines=plotlines,namefig=namefig)

namefig=pathout+'PI_'+prefix+'_All'
namey=r'$\PI_E$'
tl.plot_flux(kv,PI_E_sum,kPBL=kPBLall,
          y1lab='PI_E',y2lab='PiE',
          labels=labels,
          plotlines=plotlines,namefig=namefig, logy=False)

# LWP
namefig=pathout+'Er'+nvar+'_'+prefix+'_All'
tl.plot_flux(kv,ErLWP,kPBL=kPBLall,
             labels=labels,showobs=True, # Can be changed
          y1lab=y1labNVAR,plotlines=plotlines,namefig=namefig)
    
    
# Plot increasing variance
# only for several hours if time is too long
idxtime = np.arange(nt-1)
if nt>8:
    idxtime = idxtime[::3]

namefig=pathout+'Er'+'_'+prefix+'_DIffTime'
tl.plot_flux(kv,diff_Er[idxtime],kPBL=kPBLall[1::][idxtime],\
          y1lab='E(t) - E(t-1)',plotlines=False,
          labels=labels[idxtime],
          namefig=namefig, logy=False)

namefig=pathout+'Er'+nvar+'_'+prefix+'_DIffTime'
tl.plot_flux(kv,diff_ErLWP[idxtime],kPBL=kPBLall[1::][idxtime],\
          y1lab='E(t) - E(t-1)',plotlines=False,
          labels=labels[idxtime],
          namefig=namefig, logy=False)

# Plot temporal evolution of aspect ratio
namefig=pathout+'aspect_ratio_'+prefix
Gamma = {}
Gamma[nvar]  =kPBLall/kvmaxLWP #(2pi/kvmax)/(2pi/kPBL) 
Gamma['TKE'] =kPBLall/kvmax 
Gamma['TKEh']  =kPBLall/kvmaxRCT[varRCT.index('E1dr_TKEh'),:]
Gamma['TKEv']  =kPBLall/kvmaxRCT[varRCT.index('E1dr_TKEv'),:]
#Gamma['WT']  =kPBLall/kvmaxRCT[varRCT.index('E1dr_WT'),:]
Gamma['THL'] =kPBLall/kvmaxRCT[varRCT.index('E1dr_THL'),:] 
#Gamma['RCT'] =kPBLall/kvmaxRCT[varRCT.index('E1dr_RCT'),:] 
maxh = np.max([Gamma[ij] for ij in Gamma.keys()])
ylim = [0,math.ceil(maxh / 5) * 5]

LambdaEpsIn = kPBLall/kin
lambdaIn = {}
lambdaIn['LambdaEpsIn']=LambdaEpsIn
tl.plot_time(timecut,Gamma,
#             lambdaIn=LambdaEpsIn,
             ylim=ylim,marker='o',
             day=day_hours[indices],
             namex='Aspect Ratio (-)',
             namefig=namefig)

# Plot temporal evolution of aspect ratio LWP
namefig=pathout+'aspect_ratioLWP_'+prefix
Gamma = {}
Gamma[nvar]= kPBLall/kvmaxLWP #(2pi/kvmax)/(2pi/kPBL) 
tl.plot_time(timecut,Gamma,
             day=day_hours[indices],
             ylim=ylim,marker='o',
             namex='Aspect Ratio (-)',
             namefig=namefig)

namefig=pathout+'aspect_ratioTKE_'+prefix
Gamma = {}
Gamma[nvar]= kPBLall/kvmaxLWP #(2pi/kvmax)/(2pi/kPBL) 
Gamma['TKE'] =kPBLall/kvmax #(2pi/kvmax)/(2pi/kPBL) 
tl.plot_time(timecut,Gamma,
             #ylim=ylim,
             namex='Aspect Ratio (-)',
             namefig=namefig)

# Plot temporal evolution of aspect ratio
namefig=pathout+'Cell_Size_'+prefix
Gamma = {}
Gamma[nvar]= 2*np.pi/kvmaxLWP/1000. # in km
Gamma['TKE'] =2*np.pi/kvmax/1000. # in km
LambdaEpsIn = 2*np.pi/kin
tl.plot_time(timecut,Gamma,
             namex='Cell Size (km)',
             namefig=namefig)

# Plot temporal evolution of length scale of epsilon_in
namefig=pathout+'lambdaIn_'+prefix
title=r'Relative scale of energy injection $\epsilon_{in}$'
tl.plot_time(timecut,lambdaIn,
             namex=r'$\lambda_{in}$ /$\lambda_{PBL}$  (-)',
             title=title,
             namefig=namefig)

# Plot temporal evolution of PBL top
PBLall = {}
PBLall['PBL']= data.PBL[indices].values
namefig=pathout+'PBLtop_'+prefix
title='Evolution of the PBL height'
tl.plot_time(timecut,PBLall,
             namex='PBL height (km)',
             title=title,
             namefig=namefig)


# Plot Variance
namefig=pathout+'variance_'+prefix
Gamma = {}
Gamma['Var TKE'] =  var_sum
Gamma['Var '+nvar]= data['var_'+nvar][indices]*10
lambdaIn['LambdaEpsIn']=LambdaEpsIn
tl.plot_time(timecut,Gamma,
             namex='Variance',
             namefig=namefig)

# Asepct ratio of liquid Water Content (WT,THLM,RCT)
GammaRCT=[np.tile(kPBLall, (nz,1)).T/kvmaxRCTz[ij,:,:] for ij in range(nv)]
# A sauvegarder : plt.contourf(GammaRCT[2].T);plt.contour(GammaRCT[5].T,levels=[25,35],linestyles='--',colors='k')

# Test Emma figure
var = "spectral_slope_binned_"+nvar
x = data[var][indices]
y = np.max(ErLWPs,axis=1)

# Create scatter plot
plt.scatter(x, y, color='blue')

# Annotate each point with its corresponding number
for i, (xi, yi) in enumerate(zip(x, y), start=1):
    plt.text(xi+0.02, yi+0.02, str(i), fontsize=8, ha='center', va='center', color='red')
#             bbox=dict(facecolor='red', edgecolor='black', boxstyle='circle,pad=0.3'))

# Labels and title
plt.xlabel("Slope")
plt.ylabel("Max")
plt.title("Scatter Plot with Data Numbers")
plt.show()


# Plot Pi 2D
# namefig0=pathout+'III_2D_'+prefix+'_All'
# for index in data:
#     print('index ',index)
#     namefig=namefig0.replace('III',index)
#     datatmp = data[index]['data'][indices,:]
#     tl.plot_spectraTKE2D(datatmp,k1d_grid,z_grid,
#                       index=index,
#                       xline=xline,yline=PBLheight,xline2=kvmaxLWP,
#                       namefig=namefig,lw=lw)









