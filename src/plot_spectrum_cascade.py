#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:54:52 2026

@author: fbrient
"""

import numpy as np
import tools as tl
import pylab as plt
import glob
from netCDF4 import num2date
import datetime
import math
import sys
import os
import socket
from pathlib import Path

# On Jean Zay
#def on_jean_zay() -> bool:
#    host = socket.getfqdn().lower()
#    return (
#        "jean-zay" in host
#        or "idris" in host
#        or "IDRIS_DEVICE" in os.environ    # variable posée par l'IDRIS
#        or "SCRATCH" in os.environ and "WORK" in os.environ
#    )

def on_jean_zay() -> bool:
    return (
        os.environ.get("MY_MACHINE") == "jeanzay"
        or str(Path.home()).startswith("/linkhome")
        or Path("/lustre/fswork").exists()
    )

ON_JZ = on_jean_zay()
print(ON_JZ)
#stop
if ON_JZ:
    pathsrc= '/lustre/fswork/projects/rech/whl/rces071/Github/cascade/' 
    path1D = '/lustre/fsstor/projects/rech/whl/rces071/MNH-V5-7-0/'
else:
    pathsrc= '/home/fbrient/GitHub/cascade/'
    path1D = '/home/fbrient/MNH/V5-7-0/'


# Open all netcdf file
prefix = sys.argv[1] # name of the prefix (FIRZ4)

pathin   = pathsrc+"data/"
#prefix   = "FIRZ4" #"IHOP" #"FIR1k"
filein0  = pathin+'Cascade_'+prefix+'*XXX.nc'

# Dir for figures
pathout = pathsrc+"figures/"
pathout+= prefix+'/'
tl.mkdir(pathout)

# Variables
varall  = ['TKE','THLM','RNPM','RCT','PABST','buoyancy']
data    = {}
for var in varall:
    filein    = filein0.replace('XXX',var)
    data[var] = tl.read_netcdfs(filein, dim='time',concat=False)
    
############ Open 1D to find time of sunlight
path1D+='*/'+prefix+'/'
file0 = path1D+'*'+'.1.'+'*'+'.000'+'*'+'.nc'
# List of files with glob
files = sorted(glob.glob(file0))
print('files ',files)
timing2 = []
datarad = []
for idx,file in enumerate(files):
    # Open data from files
    print('file : ',file)
    data1D = tl.nc_dataset_list(file)
    
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

# Find day and night
DTHRADSW = datarad
DTHRADSW = np.concatenate(DTHRADSW, axis=0)
DTHRADSW = np.max(DTHRADSW,axis=1)
epsilon = 1e-10
day = (DTHRADSW>epsilon)
print(day)
############ 
    

# TIme of 3 fields (in seconds - UTC?)
time3D = [datatmp['time'].values[0] for datatmp in data['TKE']]
time_datetimes3D = num2date(time3D, units=units, calendar=calendar)
# Reference date is just the first
ref_date = time_datetimes3D[0]- datetime.timedelta(seconds=int(time3D[0]))
time_datetimes3D = [ref_date +
                  datetime.timedelta(seconds=int(s)) for s in time3D[:]]
# Convert to hours since start
time_hours3D = [(t - time_datetimes3D[0]).total_seconds() /
              3600 for t in time_datetimes3D]
time_hours3D = np.array(time_hours3D)    
time_hours3D = time_hours3D+1 # because 001 is the saving 1 hour after the start
nt         = len(time_hours3D)

# day hours
day_hours = day[[tl.near(time_hours3D[ij],time_hours) for ij in range(nt)]]

# Caption axis
y1lab,y2lab= r'E $(m^3/s^2)$',r'$\Pi (m^2/s^3)$'

# Plot k -5/3 slope
plotlines=2
# Smooth to find kmax
sigma = 1
# For 2D plot
zplot    = [0,0.1,0.25,0.5,0.75,0.90,1] 
zplotstr = [str(idx) for idx in zplot]

# Initialize variables for global plot
kmaxLWP = np.zeros(nt)
kPBLall = np.zeros(nt)

# Plotall
plotall = False


for idxt,tc in enumerate(time_hours3D):
    
    # Which time?
    print(idxt,tc)
    tst = "{:02.0f}".format(tc)

    # Save shared values
    kv    = data['TKE'][idxt].kv.values
    kperp = data['TKE'][idxt].kperp.values
    kpara = data['TKE'][idxt].kpara.values
    z     = data['TKE'][idxt].z.values
    k2D   = data['TKE'][idxt].k2D.values
    kLWP  = data['TKE'][idxt].kLWP.values
    nk,nk2= len(kv),len(k2D)
    nz    = len(z)


    # Compute altitude differences
    dz = np.diff(z)  # Differences between consecutive altitudes
    dz = np.insert(dz, 0, dz[0])  # Assume first weight equals first diff


    kall = {}
    kall[''],kall['perp'],kall['para'] = kv,kperp,kpara


# TIme of the 3D fields
#time     = data['TKE'].time.values # ERROR


#timemax  = 100*3600
#if prefix == "FIZ4c":
#    timemax=24
#idxtmax  = min(time.max(),timemax)
#cond     = (time >= 0) & (time <= idxtmax)    
#indices  = np.where(cond)[0]

    # Initialize tables
#    if idxt==0:
#        nt,nk    =np.shape(data['TKE'][idxt].E[indices])
#        _,nz,nk2 =np.shape(data['TKE'].E2D[indices])
    
    if idxt==0:
        #EsLWP   = np.zeros((nt,nk))
        kmax3D, kmax2D, Es3D, Es2D = {},{},{},{}
        kmax2Dz ={}
        for var in varall:
            kmax3D[var] =np.zeros(nt)*np.nan
            Es3D[var]   =np.zeros((nt,nk))*np.nan
            kmax2D[var] =np.zeros(nt)*np.nan
            Es2D[var]   =np.zeros((nt,nk2))*np.nan
            # nz+5 for safety
            kmax2Dz[var]=np.zeros((nt,nz+5))*np.nan
        for var in ['TKEh','TKEv']:
            kmax3D[var]=np.zeros(nt)*np.nan
            Es3D[var]  =np.zeros((nt,nk))*np.nan

#for idxt,tc in enumerate(indices):
    
    PBL     = data['TKE'][idxt].PBL.values
    kPBL    = tl.z2k(PBL) # rad/m
    kPBLall[idxt] = kPBL
    kPBL    = np.tile(kPBL, (1, 1))
    idxpbl  = tl.near(z,PBL)

    # Find LWP cell
    ELWP = data['TKE'][idxt]['ELWP']
#    cond = ~np.isnan(ELWP)
#    ELWP = ELWP[cond]
#    kLWPh= kLWP[cond] # to modify
    kmaxLWP[idxt],EsLWP,_ = tl.findkvmax(kLWP,ELWP,sigma=sigma)
    
    for var in varall:
        
        # Find cell size
        Etmp = data[var][idxt]['E_spec']
        kmax3D[var][idxt],Es3D[var][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)
        Etmp = data[var][idxt]['E2D']
        kmax2D[var][idxt],Es2D[var][idxt,1:],_ = tl.findkvmax(k2D,Etmp.T,sigma=sigma,weights=dz)
        for idxz,zz in enumerate(z):
            kmax2Dz[var][idxt,idxz],_,_        = tl.findkvmax(k2D,Etmp[idxz,:],sigma=sigma)

        # Calculate cell size for TKEv and TKEh
        if var=="TKE":
            Etmp = data[var][idxt]['E_h_spec']
            kmax3D['TKEh'][idxt],Es3D['TKEh'][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)
            Etmp = data[var][idxt]['E_v_spec']
            kmax3D['TKEv'][idxt],Es3D['TKEv'][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)


        if plotall:            
            # Plot Pi for different altitude
            namefig=pathout+'PI_z_'+var+'_'+prefix+'_'+tst
            Erzall = [data[var][idxt]['Pi2D'][tl.near(z,idx*PBL),:] for idx in zplot]
            tl.plot_flux(k2D,Erzall,
                      kPBL=kPBL,#kin=kin[idxt],
                      kcell=kmaxLWP[idxt],
                      logy=False,\
                      y1lab=y2lab,labels=zplotstr,
                      plotlines=plotlines,namefig=namefig)    
                
            
            # Compute E, PI for all, perp and para
            chtmp = ['','perp','para']
            k3D,E3D,Pi3D = [],[],[]
            for ij,ch in enumerate(chtmp):
                namefig     = 'E'+ch+'_PI_'+var+'_'+prefix+'_'+tst
                namefig     = pathout+namefig
                Etmp,Ptmp   = data[var][idxt]['E'+ch], data[var][idxt]['Pi'+ch]
                
                idxtmp = ~np.isnan(Etmp)
                Ehere  = np.tile(Etmp[idxtmp], (1, 1))
                khere  = kall[ch]
                Pihere = np.tile(Ptmp[idxtmp], (1, 1))
                
                # Divide by diff k (should be removed)
                tmp   = np.insert(np.diff(khere), 0, np.nan)
                Ehere = Ehere/tmp
                
                tl.plot_flux(khere,Ehere,PI=Pihere,
                      kPBL=kPBL,#kin=kin[idxt],
                      kcell=kmaxLWP[idxt],
                      y1lab='E'+ch,y2lab='PiE',
                      plotlines=plotlines,namefig=namefig)
                k3D  += [khere]
                E3D  += [Ehere[0,:]]
                Pi3D += [Pihere[0,:]]
            
            # Plot Ek, Eperp and Epara on the same figure  
            namefig=pathout+'E3D_'+var+'_'+prefix+'_'+tst
            #print(k3D)
            #print(k3D.shape)
            tl.plot_flux(#k3D,E3D,PI=Pi3D,
                  np.array(k3D,dtype=object),np.array(E3D,dtype=object),PI=np.array(Pi3D,dtype=object),
                  kPBL=kPBL,#kin=kin[idxt],
                  kcell=kmaxLWP[idxt],
                  y1lab=y1lab, 
                  y2lab=y2lab,
                  labels=chtmp,normalized=True,
                  plotlines=2,namefig=namefig)
            
            # Plot separation Pi_hh, Pi_hv...
            chtmp      = ['PI_k','PI_hh','PI_hv','PI_vh','PI_vv']
            chtmplab   = ['Total','hh','hv','vh','vv']
            colors     = ['b','c','m','m','c']
            linestyles = ['-','--','--',':',':']
            
            Pi3D = []
            for ij,ch in enumerate(chtmp):
                Pitmp  = data[var][idxt][ch]
    #            idxtmp = ~np.isnan(Pitmp)
                Pi3D  += [Pitmp] #[idxtmp]]
            #kbins = np.array(data['kbins'])
            
                
            namefig=pathout+'E3Dhv_'+var+'_'+prefix+'_'+tst
            tl.plot_flux(kv,Pi3D,
                  kPBL=kPBL,
                  kcell=kmaxLWP[idxt],
                  logy=False,
                  y1lab=y2lab,
                  labels=chtmplab,normalized=True,
                  colors=colors,linestyles=linestyles,
                  plotlines=plotlines,namefig=namefig)
        
# Plot all LWP spectra

namefig=pathout+'ELWP_'+var+'_'+prefix+'_'+tst
ELWPall = [data['TKE'][ij]['ELWP'].values for ij in range(nt)]
tl.plot_flux(kLWP,ELWPall,
      kPBL=kPBLall,
#      kcell=kmaxLWP,
      y1lab=y2lab,
      normalized=True,
      plotlines=2,namefig=namefig)



# Plot temporal evolution of aspect ratio
for cell in ["Cell_Size","Aspect_Ratio"]:
    for typ in ["2D","3D"]:
        namefig=pathout+cell+'_v'+typ+'_'+prefix
        kmaxplot=kmax3D
        if typ=='2D':
            kmaxplot=kmax2D
        Gamma = {}
        for var in kmaxplot.keys():
            if cell== "Cell_Size":
                Gamma[var]   = 2*np.pi/kmaxplot[var] 
                Gamma['LWP'] = 2*np.pi/kmaxLWP
            else:
                Gamma[var]   = kPBLall/kmaxplot[var] #(2pi/kvmax)/(2pi/kPBL) 
                Gamma['LWP'] = kPBLall/kmaxLWP
            
        maxh = np.max([Gamma[ij] for ij in Gamma.keys()])
        ylim = [0,math.ceil(maxh / 5) * 5]
        
        #LambdaEpsIn = kPBLall/kin
        #lambdaIn = {}
        #lambdaIn['LambdaEpsIn']=LambdaEpsIn
        tl.plot_time(time_hours3D,Gamma,
                     ylim=ylim,marker='o',
                     day=day_hours,
                     namex=''.join(cell.split('_')),
                     namefig=namefig)
