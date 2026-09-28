#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:54:52 2026

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
pathin   = "../data/"
prefix   = "FIRZ4" #"IHOP" #"FIR1k"
filein0  = pathin+'Cascade_'+prefix+'*XXX.nc'

# Dir for figures
pathout = "../figures/"
pathout+= prefix+'/'
tl.mkdir(pathout)

# Variables
varall  = ['TKE','THLM_new','RNPM_new','RCT_new','PABST_new','buoyancy_new']
data    = {}
for var in varall:
    filein    = filein0.replace('XXX',var)
    data[var] = tl.read_netcdfs(filein, dim='time')
    
# Save shared values
kv    = data['TKE'].kv.values
kperp = data['TKE'].kperp.values
kpara = data['TKE'].kpara.values
z     = data['TKE'].z.values
k2D   = data['TKE'].k2D.values

# Compute altitude differences
dz = np.diff(z)  # Differences between consecutive altitudes
dz = np.insert(dz, 0, dz[0])  # Assume first weight equals first diff


kall = {}
kall[''],kall['perp'],kall['para'] = kv,kperp,kpara

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

# TIme of the 3D fields
time     = data['TKE'].time.values # ERROR


timemax  = 100
if prefix == "FIZ4c":
    timemax=24
idxtmax  = min(time.max(),timemax)
cond     = (time >= 0) & (time <= idxtmax)    
indices  = np.where(cond)[0]

# Caption axis
y1lab,y2lab= r'E $(m^3/s^2)$',r'$\Pi (m^2/s^3)$'

# Plot k -5/3 slope
plotlines=2
# Smooth to find kmax
sigma = 1

# Initialize tables
nt,nkv    =np.shape(data['TKE'].E[indices])
_,nz,nkv2 =np.shape(data['TKE'].E2D[indices])
kvmax3D, kvmax2D, Es3D, Es2D = {},{},{},{}
for var in varall:
    kvmax3D[var]=np.zeros(nt)
    Es3D[var]   =np.zeros((nt,nkv))
    kvmax2D[var]=np.zeros(nt)
    Es2D[var]   =np.zeros((nt,nkv2))



print(indices)
for idxt,tc in enumerate(indices):
    tt  = time[tc]
    print(idxt,tc,tt)
    tst = "{:02}".format(tt)
    
    PBL     = data['TKE'].PBL[tc].values
    kPBL    = tl.z2k(PBL) # rad/m
    kPBL = np.tile(kPBL, (1, 1))
    idxpbl  = tl.near(z,PBL)

    
    for var in varall:
        
        # New calcul, more simple
        Etmp = data[var]['E_spec'][idxt]
        print(kv.shape,Etmp.shape)
        kvmax3D[var][idxt],Es3D[var][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)
        Etmp = data[var]['E2D'][idxt]
        Etmp = Etmp/np.insert(np.diff(k2D), 0, np.nan)
        kvmax2D[var][idxt],Es2D[var][idxt,1:],_  = tl.findkvmax(k2D,Etmp.T,sigma=sigma,weights=dz)


        
        # Compute E, PI for all, perp and para
        chtmp = ['','perp','para']
        k3D,E3D,Pi3D = [],[],[]
        for ij,ch in enumerate(chtmp):
            namefig     = 'E'+ch+'_PI_'+var+'_'+prefix+'_'+tst
            namefig     = pathout+namefig
            Etmp,Ptmp   = data[var]['E'+ch][idxt], data[var]['Pi'+ch][idxt]
            
            idxtmp = ~np.isnan(Etmp)
            Ehere  = np.tile(Etmp[idxtmp], (1, 1))
            khere  = kall[ch]
            Pihere = np.tile(Ptmp[idxtmp], (1, 1))
            
            # Divide by diff k (should be removed)
            tmp   = np.insert(np.diff(khere), 0, np.nan)
            Ehere = Ehere/tmp
            
            tl.plot_flux(khere,Ehere,PI=Pihere,
                  kPBL=kPBL,#kin=kin[idxt],
    #              kcell=kvmaxLWP[idxt],
                  y1lab='E'+ch,y2lab='PiE',
                  plotlines=plotlines,namefig=namefig)
            k3D  += [khere]
            E3D  += [Ehere[0,:]]
            Pi3D += [Pihere[0,:]]
        
        # Plot Ek, Eperp and Epara on the same figure  
        namefig=pathout+'E3D_'+var+'_'+prefix+'_'+tst
        tl.plot_flux(np.array(k3D),np.array(E3D),PI=np.array(Pi3D),
              kPBL=kPBL,#kin=kin[idxt],
    #          kcell=kvmaxLWP[idxt],
              y1lab=y1lab, 
              y2lab=y2lab,
              labels=chtmp,normalized=True,
              plotlines=2,namefig=namefig)
