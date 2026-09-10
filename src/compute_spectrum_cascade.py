#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:04:32 2026


# Output variables
# 

Date: 16 July 2026
@author: fbrient
"""

import numpy as np
import netCDF4 as nc
import tools as tl
import spectratools as stl
#import cloudmetrics as cm
import time
import xarray as xr
import sys

# Test on local file (by default: False)
testlocal= True
# Run Filtered cascade (by default: True)
Filter3D = True
    
if testlocal:
    file = 'FIRZ4.1.V0001.OUT.003.nc' #'IHOP0.1.NWV01.OUT.013.nc' #'FIRZ4.1.V0001.OUT.003.nc'
    fileinfo  = '../infos/info_run_Dell_FIRZ4.txt' #'../infos/info_run_Dell_IHOPNW.txt' #../infos/info_run_Dell_FIRZ4.txt'
    Filter3D = False
else:        
    file = sys.argv[1] # name of the file
    # Open information from 'info_run_JZ.txt'
    pathinfo  = '../infos/'
    fileinfo  = pathinfo+'info_run.txt'

# Find path of file, create save path
path,pathsave = tl.findpath(fileinfo)
file=path+file

# Names of vertical axis
var1D = ['level','nj','ni'] #Z,Y,X
    
print('*****')
print('Start analysis of ',file)
print('*****')

# Open the netcdf file
DATA    = nc.Dataset(file,'r')

# OPEN DATA (only here)
variables = ['UT','VT','WT']
UT,VT,WT = [np.squeeze(DATA[var]) for var in variables]
variables = ['THLM','RNPM']
THLM,RNPM = [tl.createnew(var,DATA,var1D) for var in variables]

# Open dimensions
#nxnynz,data1D,dx,dy,dz= tl.dimensions(DATA,var1D)
z,y,x  = [DATA[ij][:] for ij in var1D]
nxnynz,dz,dy,dx= tl.dimensions(DATA,var1D)

# Delete DATA to save memory
del DATA

# Clean data (remove boundaries)
UT,VT,WT,THLM,RNPM = [tl.removebounds(tmp) for tmp in [UT,VT,WT,THLM,RNPM]]
x,y,z,dz     = [tl.removebounds(tmp) for tmp in [x,y,z,dz]]

# Find Boundary-layer height (zi)
inv       = 'THLM'
threshold = 0.75
idxzi     = tl.findpbltop(THLM,z,offset=threshold)
PBLheight = z[idxzi] # km
kPBL      = tl.z2k(PBLheight) #rad/km
print('PBL Height: ',PBLheight)

# Select maximum altitude
#zmax   = z.max()
idxplus= 10 # 10 layers above zi
zmax   = min(z[idxzi+idxplus],z.max())
idxmax = (z>0) & (z<zmax)
zpbl   = z[idxmax]


# Remove layers not needed
UT,VT,WT,THLM,RNPM = [tmp[idxmax,:,:] for tmp in [UT,VT,WT,THLM,RNPM]]

# Substract horizontal mean
THLM = tl.anomcalc(THLM)
RNPM = tl.anomcalc(RNPM)
anomHor = True
if anomHor:
    UT = tl.anomcalc(UT)
    VT = tl.anomcalc(VT)
    WT = tl.anomcalc(WT)

# Calculate vertical weights
weights  = np.diff(zpbl)
weights  = np.insert(weights, 0, weights[0])  # Assume first weight equals first diff


# Need linear grid on the z-axis
dz_new = int(max(10,np.max(np.diff(zpbl))))

# Interpolate the verical axis
table = (UT, VT, WT, THLM, RNPM)
table_new, z_new = tl.interp_to_uniform_z(table, zpbl, dz_new=dz_new)
(UT_new, VT_new, WT_new, THLM_new, RNPM_new) = table_new
nx,ny,nz = len(x),len(y),len(z_new)
print(nx,ny,nz,UT_new.shape)



nbins = 100

################################################
#    Calculate 3D spectra flux and cascade     #
################################################

# Compute spectra and cascade from 3D fields
result = stl.compute_spectral_transfer(
    UT_new, VT_new, WT_new, 
    dx=dx, dy=dy, dz=dz_new,
    binning='log',nbins=nbins)

# For information, Egality is:
# Pi = PI_3d*(nx*ny*nznew)

# Check equality between the spectra energy and the field
k,kc,E =  result['k'],result['k_shell_centers'],result['E']
E_k3_mean = E/np.diff(k)/(nx*ny*nz)
TKE3D  = 0.5*(pow(tl.anomcalc(UT_new),2.)
            +pow(tl.anomcalc(VT_new),2.)\
            +pow(tl.anomcalc(WT_new),2.))
tl.checkvariance(kc,E_k3_mean,TKE3D,type='mean')

# For 3D filtering
if Filter3D:
    idxzlist   = np.arange(0,1.2,0.1)
    idxzfilter = [tl.near(z_new, ij*PBLheight) for ij in idxzlist]
    zfilter    = z_new[idxzfilter]

    Nk,Nk2,Nzz = len(k),len(kbins),len(idxzlist)
    Ek_BT,Tk_BT,Pi_BT = [np.zeros((Nk,Nzz)) for ij in range(3)]
    Pi2_BT = [np.zeros((Nk2,Nzz)) for ij in range(5)]
    Ek_TB,Tk_TB,Pi_TB = [np.zeros((Nk,Nzz)) for ij in range(3)]
    Pi2_TB = [np.zeros((Nk2,Nzz)) for ij in range(5)]

    for ij,cutoff in enumerate(idxzfilter):
        Uh,Vh,Wh    = UT_new.copy(),VT_new.copy(),WT_new.copy()
        Uh2,Vh2,Wh2 = UT_new.copy(),VT_new.copy(),WT_new.copy()

        # zero out above 75% along z
        Uh[cutoff:, :, :] = 0
        Vh[cutoff:, :, :] = 0
        Wh[cutoff:, :, :] = 0
        
        # zero below z0 (ij)
        Uh2[:cutoff, :, :] = 0
        Vh2[:cutoff, :, :] = 0
        Wh2[:cutoff, :, :] = 0
    
        # Compute spectra and cascade from truncated 3D fields
        resultTMP = stl.compute_spectral_transfer(
            (Uh,Vh,Wh),
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins)
        
        
        # Specific selection u<_a * (u_b * grad(u_c))
        # In classic case a=b=c. 
        # Cross terms are when at lest one is different
        resultTMP = stl.compute_spectral_transfer(
            (Uh, Vh, Wh),
            cross_b=(Uh, Vh, Wh),
            cross_c=(Uh2, Vh2, Wh2),
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins)


################################################
#    Calculate 2D spectra flux and cascade     #
################################################




# Saving files






