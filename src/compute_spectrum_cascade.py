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
import coarse_graining_flux as cgf
import pylab as plt
from coarse_graining_flux import compute_Pi_2D_map


# Test on local file (by default: False)
testlocal= True
# Run Filtered cascade (by default: True)
Filter3D = True
    
if testlocal:
    file = 'FIRZ4.1.V0001.OUT.003.nc' #'IHOP0.1.NWV01.OUT.013.nc' #'FIRZ4.1.V0001.OUT.003.nc'
    fileinfo  = '../infos/info_run_Dell_FIRZ4.txt' #'../infos/info_run_Dell_IHOPNW.txt' #../infos/info_run_Dell_FIRZ4.txt'
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
variables = ['UT','VT','WT','PABST']
UT,VT,WT,PABST = [np.squeeze(DATA[var]) for var in variables]
variables = ['THLM','RNPM','RVT','RCT']
THLM,RNPM,RVT,RCT = [tl.createnew(var,DATA,var1D) for var in variables]

# Open dimensions
#nxnynz,data1D,dx,dy,dz= tl.dimensions(DATA,var1D)
z,y,x  = [DATA[ij][:] for ij in var1D]
nxnynz,dz,dy,dx= tl.dimensions(DATA,var1D)

# Delete DATA to save memory
del DATA

# Clean data (remove boundaries)
UT,VT,WT,PABST    = [tl.removebounds(tmp) for tmp in [UT,VT,WT,PABST]]
THLM,RNPM,RVT,RCT = [tl.removebounds(tmp) for tmp in [THLM,RNPM,RVT,RCT]]
x,y,z,dz          = [tl.removebounds(tmp) for tmp in [x,y,z,dz]]

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
UT,VT,WT,PABST    = [tmp[idxmax,:,:] for tmp in [UT,VT,WT,PABST]]
THLM,RNPM,RVT,RCT = [tmp[idxmax,:,:] for tmp in [THLM,RNPM,RVT,RCT]]
# New variables : Buoyancy
buoyancy = stl.compute_buoyancy_from_thlm(THLM, RVT, RNPM, PABST)       # (nz, ny, nx)



# Substract horizontal mean
THLM,RNPM,RVT,RCT,PABST =  [tl.anomcalc(tmp) for tmp in [THLM,RNPM,RVT,RCT,PABST]]
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
table = (UT, VT, WT, THLM, RNPM, RVT, RCT, PABST, buoyancy)
table_new, z_new = tl.interp_to_uniform_z(table, zpbl, dz_new=dz_new)
(UT_new, VT_new, WT_new, THLM_new, RNPM_new, RVT_new, RCT_new, PABST_new, buoyancy_new) = table_new
nx,ny,nz = len(x),len(y),len(z_new)
print(nx,ny,nz,UT_new.shape)

nbins = 100

################################################
#    Calculate 3D spectra flux and cascade     #
################################################

# nmin
nmin = 20

# Compute spectra and cascade from 3D fields
winds =  (UT_new, VT_new, WT_new)
result = stl.compute_spectral_transfer(
    winds, 
    dx=dx, dy=dy, dz=dz_new,
    binning='log',nbins=nbins, nmin=nmin)

# Compute spectra and cascade for Buoyancy flux

result_b = stl.compute_spectral_transfer(
    winds, scalar=buoyancy_new,
    dx=dx, dy=dy, dz=dz_new,
    binning='log',nbins=nbins, nmin=nmin)

result_c = stl.compute_spectral_transfer(
    winds, scalar=THLM_new,
    dx=dx, dy=dy, dz=dz_new,
    binning='log',nbins=nbins, nmin=nmin)

result_d = stl.compute_spectral_transfer(
    winds, scalar=PABST_new,
    dx=dx, dy=dy, dz=dz_new,
    binning='log',nbins=nbins, nmin=nmin)

# For information, Egality is:
# Pi = PI_3d*(nx*ny*nznew)

# Check equality between the spectra energy and the field
k,kc = result['k'],result['k_shell_centers']
dk   = result['dk']
E,T  = result['E'],result['T']
E_k3_mean = E/dk #/(nx*ny*nz)
TKE3D  = 0.5*(pow(tl.anomcalc(UT_new),2.)
            +pow(tl.anomcalc(VT_new),2.)\
            +pow(tl.anomcalc(WT_new),2.))
tl.checkvariance(kc,E_k3_mean,TKE3D,type='mean')


# Eperp_k3_mean = result['Eperp']/np.diff(k)/(nx*ny*nz)
# tl.checkvariance(kc,Eperp_k3_mean,TKE3D,type='mean')

# c_k3_mean = result_c['Epara']/np.diff(result_c['kpara']).mean()/(nx*ny*nz)
# tl.checkvariance(result_c['kpara'],c_k3_mean,THLM_new,type='var')

# b_k3_mean = result_b['E']/np.diff(k)/(nx*ny*nz)
# tl.checkvariance(kc,b_k3_mean,buoyancy_new,type='var')

# c_k3_mean = result_c['E']/np.diff(k)/(nx*ny*nz)
# tl.checkvariance(kc,c_k3_mean,THLM_new,type='var')

# c_k3_mean = result_c['Eperp']/np.diff(k)/(nx*ny*nz)
# tl.checkvariance(kc,c_k3_mean,THLM_new,type='var')

# c_k3_mean = result_c['Epara']/np.diff(result_c['kpara']).mean()/(nx*ny*nz)
# tl.checkvariance(result_c['kpara'],c_k3_mean,THLM_new,type='var')

# For 3D filtering
idxzlist = None
if Filter3D:
    idxzlist   = np.arange(0,1.2,0.1) #np.arange(0,1.2,0.1)
    idxzfilter = [tl.near(z_new, ij*PBLheight) for ij in idxzlist]
    zfilter    = z_new[idxzfilter]

    Nk,Nzz = len(k),len(idxzlist)
    Ek_BT,Tk_BT,Pi_BT = [np.zeros((Nk,Nzz)) for ij in range(3)]
    Ek_TB,Tk_TB,Pi_TB = [np.zeros((Nk,Nzz)) for ij in range(3)]
    PIhh_BT,PIhv_BT,PIvh_BT,PIvv_BT = [np.zeros((Nk,Nzz)) for ij in range(5)]
    PIhh_TB,PIhv_TB,PIvh_TB,PIvv_TB = [np.zeros((Nk,Nzz)) for ij in range(5)]

    
    resultplus1,resultminus1,resultplus2 ={},{},{}
    
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
        resultplus1[str(cutoff)]  = stl.compute_spectral_transfer(
            (Uh,Vh,Wh),
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins, nmin=nmin)
        resultminus1[str(cutoff)] = stl.compute_spectral_transfer(
            (Uh2,Vh2,Wh2),
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins, nmin=nmin)        
        
        # Specific selection u<_a * (u_b * grad(u_c))
        # In classic case a=b=c. 
        # Cross terms are when at lest one is different
        # resultplus2[str(cutoff)] = stl.compute_spectral_transfer(
        #     (Uh, Vh, Wh),            # k-filtered winds
        #     windsb=(Uh, Vh, Wh),    # winds
        #     windsc=(Uh2, Vh2, Wh2), # gradient
        #     dx=dx, dy=dy, dz=dz_new,
        #     binning='log',nbins=nbins)
        
        
        # Save important variables
        Ek_BT[:,ij],Tk_BT[:,ij],Pi_BT[:,ij] = [resultplus1[ijk] for ijk in ['E_k','T_k','Pi_k']]
        PIhh_BT[:,ij],PIhv_BT[:,ij],PIvh_BT[:,ij],PIvv_BT[:,ij] = \
            [resultplus1[ijk] for ijk in ['PI_hh','PI_hv','PI_vh','PI_vv']]
            
        
        Ek_TB[:,ij],Tk_TB[:,ij],Pi_TB[:,ij] = [resultminus1[ijk] for ijk in ['E_k','T_k','Pi_k']]
        PIhh_TB[:,ij],PIhv_TB[:,ij],PIvh_TB[:,ij],PIvv_TB[:,ij] = \
            [resultminus1[ijk] for ijk in ['PI_hh','PI_hv','PI_vh','PI_vv']]


        
    
    # Integrate cascade
    cascadeneg,_ = stl.integrate_negative_cascade(kc, result['Pi'], kPBL, method="trapz")
    PinegBT,PinegTB = np.zeros(len(resultplus1)),np.zeros(len(resultplus1))
    for ij,key in enumerate(resultplus1):
        Pi = resultplus1[key]['Pi']
        PinegBT[ij], _ = stl.integrate_negative_cascade(kc, Pi, kPBL, method="trapz")
        Pi = resultminus1[key]['Pi']
        PinegTB[ij], _ = stl.integrate_negative_cascade(kc, Pi, kPBL, method="trapz")
    
#    plt.plot(zfilter/PBLheight,I/cascadeneg,'r')
#    plt.plot(zfilter[::-1]/PBLheight,J/cascadeneg,'b')
#    plt.show()
################################################
#    Calculate coarse graining                 #
################################################

coarsegraining = True
if coarsegraining:
    k_cuts = k
    Pi_z_k = cgf.compute_coarse_grained_flux_profile(
                    UT_new, VT_new, WT_new,    
                    dx, dy, z_new, k_cuts,
                    flux_type='both')
    
    print("sfs shape:", Pi_z_k['sfs'].shape, " naive shape:", Pi_z_k['naive'].shape)
    
    #Pi_z_k_gauss = cgf.compute_coarse_grained_flux_profile(
    #                UT_new, VT_new, WT_new,    
    #                dx, dy, z_new, k_cuts,
    #                flux_type='both',
    #                filter_type='gaussian')
    
    # --- both-ends zero check ---
    res_small = cgf.compute_coarse_grained_flux(UT_new, VT_new, WT_new, dx, dy, z_new, 1e-6, flux_type='both')
    res_large = cgf.compute_coarse_grained_flux(UT_new, VT_new, WT_new, dx, dy, z_new, 1e6, flux_type='both')
    print("k_cut -> 0   : max|Pi_sfs|=%.3e  max|Pi_naive|=%.3e" %
          (np.max(np.abs(res_small['sfs'][1])), np.max(np.abs(res_small['naive'][1]))))
    print("k_cut -> inf : max|Pi_sfs|=%.3e  max|Pi_naive|=%.3e" %
          (np.max(np.abs(res_large['sfs'][1])), np.max(np.abs(res_large['naive'][1]))))
    
        
    H = PBLheight      # boundary-layer depth (or a relevant plume vertical scale)
    
    if zfilter.any():
        ell_z_list = zfilter
    else:
        ell_z_list = [None, 0.05*H, 0.2*H, 0.5*H, H, 1.5*H]
    
    #sweep = compute_coarse_grained_flux_ellz_sweep(
    #    UT_new, VT_new, WT_new, dx, dy, z_new,
    #    k_cuts, ell_z_list=[None, 0.05*H, 0.2*H, 0.5*H, H],
    #    filter_type='gaussian', flux_type='sfs')
    
    sweep_sharp = cgf.compute_coarse_grained_flux_ellz_sweep(
        UT_new, VT_new, WT_new, dx, dy, z_new,
        k_cuts, ell_z_list=ell_z_list,
        filter_type='sharp', flux_type='sfs')
    
    
    # colors=['r','b','y','m','k','orange']
    # for ij,key in enumerate(sweep_sharp.keys()):
    #     plt.loglog(k/kPBL,np.mean(sweep_sharp[key]["sfs"],axis=1),color=colors[ij],label=f"ell_z={key}")
    #     plt.loglog(k/kPBL,np.mean(-sweep_sharp[key]["sfs"],axis=1),'--',color=colors[ij])
    #     plt.legend()
    # plt.figure()
    # for ij,key in enumerate(sweep_sharp.keys()):
    #     plt.semilogx(k/kPBL,np.mean(sweep_sharp[key]["sfs"],axis=1),color=colors[ij],label=f"ell_z={key}")
    #     plt.legend()
        
            
    # 2D maps?
    ell_h_list = 2 * np.pi / np.array(k_cuts[::3])
    # Calculer la carte Pi(ell_h, ell_z)
    Pi_map = compute_Pi_2D_map(UT_new, VT_new, WT_new, dx, dy, z_new, 
                               ell_h_list, ell_z_list, 
                               filter_type='sharp', flux_type='sfs')
    
    # # Visualisation
    # plt.figure(figsize=(10, 8))
    # plt.contourf(
    #     ell_z_list/PBLheight, ell_h_list/PBLheight, Pi_map[:,:,1],
    #     levels=20, cmap='RdBu'
    # #, norm=plt.Normalize(vmin=-1e-4, vmax=1e-4)
    # )
    # plt.colorbar(label='Flux $\Pi(\ell_h, \ell_z)$ [m²/s³]')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Échelle verticale $\ell_z$ [m]')
    # plt.ylabel('Échelle horizontale $\ell_h$ [m]')
    # plt.title('Carte 2D du flux $\Pi(\ell_z, \ell_h)$')
    # plt.grid(True, which='both', linestyle='--')
    # plt.show()
    
    
################################################
#    Calculate 2D spectra flux and cascade     #
################################################




################################################
#               Saving files                   #
################################################


# Name of NetCDF file to save
# Take relevant information from the name file
tab = file.split('/')[-1].split('.')
prefix,vinfo, tinfo = tab[0],tab[2],tab[4]
file_netcdf  = '_'.join(['Spectra',prefix,vinfo,tinfo])
file_netcdf2 = pathsave+file_netcdf+'.nc'

# Simplification to write netcdf file
r = result.copy()

# Save file    
ds = xr.Dataset(
    {
    "E": (("kc",), r['E']),
    "T": (("kc",), r['T']),
    "Pi": (("kc",), r['Pi']),
    },
    coords={"kv":r['k'], 
            "kc":r['k_shell_centers'],
            "kk2":r['kk2'],
            "z" :z_new,
            "kperp":r['kperp'],
            "kpara":r['kpara'],
            "nx":nx,"ny":ny,"nz":nz,
            "zlist":idxzlist
            }
)

ds["PBL"] = PBLheight  # A single value

if r['PI_k'] is not None:
    ds["PI_k"]  = (("k",), r['PI_k'])
    ds["PI_hh"] = (("k",), r['PI_hh'])
    ds["PI_hv"] = (("k",), r['PI_hv'])
    ds["PI_vh"] = (("k",), r['PI_vh'])
    ds["PI_vv"] = (("k",), r['PI_vv'])

if r['PI_hz'] is not None:
    ds["PI_hz"]  = (("kk2",), r['PI_hz'])
    
if r['Pi_perp'] is not None:
    ds["Eperp"]   = (("kperp",), r['Eperp'])
    ds["Tperp"]   = (("kperp",), r['Tperp'])
    ds["Piperp"]  = (("kperp",), r['Piperp'])
    ds["Epara"]   = (("kpara",), r['Epara'])
    ds["Tpara"]   = (("kpara",), r['Tpara'])
    ds["Pipara"]  = (("kpara",), r['Pipara'])
    
if Filter3D:
    ds["EkBT"] = (("k", "zlist"), Ek_BT)
    ds["TkBT"] = (("k", "zlist"), Tk_BT)
    ds["PiBT"] = (("k", "zlist"), Pi_BT)
    ds["PIhhBT"] = (("k", "zlist"), PIhh_BT)
    ds["PIhvBT"] = (("k", "zlist"), PIhv_BT)
    ds["PIvhBT"] = (("k", "zlist"), PIvh_BT)
    ds["PIvvBT"] = (("k", "zlist"), PIvv_BT)
    ds["EkTB"] = (("k", "zlist"), Ek_TB)
    ds["TkTB"] = (("k", "zlist"), Tk_TB)
    ds["PiTB"] = (("k", "zlist"), Pi_TB)
    ds["PIhhTB"] = (("k", "zlist"), PIhh_TB)
    ds["PIhvTB"] = (("k", "zlist"), PIhv_TB)
    ds["PIvhTB"] = (("k", "zlist"), PIvh_TB)
    ds["PIvvTB"] = (("k", "zlist"), PIvv_TB)
    ds["PinegBT"] = (("zlist",),PinegBT)
    ds["PinegTB"] = (("zlist",),PinegTB)


# Save to NetCDF (overwrites if exists)
file_netcdf2=pathsave+file_netcdf+'.nc'
ds.to_netcdf(file_netcdf2)

# Save RCT
if nvar is not None:
    ds["E1dr_"+nvar0] = (("kv", "z"), Evr)
    ds["E1da_"+nvar0] = (("kvazi", "z"), Eva)
    
ds["E1dr_WT"] = (("kv", "z"), Ewr)
ds["E1da_WT"] = (("kvazi", "z"), Ewa)
ds["E1dr_THL"] = (("kv", "z"), Ethr)
ds["E1da_THL"] = (("kvazi", "z"), Etha)
    
# Add a scalar variable (e.g., a global attribute)
for index in indLWP.keys():
    indexLWP = index+'_'+nvar
    ds[indexLWP] = indLWP[index]  # A single value

# Add spectra of LWP
ds["E1dr_"+nvar] = (("kv",), ELWPr)
ds["E1da_"+nvar] = (("kvazi",), ELWPa)







