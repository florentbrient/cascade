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
Filter3D = False
coarsegraining = False


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

        
# Create new dictionary saving values
globals_copy = list(globals().items())
table2 = {
    nom: obj 
    for nom, obj in globals_copy 
    if any(obj is valeur for valeur in table_new)
}


nbins = 100

################################################
#    Calculate 3D spectra flux and cascade     #
################################################

# nmin
nmin = 20

# Compute spectra and cascade from 3D fields
winds =  (UT_new, VT_new, WT_new)
result = stl.compute_spectral_transfer(
    winds, P=PABST_new, B=buoyancy_new,
    dx=dx, dy=dy, dz=dz_new,z=z_new,
    binning='log',nbins=nbins, nmin=nmin)

# Compute spectra and cascade for all variables in table2:
result_b ={}
for scalar in table2.keys():
    if not any(text in scalar for text in ('UT', 'VT', 'WT')):
        print(scalar)
        # Buoyancy flux
        tmp = stl.compute_spectral_transfer(
            winds, scalar=table2[scalar],
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins, nmin=nmin)
        result_b[scalar] = tmp
        del tmp
    # What do you want to save:
        
        

# For information, Egality is:
# Pi = PI_3d*(nx*ny*nznew)

# Check equality between the spectra energy and the field
k,kc = result['Eout']['k'],result['Eout']['k_shell_centers']
dk   = result['Eout']['dk']
E,T  = result['Eout']['E_k'],result['Eout']['T_k']
E_k3_mean = E/dk #/(nx*ny*nz)
TKE3D  = 0.5*(pow(tl.anomcalc(UT_new),2.)
            +pow(tl.anomcalc(VT_new),2.)\
            +pow(tl.anomcalc(WT_new),2.))
tl.checkvariance(kc,E_k3_mean,TKE3D,type='mean')


cascadeneg,_ = stl.integrate_negative_cascade(kc, result['Eout']['Pi_k'], kPBL, method="trapz")


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
    PIhh_BT,PIhv_BT,PIvh_BT,PIvv_BT = [np.zeros((Nk,Nzz)) for ij in range(4)]
    PIhh_TB,PIhv_TB,PIvh_TB,PIvv_TB = [np.zeros((Nk,Nzz)) for ij in range(4)]

    
    resultplus1,resultminus1,resultplus2 ={},{},{}
    
    for ij,cutoff in enumerate(idxzfilter):
        Uh,Vh,Wh    = UT_new.copy(),VT_new.copy(),WT_new.copy()
        Uh2,Vh2,Wh2 = UT_new.copy(),VT_new.copy(),WT_new.copy()
        stcut       = str(cutoff)

        # zero out above 75% along z
        Uh[cutoff:, :, :] = 0
        Vh[cutoff:, :, :] = 0
        Wh[cutoff:, :, :] = 0
        
        # zero below z0 (ij)
        Uh2[:cutoff, :, :] = 0
        Vh2[:cutoff, :, :] = 0
        Wh2[:cutoff, :, :] = 0
    
        # Compute spectra and cascade from truncated 3D fields
        resultplus1[stcut]  = stl.compute_spectral_transfer(
            (Uh,Vh,Wh),
            dx=dx, dy=dy, dz=dz_new,
            binning='log',nbins=nbins, nmin=nmin)
        resultminus1[stcut] = stl.compute_spectral_transfer(
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
        Ek_BT[:,ij],Tk_BT[:,ij],Pi_BT[:,ij] = [resultplus1[stcut]['Eout'][ijk] for ijk in ['E_k','T_k','Pi_k']]
        PIhh_BT[:,ij],PIhv_BT[:,ij],PIvh_BT[:,ij],PIvv_BT[:,ij] = \
            [resultplus1[stcut][ijk] for ijk in ['PI_hh','PI_hv','PI_vh','PI_vv']]
            
        
        Ek_TB[:,ij],Tk_TB[:,ij],Pi_TB[:,ij] = [resultminus1[stcut]['Eout'][ijk] for ijk in ['E_k','T_k','Pi_k']]
        PIhh_TB[:,ij],PIhv_TB[:,ij],PIvh_TB[:,ij],PIvv_TB[:,ij] = \
            [resultminus1[stcut][ijk] for ijk in ['PI_hh','PI_hv','PI_vh','PI_vv']]


        
    
    # Integrate cascade
    PinegBT,PinegTB = np.zeros(len(resultplus1)),np.zeros(len(resultplus1))
    for ij,key in enumerate(resultplus1):
        Pi = resultplus1[key]['Eout']['Pi_k']
        PinegBT[ij], _ = stl.integrate_negative_cascade(kc, Pi, kPBL, method="trapz")
        Pi = resultminus1[key]['Eout']['Pi_k']
        PinegTB[ij], _ = stl.integrate_negative_cascade(kc, Pi, kPBL, method="trapz")
    
#    plt.plot(zfilter/PBLheight,I/cascadeneg,'r')
#    plt.plot(zfilter[::-1]/PBLheight,J/cascadeneg,'b')
#    plt.show()
################################################
#    Calculate coarse graining                 #
################################################
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
var_to_plot = ['TKE']+list(result_b.keys())[:]
E2D,Pi2D = {},{}
for var in var_to_plot:
    E2D[var]  = np.zeros((nz,nbins+1))
    Pi2D[var] = np.zeros((nz,nbins+1))
for idx,zi in enumerate(z_new):
    
    # TKE full
    winds =  (UT_new[idx,:,:], VT_new[idx,:,:], WT_new[idx,:,:])
    result2D = stl.compute_spectral_transfer(
        winds, 
        dx=dx, dy=dy,
        binning='log',nbins=nbins)
    E2D['TKE'][idx,:] =result2D['Eout']['E_spec']
    Pi2D['TKE'][idx,:]=result2D['Eout']['Pi_k']
    
    if idx==0:
        k2D = result2D['Eout']['k']

    for scalar in table2.keys():
        if not any(text in scalar for text in ('UT', 'VT', 'WT')):
            tmp = table2[scalar]
            resulttmp = stl.compute_spectral_transfer(winds, 
                                                      dx=dx, dy=dy,
                                                      scalar=tmp[idx,:,:],
                                                      nbins=nbins)
            
            E2D[scalar][idx,:] =resulttmp['Eout']['E_spec']
            Pi2D[scalar][idx,:]=resulttmp['Eout']['Pi_k']
    
    



################################################
#               Saving files                   #
################################################


# Name of NetCDF file to save
# Take relevant information from the name file
tab = file.split('/')[-1].split('.')
prefix,vinfo, tinfo = tab[0],tab[2],tab[4]
file_netcdf0  = '_'.join(['Cascade',prefix,vinfo,tinfo])
file_netcdf0 += "_XXX"
file_netcdf0  = pathsave+file_netcdf0+'.nc'

for var in var_to_plot:
    file_netcdf1 = file_netcdf0.replace('XXX',var)
    if Filter3D:
        file_netcdf2 = file_netcdf1.replace('Cascade','Cascade_BT')
    if coarsegraining:
        file_netcdf3 = file_netcdf1.replace('Cascade','Cascade_CG')


    # Simplification to write netcdf file
    if var=='TKE':
        r = result.copy()
    else:
        r = result_b[var].copy()
        
    rout = r['Eout']
    # Save file    
    ds = xr.Dataset(
        {
        "E": (("kc",), rout['E_k']),
        "T": (("kc",), rout['T_k']),
        "Pi": (("kc",), rout['Pi_k']),
        },
        coords={"kv":r['k'], 
                "kc":rout['k_shell_centers'],
                "kk2":r['kk2'],
                "z" :z_new,
                "kperp":r['kperp'],
                "kpara":r['kpara'],
                "nx":nx,"ny":ny,"nz":nz,
                "zlist":idxzlist,
                "k2D":k2D
                }
    )
    
    ds["PBL"]   = PBLheight  # A single value
    ds["Pineg"] = cascadeneg
    
    if Filter3D:
        ds2 = ds.copy()
    if coarsegraining:
        ds3 = ds.copy()
    
    if r['PI_k'] is not None:
        ds["PI_k"]  = (("k",), r['PI_k'])
        ds["PI_hh"] = (("k",), r['PI_hh'])
        ds["PI_hv"] = (("k",), r['PI_hv'])
        ds["PI_vh"] = (("k",), r['PI_vh'])
        ds["PI_vv"] = (("k",), r['PI_vv'])
    
    ds["E_spec"]   = (("k",), rout['E_spec'])
    if rout['E_h_spec'] is not None:
        ds["E_h_spec"] = (("k",), rout['E_h_spec'])
        ds["E_v_spec"] = (("k",), rout['E_v_spec'])
        ds["E_h_k"]    = (("k",), rout['E_h_k'])
        ds["E_v_k"]    = (("k",), rout['E_v_k'])
        ds["Pi_h_k"]   = (("k",), rout['Pi_h_k'])
        ds["Pi_v_k"]   = (("k",), rout['Pi_v_k'])

    if r['PI_hz'] is not None:
        ds["PI_hz"]  = (("kk2",), r['PI_hz'])
        
    if r['Piperp'] is not None:
        ds["Eperp"]   = (("kperp",), r['Eperp'])
        ds["Tperp"]   = (("kperp",), r['Tperp'])
        ds["Piperp"]  = (("kperp",), r['Piperp'])
        ds["Epara"]   = (("kpara",), r['Epara'])
        ds["Tpara"]   = (("kpara",), r['Tpara'])
        ds["Pipara"]  = (("kpara",), r['Pipara'])
    ds['E2D']  = (("z","k2D"),E2D[scalar]) 
    ds['Pi2D'] = (("z","k2D"),Pi2D[scalar])

        
    if Filter3D:
        ds2 = ds.copy()
        ds2["EkBT"] = (("k", "zlist"), Ek_BT)
        ds2["TkBT"] = (("k", "zlist"), Tk_BT)
        ds2["PiBT"] = (("k", "zlist"), Pi_BT)
        ds2["PIhhBT"] = (("k", "zlist"), PIhh_BT)
        ds2["PIhvBT"] = (("k", "zlist"), PIhv_BT)
        ds2["PIvhBT"] = (("k", "zlist"), PIvh_BT)
        ds2["PIvvBT"] = (("k", "zlist"), PIvv_BT)
        ds2["EkTB"] = (("k", "zlist"), Ek_TB)
        ds2["TkTB"] = (("k", "zlist"), Tk_TB)
        ds2["PiTB"] = (("k", "zlist"), Pi_TB)
        ds2["PIhhTB"] = (("k", "zlist"), PIhh_TB)
        ds2["PIhvTB"] = (("k", "zlist"), PIhv_TB)
        ds2["PIvhTB"] = (("k", "zlist"), PIvh_TB)
        ds2["PIvvTB"] = (("k", "zlist"), PIvv_TB)
        ds2["PinegBT"] = (("zlist",),PinegBT)
        ds2["PinegTB"] = (("zlist",),PinegTB)
        ds2.to_netcdf(file_netcdf2)
        del ds2
    
    if coarsegraining:
        ds3["PicoarSFS"] = (("k","z"),Pi_z_k['sfs'])
        ds3["PicoarNAI"] = (("k","z"),Pi_z_k['naive'])
        ds3 = ds3.assign_coords(Lz=ell_z_list)
        ds3 = ds3.assign_coords(Lh=ell_h_list)
        ds3["Pi_map"]    = (("Lh","Lz","z"),Pi_map)
        ds3.to_netcdf(file_netcdf3)
        del ds3
    
    
    # Save to NetCDF (overwrites if exists)
    ds.to_netcdf(file_netcdf1)
    

    del r,file_netcdf1








