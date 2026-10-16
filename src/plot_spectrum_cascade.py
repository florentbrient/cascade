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
import pandas as pd
import spectratools as stl


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
# Mean every 2 time step
def make_moy(tmp,nb=2):    
    new_tmp = pd.Series(tmp).groupby(lambda x: x // nb).mean().to_numpy()
    return new_tmp

def make_zavg(tmp,z,idx,deltaz,PBL):
    tmp = np.mean(tmp[(z >= (idx - deltaz) * PBL) & ((z + deltaz) < idx * PBL), :], axis=0)
    return tmp
    


ON_JZ = on_jean_zay()
print(ON_JZ)
#stop
if ON_JZ:
    pathsrc= '/lustre/fswork/projects/rech/whl/rces071/Github/cascade/' 
    #path1D = '/lustre/fsstor/projects/rech/whl/rces071/MNH-V5-7-0/'
    path1D = '/lustre/fsn1/projects/rech/whl/rces071/mon_calcul/V5-7-0/'
else:
    pathsrc= '/home/fbrient/GitHub/cascade/'
    path1D = '/home/fbrient/MNH/V5-7-0/'


# Open all netcdf file
#prefix = sys.argv[1] # name of the prefix (FIRZ4)
prefix = 'FIZ4c' 


pathin   = pathsrc+"data/"
#prefix   = "FIRZ4" #"IHOP" #"FIR1k"
filein0  = pathin+'Cascade_'+prefix+'*XXX.nc'

# Dir for figures
pathout = pathsrc+"figures/"
pathout+= prefix+'/'
tl.mkdir(pathout)

# Variables
varall  = ['TKE','TKEh','TKEv','THLM','RNPM','RCT','PABST','buoyancy']
varall  = ['TKE','TKEh','TKEv','THLM','RNPM'] #,'RCT','PABST','buoyancy']
data    = {}
for var in varall:
    filein    = filein0.replace('XXX',var)
    data[var] = tl.read_netcdfs(filein, dim='time',concat=False)
    
############ Open 1D to find time of sunlight
path1D+='*/'+prefix+'/'
file0 = path1D+'*'+'.1.'+'*'+'.000'+'*'+'.nc'
# List of files with glob
files = sorted(glob.glob(file0))
print('files ',file0,files)
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
#zplot    = [0,0.1,0.25,0.5,0.75,0.90,1] 
zplot    = [0.1,0.5,0.9] 
deltaz   = 0.1 # averaged over a layer of +/-0.1
zplotstr = [str(idx) for idx in zplot]

# Initialize variables for global plot
kmaxLWP = np.zeros(nt)
kPBLall = np.zeros(nt)

# Colors separation Pi_hh, Pi_hv...
cassep   = ['Pi','PI_hh','PI_hv','PI_vh','PI_vv']
labsep   = ['Total','hh','hv','vh','vv']
colsep   = ['b','c','m','m','c']
linsep   = ['-','--','--',':',':']

# Plotall
plotall = True


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
    k2Db  = data['TKE'][idxt].k2Db.values
    kLWP  = data['TKE'][idxt].kLWP.values
    kLWP2 = data['TKE'][idxt].kLWP2.values
    nk,nk2= len(kv),len(k2Db)
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
        kmax2Dm, Es2Dm = {},{}
        kmax2Dz ={}
        for var in varall:
            kmax3D[var] =np.zeros(nt)*np.nan
            Es3D[var]   =np.zeros((nt,nk))*np.nan
            kmax2D[var] =np.zeros(nt)*np.nan
            Es2D[var]   =np.zeros((nt,nk2))*np.nan
            kmax2Dm[var]=np.zeros(nt)*np.nan
            Es2Dm[var]  =np.zeros((nt,nk2))*np.nan
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
    ELWP2 = data['TKE'][idxt]['ELWP2']
    kmaxLWP[idxt],EsLWP,_ = tl.findkvmax(kLWP2,ELWP2,sigma=sigma)


    
    for var in varall:
        
        # Find cell size
        Etmp = data[var][idxt]['E_spec']
        kmax3D[var][idxt],Es3D[var][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)
        Etmp = data[var][idxt]['E2Dbm']
        kmax2Dm[var][idxt],Es2Dm[var][idxt,:],_ = tl.findkvmax(k2Db,Etmp,sigma=sigma)
        Etmp = data[var][idxt]['E2Db']
        kmax2D[var][idxt],Es2D[var][idxt,:],_  = tl.findkvmax(k2Db,Etmp.T,sigma=sigma,weights=dz)
        for idxz,zz in enumerate(z):
            kmax2Dz[var][idxt,idxz],_,_        = tl.findkvmax(k2Db,Etmp[idxz,:],sigma=sigma)

        # Calculate cell size for TKEv and TKEh
        #if var=="TKE":
        #    Etmp = data[var][idxt]['E_h_spec']
        #    kmax3D['TKEh'][idxt],Es3D['TKEh'][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)
        #    Etmp = data[var][idxt]['E_v_spec']
        #    kmax3D['TKEv'][idxt],Es3D['TKEv'][idxt,1:],_ = tl.findkvmax(kv,Etmp,sigma=sigma)


        if plotall:           
            # Plot E/Pi for the mean field            # Plot Ek, Eperp and Epara on the same figure  
            namefig=pathout+'E2Dm_'+var+'_'+prefix+'_'+tst
            colors=['tab:purple', 'tab:gray', 'tab:olive']
            E2Dm  = data[var][idxt]['E2Dbm'].values
            tl.plot_flux([k2Db],[E2Dm],
                  kPBL=kPBL,
                  kcell=kmaxLWP[idxt],
                  y1lab=y1lab, colors=colors,
                  plotlines=2,namefig=namefig)
            namefig=pathout+'Pi2Dm_'+var+'_'+prefix+'_'+tst            
            Pi2Dm = data[var][idxt]['Pi2Dm']
            tl.plot_flux([k2D],[Pi2Dm],
                  kPBL=kPBL,
                  kcell=kmaxLWP[idxt],
                  logy=False,
                  y1lab=y2lab, colors=colors,
                  namefig=namefig)

            
            
            
            # Plot E for different altitude
            namefig=pathout+'E_z_'+var+'_'+prefix+'_'+tst
            
            #Erzall = []
            #for idx in zplot:
            #    idxm = (z>=(idx-deltaz)*PBL) & ((z+deltaz)<idx*PBL)
            #    Erzall += [np.mean(data[var][idxt]['E2Db'][idxm,:],axis=0)]
            #Erzall = [np.mean(data[var][idxt]['E2Db'][tl.near(z,idx*PBL),:]) for idx in zplot]

            Erzall = [make_zavg(data[var][idxt]['E2Db'],z,idx,deltaz,PBL) for idx in zplot]
            tl.plot_flux(k2Db,Erzall,
                      kPBL=kPBL,#kin=kin[idxt],
                      kcell=kmaxLWP[idxt],
                      y1lab=y1lab,labels=zplotstr,
                      normalized=True,
                      plotlines=0,namefig=namefig)   
            
 
            
            # Plot Pi for different altitude
            namefig=pathout+'PI_z_'+var+'_'+prefix+'_'+tst
            #Erzall = [data[var][idxt]['Pi2D'][tl.near(z,idx*PBL),:] for idx in zplot]
            Pirzall = [make_zavg(data[var][idxt]['Pi2D'],z,idx,deltaz,PBL) for idx in zplot]

            tl.plot_flux(k2D,Pirzall,
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
            #chtmp      = ['PI_k','PI_hh','PI_hv','PI_vh','PI_vv']
            #chtmplab   = ['Total','hh','hv','vh','vv']
            #colors     = ['b','c','m','m','c']
            #linestyles = ['-','--','--',':',':']
            
            Pi3D = []
            for ij,ch in enumerate(cassep):
                if ch in data[var][idxt].keys():
                    Pitmp  = data[var][idxt][ch]
    #               idxtmp = ~np.isnan(Pitmp)
                    Pi3D  += [Pitmp] #[idxtmp]]
            #kbins = np.array(data['kbins'])
            
                
            namefig=pathout+'E3Dhv_'+var+'_'+prefix+'_'+tst
            tl.plot_flux(kv,Pi3D,
                  kPBL=kPBL,
                  kcell=kmaxLWP[idxt],
                  logy=False,
                  y1lab=y2lab,
                  labels=labsep,normalized=True,
                  colors=colsep,linestyles=linsep,
                  plotlines=plotlines,namefig=namefig)
        
# Plot all LWP spectra

nb = 3
timeint = np.arange(0,nt,nb)
timehere = make_moy(time_hours3D,nb=nb)


namefig=pathout+'ELWP_'+prefix
ELWPall = [data['TKE'][ij]['ELWP'].values for ij in range(nt)]
ELWPall  = make_moy(ELWPall,nb=nb)
kPBLall2 = make_moy(kPBLall,nb=nb)

tl.plot_flux(kLWP,ELWPall,
      kPBL=kPBLall2,
#      kcell=kmaxLWP,
      y1lab=y1lab,
      normalized=True,
      plotlines=2,namefig=namefig)

namefig=pathout+'ELWP2_'+prefix
ELWPall = [data['TKE'][ij]['ELWP2'].values for ij in range(nt)]
ELWPall = make_moy(ELWPall,nb=nb)

kLWP2   = data['TKE'][0]['kLWP2'].values
tl.plot_flux(kLWP2,ELWPall,
      kPBL=kPBLall2,
#      kcell=kmaxLWP,
      y1lab=y1lab,
      normalized=True,
      labels=timehere,
      plotlines=2,namefig=namefig)

# colors, lines, markers
nb = 2
timehere = make_moy(time_hours3D,nb=nb)

clm = {}
clm['line'], clm['color'], clm['marker'] = {},{},{}
for var in varall+['LWP']:
    clm['line'] [var]= '-' 
    clm['color'][var] = 'tab:blue'
    clm['marker'][var] = ''

clm['color']['LWP'] =  "k" 
clm['line']['LWP'] =  ""  
clm['marker']['LWP'] =  "o"   
clm['line']['TKEh'] =  "--"   
clm['line']['TKEv'] =  "-."   
clm['color']['THLM'] =  "tab:red" 
clm['color']['RNPM'] =  "tab:green" 
clm['color']['buoyancy'] =  "tab:orange" 

dayhere  = make_moy(day_hours,nb=nb)

# Plot temporal evolution of aspect ratio
for cell in ["Cell_Size","Aspect_Ratio"]:
    for typ in ["2D","2Dm","3D"]:
        namefig=pathout+cell+'_v'+typ+'_'+prefix
        kmaxplot=kmax3D
        if typ=='2D':
            kmaxplot=kmax2D
        if typ=='2Dm':
            kmaxplot=kmax2Dm
        Gamma = {}
        for var in kmaxplot.keys():
            if cell== "Cell_Size":
                Gamma['LWP'] = 2*np.pi/kmaxLWP
                Gamma[var]   = 2*np.pi/kmaxplot[var] 
            else:
                Gamma['LWP'] = kPBLall/kmaxLWP
                Gamma[var]   = kPBLall/kmaxplot[var] #(2pi/kvmax)/(2pi/kPBL)
                
            Gamma[var] = make_moy(Gamma[var],nb=nb)
        Gamma['LWP'] = make_moy(Gamma['LWP'],nb=nb)
        
        maxh = np.max([Gamma[ij] for ij in Gamma.keys()])
        ylim = [0,math.ceil(maxh / 5) * 5]
        
        
        #LambdaEpsIn = kPBLall/kin
        #lambdaIn = {}
        #lambdaIn['LambdaEpsIn']=LambdaEpsIn
        tl.plot_time(timehere,Gamma,
                     ylim=ylim,
                     marker=clm['marker'],
                     colors=clm['color'],
                     lines=clm['line'],
                     day=dayhere,
                     namex=''.join(cell.split('_')),
                     namefig=namefig)


# Compute cascade
#casall = ['PI_k','PI_hh','PI_hv','PI_vh','PI_vv']

nb = 2
maxPi=10

namefig=pathout+'PiNeg_sep_'+prefix
CnegAll = {}
for cas in cassep:
    CnegAll[cas] = np.zeros(nt)*np.nan
    for ij in range(nt):
        tmp = data['TKE'][ij][cas]
        CnegAll[cas][ij],_ = stl.integrate_negative_cascade(kv, tmp, kPBLall[ij]*0.9, 
                                                            maxPi=maxPi,
                                                            method="trapz")
    CnegAll[cas] =  make_moy(CnegAll[cas],nb=nb)
    
# Plot Negative cascade
timehere = make_moy(time_hours3D,nb=nb)
dayhere  = make_moy(day_hours,nb=nb)

linsep2= {}
for ij,icas in enumerate(cassep):
    linsep2[icas] = linsep[ij]
ylim = None
tl.plot_time(timehere,CnegAll,
             ylim=ylim,
             colors=colsep,lines=linsep2,
             day=dayhere,
             namex='Integrated cascade',
             namefig=namefig)


nb = 2
timehere = make_moy(time_hours3D,nb=nb)
dayhere  = make_moy(day_hours,nb=nb)
namefig=pathout+'PiNeg_'+prefix
PinegAll = {}
PinegAll['TKE']  = np.array([data['TKE'][ij]['Pineg'].values for ij in range(nt)])
PinegAll['TKEhz'] = {}
for ij in range(nt):
    tmp = data['TKE'][ij]['PI_hz']
    PinegAll['TKEhz'][ij],_ = stl.integrate_negative_cascade(kv, tmp, kPBLall[ij], 
                                                        method="trapz")

for var in PinegAll.keys():
    PinegAll[var] = make_moy(PinegAll[var],nb=nb)
    
minTKE = np.min(PinegAll['TKE'])
for var in PinegAll.keys():
    PinegAll[var] /= minTKE

maxh = np.max(PinegAll['TKE'])
ylim = None #[0,math.ceil(maxh / 5) * 5]
tl.plot_time(timehere,PinegAll,
             ylim=ylim,marker='o',
             day=dayhere,
             namex='Neg Cascade',
             namefig=namefig)
