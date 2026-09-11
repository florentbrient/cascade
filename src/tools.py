#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:40:31 2026

Toolbox for spectral analysis
# Update to do (17/07/26)

@author: fbrient
"""

from collections import OrderedDict
import numpy as np
from scipy import integrate
import Constants as CC
from scipy.interpolate import interp1d


# Read txt file for informations
def read_info(fileinfo):
    with open(fileinfo, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    #print(lines)
    info_dict=dict()
    for ij in lines:
        tmp = ij.rstrip('\n').strip().split(': ')
        key=tmp[0]
        info_dict[key]=tmp[1]
    #print(info_dict)
    return info_dict

# Find path
def findpath(fileinfo):
    info_dict = read_info(fileinfo)
    info_list = ['vtyp','machine','path','case','sens','prefix','OUT','nc4']
    
    ivar = {}
    for tmp in info_list:
        ivar[tmp] = info_dict[tmp]
        
    
    # Path of your simulations
    path    = '/'.join([ivar[ij] for ij in ['path','vtyp','case','sens']])+'/'
    if 'subdir' in info_dict.keys():
        path+=info_dict['subdir']+'/'
    
    if 'pathsave' in info_dict.keys():
        pathsave=info_dict['pathsave']
    return path,pathsave

# Read dimensions informations in the NetCDF file
def dimensions(DATA,var1D):
    # Dimensions
    data1D,nzyx,sizezyx = [OrderedDict() for ij in range(3)]
    for ij in var1D:
      data1D[ij]  = DATA[ij][:] #/1000.
      nzyx[ij]    = data1D[ij][1]-data1D[ij][0]
      sizezyx[ij] = len(data1D[ij])

    #xy     = [data1D[var1D[1]],data1D[var1D[2]]] #x-axis and y-axis
    nxny   = nzyx[var1D[1]]*nzyx[var1D[2]] #km^2
    ALT    = data1D[var1D[0]]
    dz     = [0.5*(ALT[ij+1]-ALT[ij-1]) for ij in range(1,len(ALT)-1)]
    dz.insert(0,ALT[1]-ALT[0])
    dz.insert(-1,ALT[-1]-ALT[-2])
    nxnynz = np.array([nxny*ij for ij in dz]) # volume of each level
    nxnynz = np.repeat(np.repeat(nxnynz[:, np.newaxis, np.newaxis]
                                 , sizezyx[var1D[1]], axis=1), sizezyx[var1D[2]], axis=2)
    dx     = nzyx[var1D[1]]
    dy     = nzyx[var1D[2]]

    return nxnynz,dz,dy,dx

# Remove Ghost layers of the Meso-NH model
def removebounds(tmp):
   tmp = resiz(tmp)
   if len(tmp.shape) == 1:
     tmp = tmp[1:-1]
   elif len(tmp.shape) == 2:
     tmp = tmp[1:-1,1:-1]
   elif len(tmp.shape) == 3:
     tmp = tmp[1:-1,1:-1,1:-1]
   else:
     print('Problem removebounds')
   return tmp

# Convert altitude to wavenumber
def z2k(z):
    return 2*np.pi/z

# Squeez data
def resiz(tmp): # should be removed by pre-treatment
    return np.squeeze(tmp)

# Fin nearest point
def near(array,value):
    idx=(abs(array-value)).argmin()
    return idx

# Open a variable (try or error)
def tryopen(vv,DATA):
    try:
        tmp = resiz(DATA[vv][:])
    except:
        print('Error in opening ',vv)
        tmp = None
    return tmp

# Calculate the anamoly relative to the horizontal mean
def anomcalc(tmp):
    # ip = 0 (3D)
    ss    = tmp.shape
    if len(ss)==2:
        mean = np.mean(tmp,axis=-1)
        mean = np.repeat(mean[ :,np.newaxis],ss[-1],axis=-1)
    else:    
        mean  = np.mean(np.mean(tmp,axis=2),axis=1)
        mean    = np.repeat(np.repeat(mean[ :,np.newaxis, np.newaxis],ss[1],axis=1),ss[2],axis=2)
    data = tmp-mean
    return data

# Find Boundary-layer top (zi)
def findpbltop(tmp,zz,offset=0.25):
    #tmp     = createnew(typ,DATA,var1D)
    #print('len ',len(tmp.shape))
    if len(tmp.shape)==2:
        temp   = np.nanmean(tmp,axis=(-1))
    else:
        temp   = np.nanmean(tmp,axis=(1,2,))
    
    #zz = tryopen(var1D[0],DATA)
    THLMint = integrate.cumulative_trapezoid(temp,zz,initial=0)/(zz-zz[0])
    
    DT      = temp-(THLMint+offset)
    idx     = np.argmax(DT>0)

    return idx


# Interpolate to isotropic z-grid
def interp_to_uniform_z(U, z_old, dz_new=10.0):
    """
    Interpolate 3D wind components (U,V,W) from nonuniform vertical grid z_old
    to a new uniform vertical grid with spacing dz_new [m].

    Parameters
    ----------
    U, V, W : ndarray (Nz, Ny, Nx)
        Wind components on original grid.
    z_old : 1D ndarray (Nz,)
        Original vertical coordinates (monotonic, uneven spacing, in meters).
    dz_new : float
        Desired uniform vertical spacing in meters (default 10 m).

    Returns
    -------
    U_new, V_new, W_new : ndarray (Nz_new, Ny, Nx)
        Interpolated fields on the uniform grid.
    z_new : 1D ndarray (Nz_new,)
        The new uniform vertical coordinate array.
    """
    zmin, zmax = z_old[0], z_old[-1]
    Nz_new = int(np.floor((zmax - zmin) / dz_new)) + 1
    z_new = np.linspace(zmin, zmax, Nz_new)

    U_new = []
    for ij,tmp in enumerate(U):
        # Define interpolation function for each component along the vertical axis
        print(tmp.shape)
        fU = interp1d(z_old, tmp, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        U_new.append(fU(z_new))
        
    return U_new, z_new #V_new, W_new, z_new

def tht2temp(THT,P):
#    ss   = P.shape
    p0   = 100000. #np.ones(ss)*100000. #p0
    exner= np.power(P/p0,CC.RD/CC.RCP)
    temp = THT*exner
    return temp

# Create new variables in Meso-NH
def createnew(vv,DATA,var1D,idxzi=None):
    vc   = {'THLM' :('THT','PABST','RCT'),
            'THV'  :('THT','RVT','RCT'),
            'DIVUV':('UT',var1D[1]),\
            'TKE'  :('UT','VT','WT'),\
            'RNPM' :('RVT','RCT') }
    [vc.update({ij:('THT','PABST',var1D[0])}) for ij in ['PRW','LWP','Reflectance']]
    [vc.update({ij:('WT','RVT','RCT','PABST',var1D[0])}) for ij in ['Wstar','Tstar','Thetastar']]

    
    # ATTENTION: PEUT ETRE DES ERREURS 2D/3D !!!!
    
    tmp = tryopen(vv,DATA)

    data = []
    if vv in vc.keys() and tmp is None:
        data      = [tryopen(ij,DATA) for ij in vc[vv]]
        if vv == 'THV' : # THV = THT * (1 + 0.61 RVT - RCT)
            a1      = 0.61
            if data[1] is None:
                data[1] = np.zeros(data[0].shape)
            if data[2] is None:
                data[2] = np.zeros(data[0].shape)
            tmp     = data[0] * (np.ones(data[0].shape) +a1*data[1] - data[2] )
        if vv == 'THLM':
            #thetal = theta - L/Cp Theta/T Ql
            tmp = tryopen('THLM',DATA)
            if tmp is None:
                if data[2] is None: # No clouds
                    data[2] = np.zeros(data[0].shape)
                tmp = data[0] -(\
                     (data[0]/tht2temp(data[0],data[1]))\
                    *(CC.RLVTT/CC.RCP)*data[2])
            tmp = tmp-273.15 # Celsius
        if vv == 'RNPM':
            tmp = tryopen('RNPM',DATA)
            if tmp is None:
                tmp = data[0]
                #print(tmp)
                if data[1] is not None: # No clouds
                    tmp += data[1]
        #if vv == 'DIVUV':
            # DIVUV = DU/DX + DV/DY
        #    dx   = data[1][2]-data[1][1]; print(dx)
        #    tmpU = divergence(data[0], dx, axis=-1) # x-axis
        #    tmp  = tmpU #+ tmpV
            
        if 'TKE' in vv:
            tmp = 0.5*(anomcalc(data[0])**2.\
                  + anomcalc(data[1])**2\
                  + anomcalc(data[2])**2)
            if vv == 'TKEM':
                  tmp = np.sqrt(tmp / (data[0]**2+data[1]**2+data[2]**2) ) 
        if vv == 'PRW' or vv == 'LWP' or vv == 'Reflectance':
            TA   = tht2temp(data[0],data[1])
            rho  = createrho(TA,data[1])
            name= 'RVT'
            if vv == 'LWP' or vv == 'Reflectance':
                name = 'RCT'
            RCT = tryopen(name,DATA)
            if RCT is not None:
                ss  = RCT.shape
                print(ss)
                if len(ss)==3.:
                    zz = repeat(data[2],(ss[1],ss[2]))
                    tmp = np.zeros((1,ss[1],ss[2]))
                    for  ij in range(len(data[2])-1):
                        tmp[0,:,:] += rho[ij,:,:]*RCT[ij,:,:]*(zz[ij+1,:,:]-zz[ij,:,:])
                else:
                    zz = repeat(data[2],[ss[1]])
                    tmp = np.zeros((1,ss[1]))
                    for  ij in range(len(data[2])-1):
                        tmp[0,:] += rho[ij,:]*RCT[ij,:]*(zz[ij+1,:]-zz[ij,:])
                #zz  = np.repeat(np.repeat(data[2][ :,np.newaxis, np.newaxis],ss[1],axis=1),ss[2],axis=2)

            else:
                tmp = None
            if vv == 'Reflectance' and tmp is not None:
                rho_eau  = 1.e+6 #g/m3
                reff     = 5.*1e-6  #10.e-9 #m #
                g        = 0.85
                tmp      = tmp*1000. #kg/m2  --> g/m2
                tmp      = 1.5*tmp/(reff*rho_eau)
                trans    = 1.0/(1.0+0.75*tmp*(1.0-g))  # Quentin Libois
                tmp      = 1.-trans
        if vv == "Wstar" or vv == 'Tstar' or vv=='Thetastar':
            # W_star = g*zi*H0/theta_v(0-zi) #First version
            # Deardroff velocity
            # W_star = (g/T_v * zi * (w'th_v')_surf)^1/3
            # 'Wstar':('WT','RVT','RCT','PABST',var1D[0])
            WT  = data[0]
            print('Check wstar')
            print(WT.shape) # alt,y,x ou alt,x
            THV = createnew('THV',DATA,var1D)
            TV  = tht2temp(THV,data[3])
            #print(THV[0,15],TV[0,15])
            ss  = WT.shape 
            if len(ss)==3:
                WT  = np.reshape(WT,(ss[0],ss[1]*ss[2]))
                THV = np.reshape(THV,(ss[0],ss[1]*ss[2]))
                TV  = np.reshape(TV,(ss[0],ss[1]*ss[2]))
            zz   = data[4]
            
            # Find maximum of W'Thv' (near surface)
            wthv = 0.
            for ij,zz1 in enumerate(zz):
                tmp  = WT[ij,:]*(THV[ij,:]-np.nanmean(THV[ij,:]))
                #print(ij,wthv,np.nanmean(tmp))
                wthv = max(wthv,np.nanmean(tmp))
                #print(ij,zz1,wthv)
            
            
            
            #wthv=WT[0,:]*(THV[0,:]-np.nanmean(THV[0,:])) #surface
            #print(THV[0,:]-np.nanmean(THV[0,:]))
            #wthv=np.nanmean(wthv)
            if idxzi is not None:
                print('PBL top ',zz[idxzi])
                tmp=CC.RG*zz[idxzi]*wthv/np.nanmean(TV[0:idxzi,:])
                #print(CC.RG,zz[idxzi],wthv,np.nanmean(TV[0:idxzi,:]))
                tmp=pow(tmp,1./3.)
            #print('tmp ',tmp,zz[idxzi])
            if vv == 'Tstar' and tmp is not None :
                #t_star = H/w_star
                tmp = zz[idxzi]/tmp # en secondes
            if vv == 'Thetastar' and tmp is not None :
                #theta_star = Qs/w_star
                tmp = wthv/tmp
    return tmp


# # Calculate gradients in the 3 dimensions
# def compute_gradients(u, dx, dy, dz, v=None, w=None):
#     # Compute gradients in the x-direction
#     du_dx = np.gradient(u, dx, axis=-1)
#     # Compute gradients in the y-direction
#     du_dy = np.gradient(u, dy, axis=-2)
    
#     du_dz = None
#     if len(u.shape)>2:
#         # Compute gradients in the z-direction with varying dz
#         du_dz = np.zeros_like(u)
#         # Handle the boundaries
#         #dz_0           = (dz[1] + dz[0])/2.
#         dz_0           = dz[0]
#         du_dz[0, :, :] = (u[1, :, :] - u[0, :, :]) / dz_0
#         #dz_1           = (dz[-1] + dz[-2])/2.
#         dz_1           = dz[-1]
#         du_dz[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) / dz_1

#     dv_dx, dv_dy, dv_dz = [None for ij in range(3)]
#     dw_dx, dw_dy, dw_dz = [None for ij in range(3)]
    
#     if v is not None:
#         dv_dx = np.gradient(v, dx, axis=-1)
#         dv_dy = np.gradient(v, dy, axis=-2)
#         if len(v.shape)>2:
#             dv_dz = np.zeros_like(v)
#             dv_dz[0, :, :] = (v[1, :, :] - v[0, :, :]) / dz_0
#             dv_dz[-1, :, :] = (v[-1, :, :] - v[-2, :, :]) / dz_1
        
#     if w is not None:
#         dw_dx = np.gradient(w, dx, axis=-1)
#         dw_dy = np.gradient(w, dy, axis=-2)
#         if len(w.shape)>2:
#             dw_dz = np.zeros_like(w)
#             dw_dz[0, :, :] = (w[1, :, :] - w[0, :, :]) / dz_0
#             dw_dz[-1, :, :] = (w[-1, :, :] - w[-2, :, :]) / dz_1
    
#     for k in range(1, u.shape[0] - 1):
#         if len(u.shape)>2:
#             du_dz[k, :, :] = (u[k + 1, :, :] - u[k - 1, :, :]) / (2.*dz[k])
#         if v is not None and len(v.shape)>2:
#             dv_dz[k, :, :] = (v[k + 1, :, :] - v[k - 1, :, :]) / (2.*dz[k])
#         if w is not None and len(w.shape)>2:
#             dw_dz[k, :, :] = (w[k + 1, :, :] - w[k - 1, :, :]) / (2.*dz[k])
    

#     return du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz



def _spectral_derivative_1d(field, d, axis):
    """
    Exact derivative along `axis` for a periodic field using FFT.
    field : ndarray, periodic along `axis`
    d     : grid spacing along `axis` (uniform)
    axis  : axis index to differentiate along
    """
    n = field.shape[axis]
    k = 2. * np.pi * np.fft.fftfreq(n, d=d)
    # reshape k to broadcast against `field` along `axis`
    shape = [1] * field.ndim
    shape[axis] = n
    k = k.reshape(shape)

    fhat = np.fft.fft(field, axis=axis)
    dfield_hat = 1j * k * fhat
    dfield = np.fft.ifft(dfield_hat, axis=axis).real
    return dfield


def _vertical_derivative(u, z):
    """
    Centered (non-uniform-grid-correct) vertical derivative, one-sided
    (2nd order accurate) at the two boundaries.
    u : ndarray with vertical dimension along axis 0
    z : 1D array of vertical levels, len == u.shape[0]
    """
    du_dz = np.zeros_like(u)

    # Boundaries: simple one-sided differences
    du_dz[0, ...] = (u[1, ...] - u[0, ...]) / (z[1] - z[0])
    du_dz[-1, ...] = (u[-1, ...] - u[-2, ...]) / (z[-1] - z[-2])

    # Interior: true centered difference on a (possibly) stretched grid
    # d u/dz ~ (u[k+1]-u[k-1]) / (z[k+1]-z[k-1])   -- NOT 2*dz[k]
    z_up = z[2:]      # z[k+1]
    z_dn = z[:-2]     # z[k-1]
    denom = (z_up - z_dn)
    shape = [1] * (u.ndim - 1)
    denom = denom.reshape((-1,) + tuple(shape))

    du_dz[1:-1, ...] = (u[2:, ...] - u[:-2, ...]) / denom

    return du_dz


def compute_gradients(u, dx, dy, z, v=None, w=None):
    """
    Compute 3D gradients of u (and optionally v, w).

    Horizontal (x, y) derivatives are computed spectrally (FFT), assuming
    periodicity in x and y -- this matches the accuracy of your spectral
    workflow and avoids the boundary-discontinuity / Gibbs leakage that
    np.gradient's default one-sided edge treatment introduces on a
    periodic domain.

    Vertical (z) derivative uses a centered finite difference that
    correctly accounts for a non-uniform (stretched) grid:
        du/dz[k] = (u[k+1] - u[k-1]) / (z[k+1] - z[k-1])
    rather than dividing by 2*dz[k], which is only valid on a uniform grid.

    Parameters
    ----------
    u, v, w : ndarray, shape (nz, ny, nx) or (ny, nx)
        v, w optional.
    dx, dy : float
        uniform horizontal grid spacing
    z : 1D ndarray, len nz
        vertical levels (need not be uniformly spaced). Required if
        u.ndim > 2.

    Returns
    -------
    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
    (None for any component not provided / not applicable)
    """
    axis_x = -1
    axis_y = -2

    du_dx = _spectral_derivative_1d(u, dx, axis_x)
    du_dy = _spectral_derivative_1d(u, dy, axis_y)

    du_dz = None
    if u.ndim > 2:
        if z is None:
            raise ValueError("z levels must be provided for 3D fields")
        du_dz = _vertical_derivative(u, z)

    dv_dx = dv_dy = dv_dz = None
    if v is not None:
        dv_dx = _spectral_derivative_1d(v, dx, axis_x)
        dv_dy = _spectral_derivative_1d(v, dy, axis_y)
        if v.ndim > 2:
            dv_dz = _vertical_derivative(v, z)

    dw_dx = dw_dy = dw_dz = None
    if w is not None:
        dw_dx = _spectral_derivative_1d(w, dx, axis_x)
        dw_dy = _spectral_derivative_1d(w, dy, axis_y)
        if w.ndim > 2:
            dw_dz = _vertical_derivative(w, z)

    return du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz


def checkvariance(k,E,field,type='var'):
    varspec = np.trapz(E,x=k)
    if type=='mean':
        varfield= np.mean(field)
    else:
        varfield= np.var(field)
    print('variance Spectra ',varspec)
    print('variance Field ',varfield)
    perc=100*varspec/varfield
    sentence='The spectra represent {perc}% of the {type} of the field'
    print(sentence.format(perc=perc,type=type))
    return None