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
from scipy.ndimage.filters import gaussian_filter1d
import xarray as xr
from glob import glob
import pylab as plt

from PIL import Image
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.cm as cmplt

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

# Make Dir
def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

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
    varspec = np.trapz(E[~np.isnan(E)],x=k[~np.isnan(E)])
    if type=='mean':
        varfield= np.nanmean(field)
    else:
        varfield= np.nanvar(field)
    print('variance Spectra ',varspec)
    print('variance Field ',varfield)
    perc=100*varspec/varfield
    sentence='The spectra represent {perc}% of the {type} of the field'
    print(sentence.format(perc=perc,type=type))
    return None


def nanweighted_average(data, axis=-1, weights=None):
    """
    Compute a weighted average ignoring NaN values.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    axis : int or tuple of ints
        Axis (or axes) along which to average.
    weights : array-like, optional
        Weights for the average. Must be broadcastable to 'data'.

    Returns
    -------
    np.ndarray
        The weighted average with NaNs ignored.
    """
    # Mask NaNs
    masked = np.ma.masked_invalid(data)
    
    # Compute masked weighted average
    result = np.ma.average(masked, axis=axis, weights=weights)
    
    # Return as normal ndarray (with NaNs where all elements were masked)
    return result.filled(np.nan)

    

# Smooth line
def smooth(y,sigma=2):
    return gaussian_filter1d(y, sigma=sigma)

def findkvmax(kv,tmp,weights=None,sigma=1):
    
    if weights is not None:
        tmp = nanweighted_average(tmp,weights=weights) 

    tmp     = tmp[~np.isnan(tmp)]    
    Er      = np.tile(tmp, (1, 1))
    Ers     = smooth(tmp,sigma=sigma)
    
    plt.loglog(Er);plt.loglog(Ers,'r');plt.show()
    print(np.isnan(Ers.max()))

    kvmax   = None
    if not np.isnan(Ers.max()):
        kvmax   = kv[Ers.argmax()]
    
    return kvmax,Ers,Er

def read_netcdfs(files, dim, concat=True):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    print(len(paths),paths)
    combined = [xr.open_dataset(p) for p in paths]
    if concat:
        combined = xr.concat(combined, dim)
    return combined


############### FIGsURES ###########################

# Convert to np array
def fig_to_np_array(fig):
    """Convert a Matplotlib figure to a NumPy array."""
    fig.canvas.draw()  # Draw the canvas to update it
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

# Save figure
def savefig(fig, path):
    #Image.fromarray(mplfig_to_npimage(fig)).save(path)
    np_image = fig_to_np_array(fig)
    Image.fromarray(np_image).save(path)


# adjust spines in figures
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 0))  # outward by 10 points
            #print(dir(spine))
            # FB : Need to update with Python3
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def plot_flux(k,E,PI=None,
              kPBL=None,kin=None,
              kcell=None,\
              Euv=None,\
              y1lab='xlab',y2lab='ylab',\
              normalized=False,logy=True,\
              namefig='namefig',plotlines=False,\
              smooth=None,labels=None,\
              showobs=False,
              colors=None,linestyles=None,\
              xsize=(12,10),fts=18,lw=2.5):
    
    # Start plot
    fig, ax1 = plt.subplots(1,1,figsize=xsize)
    ax2 = False
    if PI is not None:
        xsize=(12,16)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=xsize)

    # By default
    namex = 'Wavenumber'

    # If multiple time
    ss = np.shape(E)
    if ss != np.shape(k):
        k  = np.repeat(k[np.newaxis, :], ss[0], axis=0)  # Adds new axis first
    if ss[0] != len(kPBL):
        kPBL  = np.repeat(kPBL[np.newaxis, :], ss[0], axis=0)  # Adds new axis first
    
    print(ss,ss[0],len(ss),normalized)
    if len(ss) > 1 or normalized:
        normalized=True
        # Normalisation of k by kPBL
        for ij in range(ss[0]):
            print('k,kPBL: ',ij,k[ij].shape,kPBL[ij].shape)
            #k[ij,:]=k[ij,:]/kPBL[ij]
            k[ij]=k[ij]/kPBL[ij]
            namex = r"$k/k_h$"
            
    
    if normalized:
        if kin is not None:
            kin=kin/kPBL[ij]
        if kcell is not None:
            kcell=kcell/kPBL[ij]

    kmax = max([ij.max() for ij in k])

    xline=0
    if normalized:
        xline = 1
    else:
        if kPBL is not None :
            xline = kPBL[-1]

    # colors
    dt     = 1

    if linestyles is None:
        linestyles = np.repeat('-',ss[0])
        
    if ss[0] < 6:
        if colors is None:
            colors = ['b','r','g','c','m','y']
        if labels is None:
            time   = int(namefig.split('_')[-1])+dt
            labels = ['t+'+str(time)]
    else:
        if colors is None:
            # Create colormap and labels
            # def twilight_shifted
            #colors = cmplt.Spectral(np.linspace(0, 1, ss[0]))
            colors = cmplt.seismic(np.linspace(0, 1, ss[0]))
        

        if labels is None:
            labels = ['t+'+str(ij+dt).zfill(2)  for ij in range(ss[0])] 
    print('labels ',labels)
    
    # Plot Spectra and Energy
    for ij in range(ss[0]):
        ktmp = np.squeeze(k[ij][:])
        ax1.semilogx(ktmp,E[ij][:],lw=lw, color=colors[ij], ls=linestyles[ij], label=labels[ij])
        if smooth is not None:
            ax1.semilogx(ktmp,smooth[ij][:],lw=lw-1, color='r',ls='--')
        if ax2:
            ax2.semilogx(ktmp,PI[ij][:],lw=lw, color=colors[ij], ls=linestyles[ij],label=labels[ij])

    if logy:
        ax1.set_yscale("log")

    if plotlines and logy:
        #k0max=kmax/2
        #k0min=kmax/4
        #k0 = np.linspace(k0min,k0max,1000)
        k1max=kmax/2
        k1min=kmax/6
        k1 = np.linspace(k1min,k1max,1000)
        k0 = k1
        
        # scale to plot slopes - calculate offset
        offset = xline**(-5/3.) # what the slope see

        mean   = E[-1][near(k[-1][:],xline)] # what the real y-axis plot
        #mean   = np.max(E[-1]) # second try
        
        #mean  *= 2.
        k1scale = mean/offset #3e-2
        # pentes en -5/3
        ax1.plot(k1,k1scale*k1**(-5/3.),color='gray',linewidth=3,linestyle='--',label=r'$\mathbf{k^{-5/3}}$')        
        
        if plotlines == 2:
            offset = xline**(3.) # what the slope see
            k0scale = mean*(5)/offset #kPBL[-1]**(-3.)#*(k1min/k0min) #3e-3
            #print(mean,offset,kmax,k1scale)
    
            # pentes en -3
            ax1.plot(k0,k0scale*k0**(-3),color='gray',linewidth=3,linestyle='-',label=r'$\mathbf{k^{-3}}$')
            
        # Legends
    ax1.legend(title=None,shadow=True,numpoints=1,loc=2,
               bbox_to_anchor=(1.0,1.0),
               fontsize=12,title_fontsize=20)
        
    #ax1.set_title('Original Signal')
    ax1.set_ylabel(y1lab,fontsize=fts)
    if logy:
        ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
        ax1.yaxis.set_major_formatter(LogFormatterSciNotation())
    if ax2:
        ax2.set_ylabel(y2lab,fontsize=fts)
        ax2.axhline(y=0,color='k')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Uses scientific notation when needed
        ax2.yaxis.set_major_formatter(formatter)
        ax2.yaxis.get_offset_text().set_fontsize(12)  # Adjust font size as needed
    
    # Second axis (for Spectra only)
    second_axis='bottom'
    if second_axis == 'top':
        func = z2k; namex2 = r'$\mathbf{\lambda \;(m)}$'
        if normalized:
            namex2 = r'$\mathbf{\lambda/H \;}$'
            func=(lambda x: 1 / x)
        secax = ax1.secondary_xaxis('top', functions=(func, func))
        secax.set_xlabel(namex2,fontsize=fts,labelpad=15)
        for label in secax.get_xticklabels():
            label.set_fontsize(fts)
            #label.set_fontweight('bold')
        secax.tick_params('both', length=8, width=2, which='major', direction='out')
        secax.tick_params('both', length=4, width=1, which='minor', direction='out')
    else:
        func = z2k; namex2 = r'$\lambda \;(m)}$'
        if normalized:
            namex2 = r'$\mathbf{\lambda/H \;}$'
            func=(lambda x: 1 / x)
        secax = ax1.secondary_xaxis('bottom', functions=(func, func))
        secax.spines["bottom"].set_position(("data", -1))
        secax.set_xlabel(namex2,fontsize=fts,labelpad=-50)
        for label in secax.get_xticklabels():
            label.set_fontsize(fts)
            #label.set_fontweight('bold')
        secax.tick_params(axis="x", length=8, width=2, which='major', direction='in',pad=-30, colors="red")
        secax.tick_params(axis="x", length=4, width=1, which='minor', direction='in',pad=-30, colors="red")
        secax.xaxis.label.set_color("red")
        
        # It doesn't work
        # Add a second set of tick labels below the x-axis
        # secax = ax1.twiny()  # Create a twin x-axis (it will align automatically)
        # #secax.set_scale("log")
        # secax.set_xlim(ax1.get_xlim())  # Ensure both axes align
        
        # # Define tick locations for wavelength (bottom)
        # wavelength_ticks = np.logspace(-4, -1, num=4)  # Example: 10^2, 10^3, 10^4
        # ax1.set_xticks(wavelength_ticks)
        # ax1.xaxis.set_major_locator(LogLocator(base=10.0))  # Log ticks

        # # Convert wavelength ticks to wave number for top axis
        # wavenumber_ticks = z2k(wavelength_ticks[::-1])  # Reverse order for correct alignment
        # secax.set_xticks(wavenumber_ticks)
        # secax.xaxis.set_major_locator(LogLocator(base=10.0))  # Log ticks
        # secax.set_xticklabels([f"{k:.1e}" for k in wavenumber_ticks])  # Scientific notation
        
        # # Set axis labels
        # secax.set_xlabel(r"Wave Number ($k = 2\pi/\lambda$) [1/nm]", fontsize=12, labelpad=10)

        # Add grey shaded area between aspect 30 and 40 (See Wood and Hartmann, 2006)
        # If normalized, Axis is k/kH -> Pour z/H = 40 --> k/kH = 1/40
        if normalized and showobs:
            k30 = 1./25.
            k40 = 1./40.
            ax1.axvspan(k40, k30, color="grey", alpha=0.3)  # Adjust alpha for transparency
    
      
    if not logy:
        ax1.axhline(y=0,color='k')
    # Format x-axis and y-axis in scientific notation
    #formatter.set_powerlimits((-3, 3))  # Defines range for scientific notation
   
    axall = [ax1]
    if ax2:
        axall+=[ax2]
    for ax in axall:
        if xline is not None:
            ax.axvline(x=xline,color='k',ls='--')
        if kin is not None:
            ax.axvline(x=kin,color='g',ls='--')
        if kcell is not None:
            ax.axvline(x=kcell,color='r',ls='--')
        
        ax.set_xlabel(namex,fontsize=fts)
        ax.tick_params(axis='both', labelsize=fts)
        adjust_spines(ax,['left', 'bottom'])
     
    # useful?
    #plt.tight_layout()
    # Save figure
    namefig=namefig.replace('XXXX',y2lab)+'.png'
    savefig(fig, namefig)
    plt.close('all')
    
    return None

