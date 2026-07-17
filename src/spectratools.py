#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:06:11 2026

@author: fbrient
"""

import numpy as np
import tools as tl
import time
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn, dstn, idstn


def compute_spectral_transfer(U, V, W, dx=50.0, dy=50.0, dz=10.0,
                              scalar=None,
                              norm='ortho', dealiasing_23=False,
                              mirror=False,
                              binning='log',n_bins=80):
    """
    Compute spectral KE E(kx,ky,kz), nonlinear transfer T(kx,ky,kz), 
    and shell-averaged E(k), T(k).
    
    Inputs:
      U,V,W : numpy arrays shape (Nz,Ny,Nx) - physical space velocities (m/s)
      dx,dy,dz : grid spacings (m)
      norm : FFT normalization, use 'ortho' for Parseval-friendly behaviour
      dealiasing_23 : apply 2/3 rule de-aliasing
      binning : log bin by default
      scalar: If None, numpy array with same shape as U (ex: temperature)
      mirror: Make 3D field mirror along the z=0
      fast: Run only the last PI (no energy spectra)

    Returns:
      dict with keys:
        'E_k3' : 3D array shape (Nz,Ny,Nx) of spectral KE density per mode
        'T_k3' : 3D array of spectral transfer per mode
        'kx','ky','kz' : 3D arrays of wavenumbers (rad/m)
        'k_shell_centers' : 1D k-shell center values
        'E_k' : 1D shell-averaged E(k)
        'T_k' : 1D shell-averaged T(k)
        'mode_count' : number of modes per shell
    """
    
    # Perform mirror
    # if mirror:
    #     U = np.concatenate((U[::-1, :, :], U), axis=0)
    #     V = np.concatenate((V[::-1, :, :], V), axis=0)
    #     W = np.concatenate((-W[::-1, :, :], W), axis=0)

    
    # shapes
    Nz, Ny, Nx = U.shape
    assert V.shape == U.shape and W.shape == U.shape
    if scalar is not None:
        assert scalar.shape == U.shape
    

    # build wavenumber arrays (rad/m)
    kx_1d = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
    ky_1d = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
    kz_1d = 2*np.pi * np.fft.fftfreq(Nz, d=dz)
    kz, ky, kx = np.meshgrid(kz_1d, ky_1d, kx_1d, indexing='ij')  # shape (Nz,Ny,Nx)
    
    k_mag      = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag_flat = k_mag.ravel()
    k_min      = np.min(k_mag_flat[k_mag_flat > 0])
    k_max      = np.max(k_mag_flat)
    
    if binning.lower() == 'log':
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins+1)
        # use geometric mean as center
        k_shell_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    else:
        k_bins = np.linspace(0.0, k_max, n_bins+1)
        k_shell_centers = 0.5*(k_bins[:-1] + k_bins[1:])          

    # Gradients
    #dzall     = np.repeat(dz,Nz)
    zall      = np.arange(0,Nz)*dz
    gradients = tl.compute_gradients(U, dx, dy, zall, v=V, w=W)
    (du_dx, dv_dx, dw_dx, 
     du_dy, dv_dy, dw_dy, 
     du_dz, dv_dz, dw_dz) = gradients
    
    # nonlinear term N = u · ∇u (vector)
    N_uu = U * du_dx + V * du_dy
    N_uw = W * du_dz
    N_vu = U * dv_dx + V * dv_dy
    N_vw = W * dv_dz
    N_wu = U * dw_dx + V * dw_dy
    N_ww = W * dw_dz
    N_u  = N_uu + N_uw
    N_v  = N_vu + N_vw
    N_w  = N_wu + N_ww
    if scalar is not None:
        gradientsS = tl.compute_gradients(scalar, dx, dy, dz)
        (dscalar_dx, a, d, 
         dscalar_dy, b, e, 
         dscalar_dz, c, f) = gradientsS
        # Nonlinear term n= U ∇THLM (vector)
        N_scalar_u = U * dscalar_dx + V * dscalar_dy
        N_scalar_w = W * dscalar_dz
        N_scalar   = N_scalar_u + N_scalar_w
        
        
    time1 = time.time()
    if scalar is None:
        k, PI_k, PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF_v0(U,V,W,
                                    N_u,N_v,N_w,
                                    kk=k_bins,
                                    dx=dx,dy=dy,dz=dz,
                                    Nhh=(N_uu,N_vu),Nhv=(N_uw,N_vw),
                                    Nvh=(N_wu),Nvv=(N_ww))
        
        kk1, PI_3d, *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w, 
#                                             kk=k_bins,
                                             dx=dx,dy=dy,dz=dz,filter_type='spectral_3d')
        kk2, PI_h,  *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,  
#                                             kk=k_bins,
                                             dx=dx,dy=dy,dz=dz,filter_type='spectral_h')
        kk3, PI_g,  *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,  
#                                             kk=k_bins,
                                             dx=dx,dy=dy,dz=dz,filter_type='gaussian')
        kk4, PI_hz,  *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,  
#                                             kk=k_bins,
                                             dx=dx,dy=dy,dz=dz,filter_type='spectral_hz')

    else:
        k, PI_k,PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF(U,V,W,
                                    N_u,N_v,N_w,
                                    scalar=(scalar,N_scalar_u,N_scalar_w),
                                    kk=k_bins,dx=dx,dy=dy,dz=dz)
    time2 = time.time()
    print('%s function took %0.3f ms' % ("Calculate PI_k2", (time2-time1)*1000.0))
    
    # For information, Egality is:
    # Pi_k2[1] = Pi_k/(nx*ny*nz)
    
    #for idxk,k_idx in enumerate(k_mag):
    #    tmp_PI = u_k[idxk,:,:,:]*N_u+\
    #             v_k[idxk,:,:,:]*N_v+\
    #             w_k[idxk,:,:,:]*N_w 
    #    PI_k2  = np.mean(tmp_PI,axis=(1,2,3))
    

    
    out = dict(k=k, 
               PI_k=PI_k, PI_hh=PI_hh, PI_hv=PI_hv,
               PI_vh=PI_vh, PI_vv=PI_vv,
               kk1=kk1, PI_3d=PI_3d,
               kk2=kk2, PI_h=PI_h,
               kk3=kk3, PI_g=PI_g,
               kk4=kk4, PI_hz=PI_hz)
    return out


def compute_Pi_from_uBF_v0(U, V, W, N_u, N_v, N_w,
                        scalar=None, # 3D (scalar,N_scalar_u,N_scalar_w)
                        Nhh=None,Nhv=None,
                        Nvh=None,Nvv=None,
                        kk=None, dx=1.0, dy=1.0, dz=1.0):

    nz, ny, nx = U.shape

    # --- FFTs ---
    U_hat = np.fft.fftn(U)
    V_hat = np.fft.fftn(V)
    W_hat = np.fft.fftn(W)

    # --- Wavenumber grid ---
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)

    kz, ky, kx = np.meshgrid(kz, ky, kx, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    # --- Define cutoff values if not given ---
    if kk is None:
        kk = np.linspace(0.0, k_mag.max(), 30)
        

    kk = np.asarray(kk)
    nk = len(kk)

    # --- Output ---
    PI_k2 = np.empty(nk)
    PI_hh,PI_hv,PI_vh,PI_vv=[np.empty(nk) for ij in range(4)]

    # --- Reusable buffers ---
    U_hatf = np.empty_like(U_hat)
    V_hatf = np.empty_like(V_hat)
    W_hatf = np.empty_like(W_hat)
    if scalar is not None:
        scalar_hat  = np.fft.fftn(scalar[0])
        scalar_hatf = np.empty_like(scalar_hat)
        N_scalar_u  = scalar[1]
        N_scalar_w  = scalar[2]
        N_scalar    = N_scalar_u + N_scalar_w
    

    # --- Main loop over cutoff radii ---
    for i, k_cut in enumerate(kk):

        mask = (k_mag <= k_cut)

        np.multiply(U_hat, mask, out=U_hatf)
        np.multiply(V_hat, mask, out=V_hatf)
        np.multiply(W_hat, mask, out=W_hatf)

        u_f = np.fft.ifftn(U_hatf).real
        v_f = np.fft.ifftn(V_hatf).real
        w_f = np.fft.ifftn(W_hatf).real
        
#        u_f = gaussian_filter(U,
#                      sigma=(sigma_z,
#                             sigma_y,
#                             sigma_x),
#                      mode=('nearest',
#                            'wrap',
#                            'wrap'))

        if scalar is None:
            PI_k2[i] = np.mean(
                u_f * N_u +
                v_f * N_v +
                w_f * N_w
            )
            if Nhh is not None:
                PI_hh[i]=np.mean(u_f*Nhh[0]+v_f*Nhh[1])
            if Nhv is not None:
                PI_hv[i]=np.mean(u_f*Nhv[0]+v_f*Nhv[1])
            if Nvh is not None:
                PI_vh[i]=np.mean(w_f*Nvh)
            if Nvv is not None:
                PI_vv[i]=np.mean(w_f*Nvv)
        else:
            np.multiply(scalar_hat, mask, out=scalar_hatf)
            scalar_f = np.fft.ifftn(scalar_hatf).real
            PI_k2[i] = np.mean(scalar_f * N_scalar)
            PI_hh[i] = np.mean(scalar_f * N_scalar_u)
            PI_hv[i] = np.mean(scalar_f * N_scalar_w)
            

    return kk, PI_k2, PI_hh, PI_hv, PI_vh, PI_vv



def compute_Pi_from_uBF_v1(U, V, W, N_u, N_v, N_w,
                        scalar=None,
                        Nhh=None, Nhv=None,
                        Nvh=None, Nvv=None,
                        kk=None, dx=1.0, dy=1.0, dz=1.0,
                        binning='log',nbins=80,
                        filter_type='spectral_h'):
    """
    filter_type : 'spectral_3d' | 'spectral_h' | 'gaussian'
        - spectral_3d : coupure isotrope sur k=sqrt(kx²+ky²+kz²) (suppose periodicite en z, biaisee)
        - spectral_h  : coupure sur kh=sqrt(kx²+ky²) uniquement, z inchange (recommande, exact, pas d'hyp. periodicite en z)
        - gaussian    : filtre reel anisotrope, wrap en x,y, nearest en z (test de sensibilite)
    """
    nz, ny, nx = U.shape

    # --- FFTs (necessaires pour spectral_3d et spectral_h) ---
    U_hat = np.fft.fftn(U)
    V_hat = np.fft.fftn(V)
    W_hat = np.fft.fftn(W)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    kz3, ky3, kx3 = np.meshgrid(kz, ky, kx, indexing='ij')
    k_mag  = np.sqrt(kx3**2 + ky3**2 + kz3**2)   # pour spectral_3d
    kh_mag = np.sqrt(kx3**2 + ky3**2)            # pour spectral_h (indep. de kz)

    if kk is None:
        ktmp=k_mag
        if filter_type == 'spectral_h':
            ktmp=kh_mag                     
        kmax = ktmp.max() #if filter_type == 'spectral_h' else k_mag.max()
        kmin = np.min(ktmp[ktmp > 0]) #if filter_type == 'spectral_h' else np.min(k_mag[k_mag > 0])
        print(kmin,kmax,nbins)
        if binning.lower() == 'log':
            kk = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
        else:
            kk = np.linspace(0.0, kmax, nbins+1)
        print('kk ',kk)
#        stop
    kk = np.asarray(kk)
    nk = len(kk)

    PI_k2 = np.empty(nk)
    PI_hh, PI_hv, PI_vh, PI_vv = [np.empty(nk) for _ in range(4)]

    U_hatf = np.empty_like(U_hat)
    V_hatf = np.empty_like(V_hat)
    W_hatf = np.empty_like(W_hat)
    if scalar is not None:
        scalar_hat  = np.fft.fftn(scalar[0])
        scalar_hatf = np.empty_like(scalar_hat)
        N_scalar_u  = scalar[1]
        N_scalar_w  = scalar[2]
        N_scalar    = N_scalar_u + N_scalar_w

    # borne min pour eviter sigma -> inf a k_cut=0 (gaussien)
    k_floor = kk[kk > 0].min() * 0.5 if np.any(kk > 0) else 1e-6

    for i, k_cut in enumerate(kk):

        if filter_type in ('spectral_3d', 'spectral_h'):
            mask = (k_mag <= k_cut) if filter_type == 'spectral_3d' else (kh_mag <= k_cut)
            np.multiply(U_hat, mask, out=U_hatf)
            np.multiply(V_hat, mask, out=V_hatf)
            np.multiply(W_hat, mask, out=W_hatf)
            u_f = np.fft.ifftn(U_hatf).real
            v_f = np.fft.ifftn(V_hatf).real
            w_f = np.fft.ifftn(W_hatf).real
            if scalar is not None:
                np.multiply(scalar_hat, mask, out=scalar_hatf)
                scalar_f = np.fft.ifftn(scalar_hatf).real

        elif filter_type == 'gaussian':
            sigma_phys = 1.0 / max(k_cut, k_floor)   # relation approx k_cut <-> sigma
            sigma_pix  = (sigma_phys / dz, sigma_phys / dy, sigma_phys / dx)  # ordre (z,y,x)
            modes = ('nearest', 'wrap', 'wrap')      # z non periodique, x,y periodiques
            u_f = gaussian_filter(U, sigma=sigma_pix, mode=modes)
            v_f = gaussian_filter(V, sigma=sigma_pix, mode=modes)
            w_f = gaussian_filter(W, sigma=sigma_pix, mode=modes)
            if scalar is not None:
                scalar_f = gaussian_filter(scalar[0], sigma=sigma_pix, mode=modes)
        else:
            raise ValueError(f"filter_type inconnu: {filter_type}")

        if scalar is None:
            PI_k2[i] = np.mean(u_f * N_u + v_f * N_v + w_f * N_w)
            if Nhh is not None: PI_hh[i] = np.mean(u_f*Nhh[0] + v_f*Nhh[1])
            if Nhv is not None: PI_hv[i] = np.mean(u_f*Nhv[0] + v_f*Nhv[1])
            if Nvh is not None: PI_vh[i] = np.mean(w_f*Nvh)
            if Nvv is not None: PI_vv[i] = np.mean(w_f*Nvv)
        else:
            PI_k2[i] = np.mean(scalar_f * N_scalar)
            PI_hh[i] = np.mean(scalar_f * N_scalar_u)
            PI_hv[i] = np.mean(scalar_f * N_scalar_w)

    return kk, PI_k2, PI_hh, PI_hv, PI_vh, PI_vv


def compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,
                        scalar=None,
                        Nhh=None, Nhv=None,
                        Nvh=None, Nvv=None,
                        kk=None, dx=1.0, dy=1.0, dz=1.0,
                        binning='log',nbins=80,
                        filter_type='spectral_3d',
                        w_bc='dct'):   # 'dst' (w=0 aux bords) ou 'dct' (dw/dz=0)
    """
    filter_type :
      'spectral_3d' : FFT 3D complete, coquille isotrope (biaisee: periodicite en z + sous-resolution kz)
      'spectral_h'  : coupure sur kh seul, z inchange (pas de cascade h<->v)
      'spectral_hz' : FFT en x,y (periodique) + DCT/DST en z (borne, sans wraparound) - RECOMMANDE
      'gaussian'    : noyau reel anisotrope, lisse (test de sensibilite)
    """
    nz, ny, nx = U.shape

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    # nombres d'onde DCT-II / DST-II : k_n = pi*n/(nz*dz), n=0..nz-1 (DCT), n=1..nz (DST)
    n_dct = np.arange(nz)
    n_dst = np.arange(1, nz + 1)
    kz_dct = np.pi * n_dct / (nz * dz)
    kz_dst = np.pi * n_dst / (nz * dz)

    ky2, kx2 = np.meshgrid(ky, kx, indexing='ij')
    kh_mag = np.sqrt(kx2**2 + ky2**2)  # (ny, nx), sert pour spectral_h et brique horizontale de spectral_hz

    def make_k3(kz_1d):
        kz3, ky3, kx3 = np.meshgrid(kz_1d, ky, kx, indexing='ij')
        return np.sqrt(kx3**2 + ky3**2 + kz3**2)

    if kk is None:
        #kref = kh_mag.max() if filter_type == 'spectral_h' else \
        #       make_k3(kz_dst if w_bc=='dst' else kz_dct).max()
               
        #kk = np.linspace(0.0, kref, 30)
        
        ktmp=make_k3(kz_dst if w_bc=='dst' else kz_dct)
        if filter_type == 'spectral_h':
            ktmp=kh_mag  
        kmax = ktmp.max() #if filter_type == 'spectral_h' else k_mag.max()
        kmin = np.min(ktmp[ktmp > 0]) #if filter_type == 'spectral_h' else np.min(k_mag[k_mag > 0])
        print(kmin,kmax,nbins)
        if binning.lower() == 'log':
            kk = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
        else:
            kk = np.linspace(0.0, kmax, nbins+1)
        print('kk ',kk)
        
    kk = np.asarray(kk)
    nk = len(kk)
    
    PI_k2 = np.empty(nk)
    PI_hh, PI_hv, PI_vh, PI_vv = [np.empty(nk) for _ in range(4)]

    # --- pre-transformees selon la methode ---
    if filter_type == 'spectral_3d':
        U_hat, V_hat, W_hat = np.fft.fftn(U), np.fft.fftn(V), np.fft.fftn(W)
        kz1 = 2.0*np.pi*np.fft.fftfreq(nz, d=dz)
        k_mag = make_k3(kz1)
        if scalar is not None:
            scalar_hat = np.fft.fftn(scalar[0])

    elif filter_type == 'spectral_h':
        U_hat, V_hat, W_hat = np.fft.fftn(U), np.fft.fftn(V), np.fft.fftn(W)
        k_mag = np.broadcast_to(kh_mag, (nz, ny, nx))
        if scalar is not None:
            scalar_hat = np.fft.fftn(scalar[0])

    elif filter_type == 'spectral_hz':
        # FFT horizontale (periodique, axes y,x) puis DCT/DST verticale (axe z) sur le champ deja en Fourier horizontal
        def fwd(field):
            f_hat_h = np.fft.fftn(field, axes=(1, 2))          # complexe, periodique en x,y
            # DCT/DST reelle ne gere pas le complexe directement -> traiter re/im separement
            tfun = dctn if w_bc == 'dct' else dstn
            re = tfun(f_hat_h.real, axes=(0,), type=2, norm='ortho')
            im = tfun(f_hat_h.imag, axes=(0,), type=2, norm='ortho')
            return re + 1j * im
        def bwd(field_hat):
            tfun_inv = idctn if w_bc == 'dct' else idstn
            re = tfun_inv(field_hat.real, axes=(0,), type=2, norm='ortho')
            im = tfun_inv(field_hat.imag, axes=(0,), type=2, norm='ortho')
            return np.fft.ifftn(re + 1j * im, axes=(1, 2)).real

        U_hat, V_hat, W_hat = fwd(U), fwd(V), fwd(W)
        kz1 = kz_dct if w_bc == 'dct' else kz_dst
        k_mag = make_k3(kz1)
        if scalar is not None:
            scalar_hat = fwd(scalar[0])

    elif filter_type != 'gaussian':
        raise ValueError(f"filter_type inconnu: {filter_type}")

    if scalar is not None and filter_type in ('spectral_3d','spectral_h','spectral_hz'):
        scalar_hatf = np.empty_like(scalar_hat)
    if filter_type in ('spectral_3d','spectral_h','spectral_hz'):
        U_hatf, V_hatf, W_hatf = (np.empty_like(U_hat) for _ in range(3))

    k_floor = kk[kk > 0].min() * 0.5 if np.any(kk > 0) else 1e-6

    for i, k_cut in enumerate(kk):

        if filter_type in ('spectral_3d', 'spectral_h', 'spectral_hz'):
            mask = (k_mag <= k_cut)
            np.multiply(U_hat, mask, out=U_hatf)
            np.multiply(V_hat, mask, out=V_hatf)
            np.multiply(W_hat, mask, out=W_hatf)

            if filter_type == 'spectral_hz':
                u_f, v_f, w_f = bwd(U_hatf), bwd(V_hatf), bwd(W_hatf)
            else:
                u_f = np.fft.ifftn(U_hatf).real
                v_f = np.fft.ifftn(V_hatf).real
                w_f = np.fft.ifftn(W_hatf).real

            if scalar is not None:
                np.multiply(scalar_hat, mask, out=scalar_hatf)
                scalar_f = bwd(scalar_hatf) if filter_type=='spectral_hz' else np.fft.ifftn(scalar_hatf).real

        elif filter_type == 'gaussian':
            sigma_phys = 1.0 / max(k_cut, k_floor)
            sigma_pix  = (sigma_phys/dz, sigma_phys/dy, sigma_phys/dx)
            modes = ('nearest', 'wrap', 'wrap')
            u_f = gaussian_filter(U, sigma=sigma_pix, mode=modes)
            v_f = gaussian_filter(V, sigma=sigma_pix, mode=modes)
            w_f = gaussian_filter(W, sigma=sigma_pix, mode=modes)
            if scalar is not None:
                scalar_f = gaussian_filter(scalar[0], sigma=sigma_pix, mode=modes)

        if scalar is None:
            PI_k2[i] = np.mean(u_f*N_u + v_f*N_v + w_f*N_w)
            if Nhh is not None: PI_hh[i] = np.mean(u_f*Nhh[0] + v_f*Nhh[1])
            if Nhv is not None: PI_hv[i] = np.mean(u_f*Nhv[0] + v_f*Nhv[1])
            if Nvh is not None: PI_vh[i] = np.mean(w_f*Nvh)
            if Nvv is not None: PI_vv[i] = np.mean(w_f*Nvv)
        else:
            PI_k2[i] = np.mean(scalar_f * (N_scalar := scalar[1] + scalar[2]))
            PI_hh[i] = np.mean(scalar_f * scalar[1])
            PI_hv[i] = np.mean(scalar_f * scalar[2])

    return kk, PI_k2, PI_hh, PI_hv, PI_vh, PI_vv
