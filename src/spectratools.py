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

import pylab as plt

def compute_spectral_transfer(winds, dx=50.0, dy=50.0, dz=10.0,
                              scalar=None,
                              windsb=None,windsc=None,
                              norm='ortho', dealiasing_23=False,
                              binning='log',nbins=80):
    """
    Compute spectral KE E(kx,ky,kz), nonlinear transfer T(kx,ky,kz), 
    and shell-averaged E(k), T(k).
    
    Inputs:
      winds: U,V,W: 3 numpy arrays shape (Nz,Ny,Nx) - physical space velocities (m/s)
      dx,dy,dz : grid spacings (m)
      norm : FFT normalization, use 'ortho' for Parseval-friendly behaviour
      dealiasing_23 : apply 2/3 rule de-aliasing
      binning : log bin by default
      scalar: If None, numpy array with same shape as U (ex: temperature)
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

    # Retrieve winds
    U, V, W  = winds
    Ub,Vb,Wb = winds
    Uc,Vc,Wc = winds
    if windsb is not None:
        Ub,Vb,Wb = windsb
    if windsc is not None:
        Uc,Vc,Wc = windsc

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
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), nbins+1)
        # use geometric mean as center
        k_shell_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    else:
        k_bins = np.linspace(0.0, k_max, nbins+1)
        k_shell_centers = 0.5*(k_bins[:-1] + k_bins[1:])          

    # Gradients
    #dzall     = np.repeat(dz,Nz)
    zall      = np.arange(0,Nz)*dz
    gradients = tl.compute_gradients(Uc, dx, dy, zall, v=Vc, w=Wc)
    (du_dx, dv_dx, dw_dx, 
     du_dy, dv_dy, dw_dy, 
     du_dz, dv_dz, dw_dz) = gradients
    
    # nonlinear term N = u · ∇u (vector)
    N_uu = Ub * du_dx + Vb * du_dy
    N_uw = Wb * du_dz
    N_vu = Ub * dv_dx + Vb * dv_dy
    N_vw = Wb * dv_dz
    N_wu = Ub * dw_dx + Vb * dw_dy
    N_ww = Wb * dw_dz
    N_u  = N_uu + N_uw
    N_v  = N_vu + N_vw
    N_w  = N_wu + N_ww
    if scalar is not None:
        gradientsS = tl.compute_gradients(scalar, dx, dy, dz)
        (dscalar_dx, a, d, 
         dscalar_dy, b, e, 
         dscalar_dz, c, f) = gradientsS
        # Nonlinear term n= U ∇THLM (vector)
        N_scalar_u = Ub * dscalar_dx + Vb * dscalar_dy
        N_scalar_w = Wb * dscalar_dz
#        N_scalar   = N_scalar_u + N_scalar_w
        
    # Calculate Energy spectra
    k,mode_count,E,T,Pi     =  compute_E(
                                k_bins,k_mag,U,N_u,
                                V=(V,N_v),W=(W,N_w))
    
        
    
    kk1,kk2 = [None for ij in range(2)]
    PI_k, PI_hh, PI_hv, PI_vh, PI_vv = [None for ij in range(5)]
    PI_3d, PI_hz = [None for ij in range(2)]
    
    if windsb is None and windsc is None:
        time1 = time.time()
        if scalar is None:
    
            kk1, PI_k, PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF(
                                                 U, V, W, N_u, N_v, N_w, 
                                                 dx=dx,dy=dy,dz=dz,
                                                 binning=binning, nbins=nbins,
                                                 Nhh=(N_uu,N_vu),Nhv=(N_uw,N_vw),
                                                 Nvh=(N_wu),Nvv=(N_ww),
                                                 filter_type='spectral_3d')
            
            # Test a different filter
            kk2, PI_hz2,  *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,  
                                                 kk=kk1, # To force the save k-axis
                                                 w_bc='dst',
                                                 dx=dx,dy=dy,dz=dz,
                                                 binning=binning, nbins=nbins,
                                                 filter_type='spectral_hz')
    
        else:
            kk1, PI_3d,PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF(U,V,W,
                                        N_u,N_v,N_w,
                                        scalar=(scalar,N_scalar_u,N_scalar_w),
                                        kk=k_bins,dx=dx,dy=dy,dz=dz)
        time2 = time.time()
        print('%s function took %0.3f ms' % ("Calculate PI_k2", (time2-time1)*1000.0))
        
        # For information, Egality is:
        # Pi_k2[1] = PI_3d/(nx*ny*nz)
        
        #for idxk,k_idx in enumerate(k_mag):
        #    tmp_PI = u_k[idxk,:,:,:]*N_u+\
        #             v_k[idxk,:,:,:]*N_v+\
        #             w_k[idxk,:,:,:]*N_w 
        #    PI_k2  = np.mean(tmp_PI,axis=(1,2,3))
    

    
    out = dict(k=k, k_shell_centers=k_shell_centers,
               mode_count=mode_count,
               E=E,T=T,Pi=Pi,
               PI_k=PI_k, 
               PI_hh=PI_hh, PI_hv=PI_hv,
               PI_vh=PI_vh, PI_vv=PI_vv,
               kk1=kk1, PI_3d=PI_3d,
               kk2=kk2, PI_hz5=PI_hz)
    return out

def compute_E(k,k_mag,U,N_u,
              V=None,W=None,
              dealias_mask = None,
              norm="ortho"):
    
    """
    Compute Energy spectra from U,V,W and scalal field
    # Improve by removing k_mag?
    """
    
    # forward FFT of velocity (spectral space)
    u_hat   = np.fft.fftn(U, norm=norm)
    N_u_hat = np.fft.fftn(N_u, norm=norm)
    
    # Dealias?
    if dealias_mask is None:
        dealias_mask = np.ones_like(k_mag, dtype=bool)

    # spectral KE per mode and spectral transfer per mode
    # E_mode = 0.5 * (|û|^2 + |v̂|^2 + |ŵ|^2)
    # T_mode = -Re( û* · N̂ )
    if V is not None and W is not None:
        v_hat   = np.fft.fftn(V[0], norm=norm)
        w_hat   = np.fft.fftn(W[0], norm=norm)
        N_v_hat = np.fft.fftn(V[1], norm=norm)
        N_w_hat = np.fft.fftn(W[1], norm=norm)
        
        E_k3 = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        T_k3 = - (np.real(np.conj(u_hat) * N_u_hat) + 
                  np.real(np.conj(v_hat) * N_v_hat) +
                  np.real(np.conj(w_hat) * N_w_hat))
    else:
        E_k3 = np.abs(u_hat)**2 # *0.5
        T_k3 = -np.real(np.conj(u_hat) * N_u_hat)

        
    # Optionally zero out dealiased modes in outputs (they are already small/zero in u_hat_dealias)
    # But for consistency, mask E and T where dealias_mask==False

    E_k3[~dealias_mask] = 0.0
    T_k3[~dealias_mask] = 0.0
    
    # create radial wavenumber magnitude and shells
    k_mag_flat = k_mag.ravel()
    E_flat = E_k3.ravel()
    T_flat = T_k3.ravel()

    # Compute bin indices once
    inds = np.digitize(k_mag_flat, k) - 1
    
    n_shells = len(k) - 1
    
    # Keep only valid bins
    valid = (inds >= 0) & (inds < n_shells)
    
    inds_valid = inds[valid]
    E_valid = E_flat[valid]
    T_valid = T_flat[valid]
    
    # Fast vectorized accumulation
    mode_count = np.bincount(inds_valid, minlength=n_shells)
    E_k = np.bincount(inds_valid, weights=E_valid, minlength=n_shells)
    T_k = np.bincount(inds_valid, weights=T_valid, minlength=n_shells)

    # Calculate Pi transfer
    Pi_k = -np.cumsum(T_k)

    return k,mode_count,E_k,T_k,Pi_k    

def compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,
                         scalar=None,
                         Nhh=None, Nhv=None,
                         Nvh=None, Nvv=None,
                         kk=None, dx=1.0, dy=1.0, dz=1.0,
                         binning='log', nbins=80,
                         filter_type='spectral_3d',
                         w_bc='dst',        # w=0 aux parois (Dirichlet) -> DST. 'dct' seulement si tu as une raison physique precise.
                         scalar_bc='dct'):  # flux nul par defaut (Neumann) pour un scalaire type buoyancy/theta
    """
    Corrected and final (?) version
    
    filter_type :
      'spectral_3d' : FFT 3D complete, coquille isotrope (biaisee: periodicite en z + sous-resolution kz) - legacy/reference
      'spectral_h'  : coupure sur kh seul, z inchange (pas de cascade h<->v)
      'spectral_hz' : FFT en x,y (periodique) + DCT (u,v) / DST-ou-DCT (w) en z - RECOMMANDE
      'gaussian'    : noyau reel anisotrope, lisse (test de sensibilite)

    BC verticales (filter_type='spectral_hz' uniquement) :
      u, v      : TOUJOURS DCT-II (Neumann, du/dz=dv/dz=0 aux parois -> hypothese frottement nul)
      w         : DST-II par defaut (Dirichlet, w=0 aux parois -> impermeabilite), 'dct' possible en test
      scalar    : DCT-II par defaut (Neumann, flux nul), 'dst' possible si valeur imposee aux parois

    ATTENTION grille verticale : ces formules supposent une DCT-II/DST-II "cell-centered"
    (niveaux a z_i=(i+0.5)*dz, domaine physique de longueur L=nz*dz). Si tes niveaux sont
    positionnes sur les frontieres (ex: w natif sur les faces, y compris z=0 et z=H), il
    faut en toute rigueur une DST-I avec L=(nz+1)*dz plutot que DST-II -- verifie ta grille
    avant de faire confiance aveuglement aux resultats pres des parois.
    """
    nz, ny, nx = U.shape

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    # --- Nombres d'onde verticaux (DCT-II / DST-II, "cell-centered", type=2) ---
    # kz = pi*n/L est la frequence EXACTE de la fonction de base (pas une approximation
    # aux differences finies) : cos(pi*n*(i+0.5)/N) == cos(kz*z_i) avec z_i=(i+0.5)*dz, L=N*dz.
    n_dct = np.arange(nz)          # n = 0..nz-1, le mode n=0 (moyenne) existe
    n_dst = np.arange(1, nz + 1)   # n = 1..nz, PAS de mode n=0 (w=0 impose -> pas de composante moyenne)
    kz_dct = np.pi * n_dct / (nz * dz)
    kz_dst = np.pi * n_dst / (nz * dz)
    #kz_dst = 2/dz*np.sin(np.pi*n_dst/(2*nz)) # TEST
    

    ky2, kx2 = np.meshgrid(ky, kx, indexing='ij')
    kh_mag = np.sqrt(kx2**2 + ky2**2)  # (ny, nx)

    def make_k3(kz_1d):
        kz3, ky3, kx3 = np.meshgrid(kz_1d, ky, kx, indexing='ij')
        return np.sqrt(kx3**2 + ky3**2 + kz3**2)

    def kz_of(bc):
        return kz_dct if bc == 'dct' else kz_dst

    # --- grille de coupures kk ---
    if kk is None:
        if filter_type == 'spectral_h':
            ktmp = kh_mag
        elif filter_type == 'spectral_hz':
            # on prend l'union la plus large (celle de w, generalement la plus fine resolution en kz)
            ktmp = make_k3(kz_of(w_bc))
            
            #kz1 = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
            #plt.plot(kz1[:], 'o-', label='FFT')
            #plt.plot(kz_of(w_bc)[:], 'x-', label='DCT')
            #plt.show()
            #stop
            
        else:  # spectral_3d
            kz1 = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
            ktmp = make_k3(kz1)
        kmax = ktmp.max()
        kmin = np.min(ktmp[ktmp > 0])
        if binning.lower() == 'log':
            kk = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        else:
            kk = np.linspace(0.0, kmax, nbins + 1)

    kk = np.asarray(kk)
    nk = len(kk)

    PI_k2 = np.empty(nk)
    PI_hh, PI_hv, PI_vh, PI_vv = [np.empty(nk) for _ in range(4)]

    # --- transformees generiques FFT(x,y) + DCT/DST(z) ---
    def make_transform(z_bc):
        tfun_fwd = dctn if z_bc == 'dct' else dstn
        tfun_bwd = idctn if z_bc == 'dct' else idstn

        def fwd(field):
            f_hat_h = np.fft.fftn(field, axes=(1, 2))
            re = tfun_fwd(f_hat_h.real, axes=(0,), type=2, norm='ortho')
            im = tfun_fwd(f_hat_h.imag, axes=(0,), type=2, norm='ortho')
            return re + 1j * im

        def bwd(field_hat):
            re = tfun_bwd(field_hat.real, axes=(0,), type=2, norm='ortho')
            im = tfun_bwd(field_hat.imag, axes=(0,), type=2, norm='ortho')
            return np.fft.ifftn(re + 1j * im, axes=(1, 2)).real

        return fwd, bwd

    # --- pre-transformees selon la methode ---
    if filter_type == 'spectral_3d':
        U_hat, V_hat, W_hat = np.fft.fftn(U), np.fft.fftn(V), np.fft.fftn(W)
        kz1 = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
        k_mag = make_k3(kz1)
        if scalar is not None:
            scalar_hat = np.fft.fftn(scalar[0])

    elif filter_type == 'spectral_h':
        U_hat, V_hat, W_hat = np.fft.fftn(U), np.fft.fftn(V), np.fft.fftn(W)
        k_mag = np.broadcast_to(kh_mag, (nz, ny, nx))
        if scalar is not None:
            scalar_hat = np.fft.fftn(scalar[0])

    elif filter_type == 'spectral_hz':
        fwd_uv, bwd_uv = make_transform('dct')   # u,v : Neumann, toujours DCT
        fwd_w,  bwd_w  = make_transform(w_bc)    # w : Dirichlet par defaut (DST)

        U_hat = fwd_uv(U)
        V_hat = fwd_uv(V)
        W_hat = fwd_w(W)

        k_mag_uv = make_k3(kz_of('dct'))
        k_mag_w  = make_k3(kz_of(w_bc))

        if scalar is not None:
            fwd_s, bwd_s = make_transform(scalar_bc)
            scalar_hat = fwd_s(scalar[0])
            k_mag_s = make_k3(kz_of(scalar_bc))
            
# Test
        # Ef_fft = np.abs(np.fft.fftn(U))**2
        # Uf     = fwd_uv(U)
        # Ef_dct = np.abs(Uf)**2
        # print(Ef_fft.sum(), Ef_dct.sum()) #--> Diff = 65 !
        # plt.semilogy(np.sort(Ef_fft.ravel())[::-1], label='FFT')
        # plt.semilogy(np.sort(Ef_dct.ravel())[::-1], label='DCT')
        # plt.show()
        
        # print(np.sum(U**2))
        # Uh = fwd_uv(U)
        # print(np.sum(np.abs(Uh)**2))
        # Ur = bwd_uv(Uh)
        # print(np.max(np.abs(U-Ur)))
        
        # print(k_mag_uv.min(), k_mag_uv[k_mag_uv>0].min())
        # print(k_mag_w.min(),  k_mag_w[k_mag_w>0].min())
        # print(kk[:10])
        # stop

    elif filter_type != 'gaussian':
        raise ValueError(f"filter_type inconnu: {filter_type}")

    if filter_type in ('spectral_3d', 'spectral_h', 'spectral_hz'):
        U_hatf, V_hatf, W_hatf = (np.empty_like(U_hat) for _ in range(3))
        if scalar is not None:
            scalar_hatf = np.empty_like(scalar_hat)

    k_floor = kk[kk > 0].min() * 0.5 if np.any(kk > 0) else 1e-6

    # --- Boucle principale ---
    for i, k_cut in enumerate(kk):

        if filter_type in ('spectral_3d', 'spectral_h'):
            mask = (k_mag <= k_cut)
            np.multiply(U_hat, mask, out=U_hatf)
            np.multiply(V_hat, mask, out=V_hatf)
            np.multiply(W_hat, mask, out=W_hatf)
            u_f = np.fft.ifftn(U_hatf).real
            v_f = np.fft.ifftn(V_hatf).real
            w_f = np.fft.ifftn(W_hatf).real
            if scalar is not None:
                np.multiply(scalar_hat, mask, out=scalar_hatf)
                scalar_f = np.fft.ifftn(scalar_hatf).real

        elif filter_type == 'spectral_hz':
            mask_uv = (k_mag_uv <= k_cut)
            mask_w  = (k_mag_w  <= k_cut)
            #mask = mask_uv # TEST
            np.multiply(U_hat, mask_uv, out=U_hatf)
            np.multiply(V_hat, mask_uv, out=V_hatf)
            np.multiply(W_hat, mask_w,  out=W_hatf)
            #np.multiply(W_hat, mask,  out=W_hatf)
            u_f = bwd_uv(U_hatf)
            v_f = bwd_uv(V_hatf)
            w_f = bwd_w(W_hatf)
            if scalar is not None:
                mask_s = (k_mag_s <= k_cut)
                np.multiply(scalar_hat, mask_s, out=scalar_hatf)
                scalar_f = bwd_s(scalar_hatf)

        elif filter_type == 'gaussian':
            sigma_phys = 1.0 / max(k_cut, k_floor)
            sigma_pix = (sigma_phys / dz, sigma_phys / dy, sigma_phys / dx)
            modes = ('nearest', 'wrap', 'wrap')
            u_f = gaussian_filter(U, sigma=sigma_pix, mode=modes)
            v_f = gaussian_filter(V, sigma=sigma_pix, mode=modes)
            w_f = gaussian_filter(W, sigma=sigma_pix, mode=modes)
            if scalar is not None:
                scalar_f = gaussian_filter(scalar[0], sigma=sigma_pix, mode=modes)

        if scalar is None:
            PI_k2[i] = np.mean(u_f * N_u + v_f * N_v + w_f * N_w)
            if Nhh is not None: PI_hh[i] = np.mean(u_f * Nhh[0] + v_f * Nhh[1])
            if Nhv is not None: PI_hv[i] = np.mean(u_f * Nhv[0] + v_f * Nhv[1])
            if Nvh is not None: PI_vh[i] = np.mean(w_f * Nvh)
            if Nvv is not None: PI_vv[i] = np.mean(w_f * Nvv)
        else:
            N_scalar = scalar[1] + scalar[2]
            PI_k2[i] = np.mean(scalar_f * N_scalar)
            PI_hh[i] = np.mean(scalar_f * scalar[1])
            PI_hv[i] = np.mean(scalar_f * scalar[2])

    return kk, PI_k2, PI_hh, PI_hv, PI_vh, PI_vv
