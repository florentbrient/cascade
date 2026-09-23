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

def make_kbins(k_min,k_max,nbins=50,binning='log'):
    
    if binning.lower() == 'log':
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), nbins+1)
        # use geometric mean as center
        k_shell_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    else:
        k_bins = np.linspace(0.0, k_max, nbins+1)
        k_shell_centers = 0.5*(k_bins[:-1] + k_bins[1:]) 
        
    return k_bins,k_shell_centers

def make_edges_adaptive(k_mag, nbins=80, nmin=20):
    """
    Bords log-espacés, calés sur les valeurs |k| réellement présentes,
    avec au moins nmin modes par bin. Chaque bord est un |k| existant,
    donc le test '<=' est exact en flottants.
    """
    kv = np.sort(k_mag.ravel())
    kv = kv[kv > 0]
    target = np.geomspace(kv[0], kv[-1], nbins + 1)
    idx = np.searchsorted(kv, target, side='right')   # nb de modes <= target
    keep_idx = [max(idx[0], nmin)]
    for j in idx[1:]:
        if j - keep_idx[-1] >= nmin:
            keep_idx.append(j)
    if keep_idx[-1] < len(kv):                        # inclure les derniers modes
        keep_idx.append(len(kv))
    return kv[np.array(keep_idx) - 1]


def compute_spectral_transfer(winds, 
                              dx=50.0, dy=50.0, dz=10.0,
                              z=None,
                              scalar=None,P=None,B=None,
                              windsb=None,windsc=None,
                              norm='ortho', dealiasing_23=False,
                              binning='log', k_bins=None,
                              nbins=80,nmin=None):
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
    if len(winds)==3:
        U, V, W  = winds
        Ub,Vb,Wb = winds
        Uc,Vc,Wc = winds
        if windsb is not None:
            Ub,Vb,Wb = windsb
        if windsc is not None:
            Uc,Vc,Wc = windsc
    else:
        U, V  = winds
        Ub,Vb = winds
        Uc,Vc = winds
        W,Wc  = None,None
        fieldW= None
        Wb    = 0

    # shapes
    if len(U.shape)==3:
        Nz, Ny, Nx = U.shape
        zall      = np.arange(0,Nz)*dz
    else:
        Ny, Nx = U.shape
        Nz,zall = None,None
    assert V.shape == U.shape
    if scalar is not None:
        assert scalar.shape == U.shape
    
    if k_bins is None: 
    # !! The situation with k_bins is NOT None doesn't work because of k_mag
        # build wavenumber arrays (rad/m)
        kx_1d = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
        ky_1d = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
        
        if Nz is not None:
            kz_1d = 2*np.pi * np.fft.fftfreq(Nz, d=dz)
            kz, ky, kx = np.meshgrid(kz_1d, ky_1d, kx_1d, indexing='ij')  # shape (Nz,Ny,Nx)
        else:
            ky, kx = np.meshgrid(ky_1d, kx_1d, indexing='ij')  # shape (Nz,Ny,Nx)
            kz = np.zeros(ky.shape)
        
        k_mag      = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag_flat = k_mag.ravel()
        
        k_min      = np.min(k_mag_flat[k_mag_flat > 0])
        k_max      = np.max(k_mag_flat)
        
        k_bins,_ = make_kbins(k_min,k_max,
                              nbins=nbins,binning=binning)
            
        # New: a utiliser ou non?
        # Remplacer k_bins par le nouveau kk dans compute_E?
        if nmin is not None:
            #nmin = 20
            k_bins = make_edges_adaptive(k_mag, nbins=nbins, nmin=nmin)

    # Gradients
    #dzall     = np.repeat(dz,Nz)
    gradients = tl.compute_gradients(Uc, dx, dy, zall, v=Vc, w=Wc)
    (du_dx, dv_dx, dw_dx, 
     du_dy, dv_dy, dw_dy, 
     du_dz, dv_dz, dw_dz) = gradients
    
    # nonlinear term N = u · ∇u (vector)
    N_uu = Ub * du_dx + Vb * du_dy
    N_vu = Ub * dv_dx + Vb * dv_dy
    N_uw,N_vw,N_ww,N_wu = 0,0,0,0
    if dw_dx is not None:
        N_wu = Ub * dw_dx + Vb * dw_dy
    if du_dz is not None:
        N_uw = Wb * du_dz
        N_vw = Wb * dv_dz
        N_ww = Wb * dw_dz

    N_u  = N_uu + N_uw
    N_v  = N_vu + N_vw
    N_w  = N_wu + N_ww
    if scalar is not None:
        gradientsS = tl.compute_gradients(scalar, dx, dy, zall)
        (dscalar_dx, a, d, 
         dscalar_dy, b, e, 
         dscalar_dz, c, f) = gradientsS
        # Nonlinear term n= U ∇THLM (vector)
        N_scalar_u = Ub * dscalar_dx + Vb * dscalar_dy
        N_scalar_w = 0
        if dscalar_dz is not None:
            N_scalar_w = Wb * dscalar_dz
        
    # Calculate Energy spectra
    field,N_field = U,N_u
    fieldV        = (V,N_v)
    if W is not None:
        fieldW        = (W,N_w)
    if scalar is not None:
        field, N_field = scalar, N_scalar_u+N_scalar_w
        fieldV, fieldW = None,None
         
    Eout  =  compute_E_v2(k_bins,k_mag,
                            field,N_field,
                            V=fieldV,W=fieldW,
                            P=P,k_z=kz)
    
    Eout_layer = None
    if Nz is not None:
        Eout_layer = compute_E_layer(k_bins, dx, dy,
                                     U, V, W,                       # (nz, ny, nx) physical fields
                                     N_u, N_v, N_w,                 # u·∇u components (physical, dealiased)
                                     P=P,B=B,dz=dz)

    # print(Eout.keys())
    # plt.figure()
    # plt.semilogx(Eout['k'],Eout['Pi_k'],'k')
    # plt.semilogx(Eout['k'],Eout['Pi_h_k'],'r')
    # plt.semilogx(Eout['k'],Eout['Pi_v_k'],'b')
    # plt.figure()
    # plt.loglog(Eout['k'],Eout['E_spec'],'k')
    # plt.loglog(Eout['k'],Eout['E_h_spec'],'r')
    # plt.loglog(Eout['k'],Eout['E_v_spec'],'b')
    
    # E_k3_mean = Eout['E_k']/Eout['dk'] #/(nx*ny*nz)
    # TKE3D  = 0.5*(pow(tl.anomcalc(U),2.)
    #             +pow(tl.anomcalc(V),2.)\
    #             +pow(tl.anomcalc(W),2.))
    # tl.checkvariance(Eout['k'],E_k3_mean,TKE3D,type='mean')
    # stop
    
    k_new      =  Eout['k']
    E_k3, T_k3 = Eout['E_k3'],Eout['T_k3']
    
        
    
    kk1,kk2 = [None for ij in range(2)]
    PI_k, PI_hh, PI_hv, PI_vh, PI_vv = [None for ij in range(5)]
    PI_3d, PI_hz = [None for ij in range(2)]
    
    if windsb is None and windsc is None and Nz is not None:
        time1 = time.time()
        if scalar is None:
    
            kk1, PI_k, PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF(
                                                 U, V, W, N_u, N_v, N_w, 
                                                 dx=dx,dy=dy,dz=dz,
                                                 kk=k_new, # To force the save k-axis
#                                                 binning=binning, nbins=nbins,
                                                 Nhh=(N_uu,N_vu),Nhv=(N_uw,N_vw),
                                                 Nvh=(N_wu),Nvv=(N_ww),
                                                 filter_type='spectral_3d')
            
            # Test a different filter
            kk2, PI_hz,  *_ = compute_Pi_from_uBF(U, V, W, N_u, N_v, N_w,  
                                                 kk=kk1, # To force the save k-axis
                                                 w_bc='dst',
                                                 dx=dx,dy=dy,dz=dz,
                                                 binning=binning, nbins=nbins,
                                                 filter_type='spectral_hz')
    
        else:
            kk1, PI_k ,PI_hh, PI_hv, PI_vh, PI_vv = compute_Pi_from_uBF(
                                        U, V, W, N_u, N_v, N_w,
                                        scalar=(scalar,N_scalar_u,N_scalar_w),
                                        kk=k_new,
                                        dx=dx,dy=dy,dz=dz)
        time2 = time.time()
        print('%s function took %0.3f ms' % ("Calculate PI_k2", (time2-time1)*1000.0))
        
        # For information, Egality is:
        # Pi_k2[1] = PI_3d/(nx*ny*nz)
        
        #for idxk,k_idx in enumerate(k_mag):
        #    tmp_PI = u_k[idxk,:,:,:]*N_u+\
        #             v_k[idxk,:,:,:]*N_v+\
        #             w_k[idxk,:,:,:]*N_w 
        #    PI_k2  = np.mean(tmp_PI,axis=(1,2,3))
    
    
    # Compute Parallel and perpendicular
    k_center_perp, E_perp, T_perp, Pi_perp, \
    k_center_para, E_para, T_para, Pi_para = [None for ij in range(8)]
    
    if Nz is not None:
        #print(kx.shape,kz.shape,E_k3.shape)
        k_center_perp, E_perp, T_perp, Pi_perp,\
        k_center_para, E_para, T_para, Pi_para = spectral_cyl_plane(E_k3, T_k3,
                                          kx, ky, kz, 
                                          n_bins_perp=nbins, 
                                          n_bins_para=nbins, 
                                          binning=binning)

    
    out = dict(k=k_new, 
               # dk=Eout['dk'],
               # k_shell_centers=Eout['k_shell_centers'],
               # k_shell_centers_geo=Eout['k_shell_centers_geo'],
               # mode_count=Eout['mode_count'],
               # E=Eout['E_k'],T=Eout['T_k'],Pi=Eout['Pi_k'],
               # Eh=Eout['E_h_k'],Ev=Eout['E_v_k'],
               # E_spec=Eout['E_spec'],Eh_spec=Eout['E_h_spec'],Ev_spec=Eout['E_v_spec'],
               # Pih=Eout['Pi_h_k'],Piv=Eout['Pi_v_k'],
               # Phiv=Eout['phi_v_k'],
               PI_k=PI_k, 
               PI_hh=PI_hh, PI_hv=PI_hv,
               PI_vh=PI_vh, PI_vv=PI_vv,
               kk2=kk2, PI_hz=PI_hz,
               kperp=k_center_perp, Eperp=E_perp, Tperp=T_perp, Piperp=Pi_perp,
               kpara=k_center_para, Epara=E_para, Tpara=T_para, Pipara=Pi_para,
               Eout=Eout, Eout_layer=Eout_layer)
    return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _d_dz(field, z, dz):
    """Vertical derivative, 2nd order centered, one-sided at walls.
    z: 1D array of levels (len = field.shape[0]) or None (uniform dz)."""
    f = np.empty_like(field)
    if z is not None:
        dzc = (z[2:] - z[:-2]).reshape(-1, 1, 1)
        dz0, dz1 = (z[1] - z[0]), (z[-1] - z[-2])
    else:
        if dz is None:
            raise ValueError("provide either z (1D, len nz) or dz")
        dzc = (2 * dz)
        dz0 = dz1 = dz
    f[1:-1] = (field[2:] - field[:-2]) / dzc
    f[0]    = (field[1]  - field[0])  / dz0
    f[-1]   = (field[-1] - field[-2]) / dz1
    return f


def shells_kh(kh_bins, kx_2d, ky_2d):
    """Boolean masks (ny, nx) per shell, on the FULL fftn spectral grid.
    bin i <=> lo[i] < kh <= kh_bins[i] ; bin 0 includes kh = 0."""
    kh = np.sqrt(kx_2d**2 + ky_2d**2)
    assert kh.ndim == 2, f"kh must be 2D (ky, nx), got {kh.shape}"
    lo = np.concatenate(([0.0], kh_bins[:-1]))
    return [(kh > lo[i]) & (kh <= khi) for i, khi in enumerate(kh_bins)]

def _rfft_weights(ky_2d, kx_2d): # Not used
    """Weight 2 for kx>0 modes (hermitian half-plane), 1 on kx=0 column.
    Returns 2D array (ny, nxh), constant along ky."""
    w = np.where(kx_2d[0, :] == 0.0, 1.0, 2.0)   # (nxh,)
    return np.broadcast_to(w, ky_2d.shape).copy()


def _bin_shells(A3, shell_masks, layer_norm=1.0):
    """A3: (nz, ny, nx) real-valued. Sum per shell, then average over z.
    No hermitian weights needed with full fftn."""
    out = np.zeros(len(shell_masks))
    for i, m in enumerate(shell_masks):
        if m.any():
            out[i] = A3[:, m].sum() / layer_norm
    return out


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------
def compute_E_layer(kh_bins, dx, dy,
                    U, V, W,                # (nz, ny, nx) physical, w=0 at walls
                    N_u, N_v, N_w,          # u·∇u components (physical, dealiased)
                    P=None,                 # pressure (grid of U)
                    B=None,                 # buoyancy
                    z=None, dz=None,        # vertical levels or uniform spacing
                    zw=None,                # optional: w-levels if staggered
                    norm='ortho'):
    """
    LES layer version: periodic in x,y (rFFT), NO FFT in z, w=0 at walls.
    Shells are cylindrical in kh = sqrt(kx^2+ky^2).

    Returns shell quantities averaged over the layer, with the exact
    decomposition  E = Eh + Ev,  Pi = Pih + Piv,  and pressure-exchange
    terms computed in physical space (no kz-FFT artefact).
    """
    nz, ny, nx = U.shape
    if dz is None and z is None:
        raise ValueError("provide z (1D, len nz) or dz")

    # ---- wavenumbers on the rFFT grid ----
    kx_1d = 2 * np.pi * np.fft.fftfreq(nx, d=dx)      # nx//2+1  ✓
    ky_1d = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    ky_2d, kx_2d = np.meshgrid(ky_1d, kx_1d, indexing='ij')   # (ny, nxh)
    # NOTE: kx_2d varies along axis 1 (see _rfft_weights), ky along axis 0.

    shell_masks = shells_kh(kh_bins, kx_2d, ky_2d)
    #weights2d   = _rfft_weights(ky_2d, kx_2d)
    n_shells    = len(kh_bins)

    # ---- vertical derivatives in physical space ----
    zW = zw if (zw is not None) else z     # w may live on its own levels
    dpz_w = _d_dz(W, zW, dz)              # ∂z w, uses w=0 at walls implicitly
    dpz_p = _d_dz(P, z, dz) if P is not None else None

    # ---- horizontal rFFTs (x,y only) ----
    def hh(F):  # to spectral
        return np.fft.fftn(F, axes=(1, 2), norm=norm)
    def ih(Fh):  # to physical
        return np.fft.ifftn(Fh, s=(ny, nx), axes=(1, 2), norm=norm)

    u_hat, v_hat, w_hat = hh(U), hh(V), hh(W)
    Nu_hat, Nv_hat, Nw_hat = hh(N_u), hh(N_v), hh(N_w)

    # ---- energies & transfers per mode (nz, ny, nxh) ----
    E_h_3 = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    E_v_3 = 0.5 * np.abs(w_hat)**2
    T_h_3 = -(np.real(np.conj(u_hat) * Nu_hat) +
              np.real(np.conj(v_hat) * Nv_hat))
    T_v_3 = -np.real(np.conj(w_hat) * Nw_hat)

    # ---- shell binning (hermitian weights + layer average) ----
    E_h_k = _bin_shells(E_h_3, shell_masks, nz)
    E_v_k = _bin_shells(E_v_3, shell_masks, nz)
    T_h_k = _bin_shells(T_h_3, shell_masks, nz)
    T_v_k = _bin_shells(T_v_3, shell_masks, nz)

    E_k = E_h_k + E_v_k          # exact by construction
    T_k = T_h_k + T_v_k
    Pi_k, Pi_h_k, Pi_v_k = ( -np.cumsum(T_k),
                            -np.cumsum(T_h_k),
                            -np.cumsum(T_v_k) )

    # ---- pressure exchange (physical space, per shell) ----
    phi_v_k = np.zeros(n_shells)
    phi_h_k = np.zeros(n_shells)
    if P is not None:
        p_hat    = hh(P)
        dpz_w_h = hh(dpz_w)
        divh_h  = (1j * kx_2d[None, :, :] * u_hat +
                   1j * ky_2d[None, :, :] * v_hat)     # ∂x u + ∂y v
        for i, m in enumerate(shell_masks):
            if not m.any():
                continue
            p_f   = ih(p_hat    * m[None, :, :])
            dpw_f = ih(dpz_w_h * m[None, :, :])
            divf  = ih(divh_h  * m[None, :, :])
            # layer+horizontal mean of the physical product (no weight-2:
            # irfftn reconstructs the full physical field, products are exact)
            phi_v_k[i] = np.mean(p_f * dpw_f)
            # phi_h: pressure work ON horizontal comp = -<p div_h u_h>
            phi_h_k[i] = +np.mean(p_f * divf)

    phi_res_k = phi_h_k + phi_v_k   # boundary pressure work per shell (≠0 OK)

    # ---- NEW: buoyancy production of Ev, per shell ----
    # B_v(k) = <Re( ŵ* b̂ )>_shell   [ + for Ev, m²/s³ ]
    # Sign convention: B_v > 0  <=>  buoyancy feeds the vertical component.
    B_v_k = np.zeros(n_shells)
    if B is not None:
        if B.shape != W.shape:
            raise ValueError(
                f"B must share W's grid: B{B.shape} vs W{W.shape}. "
                "Interpolate b onto the w-levels (or vice-versa) first.")
        b_hat = hh(B)                              # same rFFT in x,y
        B_v_3 = np.real(np.conj(w_hat) * b_hat)   # per (z, ky, kx)
        B_v_k = _bin_shells(B_v_3, shell_masks, nz)

    # ---- shell geometry ----
    kh_lo = np.concatenate(([0.0], kh_bins[:-1]))
    dk = kh_bins - kh_lo
    with np.errstate(invalid='ignore', divide='ignore'):
        E_spec   = np.where(dk > 0, E_k   / dk, np.nan)
        E_h_spec = np.where(dk > 0, E_h_k / dk, np.nan)
        E_v_spec = np.where(dk > 0, E_v_k / dk, np.nan)

    # ---- closure report (sanity checks) ----
    closure = dict(
        sum_Eh_plus_Ev_minus_E=float(np.nansum(E_h_k + E_v_k - E_k)),      # ~0 exact
        sum_Pih_plus_Piv_minus_Pi=float((Pi_h_k + Pi_v_k - Pi_k)[-1]),     # ~0 exact
        sum_phi_res=float(phi_res_k.sum()),   # ≈0 only in TOTAL (w=0 walls)
    )
    

    return dict(
        k=kh_bins, dk=dk,
        k_shell_centers=0.5 * (kh_lo + kh_bins),
        k_shell_centers_geo=np.sqrt(np.maximum(kh_lo * kh_bins, 1e-30)),
        E_k=E_k, E_h_k=E_h_k, E_v_k=E_v_k,
        E_k3=E_h_3+E_v_3, T_k3=T_h_3+T_v_3,
        E_spec=E_spec, E_h_spec=E_h_spec, E_v_spec=E_v_spec,
        T_k=T_k, T_h_k=T_h_k, T_v_k=T_v_k,
        Pi_k=Pi_k, Pi_h_k=Pi_h_k, Pi_v_k=Pi_v_k,
        phi_v_k=phi_v_k, phi_h_k=phi_h_k, phi_res_k=phi_res_k,
        B_v_k=B_v_k,                    # NEW: buoyancy production of Ev
        closure=closure,
    )
def compute_E_v2(k, k_mag, U, N_u,
              V=None, W=None,
              dealias_mask=None,
              norm="ortho",
              P=None, k_z=None):
    """
    Energy spectra E(k) with horizontal/vertical decomposition,
    plus spectral transfers Pi_h, Pi_v and optional pressure-exchange term.

    Parameters
    ----------
    U, N_u   : u_x and u·∇u_x (or scalar field + its NL term if V is None)
    V, W     : tuples (field, NL_term) for v and w
    P        : pressure field (optional). If given, computes the
               pressure-deformation exchange phi(k) that couples E_h and E_v.
    k_z      : 3D array of vertical wavenumbers kz (required with P).
    """

    # forward FFT of velocity (spectral space)
    u_hat   = np.fft.fftn(U, norm=norm)
    N_u_hat = np.fft.fftn(N_u, norm=norm)

    if dealias_mask is None:
        dealias_mask = np.ones_like(k_mag, dtype=bool)

    Ntot = U.size

    if V is not None:
        v_hat,   N_v_hat = np.fft.fftn(V[0], norm=norm), np.fft.fftn(V[1], norm=norm)
        w_hat,   N_w_hat = np.fft.fftn(W[0], norm=norm), np.fft.fftn(W[1], norm=norm)

        # --- spectral KE per mode, horizontal / vertical decomposition ---
        E_h_3 = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
        E_v_3 = 0.5 *  np.abs(w_hat)**2
        E_k3  = E_h_3 + E_v_3                     # exact: E = Eh + Ev

        # --- spectral transfer per mode, same decomposition ---
        # T = -Re( û* · N̂ ), component-wise
        T_h_3 = -(np.real(np.conj(u_hat) * N_u_hat) +
                  np.real(np.conj(v_hat) * N_v_hat))
        T_v_3 = -np.real(np.conj(w_hat) * N_w_hat)
        T_k3  = T_h_3 + T_v_3                     # exact: T = Th + Tv

        # --- optional pressure-deformation exchange (couples Eh <-> Ev) ---
        # phi = Re( p̂* i kz ŵ )  : gains for E_v, losses for E_h (phi_v = -phi_h)
        phi_v_3 = None
                      
        if P is not None:
            assert k_z is not None
            p_hat   = np.fft.fftn(P, norm=norm)
            phi_v_3 = np.real(np.conj(p_hat) * 1j * k_z * w_hat)
            # phi_h = -phi_v par antisymétrie exacte, pas besoin de le stocker   
        
            
            
    else:
        # scalar variance branch (unchanged)
        E_k3  = np.abs(u_hat)**2
        T_k3  = -np.real(np.conj(u_hat) * N_u_hat)
        E_h_3 = E_v_3 = T_h_3 = T_v_3 = phi_v_3 = None

    # volume normalization
    scale = 1.0 / Ntot
    def _s(a):
        return a * scale if a is not None else None
    E_k3, T_k3   = _s(E_k3), _s(T_k3)
    E_h_3, E_v_3 = _s(E_h_3), _s(E_v_3)
    T_h_3, T_v_3 = _s(T_h_3), _s(T_v_3)
    phi_v_3      = _s(phi_v_3)

    # mask dealiased modes consistently everywhere
    E_k3[~dealias_mask] = 0.0
    T_k3[~dealias_mask] = 0.0
    if E_h_3 is not None:
        for a in (E_h_3, E_v_3, T_h_3, T_v_3):
            a[~dealias_mask] = 0.0
        if phi_v_3 is not None:
            phi_v_3[~dealias_mask] = 0.0

    # ---------------- shell binning (unchanged) ----------------
    k_mag_flat = k_mag.ravel()
    n_shells   = len(k)
    inds  = np.searchsorted(k, k_mag_flat, side='left')
    valid = inds < n_shells
    iv    = inds[valid]

    def binned(a):
        if a is None:
            return None
        return np.bincount(iv, weights=a.ravel()[valid], minlength=n_shells)
    
    

    mode_count = np.bincount(iv, minlength=n_shells)
    E_k        = binned(E_k3)
    T_k        = binned(T_k3)
    E_h_k, E_v_k = binned(E_h_3), binned(E_v_3)
    T_h_k, T_v_k = binned(T_h_3), binned(T_v_3)
    phi_v_k      = binned(phi_v_3)
    k_sum        = np.bincount(iv, weights=k_mag_flat[valid], minlength=n_shells)

    # cumulative fluxes, same convention as before: Pi = -cumsum(T)
    Pi_k   = -np.cumsum(T_k)
    Pi_h_k = -np.cumsum(T_h_k) if T_h_k is not None else None
    Pi_v_k = -np.cumsum(T_v_k) if T_v_k is not None else None

    # ---------------- shell geometry (unchanged) ----------------
    k_mean = np.where(mode_count > 0, k_sum / np.maximum(mode_count, 1), np.nan)
    dk = np.diff(k, prepend=np.nan)
    E_spec   = E_k / dk
    E_h_spec = E_h_k / dk if E_h_k is not None else None
    E_v_spec = E_v_k / dk if E_v_k is not None else None
    k_lo = np.concatenate(([np.nan], k[:-1]))
    dk  = k - k_lo
    k_shell_centers     = 0.5 * (k_lo + k)
    k_shell_centers_geo = np.sqrt(k_lo * k)

    return dict(
        k=k, mode_count=mode_count,
        E_k=E_k, T_k=T_k, Pi_k=Pi_k, E_spec=E_spec,
        # horizontal / vertical decomposition (Eh + Ev = E, Ph + Pv = P exactly)
        E_h_k=E_h_k, E_v_k=E_v_k, E_h_spec=E_h_spec, E_v_spec=E_v_spec,
        T_h_k=T_h_k, T_v_k=T_v_k, Pi_h_k=Pi_h_k, Pi_v_k=Pi_v_k,
        # pressure-deformation exchange (optional): +phi for Ev, -phi for Eh
        phi_v_k=phi_v_k,
        E_k3=E_k3, T_k3=T_k3,
        k_mean=k_mean, dk=dk,
        k_shell_centers=k_shell_centers,
        k_shell_centers_geo=k_shell_centers_geo,
    )


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
    v_hat, N_v_hat = 0, 0
    if V is not None:
        v_hat   = np.fft.fftn(V[0], norm=norm)
        N_v_hat = np.fft.fftn(V[1], norm=norm)
        w_hat, N_w_hat = 0, 0
        if W is not None:
            w_hat   = np.fft.fftn(W[0], norm=norm)
            N_w_hat = np.fft.fftn(W[1], norm=norm)
        E_k3 = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        T_k3 = - (np.real(np.conj(u_hat) * N_u_hat) + 
                  np.real(np.conj(v_hat) * N_v_hat) +
                  np.real(np.conj(w_hat) * N_w_hat))
    else: # Here is the variance, not 0.5*variance
        E_k3 = np.abs(u_hat)**2 # *0.5
        T_k3 = -np.real(np.conj(u_hat) * N_u_hat)


    # Normlize to have a mean value in volume
    N    = U.size
    E_k3 = E_k3 / N   # m²/s²  (somme sur k = énergie cinétique massique moyenne)
    T_k3 = T_k3 / N   # m²/s³
    
    # Optionally zero out dealiased modes in outputs (they are already small/zero in u_hat_dealias)
    # But for consistency, mask E and T where dealias_mask==False

    E_k3[~dealias_mask] = 0.0
    T_k3[~dealias_mask] = 0.0
    
    # create radial wavenumber magnitude and shells
    k_mag_flat = k_mag.ravel()
    E_flat = E_k3.ravel()
    T_flat = T_k3.ravel()

    # Compute bin indices once
    #inds = np.digitize(k_mag_flat, k) - 1
    
    n_shells = len(k) #- 1
    
    # Keep only valid bins # old version
    #valid = (inds >= 0) & (inds < n_shells)   
    #inds = np.digitize(k_mag_flat, k) - 1

    
    # bin i  <=>  k[i-1] < |k| <= k[i]   (le bin 0 contient tout |k| <= k[0], mode k=0 inclus)
    inds = np.searchsorted(k, k_mag_flat, side='left')

    # on jette uniquement les modes |k| > k[-1] (hors de toute coupure)
    valid = inds < n_shells

    iv = inds[valid]
    
    # Fast vectorized accumulation
    #mode_count = np.bincount(inds_valid, minlength=n_shells)
    #E_k = np.bincount(inds_valid, weights=E_flat[valid], minlength=n_shells)
    #T_k = np.bincount(inds_valid, weights=T_flat[valid], minlength=n_shells)
    
    mode_count = np.bincount(iv, minlength=n_shells)
    E_k   = np.bincount(iv, weights=E_flat[valid],     minlength=n_shells)
    T_k   = np.bincount(iv, weights=T_flat[valid],     minlength=n_shells)
    k_sum = np.bincount(iv, weights=k_mag_flat[valid], minlength=n_shells)
    
    # Calculate Pi transfer    
    Pi_k = -np.cumsum(T_k)   # Pi_k[i] = -somme_{|k| <= k[i]} T  ==  PI_k2[i]

    k_mean = np.where(mode_count > 0, k_sum / np.maximum(mode_count, 1), np.nan)
    # Spectre : énergie de la coquille / largeur de la coquille
    dk = np.diff(k, prepend=np.nan)          # dk[0] = nan -> bin 0 exclu
    E_spec = E_k / dk                        # m³/s²
    
    # Sauvegarder k_shell_centers
    # --- bornes des coquilles ]k_lo[i], k[i]] ---
    k_lo = np.concatenate(([np.nan], k[:-1]))      # bin 0 = tout ce qui est <= k[0]

    dk = k - k_lo                                  # largeur (nan pour le bin 0)
    k_shell_centers     = 0.5 * (k_lo + k)         # centre arithmétique
    k_shell_centers_geo = np.sqrt(k_lo * k)        # centre géométrique (bins log)
    

#    return k_mean,mode_count,E_k,T_k,Pi_k,E_k3,T_k3,E_spec
    return dict(k=k, mode_count=mode_count, E_k=E_k, T_k=T_k, Pi_k=Pi_k,
            E_k3=E_k3, T_k3=T_k3, k_mean=k_mean, E_spec=E_spec,
            dk=dk, k_shell_centers=k_shell_centers,
            k_shell_centers_geo=k_shell_centers_geo)

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
    Calculate cascade from 3D field only
    
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
            
        kk,kshell= make_kbins(kmin,kmax,
                              nbins=nbins,
                              binning=binning)

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


def spectral_cyl_plane(E, T, kx, ky, kz, dealias_mask=None,
                       n_bins_perp=50, n_bins_para=50, binning='log'):
    """
    Compute 1D cylindrical (k_perp) and planar (k_para) averaged spectra.
    Returns both shell-integrated and count-normalized versions.
    """
    # --- Setup ---
    k_perp = np.sqrt(kx**2 + ky**2)
    k_para = np.abs(kz)
    #mask = dealias_mask

    # --- Flatten masked values ---
    kf_perp = k_perp #[mask]
    kf_para = k_para #[mask]
    Ef = E #[mask]
    Tf = T #[mask]

    # --- Cylindrical (k_perp) binning ---
    kmin_p = np.nanmin(kf_perp[kf_perp > 0])
    kmax_p = np.nanmax(kf_perp)
    
    k_bins_perp, k_center_perp = make_kbins(kmin_p,kmax_p,
                                          nbins=n_bins_perp,binning=binning)

    inds_perp = np.digitize(kf_perp, k_bins_perp) - 1

    E_perp = np.zeros(n_bins_perp)
    T_perp = np.zeros(n_bins_perp)
    count_perp = np.zeros(n_bins_perp, dtype=int)

    for s in range(n_bins_perp):
        sel = inds_perp == s
        #print(s,sel.shape,Ef.shape)
        if np.any(sel):
            count_perp[s] = np.count_nonzero(sel)
            E_perp[s] = Ef[sel].sum()
            T_perp[s] = Tf[sel].sum()

    # --- Planar (k_parallel = |kz|) binning ---
    #kmin_z = np.nanmin(kf_para[kf_para > 0])
    #kmax_z = np.nanmax(kf_para)
    # if binning == 'log':
    #     k_bins_para = np.logspace(np.log10(kmin_z), np.log10(kmax_z), n_bins_para + 1)
    #     k_center_para = np.sqrt(k_bins_para[:-1] * k_bins_para[1:])
    # else:
    #     k_bins_para = np.linspace(0, kmax_z, n_bins_para + 1)
    #     k_center_para = 0.5 * (k_bins_para[:-1] + k_bins_para[1:])
        
    # test
    k_bins_para = np.unique(k_para[:,0,0])
    k_center_para = 0.5 * (k_bins_para[:-1] + k_bins_para[1:])
    n_bins_para = len(k_center_para)

    inds_para = np.digitize(kf_para, k_bins_para) - 1

    E_para = np.zeros(n_bins_para)
    T_para = np.zeros(n_bins_para)
    count_para = np.zeros(n_bins_para, dtype=int)

    for s in range(n_bins_para):
        sel = inds_para == s
        if np.any(sel):
            count_para[s] = np.count_nonzero(sel)
            E_para[s] = Ef[sel].sum()
            T_para[s] = Tf[sel].sum()

    # Optional: normalize by count for per-mode average
    E_perp_mean = np.divide(E_perp, count_perp, out=np.zeros_like(E_perp), where=count_perp>0)
    E_para_mean = np.divide(E_para, count_para, out=np.zeros_like(E_para), where=count_para>0)

    Pi_perp = -np.cumsum(T_perp) 
    Pi_para = -np.cumsum(T_para) 

    return k_center_perp, E_perp, T_perp, Pi_perp,\
           k_center_para, E_para, T_para, Pi_para
           
          
# Integrate Cascade
"""
Intégration du flux d'énergie spectral Pi(k) sur l'axe log(k),
de -infty (= plus petit k disponible) jusqu'à kH,
en ne sommant que les contributions où Pi(k) < 0
(partie "cascade inverse" du flux).

Convention : en variable u = ln(k), du = dk/k, donc
    ∫ Pi(k) d(ln k) = ∫ Pi(k)/k dk
On intègre donc directement Pi par rapport à ln(k) (trapèzes),
ce qui gère naturellement le changement de variable log.
"""


def integrate_negative_cascade(k, Pi, kH, method="trapz", refine_crossings=True):
    """
    Intègre Pi(k) sur ln(k), de min(k) à kH, en ne gardant que Pi < 0.
    
    Appel
    ----------
    I, info = integrate_negative_cascade(k, Pi, kH, method="trapz")
    print(f"Intégrale de Pi (Pi<0 uniquement) jusqu'à kH={kH} : {I:.4f}")

    Paramètres
    ----------
    k : array_like
        Nombres d'onde (k > 0), taille N. Pas besoin d'être trié
        ni uniformément espacé.
    Pi : array_like
        Flux d'énergie spectral Pi(k), même taille que k.
    kH : float
        Borne supérieure d'intégration (ex. nombre d'onde au sommet
        de la couche limite / cloud-top).
    method : {"trapz", "simpson"}
        Méthode d'intégration numérique.
    refine_crossings : bool
        Si True, ajoute des points interpolés aux endroits où Pi
        change de signe, pour éviter de "couper" une zone négative
        au milieu d'un intervalle (plus précis près des bords
        Pi=0 et près de kH).

    Retour
    ------
    integral : float
        Valeur de l'intégrale (unités de Pi, car d(ln k) est sans
        dimension).
    mask_info : dict
        Diagnostics utiles : k_used, Pi_used, integrand_used.
    """
        
    cond=~np.isnan(k)

    k = np.asarray(k[cond], dtype=float)
    Pi = np.asarray(Pi[cond], dtype=float)

    if k.shape != Pi.shape:
        raise ValueError("k et Pi doivent avoir la même forme")
    if np.any(k <= 0):
        raise ValueError("k doit être strictement positif (échelle log)")

    # remove Nan
    # Tri croissant en k
    order = np.argsort(k)
    k, Pi = k[order], Pi[order]
    

    # Restriction à k <= kH (on tronque/interpole le dernier point si besoin)
    if kH < k.max():
        i_cut = np.searchsorted(k, kH)
        if i_cut == 0:
            raise ValueError("kH est inférieur à tous les k fournis")
        # interpolation de Pi à k=kH pour ne pas perdre la borne exacte
        Pi_kH = np.interp(kH, k, Pi)
        k = np.concatenate([k[:i_cut], [kH]])
        Pi = np.concatenate([Pi[:i_cut], [Pi_kH]])
    # si kH >= k.max(), on garde tout le tableau tel quel

    lnk = np.log(k)

    if refine_crossings:
        lnk, Pi = _insert_zero_crossings(lnk, Pi)

    # On annule les portions où Pi >= 0 : l'intégrande reste continue
    # (vaut 0 hors des zones de cascade inverse) au lieu de "sauter"
    # des points, ce qui serait faux pour une intégrale numérique.
    integrand = np.where(Pi < 0, Pi, 0.0)

    if method == "trapz":
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        integral = trapz_fn(integrand, lnk)
    elif method == "simpson":
        from scipy.integrate import simpson
        integral = simpson(integrand, x=lnk)
    else:
        raise ValueError("method doit être 'trapz' ou 'simpson'")

    mask_info = {
        "k_used": np.exp(lnk),
        "Pi_used": Pi,
        "integrand_used": integrand,
    }
    return integral, mask_info


def _insert_zero_crossings(x, y):
    """
    Insère un point interpolé (x0, 0) à chaque changement de signe de y,
    pour que le masque Pi<0 tombe pile sur les bords des zones négatives.
    """
    x_new, y_new = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i - 1] * y[i] < 0:  # changement de signe strict
            # interpolation linéaire du zéro
            t = -y[i - 1] / (y[i] - y[i - 1])
            x0 = x[i - 1] + t * (x[i] - x[i - 1])
            x_new.append(x0)
            y_new.append(0.0)
        x_new.append(x[i])
        y_new.append(y[i])
    return np.array(x_new), np.array(y_new)


def compute_buoyancy_from_thlm(thlm, rv, rt, p, rr=None, axis=(-2, -1)):
    """
    Variante quand les sorties Meso-NH archivées sont THLM (theta_l),
    RVT (r_v) et RT (r_t = r_v + r_c (+ r_r)) plutôt que THT/RCT
    directement (cas fréquent, variables conservées).
 
    Passage direct theta_l -> theta_v en une seule expression, en
    substituant theta = theta_l + (Lv/(cp*Pi))*r_c dans
    theta_v = theta*(1+eps_v*rv-rl) et en négligeant les termes
    croisés du 2e ordre (rc*rv, rc^2), négligeables en pratique :
 
        theta_v ~ theta_l*(1+eps_v*rv) + rc*(Lv/(cp*Pi) - theta_l)
 
    Evite de matérialiser theta comme tableau intermédiaire (plus
    rapide, une seule passe sur les données).
 
    Paramètres
    ----------
    thlm : ndarray (..., nz, ny, nx)
        Température potentielle liquide (Meso-NH: THLM).
    rv : ndarray, même forme
        Rapport de mélange en vapeur d'eau (Meso-NH: RVT).
    rt : ndarray, même forme
        Rapport de mélange en eau totale (Meso-NH: RT = RVT+RCT(+RRT)).
    p : ndarray, même forme (ou broadcastable, ex. profil (nz,1,1))
        Pression (Meso-NH: PABST), en Pa, utilisée pour l'Exner.
    rr : ndarray ou None
        A fournir seulement si rt n'inclut PAS déjà la pluie et que tu
        veux l'ajouter à part; sinon laisser None (rc = rt - rv suffit).
    axis : tuple
        Axes horizontaux pour la moyenne (cf. compute_buoyancy_stcu).
 
    Retour
    ------
    b : ndarray, même forme que thlm
        Champ de flottabilité Phi.
    """
    
    G = 9.81  # m/s^2
    EPS_V = 0.61  # ~ Rv/Rd - 1, coefficient de virtualisation
    LV = 2.5e6  # J/kg, chaleur latente de vaporisation
    CP = 1004.0  # J/kg/K
    RD_CP = 0.2857  # Rd/cp
    P00 = 1.0e5  # Pa, pression de référence pour l'Exner
    
    
    exner = (p / P00) ** RD_CP
    rc = rt - rv
    if rr is not None:
        rc = rc - rr  # au cas où rt n'inclurait pas la pluie séparément
 
    theta_v = thlm * (1.0 + EPS_V * rv) + rc * (LV / (CP * exner) - thlm)
 
    theta_v_mean = np.mean(theta_v, axis=axis, keepdims=True)
    b = G * (theta_v - theta_v_mean) / theta_v_mean
 
    return b

