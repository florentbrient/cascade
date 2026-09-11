"""
Local, height-resolved turbulent kinetic-energy cascade flux via
coarse-graining (real-space filtering), with an optional anisotropic
vertical-coherence scale ell_z.

Method
------
Standard spectral/global-FFT flux estimators assume homogeneity (periodicity)
in every direction they transform, which breaks down when a field is
truncated or windowed along an inhomogeneous direction (here: altitude z).

This module follows the coarse-graining / filtering framework of Eyink
(1995, J. Stat. Phys. 78:335-351); Eyink & Aluie (2009, Phys. Fluids
21:115107 & 115108, Parts I & II); Aluie (2013, Physica D 247:54-65,
which explicitly discusses generalized/anisotropic filter kernels); and
its geophysical application in Aluie, Hecht & Vallis (2018, J. Phys.
Oceanogr.) "Mapping the Energy Cascade in the North Atlantic Ocean". See
also Green, Vlaykov, Mellado & Wilczek (2020, JFM 887:A21) for the same
approach applied to Rayleigh-Benard convection.

Filter kernel
-------------
The horizontal (x,y) directions are always filtered spectrally
(periodic/homogeneous). By default (ell_z=None) z is left completely
untouched -- every altitude filtered independently, as in the original
version of this module.

Setting ell_z (a physical length, in the same units as `z`) additionally
applies a REAL-SPACE Gaussian smoothing along z, of physical width ell_z,
built from the actual z-coordinates (so it is exact for a non-uniformly
spaced vertical grid) with an OPEN boundary: weights are renormalized to
sum to 1 using only the levels that exist, so the effective kernel simply
tapers near the domain edges -- no periodic wrap-around and no mirroring
of data across a boundary that isn't physically there (unlike a spectral
vertical treatment, or scipy's 'wrap'/'reflect' boundary modes).

This turns the filter into a genuine anisotropic 3D kernel, G_ell(x,y,z) =
G_h(x,y) * G_v(z), able to "see" vertically-coherent structures (e.g.
thermal plumes spanning much of the boundary-layer depth) as a single
correlated unit rather than as unrelated features at each height
independently. Motivation: if the horizontal-only (ell_z=None) result
under-estimates the flux carried by such structures relative to the
existing 3D-isotropic |k|=sqrt(kx^2+ky^2+kz^2) truncated-field method, a
vertical coherence scale comparable to the plume/boundary-layer depth
should recover some of that missing amplitude; ell_z -> 0 should reproduce
the ell_z=None (per-level-independent) result.

Two flux forms are provided, for sensitivity comparison:

  'sfs'   : Pi_ell = -S_ell : tau_ell(u,u),  tau_ell(u_i,u_j) = filter(u_i u_j)
            - filter(u_i) filter(u_j). Galilean invariant by construction.
            Recommended by Aluie, Hecht & Vallis (2018) over the form below,
            which can be biased by mean shear/sweeping.

  'naive' : Pi_ell = mean[ ubar_i,ell * (u . grad) u_i ], the filtered field
            dotted with the advection term built from the FULL, unfiltered
            field -- direct analogue of the existing spectral
            Pi(k) = mean(u_<k . (u.grad)u) formula (same sign convention).
            NOT Galilean invariant. Only ubar_ell (not the advection term)
            uses the (possibly ell_z-extended) filter, in both forms.

Updated limiting-behaviour note (see previous version's "k_cut -> 0 / inf"
check): with ell_z=None, Pi_ell(z) -> 0 at BOTH k_cut -> 0 and k_cut -> inf
(see docstring of compute_coarse_grained_flux). With ell_z finite and
FIXED, Pi_ell(z) no longer trivially vanishes as k_cut -> inf, because an
unresolved *vertical* scale still contributes to tau even when nothing is
removed horizontally. The correct joint check is k_cut -> inf AND
ell_z -> 0 simultaneously (i.e. the total 3D filter width -> 0).
"""

import numpy as np
import pylab as plt


def _vertical_filter_matrix(z, ell_z):
    """
    Build the (Nz, Nz) weight matrix for a real-space Gaussian vertical
    smoothing of physical width ell_z, using the actual z-coordinates
    (correct for a non-uniform vertical grid) with an open (non-periodic,
    non-reflected) boundary: each row's weights are renormalized to sum to
    1 using only the levels that exist, so the kernel tapers smoothly near
    the domain edges rather than assuming periodicity or mirroring.

    Parameters
    ----------
    z : ndarray, shape (Nz,)
    ell_z : float
        Standard deviation of the vertical Gaussian kernel [same units as z].

    Returns
    -------
    Wz : ndarray, shape (Nz, Nz)
        Wz[i, j] = normalized weight of level j in the smoothed value at
        level i. Apply via np.tensordot(Wz, field, axes=(1, 0)).
    """
    z = np.asarray(z)
    dZ = z[:, None] - z[None, :]
    W = np.exp(-0.5 * (dZ / ell_z) ** 2)
    W /= W.sum(axis=1, keepdims=True)
    return W


def compute_coarse_grained_flux(U, V, W, dx, dy, z, k_cut,
                                 filter_type='sharp', flux_type='both',
                                 ell_z=None):
    """
    Local (real-space) coarse-grained kinetic-energy flux at a single
    horizontal cutoff wavenumber k_cut (and, optionally, vertical
    coherence length ell_z), resolved at every (x, y, z) point.

    Parameters
    ----------
    U, V, W : ndarray, shape (Nz, Ny, Nx)
        3D velocity components on a horizontally-periodic, uniformly-spaced
        (dx, dy) grid. z may be non-uniformly spaced.
    dx, dy : float
        Horizontal grid spacing [m].
    z : ndarray, shape (Nz,)
        Vertical coordinate [m]. Need not be uniformly spaced.
    k_cut : float
        Horizontal filter cutoff wavenumber [rad/m].
    filter_type : {'sharp', 'gaussian'}, optional
        Horizontal filter kernel (see module docstring).
    flux_type : {'sfs', 'naive', 'both'}, optional
        Which flux form(s) to compute (see module docstring).
    ell_z : float or None, optional
        Physical vertical-coherence length [same units as z]. None (default)
        reproduces the original behaviour (every z-level filtered
        independently). A finite value adds real-space Gaussian smoothing
        of that width along z to the filter kernel (see module docstring).

    Returns
    -------
    result : dict
        Keys present depend on flux_type:
          result['sfs']   = (Pi_xyz, Pi_z)   if flux_type in ('sfs','both')
          result['naive'] = (Pi_xyz, Pi_z)   if flux_type in ('naive','both')
        Pi_xyz has shape (Nz, Ny, Nx); Pi_z = Pi_xyz.mean(axis=(1,2)),
        shape (Nz,).
    """
    if flux_type not in ('sfs', 'naive', 'both'):
        raise ValueError("flux_type must be 'sfs', 'naive', or 'both'")

    Nz, Ny, Nx = U.shape

    # Horizontal wavenumbers matching rfft2's (full-ky, half-plus-one-kx) layout
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx = 2 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')   # shape (Ny, Nx//2+1)
    K2 = KX**2 + KY**2

    if filter_type == 'sharp':
        mask = (K2 <= k_cut**2).astype(float)
    elif filter_type == 'gaussian':
        ell = 2 * np.pi / k_cut
        mask = np.exp(-K2 * ell**2 / 24.0)
    else:
        raise ValueError("filter_type must be 'sharp' or 'gaussian'")

    def hfilter(field):
        """Horizontal-only low-pass filter, independently at every z-level."""
        Fhat = np.fft.rfft2(field, axes=(-2, -1))
        Fhat = Fhat * mask
        return np.fft.irfft2(Fhat, s=(Ny, Nx), axes=(-2, -1))

    def hderiv(field, direction):
        """Horizontal derivative via the (periodic) spectral representation.
        Works on ANY field (filtered or raw)."""
        Fhat = np.fft.rfft2(field, axes=(-2, -1))
        if direction == 'x':
            Fhat = 1j * KX * Fhat
        elif direction == 'y':
            Fhat = 1j * KY * Fhat
        else:
            raise ValueError("direction must be 'x' or 'y'")
        return np.fft.irfft2(Fhat, s=(Ny, Nx), axes=(-2, -1))

    if ell_z is not None:
        Wz = _vertical_filter_matrix(z, ell_z)

        def cgfilter(field):
            """Full anisotropic coarse-graining filter: horizontal spectral
            cutoff, then real-space vertical Gaussian smoothing of physical
            width ell_z (open boundary, non-uniform-z aware)."""
            return np.tensordot(Wz, hfilter(field), axes=(1, 0))
    else:
        cgfilter = hfilter   # original behaviour: every z-level independent

    result = {}

    # --- resolved (large-scale) filtered velocity field (needed by both forms) ---
    Ub, Vb, Wb = cgfilter(U), cgfilter(V), cgfilter(W)

    if flux_type in ('sfs', 'both'):
        # subfilter-scale stress tensor: tau_ij = filter(u_i u_j) - u_i_bar u_j_bar
        tau_xx = cgfilter(U * U) - Ub * Ub
        tau_yy = cgfilter(V * V) - Vb * Vb
        tau_zz = cgfilter(W * W) - Wb * Wb
        tau_xy = cgfilter(U * V) - Ub * Vb
        tau_xz = cgfilter(U * W) - Ub * Wb
        tau_yz = cgfilter(V * W) - Vb * Wb

        dUb_dx = hderiv(Ub, 'x'); dUb_dy = hderiv(Ub, 'y'); dUb_dz = np.gradient(Ub, z, axis=0)
        dVb_dx = hderiv(Vb, 'x'); dVb_dy = hderiv(Vb, 'y'); dVb_dz = np.gradient(Vb, z, axis=0)
        dWb_dx = hderiv(Wb, 'x'); dWb_dy = hderiv(Wb, 'y'); dWb_dz = np.gradient(Wb, z, axis=0)

        S_xx = dUb_dx
        S_yy = dVb_dy
        S_zz = dWb_dz
        S_xy = 0.5 * (dUb_dy + dVb_dx)
        S_xz = 0.5 * (dUb_dz + dWb_dx)
        S_yz = 0.5 * (dVb_dz + dWb_dy)

        Pi_sfs_xyz = -(S_xx * tau_xx + S_yy * tau_yy + S_zz * tau_zz
                       + 2 * S_xy * tau_xy + 2 * S_xz * tau_xz + 2 * S_yz * tau_yz)
        result['sfs'] = (Pi_sfs_xyz, Pi_sfs_xyz.mean(axis=(1, 2)))

    if flux_type in ('naive', 'both'):
        # advection term N_i = u_j d(u_i)/dx_j, built from the FULL (raw,
        # unfiltered) field -- always, regardless of ell_z.
        dU_dx = hderiv(U, 'x'); dU_dy = hderiv(U, 'y'); dU_dz = np.gradient(U, z, axis=0)
        dV_dx = hderiv(V, 'x'); dV_dy = hderiv(V, 'y'); dV_dz = np.gradient(V, z, axis=0)
        dW_dx = hderiv(W, 'x'); dW_dy = hderiv(W, 'y'); dW_dz = np.gradient(W, z, axis=0)

        N_U = U * dU_dx + V * dU_dy + W * dU_dz
        N_V = U * dV_dx + V * dV_dy + W * dV_dz
        N_W = U * dW_dx + V * dW_dy + W * dW_dz

        # (possibly ell_z-extended) filtered field dotted with raw advection
        # term -- same sign convention as compute_Pi_from_uBF (no leading minus)
        Pi_naive_xyz = Ub * N_U + Vb * N_V + Wb * N_W
        result['naive'] = (Pi_naive_xyz, Pi_naive_xyz.mean(axis=(1, 2)))

    return result


def compute_coarse_grained_flux_profile(U, V, W, dx, dy, z, k_cuts,
                                         filter_type='sharp', flux_type='both',
                                         ell_z=None):
    """
    Loop compute_coarse_grained_flux over an array of cutoff wavenumbers to
    build the (k, z) flux array(s), at a single (fixed) ell_z.

    Returns
    -------
    Pi_z_k : dict
        Pi_z_k['sfs']   : ndarray (len(k_cuts), Nz), if requested
        Pi_z_k['naive'] : ndarray (len(k_cuts), Nz), if requested
    """
    k_cuts = np.asarray(k_cuts)
    Nz = U.shape[0]
    want = ('sfs', 'naive') if flux_type == 'both' else (flux_type,)
    Pi_z_k = {key: np.zeros((len(k_cuts), Nz)) for key in want}

    for i, kc in enumerate(k_cuts):
        res = compute_coarse_grained_flux(U, V, W, dx, dy, z, kc,
                                           filter_type=filter_type,
                                           flux_type=flux_type, ell_z=ell_z)
        for key in want:
            Pi_z_k[key][i, :] = res[key][1]

    return Pi_z_k


def compute_coarse_grained_flux_ellz_sweep(U, V, W, dx, dy, z, k_cuts, ell_z_list,
                                            filter_type='sharp', flux_type='both'):
    """
    Sweep the vertical-coherence length ell_z at a fixed set of horizontal
    cutoffs k_cuts -- the diagnostic to test whether a purely horizontal
    filter under-represents flux carried by vertically-coherent structures
    (e.g. thermal plumes): if Pi_z_k amplitude grows substantially as ell_z
    increases from 0 towards a plausible plume/boundary-layer-depth scale,
    that supports the vertical-coherence hypothesis. Include None in
    ell_z_list to keep the original (per-level-independent) result as the
    baseline of the sweep.

    Parameters
    ----------
    ell_z_list : list of (float or None)
        Vertical-coherence lengths to test, e.g. [None, 0.1*zi, 0.3*zi, zi]
        for a boundary-layer depth zi.

    Returns
    -------
    sweep : dict
        sweep[ell_z] = Pi_z_k dict (as returned by
        compute_coarse_grained_flux_profile) for that ell_z. Keys are
        exactly the values passed in ell_z_list (including None).
    """
    sweep = {}
    for ell_z in ell_z_list:
        sweep[ell_z] = compute_coarse_grained_flux_profile(
            U, V, W, dx, dy, z, k_cuts, filter_type=filter_type,
            flux_type=flux_type, ell_z=ell_z)
    return sweep


# NEW PART for calculated PI(ℓh​,ℓz​)
def compute_coarse_grained_flux_2D(U, V, W, dx, dy, z, ell_h, ell_z, filter_type='sharp', flux_type='sfs'):
    """
    Wrapper pour compute_coarse_grained_flux qui prend ell_h au lieu de k_cut.
    """
    if ell_h <= 0:
        raise ValueError("ell_h doit être > 0")
    k_cut = 2 * np.pi / ell_h
    return compute_coarse_grained_flux(U, V, W, dx, dy, z, k_cut, filter_type, flux_type, ell_z)

def compute_Pi_2D_map(U, V, W, dx, dy, z, ell_h_list, ell_z_list, filter_type='sharp', flux_type='sfs'):
    """
    Calcule la carte 2D Pi(ell_h, ell_z).
    """
    Pi_map = np.zeros((len(ell_h_list), len(ell_z_list)))

    for i, ell_h in enumerate(ell_h_list):
        for j, ell_z in enumerate(ell_z_list):
            res = compute_coarse_grained_flux_2D(
                U, V, W, dx, dy, z, ell_h, ell_z, filter_type, flux_type
            )
            # Prendre la moyenne sur z (ou un autre profil)
            Pi_map[i, j] = res[flux_type][1].mean()

    return Pi_map




if __name__ == "__main__":
    # Minimal smoke test with synthetic data -- replace with your own
    # UT_new, VT_new, WT_new, z_new, dx, dy when integrating into your pipeline.
    np.random.seed(0)
    Nz, Ny, Nx = 20, 32, 32
    dx = dy = 50.0
    z = np.cumsum(np.r_[0, np.linspace(10, 40, Nz - 1)])

    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y, indexing='xy')
    U = np.zeros((Nz, Ny, Nx)); V = np.zeros((Nz, Ny, Nx)); W = np.zeros((Nz, Ny, Nx))
    for iz in range(Nz):
        amp = 1.0 + 0.5 * np.sin(2 * np.pi * iz / Nz)
        U[iz] = amp * np.sin(2 * np.pi * X / (4 * dx)) + 0.3 * np.random.randn(Ny, Nx)
        V[iz] = amp * np.cos(2 * np.pi * Y / (6 * dy)) + 0.3 * np.random.randn(Ny, Nx)
        W[iz] = 0.2 * np.random.randn(Ny, Nx)

    nyq = np.pi / dx
    k_cuts = np.linspace(0.02, 0.9, 6) * nyq

    # 1) regression check: ell_z=None must reproduce the previous behaviour
    Pi_z_k_none = compute_coarse_grained_flux_profile(U, V, W, dx, dy, z, k_cuts, flux_type='both')
    print("ell_z=None  sfs shape:", Pi_z_k_none['sfs'].shape)

    # 2) ell_z -> 0 limit must recover the ell_z=None result closely
    tiny = 1e-3 * np.min(np.diff(z))
    res_tiny = compute_coarse_grained_flux(U, V, W, dx, dy, z, k_cuts[2],
                                            flux_type='both', ell_z=tiny)
    res_none = compute_coarse_grained_flux(U, V, W, dx, dy, z, k_cuts[2],
                                            flux_type='both', ell_z=None)
    diff = np.max(np.abs(res_tiny['sfs'][1] - res_none['sfs'][1]))
    scale = np.max(np.abs(res_none['sfs'][1])) + 1e-30
    print("ell_z -> 0 vs ell_z=None: max relative diff =", diff / scale)

    # 3) weight-matrix sanity: rows sum to 1, matrix -> identity as ell_z -> 0
    Wz_small = _vertical_filter_matrix(z, tiny)
    print("Wz row sums (should all be 1):", np.allclose(Wz_small.sum(axis=1), 1.0))
    print("Wz -> identity as ell_z -> 0:", np.allclose(Wz_small, np.eye(Nz), atol=1e-6))

    # 4) ell_z sweep, fixed k_cuts, comparing a few coherence lengths
    zi = z[-1]  # placeholder "boundary-layer depth" for the synthetic test
    ell_z_list = [None, 0.05 * zi, 0.2 * zi, 0.5 * zi]
    sweep = compute_coarse_grained_flux_ellz_sweep(U, V, W, dx, dy, z, k_cuts,
                                                    ell_z_list, flux_type='sfs')
    for ell_z in ell_z_list:
        peak = np.max(np.abs(sweep[ell_z]['sfs']))
        print(f"ell_z={ell_z}: max|Pi_sfs| over (k,z) = {peak:.4e}")
        
        
        
# Exemple d'utilisation dans le main
#if __name__ == "__main__":
    # ... (ton code existant pour générer U, V, W, z, etc.)

    # Définir les gammes d'échelles
    Lx = Nx * dx
    zi = z[-1]  # Hauteur de la couche limite
    ell_h_list = np.logspace(np.log10(2*dx), np.log10(Lx/2), 15)
    ell_z_list = np.logspace(np.log10(2*np.min(np.diff(z))), np.log10(zi), 15)

    # Calculer la carte Pi(ell_h, ell_z)
    Pi_map = compute_Pi_2D_map(U, V, W, dx, dy, z, ell_h_list, ell_z_list, flux_type='sfs')

    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.contourf(
        ell_h_list, ell_z_list, Pi_map.T,
        levels=20, cmap='RdBu', norm=plt.Normalize(vmin=-1e-4, vmax=1e-4)
    )
    plt.colorbar(label='Flux $\Pi(\ell_h, \ell_z)$ [m²/s³]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Échelle horizontale $\ell_h$ [m]')
    plt.ylabel('Échelle verticale $\ell_z$ [m]')
    plt.title('Carte 2D du flux $\Pi(\ell_h, \ell_z)$')
    plt.grid(True, which='both', linestyle='--')
    plt.show()