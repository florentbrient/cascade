"""
Local, height-resolved turbulent kinetic-energy cascade flux via
horizontal-only coarse-graining (real-space filtering).

Method
------
Standard spectral/global-FFT flux estimators assume homogeneity (periodicity)
in every direction they transform, which breaks down when a field is
truncated or windowed along an inhomogeneous direction (here: altitude z).

This module instead follows the coarse-graining / filtering framework of
Eyink (1995, J. Stat. Phys. 78:335-351); Eyink & Aluie (2009, Phys. Fluids
21:115107 & 115108, Parts I & II); Aluie (2013, Physica D 247:54-65); and
its geophysical application in Aluie, Hecht & Vallis (2018, J. Phys.
Oceanogr.) "Mapping the Energy Cascade in the North Atlantic Ocean" -- which
plots exactly this kind of depth-resolved flux profile in a stratified,
horizontally-homogeneous-but-vertically-inhomogeneous flow.

The horizontal (x,y) directions are periodic/homogeneous in a typical LES
domain, so filtering is applied ONLY in x,y, independently at every z-level.
The vertical direction is never transformed or truncated, so:
  - no artificial discontinuity is introduced anywhere,
  - no vertical periodicity assumption is required,
  - there is no decomposition of u into pieces, hence no
    cross-interaction-term problem to correct for.

The flux computed is the Galilean-invariant subfilter-scale (SFS) flux,
    Pi_ell(x,y,z) = - S_ell : tau_ell(u,u)
with
    tau_ell(u_i,u_j) = filter(u_i u_j) - filter(u_i) filter(u_j)
    S_ell            = symmetric strain-rate tensor of the filtered field
This is the form recommended over the naive u.(grad u_ell).u_ell-type flux,
which is NOT Galilean invariant and can be strongly biased by mean
shear/sweeping (see Aluie, Hecht & Vallis 2018, their Fig. 3 and Sec. 2.3) --
relevant here given the mean wind shear typical of the surface layer.

See also Green, Vlaykov, Mellado & Wilczek (2020, JFM 887:A21), who apply
essentially this approach to Rayleigh-Benard convection and find a
"height-dependent energy transfer rate [with] a complex structure with
distinct bulk and boundary layer features" -- i.e. the same phenomenology
this module is meant to test for the boundary layer.
"""

import numpy as np


def compute_coarse_grained_flux(U, V, W, dx, dy, z, k_cut, filter_type='sharp'):
    """
    Local (real-space) coarse-grained kinetic-energy flux at a single
    horizontal cutoff wavenumber k_cut, resolved at every (x, y, z) point.

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
        Horizontal filter cutoff wavenumber [rad/m]. Motions with
        horizontal wavenumber magnitude <= k_cut are retained in the
        "resolved"/large-scale field; the rest is subfilter.
    filter_type : {'sharp', 'gaussian'}, optional
        'sharp'    : spectral (top-hat in k-space) cutoff filter -- matches
                     the sharp-spectral convention already used in your
                     truncated-field pipeline.
        'gaussian' : smooth roll-off in k-space (no ringing); useful as a
                     robustness/sensitivity check against 'sharp'.

    Returns
    -------
    Pi_xyz : ndarray, shape (Nz, Ny, Nx)
        Local Galilean-invariant SFS energy flux across scale k_cut, at
        every grid point. Sign convention: positive = downscale (direct
        cascade, large -> small), negative = upscale (inverse cascade).
    Pi_z : ndarray, shape (Nz,)
        Horizontal average of Pi_xyz at each altitude -- the height-resolved
        flux profile Pi_ell(z) you want to compare against Pi_+(k,z) +
        Pi_-(k,z) from the truncated-field method.
    """
    Nz, Ny, Nx = U.shape

    # Horizontal wavenumbers matching rfft2's (full-ky, half-plus-one-kx) layout
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx = 2 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')   # shape (Ny, Nx//2+1)
    K2 = KX**2 + KY**2

    if filter_type == 'sharp':
        mask = (K2 <= k_cut**2).astype(float)
    elif filter_type == 'gaussian':
        # Standard coarse-graining Gaussian kernel (Eyink & Aluie 2009),
        # ell = 2*pi/k_cut as the nominal filter width.
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
        """Horizontal derivative via the (periodic) spectral representation."""
        Fhat = np.fft.rfft2(field, axes=(-2, -1))
        if direction == 'x':
            Fhat = 1j * KX * Fhat
        elif direction == 'y':
            Fhat = 1j * KY * Fhat
        else:
            raise ValueError("direction must be 'x' or 'y'")
        return np.fft.irfft2(Fhat, s=(Ny, Nx), axes=(-2, -1))

    # --- resolved (large-scale) filtered velocity field ---
    Ub, Vb, Wb = hfilter(U), hfilter(V), hfilter(W)

    # --- subfilter-scale stress tensor: tau_ij = filter(u_i u_j) - u_i_bar u_j_bar ---
    tau_xx = hfilter(U * U) - Ub * Ub
    tau_yy = hfilter(V * V) - Vb * Vb
    tau_zz = hfilter(W * W) - Wb * Wb
    tau_xy = hfilter(U * V) - Ub * Vb
    tau_xz = hfilter(U * W) - Ub * Wb
    tau_yz = hfilter(V * W) - Vb * Wb

    # --- strain-rate tensor of the filtered field ---
    # horizontal derivatives: spectral (periodic, exact for band-limited field)
    # vertical derivative: real-space, handles non-uniform z spacing
    dUb_dx = hderiv(Ub, 'x'); dUb_dy = hderiv(Ub, 'y'); dUb_dz = np.gradient(Ub, z, axis=0)
    dVb_dx = hderiv(Vb, 'x'); dVb_dy = hderiv(Vb, 'y'); dVb_dz = np.gradient(Vb, z, axis=0)
    dWb_dx = hderiv(Wb, 'x'); dWb_dy = hderiv(Wb, 'y'); dWb_dz = np.gradient(Wb, z, axis=0)

    S_xx = dUb_dx
    S_yy = dVb_dy
    S_zz = dWb_dz
    S_xy = 0.5 * (dUb_dy + dVb_dx)
    S_xz = 0.5 * (dUb_dz + dWb_dx)
    S_yz = 0.5 * (dVb_dz + dWb_dy)

    # --- local Galilean-invariant SFS flux: Pi_ell = -S_ell : tau_ell ---
    Pi_xyz = -(S_xx * tau_xx + S_yy * tau_yy + S_zz * tau_zz
               + 2 * S_xy * tau_xy + 2 * S_xz * tau_xz + 2 * S_yz * tau_yz)

    Pi_z = Pi_xyz.mean(axis=(1, 2))

    return Pi_xyz, Pi_z


def compute_coarse_grained_flux_profile(U, V, W, dx, dy, z, k_cuts,
                                         filter_type='sharp', return_3d=False):
    """
    Loop compute_coarse_grained_flux over an array of cutoff wavenumbers to
    build the (k, z) flux array -- the direct analogue of your existing
    Pi_BT/Pi_TB arrays, but computed with no truncation and no cross terms.

    Parameters
    ----------
    k_cuts : array_like
        Horizontal cutoff wavenumbers [rad/m], e.g. reuse your existing
        `kbins`/`k3D` array (restricted to horizontal wavenumbers) for a
        like-for-like comparison against Pi_BT + Pi_TB.
    return_3d : bool, optional
        If True, also return the full (Nk, Nz, Ny, Nx) array of Pi_xyz for
        every cutoff (memory-heavy -- only enable for a small grid/coarse
        k_cuts, e.g. for the plume/w-correlation diagnostic).

    Returns
    -------
    Pi_z_k : ndarray, shape (len(k_cuts), Nz)
        Height-resolved flux spectrum, directly comparable to
        Pi_BT[:, iz] + Pi_TB[:, iz] from your truncated-field pipeline.
    Pi_xyz_k : ndarray, shape (len(k_cuts), Nz, Ny, Nx), optional
        Only returned if return_3d=True.
    """
    k_cuts = np.asarray(k_cuts)
    Nz = U.shape[0]
    Pi_z_k = np.zeros((len(k_cuts), Nz))
    Pi_xyz_k = [] if return_3d else None

    for i, kc in enumerate(k_cuts):
        Pi_xyz, Pi_z = compute_coarse_grained_flux(U, V, W, dx, dy, z, kc,
                                                     filter_type=filter_type)
        Pi_z_k[i, :] = Pi_z
        if return_3d:
            Pi_xyz_k.append(Pi_xyz)

    if return_3d:
        return Pi_z_k, np.array(Pi_xyz_k)
    return Pi_z_k


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

    k_cuts = np.linspace(0.5, 8, 6) * (2 * np.pi / (Nx * dx))
    Pi_z_k = compute_coarse_grained_flux_profile(U, V, W, dx, dy, z, k_cuts)
    print("Pi_z_k shape:", Pi_z_k.shape)
    print(Pi_z_k)
