#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:11:30 2026

@author: fbrient
"""

import numpy as np


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


if __name__ == "__main__":
    # --- quick sanity checks ---

    # 1) horizontal: compare spectral derivative to analytic on a periodic field
    nx = 64
    Lx = 2 * np.pi
    dx = Lx / nx
    x = np.arange(nx) * dx
    kmode = 5
    u2d = np.tile(np.sin(kmode * x), (10, 1))  # shape (ny, nx), constant in y
    du_dx_analytic = kmode * np.cos(kmode * x)

    du_dx, *_ = compute_gradients(u2d, dx, dx, z=None)
    err = np.max(np.abs(du_dx[0, :] - du_dx_analytic))
    print("horizontal spectral derivative max error:", err)

    # 2) vertical: compare corrected stencil to analytic on a STRETCHED grid
    nz = 20
    z = np.cumsum(np.linspace(1.0, 3.0, nz))  # stretched, increasing spacing
    z = z - z[0]
    u3d = (z**2)[:, None, None] * np.ones((1, 4, 4))  # u = z^2 -> du/dz = 2z
    du_dz_analytic = 2 * z

    _, _, _, _, _, _, du_dz, _, _ = compute_gradients(
        u3d, dx=1.0, dy=1.0, z=z
    )
    err_interior = np.max(
        np.abs(du_dz[1:-1, 0, 0] - du_dz_analytic[1:-1])
    )
    print("vertical (stretched grid) interior max error:", err_interior)

    # compare against the OLD buggy formula to show the difference
    du_dz_old = np.zeros_like(u3d[:, 0, 0])
    dz_arr = np.diff(z)
    dz_arr = np.append(dz_arr, dz_arr[-1])  # rough stand-in for old dz[k]
    for k in range(1, nz - 1):
        du_dz_old[k] = (u3d[k + 1, 0, 0] - u3d[k - 1, 0, 0]) / (2. * dz_arr[k])
    err_old = np.max(np.abs(du_dz_old[1:-1] - du_dz_analytic[1:-1]))
    print("vertical OLD (2*dz[k]) interior max error:", err_old)