#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectre radial isotrope d'un champ 2D périodique (LWP, thl, ...), plus
échelles de longueur calculées SANS binning.

Idée : garder ce qu'il y a de bon dans les deux approches existantes.
  - comme compute_spectra (spectral.py) : estimateur ISOTROPE
        E(k) = <P>_{modes du bin} * 2*pi*k / (dkx*dky)
    (moyenne par mode x aire de l'anneau) -> pas de bruit lié au nombre
    aléatoire de modes par bin, pas de biais sur les coquilles incomplètes ;
  - comme compute_spectral_transfer (spectratools.py) : bins log-espacés
    qui montent jusqu'au grand k, avec un minimum de nmin modes par bin
    (aucun bin vide ou à 1 seul mode aux petits k).

Conventions
  P(kx,ky) = |FFT2(f - <f>)|^2 / (Nx*Ny)^2   ->  sum(P) = var(f)  (Parseval)
  Ek[i]    = sum(P) dans le bin i          (variance du bin, somme EXACTE)
  dk_eff   = n_i * dkx*dky / (2*pi*kbar_i) (largeur de l'anneau idéal à n_i modes)
  E[i]     = Ek[i] / dk_eff[i]              ->  sum(E*dk_eff) = var(f) exactement
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. modes 2D : P(kx, ky) et |k|
# ----------------------------------------------------------------------
def power_modes(field, dx, dy=None, remove_mean=True, window=None):
    """Retourne kh (Ny,Nx), P (Ny,Nx), dA = dkx*dky.  sum(P) = variance."""
    f = np.asarray(field, dtype=float)
    dy = dx if dy is None else dy
    Ny, Nx = f.shape
    if remove_mean:
        f = f - f.mean()
    if window is not None:                      # seulement si domaine NON périodique
        wy, wx = np.hanning(Ny), np.hanning(Nx)
        w = np.outer(wy, wx)
        f = f * w / np.sqrt(np.mean(w**2))      # garde la variance
    F = np.fft.fft2(f)

    P = np.abs(F) ** 2 / (Nx * Ny) ** 2

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    kh = np.hypot(KX, KY)
    dA = (2 * np.pi / (Nx * dx)) * (2 * np.pi / (Ny * dy))
    return kh, P, dA


# ----------------------------------------------------------------------
# 2. spectre radial isotrope, bins log avec >= nmin modes
# ----------------------------------------------------------------------
def radial_spectrum(field, dx, dy=None, nbins=150, nmin=8, kmax="nyquist",
                    remove_mean=True, window=None,
                    field1 = None, field2 = None):
    """
    Parameters
    ----------
    nbins : nb de bins log visé (réduit automatiquement si nmin l'impose).
            Aux grands k les bins ont des milliers de modes : en mettre beaucoup ne
            coûte rien en bruit et évite un biais sur la queue raide (avec 60 bins
            larges de ~9 %, +10 % près de Nyquist sur mon test ; avec 150, +2 %).
    nmin  : nb minimum de modes 2D par bin. 8 = la première coquille entière
            (comme compute_spectra) ; monter à 15-20 pour un spectre plus lisse.
    kmax  : 'nyquist'  -> coupe à pi/dx : coquilles complètes, spectre isotrope
                           bien défini (recommandé)
            'corner'   -> garde aussi les coins du carré (jusqu'à sqrt(2)*k_Nyq).
                           Coquilles partielles, uniquement directions diagonales :
                           à tracer en pointillé, à ne pas interpréter physiquement.
            float      -> valeur explicite (rad/m)

    Returns
    -------
    dict : k (moyenne des |k| des modes du bin = abscisse à tracer),
           E (densité par unité de k), Ek (variance du bin), n (modes),
           dk_eff, k_lo, k_hi, var (variance du champ), var_kept
    """
    kh, P, dA = power_modes(field, dx, dy, remove_mean, window)
    if field1 is not None:
        _, P1, _ = power_modes(field, dx, dy, remove_mean, window)
        P += P1
    if field2 is not None:
        _, P2, _ = power_modes(field, dx, dy, remove_mean, window)
        P += P2
    #print(kh[kh>0].min())
    Ny, Nx = kh.shape
    dy_ = dx if dy is None else dy

    k_nyq = min(np.pi / dx, np.pi / dy_)
    if kmax == "nyquist":
        kcut = k_nyq * (1 + 1e-9)
    elif kmax == "corner":
        kcut = kh.max() * (1 + 1e-9)
    else:
        kcut = float(kh.max())

    sel = (kh > 0) & (kh <= kcut)
    kv, Pv = kh[sel], P[sel]
    order = np.argsort(kv, kind="stable")
    ks, Ps = kv[order], Pv[order]
    n = ks.size

    # --- bornes de bins : cibles log, puis fusion pour avoir >= nmin modes
    targets = np.geomspace(ks[0], ks[-1], nbins + 1)[1:]
    cuts = np.searchsorted(ks, targets * (1 + 1e-9), side="right")
    bounds = [0]
    for c in cuts:
        if c - bounds[-1] >= nmin:
            bounds.append(int(c))
    if bounds[-1] < n:                       # reste en queue
        if n - bounds[-1] >= nmin or len(bounds) == 1:
            bounds.append(n)
        else:
            bounds[-1] = n                   # fusionne la queue trop petite
    bounds = np.asarray(bounds)

    i0, i1 = bounds[:-1], bounds[1:]
    cs = np.concatenate(([0.0], np.cumsum(Ps)))
    ck = np.concatenate(([0.0], np.cumsum(ks)))
    cnt = (i1 - i0).astype(float)
    Ek = cs[i1] - cs[i0]
    kbar = (ck[i1] - ck[i0]) / cnt
    dk_eff = cnt * dA / (2 * np.pi * kbar)
    E = Ek / dk_eff

    return dict(k=kbar, E=E, Ek=Ek, n=cnt.astype(int), dk_eff=dk_eff,
                k_lo=ks[i0], k_hi=ks[i1 - 1],
                var=float(np.var(np.asarray(field, float))),
                var_kept=float(Ek.sum()), k_nyq=k_nyq)


# ----------------------------------------------------------------------
# 3. échelles de longueur SANS binning (mode par mode)
# ----------------------------------------------------------------------
def length_scales(field, dx, dy=None, kmax="nyquist", frac_small=2.0 / 3.0,
                  nbins_peak=40, nmin_peak=15, remove_mean=True, window=None,
                  field1 = None, field2 = None):
    """
    lambda_mean : 2*pi / <k>,  <k> = sum(k P) / sum(P)   (Pino et al. 2006, ordre 1)
    lambda_ogive: longueur d'onde où une fraction `frac_small` de la variance
                  est à plus petite échelle (de Roode et al. 2004, 2/3 par défaut)
                  -> ogive cumulée mode par mode, monotone, aucun bruit de binning
    lambda_peak : max de k*E(k) (forme "variance-preserving"), raffiné par une
                  parabole en log k autour du max d'un spectre lissé
    """
    kh, P, dA = power_modes(field, dx, dy, remove_mean, window)
    if field1 is not None:
        _, P1, _ = power_modes(field, dx, dy, remove_mean, window)
        P += P1
    if field2 is not None:
        _, P2, _ = power_modes(field, dx, dy, remove_mean, window)
        P += P2
    
    dy_ = dx if dy is None else dy
    k_nyq = min(np.pi / dx, np.pi / dy_)
    kcut = k_nyq * (1 + 1e-9) if kmax == "nyquist" else (
        kh.max() if kmax == "corner" else float(kmax))
    sel = (kh > 0) & (kh <= kcut)
    kv, Pv = kh[sel], P[sel]
    o = np.argsort(kv, kind="stable")
    kv, Pv = kv[o], Pv[o]

    # moyenne
    kmean = np.sum(kv * Pv) / np.sum(Pv)

    # ogive (cumul depuis les grandes échelles)
    C = np.cumsum(Pv) / np.sum(Pv)
    kc = np.interp(1.0 - frac_small, C, kv)

    # pic de k*E(k) sur un spectre lissé (bins plus gros => peu de bruit)
    sp = radial_spectrum(field, dx, dy, nbins=nbins_peak, nmin=nmin_peak,
                         kmax=kmax, remove_mean=remove_mean, window=window)
    y = sp["k"] * sp["E"]
    j = int(np.argmax(y))
    kpk = sp["k"][j]
    if 0 < j < len(y) - 1:                       # parabole sur (log k, log y)
        x = np.log(sp["k"][j - 1:j + 2])
        a, b, _ = np.polyfit(x, np.log(y[j - 1:j + 2]), 2)
        if a < 0:
            kpk = float(np.exp(-b / (2 * a)))

    return dict(k_mean=kmean, lambda_mean=2 * np.pi / kmean,
                k_ogive=kc, lambda_ogive=2 * np.pi / kc,
                k_peak=kpk, lambda_peak=2 * np.pi / kpk)
