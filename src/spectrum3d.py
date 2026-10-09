#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectrum2d.py -- spectre radial isotrope d'un champ 2D périodique (LWP, thl, ...)
ou 3D (Nz,Ny,Nx), plus échelles de longueur 2D calculées SANS binning.
(Le nom du fichier est conservé ; le 3D est géré via dz / zext.)

Idée : garder ce qu'il y a de bon dans les deux approches existantes.
  - comme compute_spectra (spectral.py) : estimateur ISOTROPE
        E(k) = <P>_{modes du bin} * mesure(k) / dV
    (moyenne par mode x aire de l'anneau 2*pi*k en 2D, surface de la coquille
    4*pi*k^2 en 3D) -> pas de bruit lié au nombre aléatoire de modes par bin ;
  - comme compute_spectral_transfer (spectratools.py) : bins log-espacés qui
    montent jusqu'au grand k, avec un minimum de nmin modes par bin
    (aucun bin vide ou à 1 seul mode aux petits k).

Conventions
  P        = |FFTn(f - <f>)|^2 / N^2   (N = nb total de points) -> sum(P) = var(f)
  Ek[i]    = sum(P) dans le bin i        (variance du bin, somme EXACTE)
  dk_eff   = n_i * dV / mesure(kbar_i)   (largeur de la coquille idéale à n_i modes)
  E[i]     = Ek[i] / dk_eff[i]           ->  sum(E*dk_eff) = sum(Ek)
  abscisse = kbar_i = moyenne des |k| des modes du bin (PAS la borne du bin)

Usage
  import spectrum2d as s2
  sp = s2.radial_spectrum(LWP, dx)                          # 2D
  plt.loglog(sp['k'], sp['E'])
  ls = s2.length_scales(LWP, dx)                            # lambda_mean/ogive/peak
  sp3 = s2.radial_spectrum(thl3d, dx, dy, dz, zext='even')  # 3D scalaire (DCT)
  spw = s2.radial_spectrum(w3d,   dx, dy, dz, zext='odd')   # 3D, w=0 aux parois (DST)

Points à retenir
  - bilan : utiliser Ek (sommes, tous les modes, kmax='corner') ;
    affichage : E (densité isotrope), coupée par défaut au plus petit Nyquist.
    var_kept/var dit quelle part de la variance reste dans la courbe.
  - 3D : tant que k < pi/Lz (DCT) la coquille ne contient que le plan kz ~ 0
    = spectre horizontal du champ moyenné en z ; pas un spectre 3D isotrope.
  - validé sur champs synthétiques (2D et 3D) ; spectre de VARIANCE d'un scalaire
    seulement. Pas de E_h/E_v, de transfert T ni de flux Pi ici.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. modes 2D : P(kx, ky) et |k|
# ----------------------------------------------------------------------
def power_modes(field, dx, dy=None, dz=None, remove_mean=True, window=None,
                zext=None):
    """
    Retourne kh (|k|, même shape que le champ), P, dA (= volume d'un mode dk^n), ndim.
    sum(P) = variance (Parseval).

    3D (Nz,Ny,Nx) : la moyenne horizontale de CHAQUE niveau est retirée
    (remove_mean=True) pour ne pas mettre le profil moyen dans les coquilles.
    zext : traitement de la non-périodicité verticale (3D uniquement)
        None   -> FFT périodique en z (légacy ; fuite spectrale si le champ
                  n'est pas le même en bas et en haut)
        'even' -> extension miroir paire = DCT-II (u, v, scalaires, flux nul)
        'odd'  -> extension miroir impaire = DST-II (w = 0 aux parois)
        Les kz sont alors kz = pi*m/(Nz*dz) (deux fois plus fins que 2*pi/(Nz*dz)).
    """
    f = np.asarray(field, dtype=float)
    dy = dx if dy is None else dy

    if f.ndim == 2:
        Ny, Nx = f.shape
        if remove_mean:
            f = f - f.mean()
        if window is not None:                  # seulement si domaine NON périodique
            w = np.outer(np.hanning(Ny), np.hanning(Nx))
            f = f * w / np.sqrt(np.mean(w**2))  # garde la variance
        F = np.fft.fft2(f)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        KY, KX = np.meshgrid(ky, kx, indexing="ij")
        kh = np.hypot(KX, KY)
        dA = (2 * np.pi / (Nx * dx)) * (2 * np.pi / (Ny * dy))
    elif f.ndim == 3:
        if dz is None:
            raise ValueError("champ 3D : dz est requis")
        if window is not None:
            raise ValueError("window n'est pas géré en 3D (utilise zext pour z)")
        if remove_mean:
            f = f - f.mean(axis=(1, 2), keepdims=True)   # moyenne horizontale par niveau
        if zext == "even":
            f = np.concatenate([f, f[::-1]], axis=0)
        elif zext == "odd":
            f = np.concatenate([f, -f[::-1]], axis=0)
        elif zext is not None:
            raise ValueError("zext doit valoir None, 'even' ou 'odd'")
        Nz, Ny, Nx = f.shape
        F = np.fft.fftn(f)
        kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
        kh = np.sqrt(KX**2 + KY**2 + KZ**2)
        dA = (2 * np.pi / (Nx * dx)) * (2 * np.pi / (Ny * dy)) * (2 * np.pi / (Nz * dz))
    else:
        raise ValueError("champ 2D ou 3D attendu")

    P = np.abs(F) ** 2 / f.size ** 2
    return kh, P, dA, f.ndim


def _shell_measure(k, ndim):
    """Aire (2D) / surface (3D) de la coquille de rayon k : mesure de d(variance)/dk."""
    return 2 * np.pi * k if ndim == 2 else 4 * np.pi * k ** 2


def _k_nyquist(dx, dy, dz, ndim):
    out = [np.pi / dx, np.pi / dy]
    if ndim == 3:
        out.append(np.pi / dz)
    return min(out)


# ----------------------------------------------------------------------
# 2. spectre radial isotrope, bins log avec >= nmin modes
# ----------------------------------------------------------------------
def radial_spectrum(field, dx, dy=None, dz=None, nbins=150, nmin=8, kmax="nyquist",
                    remove_mean=True, window=None, zext=None):
    """
    Parameters
    ----------
    field : (Ny,Nx) ou (Nz,Ny,Nx). En 3D : dz requis, moyenne horizontale retirée
            par niveau, mesure de coquille 4*pi*k^2, coupure au plus petit Nyquist
            des 3 axes, et zext='even'/'odd' pour éviter la fuite spectrale en z.
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
    kh, P, dA, ndim = power_modes(field, dx, dy, dz, remove_mean, window, zext)
    dy_ = dx if dy is None else dy

    # Nyquist = le plus petit des axes. En 3D, au-delà, la coquille n'est plus
    # remplie que par des modes à grand kz (ou kh) : anisotrope, pas un spectre isotrope.
    k_nyq = _k_nyquist(dx, dy_, dz, ndim)
    if kmax == "nyquist":
        kcut = k_nyq * (1 + 1e-9)
    elif kmax == "corner":
        kcut = kh.max() * (1 + 1e-9)
    else:
        kcut = float(kmax)

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
    dk_eff = cnt * dA / _shell_measure(kbar, ndim)
    E = Ek / dk_eff

    return dict(k=kbar, E=E, Ek=Ek, n=cnt.astype(int), dk_eff=dk_eff,
                k_lo=ks[i0], k_hi=ks[i1 - 1],
                var=float(P.sum()),
                var_kept=float(Ek.sum()), k_nyq=k_nyq)


# ----------------------------------------------------------------------
# 3. échelles de longueur SANS binning (mode par mode)
# ----------------------------------------------------------------------
def length_scales(field, dx, dy=None, kmax="nyquist", frac_small=2.0 / 3.0,
                  nbins_peak=40, nmin_peak=15, remove_mean=True, window=None):
    """
    lambda_mean : 2*pi / <k>,  <k> = sum(k P) / sum(P)   (Pino et al. 2006, ordre 1)
    lambda_ogive: longueur d'onde où une fraction `frac_small` de la variance
                  est à plus petite échelle (de Roode et al. 2004, 2/3 par défaut)
                  -> ogive cumulée mode par mode, monotone, aucun bruit de binning
    lambda_peak : max de k*E(k) (forme "variance-preserving"), raffiné par une
                  parabole en log k autour du max d'un spectre lissé
    """
    if np.ndim(field) != 2:
        raise ValueError("length_scales : champ 2D attendu (LWP, coupe horizontale)")
    kh, P, dA, _ = power_modes(field, dx, dy, None, remove_mean, window)
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
