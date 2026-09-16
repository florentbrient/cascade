#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:55:07 2026

New version cascade and coarse graining

@author: fbrient
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import os
import sys
import tools as tl

# =============================================
# 1. PARAMÈTRES ET CHARGEMENT DES DONNÉES
# =============================================

# Test on local file (by default: False)
testlocal= True
# Run Filtered cascade (by default: True)
Filter3D = True
    
if testlocal:
    file = 'FIRZ4.1.V0001.OUT.003.nc' #'IHOP0.1.NWV01.OUT.013.nc' #'FIRZ4.1.V0001.OUT.003.nc'
    fileinfo  = '../infos/info_run_Dell_FIRZ4.txt' #'../infos/info_run_Dell_IHOPNW.txt' #../infos/info_run_Dell_FIRZ4.txt'
    Filter3D = True
else:        
    file = sys.argv[1] # name of the file
    # Open information from 'info_run_JZ.txt'
    pathinfo  = '../infos/'
    fileinfo  = pathinfo+'info_run.txt'

# Find path of file, create save path
path,pathsave = tl.findpath(fileinfo)
nc_file=path+file

# --- Paramètres ---
#nc_file = "wrfout_d01.nc"  # Remplacez par votre fichier NetCDF
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Paramètres physiques
g = 9.81  # m/s²
theta_0 = 300.0  # Température potentielle de référence (K)
dz = 10.0  # Résolution verticale (m)
dx = 300.0  # Résolution horizontale (m)

# Échelles pour le coarse-graining (en mètres)
vertical_scales = [50, 300, 600]  # Panaches (100–1000 m)
horizontal_scales = [300, 1200, 3000]  # Cellules (5 km, 25 km)

# --- Chargement des données ---
print("Chargement des données...")
ds = xr.open_dataset(nc_file)

# Extraire les champs 3D (z, y, x)
u = ds["UT"].values  # Vitesse horizontale u (m/s)
v = ds["VT"].values  # Vitesse horizontale v (m/s)
w = ds["WT"].values  # Vitesse verticale w (m/s)
theta = ds["THT"].values  # Température potentielle (K)
z = ds["level"].values  # Altitude (m)

u,v,w,theta,z = [tl.removebounds(np.squeeze(tmp)) for tmp in (u,v,w,theta,z)]

# Vérifier les dimensions
print(f"Dimensions : u {u.shape}, v {v.shape}, w {w.shape}, theta {theta.shape}")

# =============================================
# 2. CALCUL DES MOYENNES ET FLUCTUATIONS
# =============================================

print("Calcul des moyennes et fluctuations...")

# Moyennes horizontales (sur y et x)
u_mean = u.mean(axis=(1, 2))  # (z)
v_mean = v.mean(axis=(1, 2))
w_mean = w.mean(axis=(1, 2))
theta_mean = theta.mean(axis=(1, 2))

# Fluctuations (u' = u - u_mean)
u_prime = u - u_mean[:, np.newaxis, np.newaxis]
v_prime = v - v_mean[:, np.newaxis, np.newaxis]
w_prime = w - w_mean[:, np.newaxis, np.newaxis]
theta_prime = theta - theta_mean[:, np.newaxis, np.newaxis]

# =============================================
# 3. CALCUL DES TERMES CROISÉS (INTERACTIONS w-(u,v))
# =============================================

print("Calcul des termes croisés...")

# --- Gradients verticaux ---
dU_dz = np.gradient(u, dz, axis=0)  # ∂u/∂z (z, y, x)
dV_dz = np.gradient(v, dz, axis=0)  # ∂v/∂z
dW_dz = np.gradient(w, dz, axis=0)  # ∂w/∂z

# --- Termes croisés (moyenne horizontale) ---
# 1. Flux de moment vertical-horizontal
w_prime_u_prime = (w_prime * u_prime).mean(axis=(1, 2))  # (z)
w_prime_v_prime = (w_prime * v_prime).mean(axis=(1, 2))

# 2. Production d'énergie horizontale par w
w_prime_dU_dz = (w_prime * dU_dz).mean(axis=(1, 2))  # (z)
w_prime_dV_dz = (w_prime * dV_dz).mean(axis=(1, 2))

# 3. Redistribution non-linéaire
w_prime_u_prime_dU_dz = (w_prime * u_prime * dU_dz).mean(axis=(1, 2))  # (z)
w_prime_v_prime_dV_dz = (w_prime * v_prime * dV_dz).mean(axis=(1, 2))
u_prime_v_prime_dV_dx = (u_prime * v_prime * np.gradient(v, dx, axis=2)).mean(axis=(1, 2))  # ∂v/∂x

# 4. Production par flottabilité (pour comparaison)
w_prime_theta_prime = (w_prime * theta_prime).mean(axis=(1, 2)) * g / theta_0  # P_b

# 5. Production par cisaillement
dU_dz_mean = np.gradient(u_mean, dz)  # ∂U/∂z (moyenne)
dV_dz_mean = np.gradient(v_mean, dz)
P_s_u = - (u_prime * w_prime * dU_dz_mean[:, np.newaxis, np.newaxis]).mean(axis=(1, 2))  # (z)
P_s_v = - (v_prime * w_prime * dV_dz_mean[:, np.newaxis, np.newaxis]).mean(axis=(1, 2))

# 6. Transport vertical de TKE
e = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)  # TKE
T_p = -np.gradient((w_prime * e).mean(axis=(1, 2)), dz)  # -∂/∂z (w'e)

# =============================================
# 4. VISUALISATION DES PROFILS VERTICAUX
# =============================================
print("Visualisation des profils verticaux...")

plt.figure(figsize=(15, 10))
zmax = 700.
# --- Flux de moment ---
plt.subplot(2, 3, 1)
plt.plot(w_prime_u_prime, z, label="$\overline{w' u'}$", color="blue")
plt.plot(w_prime_v_prime, z, label="$\overline{w' v'}$", color="orange")
plt.xlabel("Flux de moment (m²/s²)")
plt.ylabel("Altitude z (m)")
plt.title("Flux de moment vertical-horizontal")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- Production d'énergie horizontale ---
plt.subplot(2, 3, 2)
plt.plot(w_prime_dU_dz, z, label="$\overline{w' \\partial_z u}$", color="green")
plt.plot(w_prime_dV_dz, z, label="$\overline{w' \\partial_z v}$", color="red")
plt.xlabel("Production d'énergie (m²/s³)")
plt.title("Production d'énergie horizontale par $w$")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- Redistribution non-linéaire ---
plt.subplot(2, 3, 3)
plt.plot(w_prime_u_prime_dU_dz, z, label="$\overline{w' u' \\partial_z u}$", color="purple")
plt.plot(w_prime_v_prime_dV_dz, z, label="$\overline{w' v' \\partial_z v}$", color="brown")
plt.xlabel("Redistribution non-linéaire (m³/s³)")
plt.title("Redistribution par $w-u$ et $w-v$")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- Production par flottabilité et cisaillement ---
plt.subplot(2, 3, 4)
plt.plot(w_prime_theta_prime, z, label="$P_b$ (Flottabilité)", color="cyan")
plt.plot(P_s_u + P_s_v, z, label="$P_s$ (Cisaillement)", color="magenta")
plt.xlabel("Production (m²/s³)")
plt.title("Production d'énergie par flottabilité et cisaillement")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- Transport vertical ---
plt.subplot(2, 3, 5)
plt.plot(T_p, z, label="$T_p$ (Transport vertical)", color="black")
plt.xlabel("Transport (m²/s³)")
plt.title("Transport vertical de TKE")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(f"{output_dir}/profiles_vertical_terms.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 5. COARSE-GRAINING ÉCHELLE-PAR-ÉCHELLE
# =============================================

print("Coarse-graining échelle-par-échelle...")

from scipy.ndimage import gaussian_filter1d

# Fonction corrigée pour appliquer un filtre gaussien 3D
def apply_3d_filter(field, scale_z, scale_xy, dz, dx):
    """
    Applique un filtre gaussien 3D avec des échelles différentes pour z et (x,y).
    """
    sigma_z = scale_z / (2 * np.sqrt(2 * np.log(2))) / dz
    sigma_xy = scale_xy / (2 * np.sqrt(2 * np.log(2))) / dx

    filtered = np.zeros_like(field)

    # Filtrer horizontalement (y, x)
    for i in range(field.shape[0]):
        filtered[i] = gaussian_filter1d(field[i], sigma=sigma_xy, axis=0,mode='wrap')  # Filtre sur y
        filtered[i] = gaussian_filter1d(filtered[i], sigma=sigma_xy, axis=1,mode='wrap')  # Filtre sur x

    # Filtrer verticalement (z)
    for j in range(filtered.shape[1]):
        for k in range(filtered.shape[2]):
            filtered[:, j, k] = gaussian_filter1d(filtered[:, j, k], sigma=sigma_z, axis=0,mode='wrap')

    return filtered

# Filtrer u, v, w à différentes échelles
u_filtered = {}
v_filtered = {}
w_filtered = {}

for scale_z in vertical_scales:
    for scale_xy in horizontal_scales:
        key = f"z{scale_z}_xy{scale_xy}"
        print(f"Filtrage à l'échelle Δz={scale_z} m, Δxy={scale_xy} m...")
        u_filtered[key] = apply_3d_filter(u, scale_z, scale_xy, dz, dx)
        v_filtered[key] = apply_3d_filter(v, scale_z, scale_xy, dz, dx)
        w_filtered[key] = apply_3d_filter(w, scale_z, scale_xy, dz, dx)

# Calculer les termes croisés pour chaque échelle
terms_by_scale = {
    "w_prime_u_prime": {},
    "w_prime_dU_dz": {},
    "w_prime_u_prime_dU_dz": {}
}

for scale_z in vertical_scales:
    for scale_xy in horizontal_scales:
        key = f"z{scale_z}_xy{scale_xy}"
        u_f = u_filtered[key]
        v_f = v_filtered[key]
        w_f = w_filtered[key]

        # Fluctuations à cette échelle
        u_f_mean = u_f.mean(axis=(1, 2))
        v_f_mean = v_f.mean(axis=(1, 2))
        w_f_mean = w_f.mean(axis=(1, 2))

        u_f_prime = u_f - u_f_mean[:, np.newaxis, np.newaxis]
        v_f_prime = v_f - v_f_mean[:, np.newaxis, np.newaxis]
        w_f_prime = w_f - w_f_mean[:, np.newaxis, np.newaxis]

        # Gradients verticaux
        dU_dz_f = np.gradient(u_f, dz, axis=0)

        # Termes croisés (moyenne horizontale)
        terms_by_scale["w_prime_u_prime"][key] = (w_f_prime * u_f_prime).mean(axis=(1, 2))
        terms_by_scale["w_prime_dU_dz"][key] = (w_f_prime * dU_dz_f).mean(axis=(1, 2))
        terms_by_scale["w_prime_u_prime_dU_dz"][key] = (w_f_prime * u_f_prime * dU_dz_f).mean(axis=(1, 2))

# Visualisation des termes pour différentes échelles
plt.figure(figsize=(15, 5))

# --- w' u' ---
plt.subplot(1, 3, 1)
for scale_z in vertical_scales:
    for scale_xy in horizontal_scales:
        key = f"z{scale_z}_xy{scale_xy}"
        plt.plot(
            terms_by_scale["w_prime_u_prime"][key],
            z,
            label=f"Δz={scale_z} m, Δxy={scale_xy//1000} km"
        )
plt.xlabel("$\overline{w' u'}$ (m²/s²)")
plt.ylabel("Altitude z (m)")
plt.title("Flux de moment par échelle")
plt.ylim(0, zmax)
plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- w' ∂z u ---
plt.subplot(1, 3, 2)
for scale_z in vertical_scales:
    for scale_xy in horizontal_scales:
        key = f"z{scale_z}_xy{scale_xy}"
        plt.plot(
            terms_by_scale["w_prime_dU_dz"][key],
            z,
            label=f"Δz={scale_z} m, Δxy={scale_xy//1000} km"
        )
plt.xlabel("$\overline{w' \\partial_z u}$ (m²/s³)")
plt.title("Production d'énergie par échelle")
plt.ylim(0, zmax)

plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

# --- w' u' ∂z u ---
plt.subplot(1, 3, 3)
for scale_z in vertical_scales:
    for scale_xy in horizontal_scales:
        key = f"z{scale_z}_xy{scale_xy}"
        plt.plot(
            terms_by_scale["w_prime_u_prime_dU_dz"][key],
            z,
            label=f"Δz={scale_z} m, Δxy={scale_xy//1000} km"
        )
plt.xlabel("$\overline{w' u' \\partial_z u}$ (m³/s³)")
plt.title("Redistribution non-linéaire par échelle")
plt.ylim(0, zmax)

plt.legend()
plt.grid(True)
#plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(f"{output_dir}/terms_by_scale.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 6. ANALYSE SPECTRALE 2D (MOYENNE VERTICALE)
# =============================================

print("Analyse spectrale 2D...")

# Fonction corrigée pour le spectre 1D (avec option L ou k)
def azimuthal_average(spectrum_2d, dx, output_scale="k"):
    """
    Calcule le spectre 1D par moyenne azimutale.

    Args:
        spectrum_2d: Spectre 2D (ny, nx).
        dx: Résolution spatiale (m).
        output_scale: "k" (nombre d'onde, m^{-1}) ou "L" (échelle, m).

    Returns:
        Si output_scale="k": (k, E(k)).
        Si output_scale="L": (L, E(L)).
    """
    ny, nx = spectrum_2d.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)  # kx en m^{-1}
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)  # ky en m^{-1}
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)     # k en m^{-1}

    # Masquer k=0 et les valeurs négatives (FFT shift)
    k_flat = k.flatten()
    spectrum_flat = spectrum_2d.flatten()
    mask = (k_flat > 0)
    print('test ',len(k_flat),mask.sum())
    k_flat = k_flat[mask]
    spectrum_flat = spectrum_flat[mask]

    # Bins logarithmique pour k
    k_bins = np.logspace(np.log10(np.min(k_flat)), np.log10(np.max(k_flat)), 50)
    spectrum_1d = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask_bin = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.sum(mask_bin) > 0:
            spectrum_1d[i] = np.mean(spectrum_flat[mask_bin])
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    if output_scale == "L":
        L_centers = 2 * np.pi / k_centers
        L_centers = L_centers[::-1]  # Inverser pour L croissant
        spectrum_1d = spectrum_1d[::-1]
        return L_centers, spectrum_1d
    else:
        return k_centers, spectrum_1d

# Sélectionner une couche (ex. : entre 500 m et 1500 m)
z_min, z_max = 0, 61  # Indices (à ajuster selon votre domaine)
u_layer = u[z_min:z_max].mean(axis=0)  # Moyenne verticale sur la couche (y, x)
v_layer = v[z_min:z_max].mean(axis=0)
w_layer = w[z_min:z_max].mean(axis=0)

# Fluctuations dans la couche
u_prime_layer = u_layer - u_layer.mean()
v_prime_layer = v_layer - v_layer.mean()
w_prime_layer = w_layer - w_layer.mean()

# Calculer les termes d'interaction dans la couche
w_prime_u_prime_layer = w_prime_layer * u_prime_layer
dU_dz_layer = np.gradient(u, dz, axis=0)[z_min:z_max].mean(axis=0)
w_prime_dU_dz_layer = w_prime_layer * dU_dz_layer

# FFT 2D des champs
def compute_fft2(field):
#    return fftshift(fft2(field))
    return fft2(field)

u_hat = compute_fft2(u_layer)
v_hat = compute_fft2(v_layer)
w_hat = compute_fft2(w_layer)
w_prime_u_prime_hat = compute_fft2(w_prime_u_prime_layer)
w_prime_dU_dz_hat = compute_fft2(w_prime_dU_dz_layer)

# Spectre d'énergie 2D
E_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
E_wu_2d = np.abs(w_prime_u_prime_hat)**2
E_wdU_2d = np.abs(w_prime_dU_dz_hat)**2

# Calcul des spectres 1D (en L et en k)
L, E_L = azimuthal_average(E_2d, dx=dx, output_scale="L")
L, E_wu_L = azimuthal_average(E_wu_2d, dx=dx, output_scale="L")
L, E_wdU_L = azimuthal_average(E_wdU_2d, dx=dx, output_scale="L")

k, E_k = azimuthal_average(E_2d, dx=dx, output_scale="k")
k, E_wu_k = azimuthal_average(E_wu_2d, dx=dx, output_scale="k")
k, E_wdU_k = azimuthal_average(E_wdU_2d, dx=dx, output_scale="k")

# Tracer les spectres en fonction de L (m)
plt.figure(figsize=(10, 6))
plt.loglog(L, E_L, label="Spectre d'énergie $E(L)$", color="blue")
plt.loglog(L, E_wu_L, label="Spectre de $\overline{w' u'}(L)$", color="orange")
plt.loglog(L, E_wdU_L, label="Spectre de $\overline{w' \partial_z u}(L)$", color="green")

# Ajouter les pentes théoriques en L
L_ref = np.logspace(np.log10(L[0]), np.log10(L[-1]), 100)
plt.loglog(
    L_ref,
    10 * (L_ref / L_ref[0])**(5/3),
    label="Cascade directe ($L^{5/3}$)",
    linestyle="--",
    color="gray"
)
plt.loglog(
    L_ref,
    10 * (L_ref / L_ref[0])**3,
    label="Cascade inverse ($L^{3}$)",
    linestyle="--",
    color="black"
)

# Ajouter une ligne verticale à L = 25 km
plt.axvline(x=25000, color="red", linestyle=":", label="L = 25 km")
plt.xlabel("Échelle $L$ (m)")
plt.ylabel("Spectre")
plt.title("Spectres en fonction de $L$ (m)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f"{output_dir}/spectra_vs_L.png", dpi=300, bbox_inches="tight")
plt.close()

# Tracer les spectres en fonction de k (m^{-1})
plt.figure(figsize=(10, 6))
plt.loglog(k, E_k, label="Spectre d'énergie $E(k)$", color="blue")
plt.loglog(k, E_wu_k, label="Spectre de $\overline{w' u'}(k)$", color="orange")
plt.loglog(k, E_wdU_k, label="Spectre de $\overline{w' \partial_z u}(k)$", color="green")

# Ajouter les pentes théoriques en k
k_ref = np.logspace(np.log10(k[1]), np.log10(k[-1]), 100)
plt.loglog(
    k_ref,
    10 * (k_ref / k_ref[0])**(-5/3),
    label="Cascade directe ($k^{-5/3}$)",
    linestyle="--",
    color="gray"
)
plt.loglog(
    k_ref,
    10 * (k_ref / k_ref[0])**(-3),
    label="Cascade inverse ($k^{-3}$)",
    linestyle="--",
    color="black"
)

# Ajouter une ligne verticale à k = 2π / 25000 ≈ 0.00025 m^{-1}
plt.axvline(x=2 * np.pi / 25000, color="red", linestyle=":", label="$k = 2\\pi / 25$ km")
plt.xlabel("Nombre d'onde $k$ (m$^{-1}$)")
plt.ylabel("Spectre")
plt.title("Spectres en fonction de $k$ (m$^{-1}$)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f"{output_dir}/spectra_vs_k.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 7. VISUALISATION DES CHAMPS FILTRÉS
# =============================================

print("Visualisation des champs filtrés...")

# Sélectionner une échelle pour les panaches et les cellules
scale_z_panaches = vertical_scales[0]  # m
scale_xy_panaches = horizontal_scales[0]  # m
scale_xy_cellules = horizontal_scales[-1]  # m
key_panaches = f"z{scale_z_panaches}_xy{scale_xy_panaches}"  # Échelle fine
key_cellules = f"z{scale_z_panaches}_xy{scale_xy_cellules}"  # Échelle grande

# Extraire les champs filtrés
u_panaches = u_filtered[key_panaches]
w_panaches = w_filtered[key_panaches]
u_cellules = u_filtered[key_cellules]
w_cellules = w_filtered[key_cellules]

# Sélectionner une altitude (ex. : z = 1000 m)
z_slice = 30  # Indice
u_panaches_slice = u_panaches[z_slice]
w_panaches_slice = w_panaches[z_slice]
u_cellules_slice = u_cellules[z_slice]
w_cellules_slice = w_cellules[z_slice]

# Tracer les champs
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(u_panaches_slice, cmap="RdBu", origin="lower")
plt.colorbar(label="$u$ (m/s)")
plt.title(f"Champ $u$ (panaches, Δz={scale_z_panaches} m, Δxy={scale_z_panaches} m)")

plt.subplot(2, 2, 2)
plt.imshow(w_panaches_slice, cmap="RdBu", origin="lower")
plt.colorbar(label="$w$ (m/s)")
plt.title(f"Champ $w$ (panaches, Δz={scale_z_panaches} m, Δxy={scale_z_panaches} m)")

plt.subplot(2, 2, 3)
plt.imshow(u_cellules_slice, cmap="RdBu", origin="lower")
plt.colorbar(label="$u$ (m/s)")
plt.title(f"Champ $u$ (cellules, Δz={scale_z_panaches} m, Δxy={scale_xy_cellules//1000} km)")

plt.subplot(2, 2, 4)
plt.imshow(w_cellules_slice, cmap="RdBu", origin="lower")
plt.colorbar(label="$w$ (m/s)")
plt.title(f"Champ $w$ (cellules, Δz={scale_z_panaches} m, Δxy={scale_xy_cellules//1000} km)")

plt.tight_layout()
plt.savefig(f"{output_dir}/filtered_fields.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================
# 8. SAUVEGARDE DES RÉSULTATS
# =============================================

print("Sauvegarde des résultats...")

# Sauvegarder les profils verticaux
results = {
    "z": z,
    "w_prime_u_prime": w_prime_u_prime,
    "w_prime_v_prime": w_prime_v_prime,
    "w_prime_dU_dz": w_prime_dU_dz,
    "w_prime_dV_dz": w_prime_dV_dz,
    "w_prime_u_prime_dU_dz": w_prime_u_prime_dU_dz,
    "w_prime_v_prime_dV_dz": w_prime_v_prime_dV_dz,
    "P_b": w_prime_theta_prime,
    "P_s": P_s_u + P_s_v,
    "T_p": T_p,
    "L": L,
    "E_L": E_L,
    "E_wu_L": E_wu_L,
    "E_wdU_L": E_wdU_L,
    "k": k,
    "E_k": E_k,
    "E_wu_k": E_wu_k,
    "E_wdU_k": E_wdU_k,
}

np.savez(f"{output_dir}/results.npz", **results)

print("✅ Analyse terminée ! Les résultats sont sauvegardés dans le dossier 'results'.")
print(f"   - Profils verticaux : profiles_vertical_terms.png")
print(f"   - Termes par échelle : terms_by_scale.png")
print(f"   - Spectres en L : spectra_vs_L.png")
print(f"   - Spectres en k : spectra_vs_k.png")
print(f"   - Champs filtrés : filtered_fields.png")
print(f"   - Données numériques : results.npz")