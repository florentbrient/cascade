#!/usr/bin/env python3
"""
Calcul du nombre de Rayleigh de convection libre (Ra_f), du nombre de
Nusselt (Nu) et du nombre de Prandtl (Pr) pour un cas StCu Meso-NH
forcé par refroidissement radiatif au sommet + flux de surface interactif.

Toutes les grandeurs sont lues dans le groupe LES_budgets du fichier
de sortie Meso-NH (.000.nc ou tout fichier de segment portant ce groupe).

Usage:
    python compute_Ra_Nu_Pr.py chemin/vers/FICHIER.nc [--dz-window 50]

Formules :
    Ra_f = g * beta * dT * L^3 / (nu * alpha)
    Nu   = Q0 * L / (K * dT)
    Pr   = nu / alpha
"""

import argparse
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# Constantes physiques (Meso-NH / air sec)
# ---------------------------------------------------------------------
G = 9.81                  # m s-2
RD = 287.05                # J kg-1 K-1 (constante des gaz parfaits, air sec)
CPD = 1004.71               # J kg-1 K-1 (chaleur spécifique air sec, pression cst)
P00 = 1.0e5                 # Pa (pression de référence pour Exner)

# Viscosité dynamique (loi de Sutherland), valable ~200-400 K
MU0 = 1.716e-5               # Pa.s
T0_SUTH = 273.15             # K
S_SUTH = 110.4                # K

# Nombre de Prandtl de l'air : quasi-constant (0.70-0.72) sur la gamme
# de températures d'une CLA marine -> on le traite comme une constante,
# ce qui donne alpha = nu / PR_AIR de façon auto-cohérente.
PR_AIR = 0.71

GROUP_MEAN = "LES_budgets/Mean/Cartesian/Not_time_averaged/Not_normalized/cart"
GROUP_RAD = "LES_budgets/Radiation/Cartesian/Not_time_averaged/Not_normalized/cart"
GROUP_MISC = "LES_budgets/Miscellaneous/Cartesian/Not_time_averaged/Not_normalized/cart"
GROUP_SURF = "LES_budgets/Surface/Cartesian/Not_time_averaged/Not_normalized/cart"


def dynamic_viscosity(T):
    """Viscosité dynamique de l'air (Pa.s) via la loi de Sutherland."""
    return MU0 * (T / T0_SUTH) ** 1.5 * (T0_SUTH + S_SUTH) / (T + S_SUTH)


def temperature_from_theta(theta, pressure):
    """Température réelle (K) à partir de theta (K) et p (Pa) via Exner."""
    return theta * (pressure / P00) ** (RD / CPD)


def interp_profile_at_height(level, profile, z_target):
    """Interpolation linéaire d'un profil vertical à une hauteur donnée."""
    return np.interp(z_target, level, profile)


def mean_between(level, profile, z_lo, z_hi):
    """Moyenne d'un profil entre deux hauteurs (intégration trapézoïdale
    normalisée), utilisé pour lisser le saut à l'inversion sur une
    fenêtre plutôt qu'une valeur ponctuelle."""
    mask = (level >= z_lo) & (level <= z_hi)
    if mask.sum() < 2:
        # fenêtre trop étroite pour les niveaux disponibles : interpolation ponctuelle
        return interp_profile_at_height(level, profile, 0.5 * (z_lo + z_hi))
    z_sel = level[mask]
    p_sel = profile[mask]
    return np.trapz(p_sel, z_sel) / (z_sel[-1] - z_sel[0])


def compute_diagnostics(nc_path, dz_window=50.0):
    
    ds_mean = xr.open_dataset(nc_path, group=GROUP_MEAN)
    ds_rad = xr.open_dataset(nc_path, group=GROUP_RAD)
    ds_misc = xr.open_dataset(nc_path, group=GROUP_MISC)
    ds_surf = xr.open_dataset(nc_path, group=GROUP_SURF)

#    level = ds_mean["level_les"].values          # (level_les,)
    level = xr.open_dataset(nc_path)['level_les']
    time  = ds_mean["time_les"].values             # (time_les,)
    n_t = len(time)

    zi_t = ds_misc["BL_H"].values                  # (time_les,) hauteur de CLA
    thv_t = ds_mean["MEAN_THV"].values              # (time_les, level_les)
    th_t = ds_mean["MEAN_TH"].values
    pre_t = ds_mean["MEAN_PRE"].values
    rho_t = ds_mean["MEAN_RHO"].values

    dthradlw_t = ds_rad["DTHRADLW"].values           # K s-1
    dthradsw_t = ds_rad["DTHRADSW"].values

    q0_surf_kin_t = ds_surf["Q0"].values              # (time_les,) m K s-1 (flux cinématique)

    results = {
        "time": time,
        "zi": np.full(n_t, np.nan),
        "dT": np.full(n_t, np.nan),
        "T_ref": np.full(n_t, np.nan),
        "Q0_surf": np.full(n_t, np.nan),
        "Q0_rad": np.full(n_t, np.nan),
        "Q0_total": np.full(n_t, np.nan),
        "nu": np.full(n_t, np.nan),
        "alpha": np.full(n_t, np.nan),
        "K": np.full(n_t, np.nan),
        "Pr": np.full(n_t, np.nan),
        "Ra_f": np.full(n_t, np.nan),
        "Nu": np.full(n_t, np.nan),
    }
    for it in range(n_t):
        zi = float(zi_t[it])

        if not np.isfinite(zi) or zi <= 0:
            continue

        theta_v = thv_t[it, :]
        theta = th_t[it, :]
        pres = pre_t[it, :]
        rho = rho_t[it, :]

        # --- Delta T : saut de theta_v à travers l'inversion, moyenné
        # sur une fenêtre +/- dz_window autour de zi pour lisser le bruit
        thv_below = mean_between(level, theta_v, zi - dz_window, zi)
        thv_above = mean_between(level, theta_v, zi, zi + dz_window)
        dT = thv_above - thv_below
        if dT <= 0:
            # saut non physique (bruit d'échantillonnage) -> on garde |dT|
            dT = abs(dT)
        if dT == 0:
            continue

        # --- Temperature et densite de reference (moyenne sur la CLA)
        T_profile = temperature_from_theta(theta, pres)
        T_ref = mean_between(level, T_profile, 0.0, zi)
        rho_ref = mean_between(level, rho, 0.0, zi)

        # --- proprietes moleculaires de l'air a T_ref
        mu = dynamic_viscosity(T_ref)
        nu = mu / rho_ref
        alpha = nu / PR_AIR
        K = alpha * rho_ref * CPD  # conductivite thermique, W/m/K

        # --- Q0 radiatif : integrale verticale de rho*cp*(dTHRADLW+dTHRADSW)
        # sur [0, zi], signe negatif car refroidissement -> on prend
        # l'oppose pour obtenir un flux de forcage positif (W/m2)
        cooling_rate = dthradlw_t[it, :] + dthradsw_t[it, :]  # K/s
        mask_bl = level <= zi
        if mask_bl.sum() >= 2:
            integrand = rho[mask_bl] * CPD * cooling_rate[mask_bl]
            Q0_rad = -np.trapz(integrand, level[mask_bl])
        else:
            Q0_rad = 0.0

        # --- Q0 de surface : flux cinematique (m K/s) -> W/m2
        rho_surf = rho[0]
        Q0_surf = q0_surf_kin_t[it] * rho_surf * CPD

        print(Q0_surf,Q0_rad)
        Q0_total = Q0_surf + Q0_rad

        beta = 1.0 / T_ref
        #print(G ,beta, dT, zi ,nu ,alpha)
        Ra_f = G * beta * dT * zi ** 3 / (nu * alpha)
        Nu_num = Q0_total * zi / (K * dT)

        results["zi"][it] = zi
        results["dT"][it] = dT
        results["T_ref"][it] = T_ref
        results["Q0_surf"][it] = Q0_surf
        results["Q0_rad"][it] = Q0_rad
        results["Q0_total"][it] = Q0_total
        results["nu"][it] = nu
        results["alpha"][it] = alpha
        results["K"][it] = K
        results["Pr"][it] = PR_AIR
        results["Ra_f"][it] = Ra_f
        results["Nu"][it] = Nu_num

    ds_mean.close()
    ds_rad.close()
    ds_misc.close()
    ds_surf.close()

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc_path", help="Chemin vers le fichier Meso-NH .000.nc")
    parser.add_argument(
        "--dz-window",
        type=float,
        default=50.0,
        help="Demi-largeur (m) de la fenetre utilisee pour lisser le saut "
        "de theta_v a l'inversion (defaut: 50 m)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Si fourni, ecrit les series temporelles dans ce fichier CSV",
    )
    args = parser.parse_args()

    res = compute_diagnostics(args.nc_path, dz_window=args.dz_window)

    print(f"{'t(s)':>10} {'zi(m)':>8} {'dT(K)':>8} {'Q0_tot(W/m2)':>13} "
          f"{'Ra_f':>12} {'Nu':>10} {'Pr':>6}")
    for it in range(len(res["time"])):
        if not np.isfinite(res["Ra_f"][it]):
            continue
        print(
            f"{res['time'][it]:10.0f} {res['zi'][it]:8.1f} {res['dT'][it]:8.3f} "
            f"{res['Q0_total'][it]:13.2f} {res['Ra_f'][it]:12.4e} "
            f"{res['Nu'][it]:10.2f} {res['Pr'][it]:6.3f}"
        )

    if args.csv_out:
        import csv

        keys = list(res.keys())
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for it in range(len(res["time"])):
                writer.writerow([res[k][it] for k in keys])
        print(f"\nSeries temporelles ecrites dans {args.csv_out}")


if __name__ == "__main__":
    main()
