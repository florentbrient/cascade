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

import numpy as np


def integrate_negative_cascade(k, Pi, kH, method="trapz", refine_crossings=True):
    """
    Intègre Pi(k) sur ln(k), de min(k) à kH, en ne gardant que Pi < 0.

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
    k = np.asarray(k, dtype=float)
    Pi = np.asarray(Pi, dtype=float)

    if k.shape != Pi.shape:
        raise ValueError("k et Pi doivent avoir la même forme")
    if np.any(k <= 0):
        raise ValueError("k doit être strictement positif (échelle log)")

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


if __name__ == "__main__":
    # --- Exemple synthétique ---
    k = np.logspace(-3, 1, 400)          # k de 1e-3 à 10
    # Pi(k) fictif : négatif aux grandes échelles (cascade inverse),
    # positif aux petites échelles
    Pi = -np.exp(-((np.log(k) + 2) ** 2) / 2) + 0.3 * np.exp(-((np.log(k) - 1) ** 2) / 0.5)

    kH = 2.0
    I, info = integrate_negative_cascade(k, Pi, kH, method="trapz")
    print(f"Intégrale de Pi (Pi<0 uniquement) jusqu'à kH={kH} : {I:.4f}")
