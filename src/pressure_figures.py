"""
pressure_figures.py
Figures de diagnostic des termes de pression dans le bilan spectral d'énergie
cinétique (voir pressure_terms.py).

  fig_pressure_check : la pression perturbe-t-elle le bilan des transferts ? (3 panneaux)
  fig_hv_budgets     : bilans de E_h(k) et E_v(k) intégrés sur la couche
  fig_R_map          : carte (k, z) de la pression-déformation R, + marginales
  fig_R_kxky         : carte (kx, ky) de R intégrée en z (anisotropie des échanges h<->v)

Toutes les courbes en k sont tracées "par unité de ln k" (variance-preserving) :
l'aire sous la courbe en axe log vaut la somme sur les coquilles.
`adv` = sortie de compute_advective_transfer ; `r` = sortie de compute_pressure_terms
(avec les mêmes bornes de coquilles : edges=r["edges"]).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pressure_terms import zint

KLAB = r"$k_h$ (rad m$^{-1}$)"


def _vp(X, edges):
    """Somme par coquille -> densité par unité de ln k."""
    return X / np.diff(np.log(edges))


def _sym_norm(F, pct=99.0):
    vmax = np.nanpercentile(np.abs(F), pct)
    vmax = vmax if vmax > 0 else 1.0
    return Normalize(-vmax, vmax)


def _save(fig, savepath):
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


# ----------------------------------------------------------------------------
def fig_pressure_check(r, adv, z, rho=None, zmin=None, zmax=None, zi=None, savepath=None):
    """
    (a) transfert advectif et travail de pression intégrés sur la couche, vs k_h ;
    (b) rapport |∫rho (P_h+P_v) dz| / ∫rho (|T_h|+|T_v|) dz ; trait plein : jusqu'à zmax,
        tirets : jusqu'à zi (écart = flux de pression à l'inversion) ; pointillés : résidu
        de fermeture (plancher numérique) ; ligne à 5 % ;
    (c) profil <p'w'>(z) = somme_k F_p.
    """
    ed, ok = r["edges"], r["nmodes"] > 0
    k = r["k"][ok]
    I = lambda X, top=zmax: zint(X, z, rho, zmin, top)
    vp = lambda X: _vp(X, ed)[ok]
    Tadv = I(adv["T_h"] + adv["T_v"])
    Ph, Pv, Pp = I(r["P_h"]), I(r["P_v"]), I(r["P_h"] + r["P_v"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    a = ax[0]
    a.axhline(0, color="0.7", lw=0.8)
    a.semilogx(k, vp(Tadv), "k", lw=2, label=r"transfert advectif $\int\rho(T_h+T_v)\,dz$")
    a.semilogx(k, vp(Ph), "C0", label=r"$\int\rho P_h\,dz$")
    a.semilogx(k, vp(Pv), "C3", label=r"$\int\rho P_v\,dz$")
    a.semilogx(k, vp(Pp), "C2", lw=2, label=r"$\int\rho(P_h+P_v)\,dz$")
    a.set_xlabel(KLAB); a.set_ylabel(r"W m$^{-2}$ par ln $k$")
    a.set_title("(a) budget intégré sur la couche"); a.legend(fontsize=8)

    def ratio(X, top):
        den = I(np.abs(adv["T_h"]) + np.abs(adv["T_v"]), top)
        good = den > 1e-3 * den.max()          # rapport non significatif si transfert négligeable
        return np.where(good, np.abs(X) / np.where(good, den, 1.0), np.nan)[ok]

    b = ax[1]
    b.loglog(k, ratio(Pp, zmax), "C2", lw=2, label=r"pression, jusqu'à $z_{max}$")
    if zi is not None:
        b.loglog(k, ratio(I(r["P_h"] + r["P_v"], zi), zi), "C2", ls="--", label=r"pression, jusqu'à $z_i$")
    b.loglog(k, ratio(I(r["resid"]), zmax), "0.5", ls=":", label="résidu de fermeture")
    b.axhline(0.05, color="r", lw=0.8)
    b.set_xlabel(KLAB); b.set_ylabel("rapport au transfert advectif")
    b.set_title("(b) |pression nette| / transfert advectif"); b.legend(fontsize=8)

    c = ax[2]
    y = z / zi if zi else z
    c.plot(r["F_p"].sum(axis=1), y, "C2")
    c.axvline(0, color="0.7", lw=0.8)
    if zi:
        c.axhline(1, color="0.7", lw=0.8)
    c.set_xlabel(r"$\langle p'w'\rangle$ (Pa m s$^{-1}$)"); c.set_ylabel(r"$z/z_i$" if zi else "z (m)")
    c.set_title(r"(c) flux de pression $\sum_k F_p$")
    return _save(fig, savepath)


def fig_hv_budgets(r, adv, z, rho=None, zmin=None, zmax=None, extra=None, savepath=None):
    """
    Bilans de E_h(k) (gauche) et E_v(k) (droite) intégrés sur [zmin, zmax].
    extra : dict optionnel de tableaux (nz, nb), par unité de masse, mêmes coquilles :
            'B' flottabilité (cospectrum(w, b)), 'D_h', 'D_v' dissipation.
    La courbe grise en tirets est la somme des termes tracés (≈ tendance, ~0 si stationnaire).
    """
    extra = extra or {}
    ed, ok = r["edges"], r["nmodes"] > 0
    k = r["k"][ok]
    vp = lambda X: _vp(zint(X, z, rho, zmin, zmax), ed)[ok]
    panels = [
        (r"$E_h$", [(r"transfert advectif", adv["T_h"], "k"), (r"pression $P_h=-R$", r["P_h"], "C0"),
                    ("dissipation", extra.get("D_h"), "C1")]),
        (r"$E_v$", [(r"transfert advectif", adv["T_v"], "k"), (r"pression $P_v$", r["P_v"], "C3"),
                    (r"flottabilité $B$", extra.get("B"), "C2"), ("dissipation", extra.get("D_v"), "C1")]),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", sharex=True)
    for a, (title, terms) in zip(ax, panels):
        terms = [(lab, X, c) for lab, X, c in terms if X is not None]
        a.axhline(0, color="0.7", lw=0.8)
        for lab, X, c in terms:
            a.semilogx(k, vp(X), c, lw=2 if c == "k" else 1.4, label=lab)
        a.semilogx(k, sum(vp(X) for _, X, _ in terms), color="0.45", ls="--", label="somme des termes")
        if title == r"$E_v$" and extra.get("B") is not None:
            a.semilogx(k, vp(r["P_v"] + extra["B"]), "C3", ls=":", label=r"pression + flottabilité")
        a.set_xlabel(KLAB); a.set_title(f"Bilan de {title} (intégré sur la couche)")
        a.legend(fontsize=8)
    ax[0].set_ylabel(r"W m$^{-2}$ par ln $k$" if rho is not None
                     else r"$\int$ terme $dz$ ($\rho=1$, m$^3$ s$^{-3}$) par ln $k$")
    return _save(fig, savepath)


def fig_R_map(r, z, rho=None, zi=None, zmin=None, zmax=None, overlay=None, savepath=None, pct=99.0):
    """
    Carte (k_h, z) de la pression-déformation R (> 0 : E_h -> E_v).
    Marginales : ∫rho R dz en fonction de k (haut), rho ΣR_k en fonction de z (droite).
    overlay : tableau optionnel (nz, nb), par unité de masse (p. ex. ton transfert advectif
              croisé h<->v), tracé en contours noirs (pointillés si négatif).
    """
    ed, ok = r["edges"], r["nmodes"] > 0
    z = np.asarray(z, dtype=float)
    y = z / zi if zi else z
    ye = np.r_[y[0] - 0.5 * (y[1] - y[0]), 0.5 * (y[1:] + y[:-1]), y[-1] + 0.5 * (y[-1] - y[-2])]
    Rvp = _vp(r["R"], ed)
    rho_ = np.ones_like(z) if rho is None else np.asarray(rho, dtype=float)

    fig = plt.figure(figsize=(9, 7), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.15], height_ratios=[1, 4])
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)
    cax = fig.add_subplot(gs[1, 2])
    pcm = axm.pcolormesh(ed, ye, Rvp, cmap="RdBu_r", norm=_sym_norm(Rvp, pct), shading="flat")
    axm.set_xscale("log")
    if overlay is not None:
        ov = _vp(overlay, ed)[:, ok]
        vmax = np.nanpercentile(np.abs(ov), 99)
        if vmax > 0:
            axm.contour(r["k"][ok], y, ov, levels=vmax * np.array([-.8, -.4, -.2, .2, .4, .8]),
                        colors="k", linewidths=0.7)
    axm.set_xlabel(KLAB); axm.set_ylabel(r"$z/z_i$" if zi else "z (m)")
    fig.colorbar(pcm, cax=cax, label=r"$R$ (m$^2$ s$^{-3}$ par ln $k$)   [>0 : $E_h\to E_v$]")

    axt.axhline(0, color="0.7", lw=0.8)
    axt.semilogx(r["k"][ok], _vp(zint(r["R"], z, rho, zmin, zmax), ed)[ok], "k")
    axt.set_ylabel(r"$\int\rho R\,dz$"); plt.setp(axt.get_xticklabels(), visible=False)
    axr.axvline(0, color="0.7", lw=0.8)
    axr.plot(rho_ * r["R"].sum(axis=1), y, "k")
    axr.set_xlabel(r"$\rho\sum_k R$"); plt.setp(axr.get_yticklabels(), visible=False)
    axr.xaxis.set_major_locator(plt.MaxNLocator(3))
    return _save(fig, savepath)


def fig_R_kxky(res2d, kmax_plot=None, savepath=None, pct=99.5):
    """
    Carte (kx, ky) de ∫rho R dz (sortie de compute_pressure_terms_2d), axes en rad/km.
    Une structure orientée (rouleaux alignés avec le vent ou le cisaillement) apparaît
    comme un lobe le long d'un axe ; un échange isotrope comme un anneau.
    """
    kx, ky, R = res2d["kx"] * 1e3, res2d["ky"] * 1e3, res2d["R"]
    kmax_plot = kmax_plot if kmax_plot else min(np.abs(kx).max(), np.abs(ky).max())
    sx, sy = np.abs(kx) <= kmax_plot, np.abs(ky) <= kmax_plot
    Rw = R[np.ix_(sy, sx)]
    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    pcm = ax.pcolormesh(kx[sx], ky[sy], Rw, shading="nearest", cmap="RdBu_r", norm=_sym_norm(Rw, pct))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (rad km$^{-1}$)"); ax.set_ylabel(r"$k_y$ (rad km$^{-1}$)")
    fig.colorbar(pcm, ax=ax, label=r"$\int\rho R\,dz$ (W m$^{-2}$ par mode)   [>0 : $E_h\to E_v$]")
    return _save(fig, savepath)


# ----------------------------------------------------------------------------
def demo(outdir="."):
    """Démo sur champ synthétique (vérifie seulement que le tracé fonctionne)."""
    import os
    from pressure_terms import (synthetic_fields, compute_pressure_terms, compute_advective_transfer,
                                cospectrum, compute_pressure_terms_2d)
    u, v, w, p, b, z = synthetic_fields()
    dx = dy = 200.0
    r = compute_pressure_terms(u, v, w, p, z, dx, dy, scale="log", nbins=30, nmin=8)
    adv = compute_advective_transfer(u, v, w, z, dx, dy, edges=r["edges"])
    B = cospectrum(w, b, dx, dy, edges=r["edges"])
    zi = 0.6 * z[-1]
    r2 = compute_pressure_terms_2d(u, v, w, p, z, dx, dy)
    fig_pressure_check(r, adv, z, zi=zi, savepath=os.path.join(outdir, "fig1_pressure_check.png"))
    fig_hv_budgets(r, adv, z, extra={"B": B}, savepath=os.path.join(outdir, "fig2_hv_budgets.png"))
    fig_R_map(r, z, zi=zi, overlay=adv["T_v"], savepath=os.path.join(outdir, "fig3_R_map.png"))
    fig_R_kxky(r2, savepath=os.path.join(outdir, "fig4_R_kxky.png"))


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else ".")
