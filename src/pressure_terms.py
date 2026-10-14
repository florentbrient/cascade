"""
pressure_terms.py
Termes de pression et transferts advectifs du bilan spectral d'énergie cinétique
en (k_h, z), pour une LES à domaine horizontal périodique (ex. Meso-NH, PABST).

Convention : tout est sommé par coquille isotrope de k_h = sqrt(kx²+ky²).
Normalisation de Parseval : avec â = fft2(a)/(nx*ny), la somme sur tous les
modes de Re[â* b̂] vaut <a b>_xy. Pour une densité par unité de ln k, voir
pressure_figures._vp (division par dln k).

Tous les termes sont PAR UNITÉ DE MASSE (m² s⁻³) ; l'intégrale verticale
zint(X, z, rho, zmin, zmax) = ∫ rho X dz les convertit en W m⁻².

  P_h  : travail de la pression sur (u, v)   = Re[û* a_x + v̂* a_y],  a = -(1/rho) grad_h p'
  P_v  : travail de la pression sur w        = Re[ŵ* a_z],          a_z = -(1/rho) dp'/dz
  R    : pression-déformation (1/rho) Re[p̂ dŵ*/dz]   ; R > 0 : E_h -> E_v
  T_p  : transport vertical par la pression  = -(1/rho) d/dz F_p
  F_p  : flux vertical de pression Re[p̂ ŵ*]  (W m⁻², par unité de volume)
  T_h, T_v : transfert advectif -Re[û_h* FT(u.grad u_h)], -Re[ŵ* FT(u.grad w)]
Identités :  P_v = T_p + R  (à l'erreur de discrétisation près)
             P_h = -R       (si div(rho u) = 0 avec rho = cste)
  resid = P_h + P_v - T_p  : diagnostic de fermeture (≈ 0 si Boussinesq ;
          sinon terme de stratification en densité, ordre dlnrho/dz * p'w').
"""
import numpy as np


# ----------------------------------------------------------------------------
# Outils de base
# ----------------------------------------------------------------------------
def _deriv_wavenumber(n, d):
    """Nombre d'onde pour la dérivée spectrale d'ordre 1 (mode de Nyquist annulé)."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=d)
    if n % 2 == 0:
        k[n // 2] = 0.0
    return k


def make_edges(nx, ny, dx, dy, scale="log", nbins=40, nmin=1, kmax=None):
    """Bornes des coquilles en k_h. nmin = nb minimal de modes par coquille
    (les coquilles trop peu peuplées sont fusionnées avec la suivante)."""
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kmag = np.hypot(*np.meshgrid(kx, ky))
    dk = min(2 * np.pi / (nx * dx), 2 * np.pi / (ny * dy))
    if kmax is None:
        kmax = min(np.pi / dx, np.pi / dy)          # Nyquist isotrope
    if scale == "lin":
        edges = (np.arange(0, int(np.ceil(kmax / dk)) + 1) + 0.5) * dk
    else:
        edges = np.geomspace(0.5 * dk, kmax, nbins + 1)
    nmin = max(int(nmin), 1)
    counts, _ = np.histogram(kmag[kmag > 0], bins=edges)
    keep, acc = [edges[0]], 0
    for i, c in enumerate(counts):
        acc += c
        if acc >= nmin:
            keep.append(edges[i + 1])
            acc = 0
    if len(keep) < 2:
        keep = [edges[0], edges[-1]]
    elif keep[-1] != edges[-1]:
        keep[-1] = edges[-1]                         # reliquat fusionné à la dernière coquille
    return np.array(keep), kmag


class _Shells:
    """Sommation par coquilles isotropes d'un champ 2D de modes (ny, nx)."""

    def __init__(self, nx, ny, dx, dy, edges=None, scale="log", nbins=40, nmin=1, kmax=None):
        if edges is None:
            edges, kmag = make_edges(nx, ny, dx, dy, scale, nbins, nmin, kmax)
        else:
            kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
            kmag = np.hypot(*np.meshgrid(kx, ky))
        self.edges = np.asarray(edges, dtype=float)
        self.nb = len(self.edges) - 1
        idx = np.digitize(kmag.ravel(), self.edges) - 1
        self.ok = (idx >= 0) & (idx < self.nb)
        self.idx = idx[self.ok]
        self.nmodes = np.bincount(self.idx, minlength=self.nb)
        self.k = (np.bincount(self.idx, weights=kmag.ravel()[self.ok], minlength=self.nb)
                  / np.maximum(self.nmodes, 1))

    def sum(self, C):
        return np.bincount(self.idx, weights=C.ravel()[self.ok], minlength=self.nb)

    def meta(self):
        return dict(k=self.k, dk=np.diff(self.edges), edges=self.edges, nmodes=self.nmodes)


def _zweights(z, mask):
    """Poids trapézoïdaux de l'intégrale verticale, restreints aux niveaux de `mask`."""
    zz = np.asarray(z, dtype=float)[mask]
    wts = np.zeros_like(zz)
    if zz.size > 1:
        dz = np.diff(zz)
        wts[:-1] += 0.5 * dz
        wts[1:] += 0.5 * dz
    return wts


def zint(X, z, rho=None, zmin=None, zmax=None):
    """Intégrale verticale pondérée  ∫ rho X dz  sur [zmin, zmax] (axe 0 de X)."""
    z = np.asarray(z, dtype=float)
    rho = np.ones_like(z) if rho is None else np.asarray(rho, dtype=float)
    m = (z >= (z[0] if zmin is None else zmin)) & (z <= (z[-1] if zmax is None else zmax))
    return np.tensordot(_zweights(z, m) * rho[m], np.asarray(X)[m], axes=(0, 0))


# ----------------------------------------------------------------------------
# Termes de pression
# ----------------------------------------------------------------------------
def compute_pressure_terms(u, v, w, p, z, dx, dy, rho=None,
                           scale="log", nbins=40, nmin=1, edges=None, kmax=None):
    """
    u, v, w, p : tableaux (nz, ny, nx), TOUS sur la même grille (points de masse :
                 interpoler UT, VT, WT comme pour ton calcul de Pi(k)).
                 p = PABST (Pa) ; la moyenne horizontale est retirée niveau par niveau.
    z          : (nz,) altitudes (m), grille verticale éventuellement étirée.
    rho        : (nz,) masse volumique de référence (kg m-3) ; None -> 1 (pas de poids).
    edges      : bornes de coquilles déjà calculées (pour réutiliser ton binning).
    Retourne un dict de tableaux (nz, nbins) : P_h, P_v, R, F_p, T_p, resid,
    + 'k' (k moyen des modes de la coquille), 'dk', 'edges', 'nmodes'.
    """
    nz, ny, nx = w.shape
    z = np.asarray(z, dtype=float)
    rho = np.ones(nz) if rho is None else np.asarray(rho, dtype=float)
    p = p - p.mean(axis=(1, 2), keepdims=True)
    sh = _Shells(nx, ny, dx, dy, edges, scale, nbins, nmin, kmax)

    kxd = _deriv_wavenumber(nx, dx)[None, :]
    kyd = _deriv_wavenumber(ny, dy)[:, None]
    dpdz = np.gradient(p, z, axis=0, edge_order=2)
    dwdz = np.gradient(w, z, axis=0, edge_order=2)

    out = {name: np.zeros((nz, sh.nb)) for name in ("P_h", "P_v", "R", "F_p")}
    N = nx * ny
    f2 = lambda a: np.fft.fft2(a) / N
    for k in range(nz):
        uh, vh, wh, ph = f2(u[k]), f2(v[k]), f2(w[k]), f2(p[k])
        dph, dwh = f2(dpdz[k]), f2(dwdz[k])
        ax = -1j * kxd * ph / rho[k]
        ay = -1j * kyd * ph / rho[k]
        out["P_h"][k] = sh.sum(np.real(np.conj(uh) * ax + np.conj(vh) * ay))
        out["P_v"][k] = sh.sum(np.real(np.conj(wh) * (-dph / rho[k])))
        out["R"][k] = sh.sum(np.real(ph * np.conj(dwh))) / rho[k]
        out["F_p"][k] = sh.sum(np.real(ph * np.conj(wh)))

    out["T_p"] = -np.gradient(out["F_p"], z, axis=0, edge_order=2) / rho[:, None]
    out["resid"] = out["P_h"] + out["P_v"] - out["T_p"]
    out.update(sh.meta())
    return out


def compute_pressure_terms_2d(u, v, w, p, z, dx, dy, rho=None, zmin=None, zmax=None):
    """
    Cartes en (kx, ky) de R, P_h, P_v intégrées sur [zmin, zmax] avec le poids rho dz
    (W m-2 par mode). Axes fftshiftés : kx, ky croissants, mode (0,0) au centre.
    """
    nz, ny, nx = w.shape
    z = np.asarray(z, dtype=float)
    rho = np.ones(nz) if rho is None else np.asarray(rho, dtype=float)
    p = p - p.mean(axis=(1, 2), keepdims=True)
    m = (z >= (z[0] if zmin is None else zmin)) & (z <= (z[-1] if zmax is None else zmax))
    levels = np.where(m)[0]
    wts = _zweights(z, m)
    kxd = _deriv_wavenumber(nx, dx)[None, :]
    kyd = _deriv_wavenumber(ny, dy)[:, None]
    dpdz = np.gradient(p, z, axis=0, edge_order=2)
    dwdz = np.gradient(w, z, axis=0, edge_order=2)
    N = nx * ny
    f2 = lambda a: np.fft.fft2(a) / N
    R2 = np.zeros((ny, nx)); Ph2 = np.zeros((ny, nx)); Pv2 = np.zeros((ny, nx))
    for wz, k in zip(wts, levels):
        uh, vh, wh, ph = f2(u[k]), f2(v[k]), f2(w[k]), f2(p[k])
        dph, dwh = f2(dpdz[k]), f2(dwdz[k])
        R2 += wz * np.real(ph * np.conj(dwh))                       # rho * (1/rho)
        Ph2 += wz * np.real(np.conj(uh) * (-1j * kxd * ph) + np.conj(vh) * (-1j * kyd * ph))
        Pv2 += wz * np.real(np.conj(wh) * (-dph))
    sf = lambda a: np.fft.fftshift(a)
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=dy))
    return dict(kx=kx, ky=ky, R=sf(R2), P_h=sf(Ph2), P_v=sf(Pv2))


# ----------------------------------------------------------------------------
# Cospectres génériques et transfert advectif
# ----------------------------------------------------------------------------
def cospectrum(a, b, dx, dy, edges=None, **kw):
    """Re[â* b̂] sommé par coquille et par niveau -> (nz, nb).
    Ex. flottabilité B(k,z) = cospectrum(w, b, ...) avec b = g (thv - <thv>_xy)/thv_ref."""
    nz, ny, nx = a.shape
    sh = _Shells(nx, ny, dx, dy, edges, **kw)
    N = nx * ny
    out = np.zeros((nz, sh.nb))
    for k in range(nz):
        out[k] = sh.sum(np.real(np.conj(np.fft.fft2(a[k]) / N) * (np.fft.fft2(b[k]) / N)))
    return out


def compute_advective_transfer(u, v, w, z, dx, dy, edges=None, remove_mean=True, **kw):
    """
    Transfert advectif d'énergie cinétique par coquille de k_h et par niveau :
        T_h = -Re[û_h* . FT((u.grad) u_h)],   T_v = -Re[ŵ* FT((u.grad) w)]   (m² s⁻³)
    Dérivées horizontales spectrales, verticale en différences finies (grille étirée).
    remove_mean=True : retire la moyenne horizontale de u, v (pas de production par le
    cisaillement moyen -> pur transfert fluctuation-fluctuation).
    Pas de dé-aliasing : à n'utiliser que si les champs sont bien résolus.
    """
    nz, ny, nx = w.shape
    z = np.asarray(z, dtype=float)
    if remove_mean:
        u = u - u.mean(axis=(1, 2), keepdims=True)
        v = v - v.mean(axis=(1, 2), keepdims=True)
        w = w - w.mean(axis=(1, 2), keepdims=True)
    sh = _Shells(nx, ny, dx, dy, edges, **kw)
    kxd = _deriv_wavenumber(nx, dx)[None, :]
    kyd = _deriv_wavenumber(ny, dy)[:, None]
    dudz = np.gradient(u, z, axis=0, edge_order=2)
    dvdz = np.gradient(v, z, axis=0, edge_order=2)
    dwdz = np.gradient(w, z, axis=0, edge_order=2)
    N = nx * ny
    T_h = np.zeros((nz, sh.nb)); T_v = np.zeros((nz, sh.nb))
    ifx = lambda A: np.fft.ifft2(1j * kxd * A).real
    ify = lambda A: np.fft.ifft2(1j * kyd * A).real
    for k in range(nz):
        U, V, W = np.fft.fft2(u[k]), np.fft.fft2(v[k]), np.fft.fft2(w[k])
        au = u[k] * ifx(U) + v[k] * ify(U) + w[k] * dudz[k]
        av = u[k] * ifx(V) + v[k] * ify(V) + w[k] * dvdz[k]
        aw = u[k] * ifx(W) + v[k] * ify(W) + w[k] * dwdz[k]
        Au, Av, Aw = (np.fft.fft2(a) / N for a in (au, av, aw))
        uh, vh, wh = U / N, V / N, W / N
        T_h[k] = sh.sum(-np.real(np.conj(uh) * Au + np.conj(vh) * Av))
        T_v[k] = sh.sum(-np.real(np.conj(wh) * Aw))
    out = dict(T_h=T_h, T_v=T_v)
    out.update(sh.meta())
    return out


# ----------------------------------------------------------------------------
# Champ synthétique incompressible (tests et démo des figures)
# ----------------------------------------------------------------------------
def synthetic_fields(nz=48, ny=64, nx=64, dx=200.0, dy=200.0, H=1000.0, kz_width=0.6, seed=0):
    """Champ incompressible : continuité exacte avec les mêmes opérateurs que le code,
    w = 0 aux bords, grille verticale étirée. Retourne u, v, w, p, b, z."""
    rng = np.random.default_rng(seed)
    z = H * np.sinh(np.linspace(0, 2, nz)) / np.sinh(2)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[None, None, :]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :, None]
    kz = 2 * np.pi * np.fft.fftfreq(nz)[:, None, None]
    kh, kc = np.hypot(kx, ky), 0.5 * np.pi / min(dx, dy)

    def noise():
        A = np.fft.fftn(rng.standard_normal((nz, ny, nx)))
        A *= np.exp(-(kh / (0.3 * kc)) ** 2 - (kz / kz_width) ** 2) * (kh < kc)
        return np.fft.ifftn(A).real

    w = noise() * np.sin(np.pi * z / z[-1])[:, None, None]
    w -= w.mean(axis=(1, 2), keepdims=True)
    p = noise() * 50.0
    psi = noise()
    kxd, kyd = _deriv_wavenumber(nx, dx)[None, :], _deriv_wavenumber(ny, dy)[:, None]
    k2 = kxd ** 2 + kyd ** 2
    k2[k2 == 0] = np.inf
    D = -np.gradient(np.fft.fft2(w), z, axis=0, edge_order=2)     # ik.u_h = -dw/dz
    Psi = np.fft.fft2(psi)
    u = np.fft.ifft2(-1j * kxd * D / k2 - 1j * kyd * Psi * 1e3).real
    v = np.fft.ifft2(-1j * kyd * D / k2 + 1j * kxd * Psi * 1e3).real
    b = 0.05 * w + 0.01 * noise()                                 # flottabilité corrélée à w
    return u, v, w, p, b, z


def _selftest():
    u, v, w, p, b, z = synthetic_fields()
    nz, ny, nx = w.shape
    dx = dy = 200.0
    kmax = 1.01 * np.hypot(np.pi / dx, np.pi / dy)                 # couvre tous les modes
    res = compute_pressure_terms(u, v, w, p, z, dx, dy, kmax=kmax, scale="lin")
    pw = ((p - p.mean(axis=(1, 2), keepdims=True)) * w).mean(axis=(1, 2))
    print("Parseval          max|sum_k F_p - <p'w'>| / max|<p'w'>|   = %.2e"
          % (np.abs(res["F_p"].sum(1) - pw).max() / np.abs(pw).max()))
    print("P_h = -R          max|P_h + R| / max|R|                   = %.2e"
          % (np.abs(res["P_h"] + res["R"]).max() / np.abs(res["R"]).max()))
    s = slice(2, -2)
    print("fermeture         rms(P_h+P_v-T_p) / rms(P_v) (hors bords) = %.2e"
          % np.sqrt((res["resid"][s] ** 2).mean() / (res["P_v"][s] ** 2).mean()))
    r2 = compute_pressure_terms_2d(u, v, w, p, z, dx, dy)
    Rint = zint(res["R"], z).sum()
    print("carte (kx,ky)     |sum R2 - zint(sum_k R)| / |zint|        = %.2e"
          % (abs(r2["R"].sum() - Rint) / abs(Rint)))
    adv = compute_advective_transfer(u, v, w, z, dx, dy, edges=res["edges"])
    Tint = zint(adv["T_h"] + adv["T_v"], z).sum()
    Tabs = zint(np.abs(adv["T_h"]) + np.abs(adv["T_v"]), z).sum()
    print("conservation adv. |int sum_k (T_h+T_v)| / int sum_k |T|    = %.2e" % (abs(Tint) / Tabs))
    Bk = cospectrum(w, b, dx, dy, edges=res["edges"])
    print("cospectre B       |sum_k B - <w b>| / |<w b>| (niveau max)  = %.2e"
          % (np.abs(Bk.sum(1) - (w * b).mean(axis=(1, 2))).max() / np.abs((w * b).mean(axis=(1, 2))).max()))


if __name__ == "__main__":
    _selftest()
