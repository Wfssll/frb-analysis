"""Production CHIME detection-probability quadrature.

The archived sightline draw and interpolation table separate model generation
from expensive Galactic electron-density calculations.
"""

from cosmology import production_cosmology
from datetime import datetime
import time
import numpy as np
import pyccl as ccl
from astropy.constants import G, c, m_p, pc
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import map_coordinates
Z_MAX=5.0
F_IGM=0.93
X_E=0.875
DELTA_NU=400e6
MPC_TO_M=3.08567758e22
JYMS_FROM_ERG_MPC2_HZ=1e22 / MPC_TO_M**2
COSMO=production_cosmology(Omega_b=0.04897,Omega_c=0.3111-0.04897,h=0.6766,sigma8=0.81,n_s=0.96)
def log(msg: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def erg_per_mpc2_hz_to_jy_ms(value):
    return np.asarray(value, dtype=np.float64) * JYMS_FROM_ERG_MPC2_HZ


def dN_dE(E, alpha, e_cut):
    E = np.asarray(E, dtype=np.float64)
    return (E / e_cut) ** (-alpha) * np.exp(-E / e_cut)


def energy_weight(E, alpha, e_cut):
    dE = np.gradient(E)
    w = np.maximum(dN_dE(E, alpha, e_cut) * dE, 0.0)
    return w, float(w.sum())


def F_v(E, z, D_L, delta_nu=DELTA_NU):
    E = np.asarray(E, dtype=np.float64)
    D_L = np.maximum(np.asarray(D_L, dtype=np.float64), 1e-8)
    return E * (1.0 + z) / (4.0 * np.pi * D_L ** 2 * delta_nu)


def dm_igm_integrand(z):
    z = np.asarray(z, dtype=np.float64)
    pref = (
        3.0 * c.value * COSMO["Omega_b"] * COSMO["h"] * 100.0
        / (8.0 * np.pi * G.value * m_p.value)
        * F_IGM * X_E
        * (1e3 / (pc.value * 1e6))
        * (1e-6 / pc.value)
    )
    ez = ccl.h_over_h0(COSMO, 1.0 / (1.0 + z))
    return pref * (1.0 + z) / ez


def build_dm_igm(z):
    z_tab = np.linspace(0.0, Z_MAX, 8000)
    dm_tab = cumulative_trapezoid(dm_igm_integrand(z_tab), z_tab, initial=0.0)
    log("DM_IGM(%g)=%.2f pc/cc" % (Z_MAX, dm_tab[-1]))
    return np.interp(z, z_tab, dm_tab)


def fill_psel_table(p_raw, f_grid):
    p = np.array(p_raw, dtype=np.float64, copy=True)
    log_f = np.log10(f_grid)
    n_f, n_dm = p.shape
    for j in range(n_dm):
        orig = p_raw[:, j]
        ok = np.isfinite(orig)
        if not ok.any():
            p[:, j] = 0.0
            continue
        i0 = int(np.argmax(ok))
        i1 = n_f - 1 - int(np.argmax(ok[::-1]))
        col = p[:, j]
        if i0 > 0:
            col[:i0] = 0.0
        if i1 < n_f - 1:
            col[i1 + 1:] = orig[i1]
        hole = ~np.isfinite(col)
        if hole.any():
            col[hole] = np.interp(log_f[hole], log_f[ok], orig[ok])
        np.clip(col, 0.0, 1.0, out=col)
    return p


def training_domain(sf):
    X = np.asarray(sf.knn_envelope._Xtrain_log, dtype=np.float64)
    names = list(sf.predictor_names)
    flu = 10.0 ** X[:, names.index("fluence")]
    dm = 10.0 ** X[:, names.index("dm")]
    return float(flu.min()), float(flu.max()), float(dm.min()), float(dm.max())


def tabulate_selection(sf, f_min, f_max, dm_min, dm_max, n_f=400, n_dm=400):
    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), n_f)
    dm_grid = np.logspace(np.log10(dm_min), np.log10(dm_max), n_dm)
    FF, DD = np.meshgrid(f_grid, dm_grid, indexing="ij")
    log("tabulate P_sel %d x %d" % (n_f, n_dm))
    t0 = time.time()
    p_raw = np.asarray(
        sf.calculate_selection_probability({
            "fluence_jy_ms": FF.ravel(),
            "dm": DD.ravel(),
        }),
        dtype=np.float64,
    ).reshape(n_f, n_dm)
    log("KNN table in %.1fs  finite=%.3f  max=%.4f"
        % (time.time() - t0, np.isfinite(p_raw).mean(), np.nanmax(p_raw)))
    return f_grid, dm_grid, fill_psel_table(p_raw, f_grid)


def interp_psel(p_tab, f_grid, dm_grid, fluence, dm):
    log_f0 = np.log10(f_grid[0])
    dlog_f = np.log10(f_grid[1] / f_grid[0])
    log_d0 = np.log10(dm_grid[0])
    dlog_d = np.log10(dm_grid[1] / dm_grid[0])
    n_f, n_dm = p_tab.shape
    xi = (np.log10(np.clip(fluence, 1e-300, None)) - log_f0) / dlog_f
    yi = (np.log10(np.clip(dm, 1e-300, None)) - log_d0) / dlog_d
    xi = np.where(fluence < f_grid[0], -1.0, np.minimum(xi, n_f - 1.0))
    in_dm = (dm >= dm_grid[0]) & (dm <= dm_grid[-1])
    yi = np.where(in_dm, np.clip(yi, 0.0, n_dm - 1.0), -1.0)
    xx, yy = np.meshgrid(xi, yi, indexing="ij")
    return map_coordinates(
        p_tab, [xx, yy], order=1, mode="constant", cval=0.0, prefilter=False,
    )


def pdet_for_host(z, E, D_L, dm_igm, dm_gal, f_grid, dm_grid, p_tab,
                  dm_host, weights):
    """weights: list of (name, w, wsum). Returns {name: P_det(z)}."""
    out = {name: np.empty(z.size, dtype=np.float64) for name, _w, _s in weights}
    nsky = dm_gal.size
    t0 = time.time()
    for k in range(z.size):
        flu = erg_per_mpc2_hz_to_jy_ms(F_v(E, z[k], D_L[k]))
        dm = dm_igm[k] + dm_host / (1.0 + z[k]) + dm_gal
        P = interp_psel(p_tab, f_grid, dm_grid, flu, dm)
        for name, w, wsum in weights:
            out[name][k] = np.sum(P * w[:, None]) / (wsum * nsky)
        if (k + 1) % 25 == 0 or k == 0 or k + 1 == z.size:
            done = k + 1
            dt = time.time() - t0
            eta = dt / done * (z.size - done)
            log("DM_host=%g  z %d/%d  z=%.3f  elapsed=%.1fs  ETA=%.1fs"
                % (dm_host, done, z.size, z[k], dt, eta))
    for name in out:
        out[name] = np.maximum(out[name], 0.0)
    return out
