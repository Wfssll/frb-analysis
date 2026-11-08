# -*- coding: utf-8 -*-
"""DM_IGM × DM_host convolution for the DM–z likelihood.

IGM_MODEL selects the IGM PDF:
  macquart  — Macquart p_Δ (scatter F), default production path
  lognormal — lognormal IGM with fixed A_IGM, mean = Macquart 〈DM〉
  gauss     — truncated-Gaussian IGM ⊗ truncated-Gaussian host (analytic)

P(DM_eg | z) is IGM ⊗ host. Production dmz/joint use Macquart p_Δ with
free F, rest-frame lognormal host, FFT 2000×4000, and make_z_grid()
(50+20+20+60 on [0, 0.2, 0.5, 1, 5]). The CHIME DM=1.60 event is not
on that FFT grid: Macquart integrates it on its own [0, DM_i] support
and on linspace(0.005, 0.008, 61). Other IGM models still drop it.
IGM_MODEL=gauss uses the closed-form convolution of two Gaussians each
truncated to DM>0 and renormalised. Observer-frame host is DM_rest/(1+z).
"""

from cosmology import production_cosmology

import os
from pathlib import Path

import numpy as np
import pyccl as ccl
from astropy.constants import G, c, m_p, pc
from astropy.io import fits
from scipy.integrate import cumulative_trapezoid, quad, trapezoid
from scipy.special import ndtr

import P_IGM

ROOT = Path(__file__).resolve().parent
FRB_DATA_FILE = ROOT / "data" / "frb" / "chimefrbcat2_process.fits"

F_IGM = 0.93
X_E = 0.875
DM_MW_RESIDUAL = 0.0
MEAN_IGM_MIN = 1.0e-3
F_SPIKE = 1.0e-4
PDF_FLOOR = 1.0e-300
LOG_SQRT_TWO_PI = 0.5 * np.log(2.0 * np.pi)
A_IGM = 230.0

# 2000 x 4000: |Δln L| ≲ 0.5 vs 4096 x 6000 at F=0.01, 0.3, 1; ~0.49x wall.
N_DM_GRID = 2000
DM_GRID_MAX = 4.0e3
# Lognormal IGM FFT grid; locked by test_igm_lognormal_conv.py vs per-source
# trapz/quad gold (z/10 ln L). 5000 x 4000: ΔlnL = -0.21; fastest with |ΔlnL|≤0.5.
# 10000 x 4000 is closer (ΔlnL = +0.05) at ~2.2x wall.
N_DM_GRID_LN = 5000
DM_GRID_MAX_LN = 4.0e3
N_DELTA_REF = 900
N_TRAPZ_DM = 8000
DM_TRAPZ_MAX = 1.2e4
N_TRAPZ_SOURCE = 20000
# FFT leaks ~1e-20 at DM=1.6. The bulk sample still drops DM < DM_EG_MIN.
# Macquart puts that one source on its own short redshift window (test.md §20).
# Header DMMEAN is unchanged (C_ell d_bar uses catalog mean).
DM_EG_MIN = 5.0
# P_det table starts at z=0.005; below that p_sel is identically zero.
TINY_Z_LO = 0.005
TINY_Z_HI = 0.008
N_TINY_Z = 61
N_TINY_DM = 384
# Production DM–z quadrature: piecewise linspace, junctions shared once.
Z_SEGMENTS = (
    (0.0, 0.2, 50),
    (0.2, 0.5, 20),
    (0.5, 1.0, 20),
    (1.0, 5.0, 60),
)

_IGM_ALIASES = {
    "macquart": "macquart",
    "p_delta": "macquart",
    "pdelta": "macquart",
    "lognormal": "lognormal",
    "ln": "lognormal",
    "gauss": "gauss",
    "gaussian": "gauss",
}


def normalize_igm_model(name=None):
    if name is None or str(name).strip() == "":
        name = os.getenv("IGM_MODEL", "macquart")
    key = str(name).strip().lower().replace("-", "_")
    if key not in _IGM_ALIASES:
        raise ValueError("unknown IGM_MODEL %r" % name)
    return _IGM_ALIASES[key]


IGM_MODEL = normalize_igm_model()


def make_z_grid(segments=None):
    """Piecewise-uniform redshift grid for the DM–z marginal.

    Production: 50+20+20+60 on [0, 0.2, 0.5, 1, 5] (147 unique nodes
    after sharing junctions). C_ell keeps a separate uniform 500-point Z.
    """
    if segments is None:
        segments = Z_SEGMENTS
    chunks = [np.linspace(float(a), float(b), int(n)) for a, b, n in segments]
    return np.unique(np.concatenate(chunks))


def make_cosmology():
    return production_cosmology(
        Omega_b=0.04897, Omega_c=0.3111 - 0.04897,
        h=0.6766, sigma8=0.81, n_s=0.96)


def load_extragalactic_dm(path=None, dm_min=None):
    """Observer-frame extra-galactic DM (NE2001 already subtracted).

    Default dm_min=DM_EG_MIN drops the DM=1.60 event from the bulk FFT
    sample. Macquart adds that event back through log_tiny_macquart_likelihood.
    Pass dm_min=0 to keep the full FITS table.
    Header DMMEAN is unchanged (C_ell d_bar still uses the catalog mean).
    """
    path = Path(path) if path is not None else FRB_DATA_FILE
    if dm_min is None:
        dm_min = DM_EG_MIN
    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        dm_mean = float(hdul[1].header["DMMEAN"])
        valid = np.isfinite(table["dm_exc_ne2001"])
        dm_res = np.asarray(table["dm_exc_ne2001"][valid], dtype=float)
    dm = dm_res + dm_mean
    ok = np.isfinite(dm) & (dm > 0.0)
    if float(dm_min) > 0.0:
        ok &= dm >= float(dm_min)
    return dm[ok], float(dm_mean)


def load_tiny_extragalactic_dm(path=None):
    """Sources with 0 < DM < DM_EG_MIN. The bulk loader drops these."""
    dm_all, dm_mean = load_extragalactic_dm(path, dm_min=0.0)
    tiny = dm_all[dm_all < DM_EG_MIN]
    return tiny, float(dm_mean)


def make_tiny_z_window():
    """Redshift nodes for the DM < DM_EG_MIN source. dz = 5e-5."""
    return np.linspace(TINY_Z_LO, TINY_Z_HI, N_TINY_Z)


def make_tiny_composite_z(z_bulk=None):
    """Short window union the bulk DM–z grid, for p_sel normalisation."""
    if z_bulk is None:
        z_bulk = make_z_grid()
    z_bulk = np.asarray(z_bulk, dtype=float)
    return np.unique(np.concatenate([make_tiny_z_window(), z_bulk]))


def mean_igm_dm_derivative(cosmo, z, f_igm=F_IGM):
    z = np.asarray(z, dtype=float)
    return (3.0 * c.value * cosmo["Omega_b"] * cosmo["h"] * 100.0
            / (8.0 * np.pi * G.value * m_p.value) * (1.0 + z) * f_igm * X_E
            / ccl.h_over_h0(cosmo, 1.0 / (1.0 + z))
            * (1e3 / (pc.value * 1e6)) * (1e-6 / pc.value))


def mean_cosmic_dm(cosmo, z, f_igm=F_IGM):
    z = np.asarray(z, dtype=float)
    z_dense = np.linspace(0.0, max(5.0, 1.02 * float(np.max(z))), 4000)
    dm_dense = cumulative_trapezoid(
        mean_igm_dm_derivative(cosmo, z_dense, f_igm=f_igm),
        z_dense, initial=0.0)
    return np.interp(z, z_dense, dm_dense)


def dm_host_mean(mu_ln, sigma_ln):
    """Rest-frame mean of the lognormal host DM."""
    return float(np.exp(float(mu_ln) + 0.5 * float(sigma_ln) ** 2))


def _truncnorm_pos_mass(mean, sig):
    """P(X>0) for X ~ N(mean, sig^2)."""
    return np.clip(ndtr(mean / np.maximum(sig, 1e-12)), 1e-300, 1.0)


def truncnorm_pdf_positive(x, mean, sig):
    """N(mean, sig^2) truncated to x>0 and renormalised. Broadcasts."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-12)
    zscore = (x - mean) / sig
    log_pdf = (
        -np.log(sig) - LOG_SQRT_TWO_PI - 0.5 * zscore ** 2
        - np.log(_truncnorm_pos_mass(mean, sig))
    )
    return np.where(x > 0.0, np.exp(np.clip(log_pdf, -700.0, 700.0)), 0.0)


def dm_host_gauss_mean(mu_h, sigma_h):
    """Rest-frame mean of N(μ_h, σ_h²) truncated to DM_rest > 0."""
    mu = float(mu_h)
    sig = max(float(sigma_h), 1e-12)
    mass = float(_truncnorm_pos_mass(mu, sig))
    phi = np.exp(-0.5 * (mu / sig) ** 2) / np.sqrt(2.0 * np.pi)
    return mu + sig * phi / mass


def p_host_obs(dm, z, mu_ln, sigma_ln):
    """Observer-frame host PDF. dm broadcasts against z.

    Rest-frame: ln DM_rest ~ N(μ, σ).  DM_obs = DM_rest / (1+z).
    """
    dm = np.asarray(dm, dtype=float)
    z = np.asarray(z, dtype=float)
    mu_ln = float(mu_ln)
    sigma_ln = float(sigma_ln)
    one_plus_z = 1.0 + z
    rest = dm * one_plus_z
    ok = rest > 0.0
    rest_ok = np.where(ok, rest, 1.0)
    log_rest = np.log(rest_ok)
    log_pdf = (
        -np.log(rest_ok) - np.log(max(sigma_ln, 1e-12)) - LOG_SQRT_TWO_PI
        - 0.5 * ((log_rest - mu_ln) / max(sigma_ln, 1e-12)) ** 2
        + np.log(np.maximum(one_plus_z, 1e-12))
    )
    return np.where(ok, np.exp(np.clip(log_pdf, -700.0, 700.0)), 0.0)


def p_host_gauss_obs(dm, z, mu_h, sigma_h):
    """Observer-frame Gaussian host, truncated to DM>0 and renormalised.

    Rest-frame N(μ_h, σ_h²) truncated at DM_rest>0 is equivalent to an
    observer-frame truncated normal with mean μ_h/(1+z) and width σ_h/(1+z).
    """
    dm = np.asarray(dm, dtype=float)
    z = np.asarray(z, dtype=float)
    one_plus_z = 1.0 + z
    mean = float(mu_h) / one_plus_z
    sig = np.maximum(float(sigma_h) / one_plus_z, 1e-12)
    return truncnorm_pdf_positive(dm, mean, sig)


def _p_delta_grid(delta, F, z):
    """p_Δ on a (nz, n_delta) grid. delta, z broadcast."""
    P_IGM.load_table()
    A, C0 = P_IGM.get_A_C0(F, z)
    sig = np.clip(P_IGM.sigma_diff(F, z), P_IGM.SIGMA_MIN, P_IGM.SIGMA_MAX)
    A = np.asarray(A, dtype=float)
    C0 = np.asarray(C0, dtype=float)
    sig = np.asarray(sig, dtype=float)
    delta = np.asarray(delta, dtype=float)
    ok = delta > 0.0
    d = np.where(ok, delta, 1.0)
    exponent = -((np.power(d, -P_IGM.ALPHA) - C0) ** 2) / (
        2.0 * P_IGM.ALPHA ** 2 * np.maximum(sig, 1e-12) ** 2)
    pdf = A * np.power(d, -P_IGM.BETA) * np.exp(np.clip(exponent, -700.0, 0.0))
    return np.where(ok, pdf, 0.0)


def p_igm_dm(dm_grid, z, F, cosmic_dm):
    """p(DM_IGM | z, F) = p_Δ(DM/〈DM〉) / 〈DM〉. Shape (nz, n_dm)."""
    dm_grid = np.asarray(dm_grid, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    mean = np.maximum(cosmic_dm, 0.0)
    small = mean < MEAN_IGM_MIN
    mean_safe = np.where(small, 1.0, mean)
    delta = dm_grid[None, :] / mean_safe[:, None]
    p_d = _p_delta_grid(delta, F, z[:, None])
    p_dm = p_d / mean_safe[:, None]
    p_dm[small] = 0.0
    return p_dm, small


def igm_lognormal_params(z, cosmic_dm, a_igm=A_IGM):
    """Log-space (μ, σ) such that 〈D〉=D̄(z) and Var(D)=A_IGM² z."""
    z = np.asarray(z, dtype=float).reshape(-1)
    mean = np.maximum(np.asarray(cosmic_dm, dtype=float).reshape(-1), 0.0)
    small = mean < MEAN_IGM_MIN
    mean_safe = np.where(small, 1.0, mean)
    var = (float(a_igm) ** 2) * np.maximum(z, 0.0)
    sigma_ln2 = np.log1p(var / np.square(mean_safe))
    sigma_ln = np.sqrt(np.maximum(sigma_ln2, 0.0))
    mu_ln = np.log(mean_safe) - 0.5 * sigma_ln2
    return mu_ln, sigma_ln, small


def p_igm_lognormal(dm_grid, z, cosmic_dm, a_igm=A_IGM):
    """p(DM_IGM | z, A_IGM) lognormal. Shape (nz, n_dm)."""
    dm_grid = np.asarray(dm_grid, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    mu_ln, sigma_ln, small = igm_lognormal_params(z, cosmic_dm, a_igm=a_igm)
    dm = dm_grid[None, :]
    ok = dm > 0.0
    dm_ok = np.where(ok, dm, 1.0)
    sig = np.maximum(sigma_ln, 1e-12)[:, None]
    log_pdf = (
        -np.log(dm_ok) - np.log(sig) - LOG_SQRT_TWO_PI
        - 0.5 * ((np.log(dm_ok) - mu_ln[:, None]) / sig) ** 2
    )
    pdf = np.where(ok, np.exp(np.clip(log_pdf, -700.0, 700.0)), 0.0)
    pdf[small] = 0.0
    return pdf, small


def p_igm_gauss(dm_grid, z, cosmic_dm, a_igm=A_IGM):
    """Gaussian IGM truncated to DM>0 and renormalised. Shape (nz, n_dm)."""
    dm_grid = np.asarray(dm_grid, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    mean = np.maximum(cosmic_dm, 0.0)
    small = mean < MEAN_IGM_MIN
    sig = np.maximum(float(a_igm) * np.sqrt(np.maximum(z, 0.0)), 1e-12)
    pdf = truncnorm_pdf_positive(dm_grid[None, :], mean[:, None], sig[:, None])
    pdf[small] = 0.0
    return pdf, small


def make_dm_grid(dm_obs=None, n_grid=None, dm_max=None, igm_model=None):
    model = normalize_igm_model(igm_model) if igm_model is not None else IGM_MODEL
    if n_grid is None:
        n_grid = N_DM_GRID_LN if model == "lognormal" else N_DM_GRID
    if dm_max is None:
        dm_max = DM_GRID_MAX_LN if model == "lognormal" else DM_GRID_MAX
    return np.linspace(0.0, float(dm_max), int(n_grid))


def _fft_convolve_pdfs(p_a, p_b, dx):
    n = p_a.shape[-1]
    nfft = 1 << (int(2 * n - 1).bit_length())
    fa = np.fft.rfft(p_a, n=nfft, axis=-1)
    fb = np.fft.rfft(p_b, n=nfft, axis=-1)
    conv = np.fft.irfft(fa * fb, n=nfft, axis=-1) * dx
    out = np.maximum(conv[..., :n], 0.0)
    return out


def _interp_pdf(pdf, dm_grid, query):
    """Linear interpolate pdf (nz, n_grid) at query (n_frb,). Return (n_frb, nz)."""
    dm_grid = np.asarray(dm_grid, dtype=float)
    query = np.asarray(query, dtype=float)
    dx = float(dm_grid[1] - dm_grid[0])
    idx = (query - dm_grid[0]) / dx
    n = dm_grid.size
    i0 = np.floor(idx).astype(int)
    valid = (query > 0.0) & (i0 >= 0) & (i0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    frac = np.clip(idx - i0c, 0.0, 1.0)
    p0 = pdf[:, i0c]
    p1 = pdf[:, i0c + 1]
    out = (p0 * (1.0 - frac)[None, :] + p1 * frac[None, :]).T
    out[~valid] = 0.0
    return out


def _shift_host_to_mean(p_host, dm_grid, shift):
    """p_host(DM - shift) on the same grid. p_host (nz, n), shift (nz,)."""
    query = dm_grid[None, :] - np.asarray(shift, dtype=float)[:, None]
    dx = float(dm_grid[1] - dm_grid[0])
    idx = (query - dm_grid[0]) / dx
    n = dm_grid.size
    i0 = np.floor(idx).astype(int)
    valid = (query > 0.0) & (i0 >= 0) & (i0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    frac = np.clip(idx - i0c, 0.0, 1.0)
    rows = np.arange(p_host.shape[0])[:, None]
    p0 = p_host[rows, i0c]
    p1 = p_host[rows, i0c + 1]
    out = p0 * (1.0 - frac) + p1 * frac
    out[~valid] = 0.0
    return out


def p_dm_given_z_fft(dm, z, cosmic_dm, F, mu_ln, sigma_ln, dm_grid=None,
                     dm_mw=DM_MW_RESIDUAL, igm_model=None, a_igm=A_IGM):
    """P(DM_i | z_j) by FFT convolution. Shape (n_frb, n_z)."""
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    model = normalize_igm_model(igm_model) if igm_model is not None else IGM_MODEL
    if dm_grid is None:
        dm_grid = make_dm_grid(dm, igm_model=model)
    dx = float(dm_grid[1] - dm_grid[0])
    if model == "gauss":
        p_host = p_host_gauss_obs(dm_grid[None, :], z[:, None], mu_ln, sigma_ln)
    else:
        p_host = p_host_obs(dm_grid[None, :], z[:, None], mu_ln, sigma_ln)
    p_host = np.maximum(p_host, 0.0)

    if model == "lognormal":
        p_igm, small = p_igm_lognormal(dm_grid, z, cosmic_dm, a_igm=a_igm)
    elif model == "gauss":
        p_igm, small = p_igm_gauss(dm_grid, z, cosmic_dm, a_igm=a_igm)
    else:
        if float(F) < F_SPIKE:
            p_tot = _shift_host_to_mean(p_host, dm_grid, cosmic_dm + float(dm_mw))
            return _interp_pdf(p_tot, dm_grid, dm)
        p_igm, small = p_igm_dm(dm_grid, z, F, cosmic_dm)

    p_tot = _fft_convolve_pdfs(p_igm, p_host, dx)
    if np.any(small):
        p_tot[small] = _shift_host_to_mean(
            p_host[small], dm_grid, cosmic_dm[small])
    if abs(float(dm_mw)) > 0.0:
        p_tot = _shift_host_to_mean(p_tot, dm_grid, np.full(z.size, float(dm_mw)))
    return _interp_pdf(p_tot, dm_grid, dm)


def _delta_ref_grid():
    return np.unique(np.concatenate([
        np.geomspace(1.0e-4, 0.3, 250),
        np.linspace(0.3, 3.0, 400),
        np.geomspace(3.0, 25.0, 250),
    ]))


def p_dm_given_z_trapz(dm, z, cosmic_dm, F, mu_ln, sigma_ln,
                       dm_mw=DM_MW_RESIDUAL, delta_grid=None):
    """P(DM_i | z_j) by trapezoid in Δ: ∫ p_Δ(Δ) p_host(DM-Δ〈DM〉) dΔ."""
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    if delta_grid is None:
        delta_grid = _delta_ref_grid()
    n_frb = dm.size
    n_z = z.size
    out = np.empty((n_frb, n_z), dtype=float)
    mean = np.maximum(cosmic_dm, 0.0)
    small = mean < MEAN_IGM_MIN

    if float(F) < F_SPIKE:
        dm_host = (dm[:, None] - cosmic_dm[None, :] - float(dm_mw))
        return p_host_obs(dm_host, z[None, :], mu_ln, sigma_ln)

    p_d = _p_delta_grid(delta_grid[None, :], F, z[:, None])
    for j in range(n_z):
        if small[j]:
            dm_host = dm - cosmic_dm[j] - float(dm_mw)
            out[:, j] = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
            continue
        dm_igm = delta_grid * mean[j]
        dm_host = dm[:, None] - dm_igm[None, :] - float(dm_mw)
        p_h = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
        out[:, j] = trapezoid(p_d[j][None, :] * p_h, delta_grid, axis=1)
    return out


def p_dm_given_z_trapz_dm(dm, z, cosmic_dm, mu_ln, sigma_ln,
                          dm_mw=DM_MW_RESIDUAL, igm_grid=None,
                          a_igm=A_IGM, n_igm=N_TRAPZ_DM, dm_max=None):
    """Hard integral in DM: ∫ p_IGM(x|z) p_host(DM-x|z) dx, x ∈ [0, DM]."""
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    if igm_grid is None:
        if dm_max is None:
            dm_max = max(float(np.max(dm)) * 1.05, float(DM_TRAPZ_MAX))
        igm_grid = np.linspace(0.0, float(dm_max), int(n_igm))
    p_igm, small = p_igm_lognormal(igm_grid, z, cosmic_dm, a_igm=a_igm)
    n_frb = dm.size
    n_z = z.size
    out = np.empty((n_frb, n_z), dtype=float)
    x = igm_grid
    for j in range(n_z):
        if small[j]:
            dm_host = dm - cosmic_dm[j] - float(dm_mw)
            out[:, j] = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
            continue
        dm_host = dm[:, None] - x[None, :] - float(dm_mw)
        p_h = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
        out[:, j] = trapezoid(p_igm[j][None, :] * p_h, x, axis=1)
    return out


def p_dm_given_z_trapz_source(dm, z, cosmic_dm, mu_ln, sigma_ln,
                              dm_mw=DM_MW_RESIDUAL, a_igm=A_IGM,
                              n_igm=N_TRAPZ_SOURCE):
    """Per-source trapezoid on [0, DM_i]: no global DM_max cutoff."""
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    mu_igm, sig_igm, small = igm_lognormal_params(z, cosmic_dm, a_igm=a_igm)
    n_frb = dm.size
    n_z = z.size
    n_igm = int(n_igm)
    frac = np.linspace(0.0, 1.0, n_igm)
    out = np.empty((n_frb, n_z), dtype=float)
    for j in range(n_z):
        if small[j]:
            dm_host = dm - cosmic_dm[j] - float(dm_mw)
            out[:, j] = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
            continue
        x = dm[:, None] * frac[None, :]
        ok = x > 0.0
        x_ok = np.where(ok, x, 1.0)
        sig = max(float(sig_igm[j]), 1e-12)
        log_p = (
            -np.log(x_ok) - np.log(sig) - LOG_SQRT_TWO_PI
            - 0.5 * ((np.log(x_ok) - mu_igm[j]) / sig) ** 2
        )
        p_igm = np.where(ok, np.exp(np.clip(log_p, -700.0, 700.0)), 0.0)
        p_h = p_host_obs(dm[:, None] - x - float(dm_mw), z[j], mu_ln, sigma_ln)
        out[:, j] = trapezoid(p_igm * p_h, x, axis=1)
    return out


def _igm_host_product(x, dm_i, z_j, mu_igm, sig_igm, mu_ln, sigma_ln, dm_mw):
    """p_IGM(x|z) p_host(DM_i - x | z) for a scalar x in (0, DM_i)."""
    x = float(x)
    if x <= 0.0 or x >= float(dm_i):
        return 0.0
    sig = max(float(sig_igm), 1e-12)
    log_igm = (
        -np.log(x) - np.log(sig) - LOG_SQRT_TWO_PI
        - 0.5 * ((np.log(x) - mu_igm) / sig) ** 2
    )
    p_igm = float(np.exp(np.clip(log_igm, -700.0, 700.0)))
    p_h = float(p_host_obs(float(dm_i) - x - float(dm_mw), z_j, mu_ln, sigma_ln))
    return p_igm * p_h


def p_dm_given_z_quad(dm, z, cosmic_dm, mu_ln, sigma_ln,
                      dm_mw=DM_MW_RESIDUAL, a_igm=A_IGM,
                      epsabs=1.0e-10, epsrel=1.0e-8):
    """Adaptive quad of ∫_0^{DM_i} p_IGM(x) p_host(DM_i-x) dx. Shape (n_frb, n_z)."""
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    mu_igm, sig_igm, small = igm_lognormal_params(z, cosmic_dm, a_igm=a_igm)
    n_frb = dm.size
    n_z = z.size
    out = np.empty((n_frb, n_z), dtype=float)
    for j in range(n_z):
        if small[j]:
            dm_host = dm - cosmic_dm[j] - float(dm_mw)
            out[:, j] = p_host_obs(dm_host, z[j], mu_ln, sigma_ln)
            continue
        for i in range(n_frb):
            dm_i = float(dm[i])
            if dm_i <= 0.0:
                out[i, j] = 0.0
                continue
            val, _err = quad(
                _igm_host_product, 0.0, dm_i,
                args=(dm_i, float(z[j]), float(mu_igm[j]), float(sig_igm[j]),
                      float(mu_ln), float(sigma_ln), float(dm_mw)),
                epsabs=float(epsabs), epsrel=float(epsrel), limit=200)
            out[i, j] = max(val, 0.0)
    return out


def p_dm_given_z_gauss_analytic(dm, z, cosmic_dm, mu_h, sigma_h,
                                a_igm=A_IGM, dm_mw=DM_MW_RESIDUAL):
    """P(DM|z) for truncated-Gaussian IGM ⊗ truncated-Gaussian host.

    Each component is N(·) restricted to DM>0 and renormalised. The density of
    the sum at s = DM - DM_MW is the untruncated Gaussian convolution times
    P(0 < X < s | X+Y = s), divided by Φ(μ_IGM/σ_IGM) Φ(μ_h/σ_h).
    """
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    one_plus_z = 1.0 + z
    mu1 = np.maximum(cosmic_dm, 0.0)
    sig1 = np.maximum(float(a_igm) * np.sqrt(np.maximum(z, 0.0)), 1e-12)
    mu2 = float(mu_h) / one_plus_z
    sig2 = np.maximum(float(sigma_h) / one_plus_z, 1e-12)
    s = dm[:, None] - float(dm_mw)
    v1 = np.square(sig1)
    v2 = np.square(sig2)
    var = v1 + v2
    sig_sum = np.sqrt(np.maximum(var, 1e-24))
    mu_sum = mu1 + mu2
    x = (s - mu_sum[None, :]) / sig_sum[None, :]
    log_phi = (
        -LOG_SQRT_TWO_PI - np.log(sig_sum[None, :]) - 0.5 * x ** 2
    )
    mass1 = _truncnorm_pos_mass(mu1, sig1)
    mass2 = _truncnorm_pos_mass(mu2, sig2)
    sig_x = np.sqrt(np.maximum(v1 * v2 / var, 1e-24))
    m_x = (mu1 * v2 + (s - mu2) * v1) / var
    inner = ndtr((s - m_x) / sig_x[None, :]) - ndtr((0.0 - m_x) / sig_x[None, :])
    inner = np.clip(inner, 0.0, 1.0)
    log_pdf = (
        log_phi + np.log(np.maximum(inner, 1e-300))
        - np.log(mass1[None, :]) - np.log(mass2[None, :])
    )
    pdf = np.exp(np.clip(log_pdf, -700.0, 700.0))
    return np.where((s > 0.0) & (inner > 0.0), pdf, 0.0)


def p_macquart_source_trapz(dm, z, cosmic_dm, F, mu_ln, sigma_ln,
                            n_dm=N_TINY_DM, dm_mw=DM_MW_RESIDUAL):
    """P(DM_i|z) by trapezoid on [0, DM_i]. Shape (n_src, n_z).

    Each source keeps its own DM abscissa. Do not evaluate this on the
    catalog FFT grid: that grid leaks at DM ~ 1.6.
    """
    dm = np.asarray(dm, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    cosmic_dm = np.asarray(cosmic_dm, dtype=float).reshape(-1)
    out = np.empty((dm.size, z.size), dtype=float)
    for i, dm_i in enumerate(dm):
        dm_i = float(dm_i)
        if dm_i <= 0.0:
            out[i] = 0.0
            continue
        x = np.linspace(0.0, dm_i, int(n_dm))
        p_igm, small = p_igm_dm(x, z, F, cosmic_dm)
        p_h = p_host_obs(
            (dm_i - x - float(dm_mw))[None, :], z[:, None], mu_ln, sigma_ln)
        col = np.maximum(trapezoid(p_igm * p_h, x, axis=1), 0.0)
        if np.any(small):
            col = np.array(col, copy=True)
            col[small] = p_host_obs(
                dm_i - cosmic_dm[small] - float(dm_mw), z[small],
                mu_ln, sigma_ln)
        out[i] = col
    return out


def log_tiny_macquart_likelihood(z_comp, p_sel, dm_tiny, cosmic_comp,
                                 F, mu_ln, sigma_ln,
                                 z_lo=TINY_Z_LO, z_hi=TINY_Z_HI):
    """ln L of the DM < DM_EG_MIN sources.

    ``p_sel`` is unit-normalised on ``z_comp`` (the short window union the
    bulk redshift grid). The DM convolution and the z integral use only
    the nodes inside [z_lo, z_hi].
    """
    dm_tiny = np.asarray(dm_tiny, dtype=float).reshape(-1)
    if dm_tiny.size == 0:
        return 0.0
    z_comp = np.asarray(z_comp, dtype=float)
    p_sel = np.clip(np.asarray(p_sel, dtype=float), 0.0, None)
    cosmic_comp = np.asarray(cosmic_comp, dtype=float)
    inside = (z_comp >= float(z_lo) - 1e-12) & (z_comp <= float(z_hi) + 1e-12)
    z_use = z_comp[inside]
    pdf = p_macquart_source_trapz(
        dm_tiny, z_use, cosmic_comp[inside], F, mu_ln, sigma_ln)
    marg = trapezoid(pdf * p_sel[inside][None, :], z_use, axis=1)
    marg = np.maximum(marg, PDF_FLOOR)
    return float(np.sum(np.log(marg)))


def log_marginal_dm_likelihood(z, p_sel, dm, cosmic_dm, mu_ln, sigma_ln, F,
                               dm_mw=DM_MW_RESIDUAL, method="fft",
                               dm_grid=None, pdf=None, igm_model=None,
                               a_igm=A_IGM):
    """Σ_i ln ∫ dz p_sel(z) P(DM_i|z). Returns (logL, per-source integrands).

    For IGM_MODEL=gauss, mu_ln/sigma_ln are the rest-frame linear host mean
    and scatter (pc cm^{-3}), not log-space parameters. F is unused. Both
    Gaussians are truncated to DM>0 and renormalised before convolution.
    """
    z = np.asarray(z, dtype=float)
    p_sel = np.clip(np.asarray(p_sel, dtype=float), 0.0, None)
    model = normalize_igm_model(igm_model) if igm_model is not None else IGM_MODEL
    if pdf is None:
        if model == "gauss":
            pdf = p_dm_given_z_gauss_analytic(
                dm, z, cosmic_dm, mu_ln, sigma_ln, a_igm=a_igm, dm_mw=dm_mw)
        elif method == "trapz":
            if model == "lognormal":
                pdf = p_dm_given_z_trapz_dm(
                    dm, z, cosmic_dm, mu_ln, sigma_ln, dm_mw=dm_mw,
                    a_igm=a_igm)
            else:
                pdf = p_dm_given_z_trapz(
                    dm, z, cosmic_dm, F, mu_ln, sigma_ln, dm_mw=dm_mw)
        elif method == "trapz_source":
            pdf = p_dm_given_z_trapz_source(
                dm, z, cosmic_dm, mu_ln, sigma_ln, dm_mw=dm_mw, a_igm=a_igm)
        elif method == "quad":
            pdf = p_dm_given_z_quad(
                dm, z, cosmic_dm, mu_ln, sigma_ln, dm_mw=dm_mw, a_igm=a_igm)
        else:
            pdf = p_dm_given_z_fft(
                dm, z, cosmic_dm, F, mu_ln, sigma_ln,
                dm_grid=dm_grid, dm_mw=dm_mw, igm_model=model, a_igm=a_igm)
    integrand = pdf * p_sel[None, :]
    marg = trapezoid(integrand, z, axis=1)
    marg = np.maximum(marg, PDF_FLOOR)
    return float(np.sum(np.log(marg))), marg
