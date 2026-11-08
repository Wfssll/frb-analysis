"""SFR + delay FRB redshift distribution, with CHIME P_det.

p_sel(z) ∝ [d_c^2 / (H(z) (1+z))] R_SFR-delay(z) P_det(z), then unit-normalised.
p_int(z) is the same without P_det.

Uses Madau & Fragos (2017) SFH by default (set SFH_YEAR=2014 for
Madau & Dickinson 2014) and the DM-dependent CHIME P_det
(pdet_chime.npz, simulated with DM_host = 100 pc cm^-3). DELAY_MODEL selects
the delay PDF, or SFH:
  lognormal  — tau_NL [Gyr], sigma_NL (natural-log width)
  gauss      — tau_gauss [Gyr], sigma_gauss [Gyr]; truncated to tau>0
  powerlaw   — alpha on [tau_min, t_max]; tau_min=0.02 Gyr,
               t_max = lookback to z_f=100 (same as gauss/lognormal)
  sfh        — p(z) ∝ vol × R_SFH(z) × P_det(z); no delay parameters
"""

import os
from pathlib import Path

import numpy as np
import pyccl as ccl
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import erf

ROOT = Path(__file__).resolve().parent
PDET_CANDIDATES = (
    ROOT / "CHIME_selection" / "pdet_chime.npz",
)

TAU_NL_FID = 6.0
SIGMA_NL_FID = 1.0
TAU_NL_PRIOR = (0.1, 10.0)
SIGMA_NL_PRIOR = (0.1, 3.0)
TAU_GAUSS_FID = 0.15
SIGMA_GAUSS_FID = 4.0
TAU_GAUSS_PRIOR = (0.1, 8.0)
SIGMA_GAUSS_PRIOR = (0.1, 8.0)
ALPHA_PL_FID = 1.0
ALPHA_PL_PRIOR = (0.01, 2.0)
TAU_PL_MIN = 0.02
TAU_MIN_GYR = 1.0e-4
Z_FORMATION_MAX = 100.0
N_FORMATION_TAIL = 1500
N_TAU_GAUSS = 800
N_TAU_POWERLAW = 400
REF_PZ_ALPHA = 2.0
REF_PZ_BETA = 6.7

_DELAY_ALIASES = {
    "lognormal": "lognormal",
    "log-normal": "lognormal",
    "ln": "lognormal",
    "gauss": "gauss",
    "gaussian": "gauss",
    "powerlaw": "powerlaw",
    "power-law": "powerlaw",
    "pl": "powerlaw",
    "sfh": "sfh",
}


def normalize_delay_model(name=None):
    if name is None or str(name).strip() == "":
        name = os.getenv("DELAY_MODEL", "lognormal")
    key = str(name).strip().lower().replace("_", "-")
    if key not in _DELAY_ALIASES:
        raise ValueError("unknown DELAY_MODEL %r" % name)
    return _DELAY_ALIASES[key]


DELAY_MODEL = normalize_delay_model()


def delay_param_spec(delay_model=None):
    """Return (params, plot_names). Each param is [name, start, lo, hi]."""
    model = normalize_delay_model(delay_model)
    if model == "gauss":
        return (
            [
                ["tau_gauss", TAU_GAUSS_FID,
                 TAU_GAUSS_PRIOR[0], TAU_GAUSS_PRIOR[1]],
                ["sigma_gauss", SIGMA_GAUSS_FID,
                 SIGMA_GAUSS_PRIOR[0], SIGMA_GAUSS_PRIOR[1]],
            ],
            [r"\tau_{\rm gauss}", r"\sigma_{\rm gauss}"],
        )
    if model == "powerlaw":
        return (
            [["alpha", ALPHA_PL_FID, ALPHA_PL_PRIOR[0], ALPHA_PL_PRIOR[1]]],
            [r"\alpha_{\tau}"],
        )
    if model == "sfh":
        return ([], [])
    return (
        [
            ["tau_NL", TAU_NL_FID, TAU_NL_PRIOR[0], TAU_NL_PRIOR[1]],
            ["sigma_NL", SIGMA_NL_FID, SIGMA_NL_PRIOR[0], SIGMA_NL_PRIOR[1]],
        ],
        [r"\tau_{\rm NL}", r"\sigma_{\rm NL}"],
    )


def delay_names(delay_model=None):
    return [row[0] for row in delay_param_spec(delay_model)[0]]


def delay_fiducial(delay_model=None):
    return tuple(float(row[1]) for row in delay_param_spec(delay_model)[0])


def format_delay_values(delay, fmt="%.6f", sep="  ", delay_model=None):
    names = delay_names(delay_model)
    if not names:
        return "SFH"
    return sep.join(
        "%s=%s" % (name, fmt % value)
        for name, value in zip(names, delay)
    )


def format_delay_tex(delay, delay_model=None):
    model = normalize_delay_model(delay_model)
    delay = tuple(float(x) for x in delay)
    if model == "sfh":
        return r"${\rm SFH}$"
    if model == "gauss":
        return (r"$\tau_{\rm gauss}=%.2f$, $\sigma_{\rm gauss}=%.2f$"
                % (delay[0], delay[1]))
    if model == "powerlaw":
        return r"$\alpha_{\tau}=%.2f$" % delay[0]
    return (r"$\tau_{\rm NL}=%.2f$, $\sigma_{\rm NL}=%.2f$"
            % (delay[0], delay[1]))


def analytic_reference_pz(z, alpha=REF_PZ_ALPHA, beta=REF_PZ_BETA):
    """Unit-normalised reference p(z) ∝ z^alpha * exp(-beta * z)."""
    z = np.asarray(z, dtype=float)
    p = np.where(z > 0.0, (z ** alpha) * np.exp(-beta * z), 0.0)
    norm = float(trapezoid(p, z))
    return p / norm if norm > 0.0 else p


def pdet_path():
    env = os.getenv("PDET_FILE", "").strip()
    candidates = (Path(env),) if env else PDET_CANDIDATES
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "CHIME P_det file not found; tried: "
        + ", ".join(str(p) for p in candidates))


# Default is Madau & Fragos (2017). Set SFH_YEAR=2014 for Madau & Dickinson.
_SFH_FITS = {
    2014: (2.7, 2.9, 5.6, 0.015, "Madau & Dickinson 2014"),
    2017: (2.6, 3.2, 6.2, 0.01, "Madau & Fragos 2017"),
}


def sfh_year():
    raw = os.getenv("SFH_YEAR", "2017").strip()
    try:
        year = int(raw)
    except ValueError:
        raise ValueError("unknown SFH_YEAR %r" % raw)
    if year not in _SFH_FITS:
        raise ValueError("unknown SFH_YEAR %r; expected 2014 or 2017" % raw)
    return year


def sfh_label(year=None):
    if year is None:
        year = sfh_year()
    return _SFH_FITS[year][4]


def R_SFH(z, a=None, b=None, c=None, psi_0=None):
    """Cosmic SFR density [M_sun yr^-1 Mpc^-3].

    Defaults follow SFH_YEAR (2017 or 2014). Explicit a, b, c, psi_0 override.
    """
    a0, b0, c0, psi0, _label = _SFH_FITS[sfh_year()]
    if a is None:
        a = a0
    if b is None:
        b = b0
    if c is None:
        c = c0
    if psi_0 is None:
        psi_0 = psi0
    z = np.asarray(z, dtype=float)
    return psi_0 * (1.0 + z) ** a / (1.0 + ((1.0 + z) / b) ** c)


def P_lognormal_delay(tau, tau_nl, sigma_nl):
    """Lognormal P(tau) dtau; tau/tau_nl are Gyr, sigma_nl is ln-width."""
    tau = np.asarray(tau, dtype=float)
    out = np.zeros_like(tau, dtype=float)
    ok = tau > 0.0
    t = tau[ok]
    out[ok] = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_nl * t)
               * np.exp(-0.5 * ((np.log(t) - np.log(tau_nl)) / sigma_nl) ** 2))
    return out


def P_gaussian_delay(tau, tau_0, sigma):
    """Gaussian delay PDF, truncated to tau > 0 and renormalised."""
    tau = np.asarray(tau, dtype=float)
    tau_0 = float(tau_0)
    sigma = float(sigma)
    out = np.zeros_like(tau, dtype=float)
    surv = 0.5 * (1.0 + erf(tau_0 / (sigma * np.sqrt(2.0))))
    if surv <= 0.0:
        return out
    ok = tau > 0.0
    out[ok] = (
        np.exp(-0.5 * ((tau[ok] - tau_0) / sigma) ** 2)
        / (sigma * np.sqrt(2.0 * np.pi) * surv)
    )
    return out


def P_powerlaw_delay(tau, alpha, tau_min=TAU_PL_MIN, tau_max=None):
    """Power-law delay PDF on [tau_min, tau_max] Gyr. alpha=1 uses the log limit.

    tau_max must be the formation-grid t_max (lookback to z_f=100), the same
    upper limit used by the gauss and lognormal delay grids.
    """
    if tau_max is None:
        raise TypeError("tau_max is required (use engine.t_max)")
    tau = np.asarray(tau, dtype=float)
    alpha = float(alpha)
    tau_max = float(tau_max)
    out = np.zeros_like(tau, dtype=float)
    ok = (tau >= tau_min) & (tau <= tau_max)
    if np.isclose(alpha, 1.0):
        out[ok] = 1.0 / (tau[ok] * np.log(tau_max / tau_min))
    else:
        c = (1.0 - alpha) / (tau_max ** (1.0 - alpha) - tau_min ** (1.0 - alpha))
        out[ok] = c * tau[ok] ** (-alpha)
    return out


class SFRLognormalSelectedNz:
    """Vectorized selected n(z); formation grid to z=100 like gauss/power-law."""

    def __init__(self, cosmo, z, n_tau=400):
        self.z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + np.clip(self.z, 0.0, None))
        self.t_L = np.asarray(ccl.lookback_time(cosmo, a), dtype=float)

        z_obs_max = max(float(self.z.max()), 5.0)
        z_form = np.unique(np.concatenate([
            np.array([0.0]),
            self.z[self.z > 0.0],
            np.geomspace(z_obs_max, Z_FORMATION_MAX, N_FORMATION_TAIL),
        ]))
        t_map = np.asarray(
            ccl.lookback_time(cosmo, 1.0 / (1.0 + z_form)), dtype=float)
        order = np.argsort(t_map)
        t_map = t_map[order]
        z_form = z_form[order]
        self.t_max = float(t_map[-1])
        self._t2z = PchipInterpolator(t_map, z_form, extrapolate=False)
        self._t_map_lo = float(t_map[0])
        self._t_map_hi = float(t_map[-1])

        self.tau = np.logspace(np.log10(TAU_MIN_GYR), np.log10(self.t_max),
                               int(n_tau))
        t_form = self.t_L[:, None] + self.tau[None, :]
        valid = t_form <= self.t_max
        t_clip = np.clip(t_form, self._t_map_lo, self._t_map_hi)
        z_at_form = self._t2z(t_clip)
        self.R_sfh_form = np.where(valid, R_SFH(z_at_form), 0.0)

        d_c = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float)
        h_z = np.asarray(ccl.h_over_h0(cosmo, a), dtype=float)
        self.vol = np.where(self.z > 0.0, d_c ** 2 / (h_z * (1.0 + self.z)), 0.0)

        data = np.load(pdet_path())
        pdet_z = interp1d(data["z"], data["p_raw"], bounds_error=False,
                          fill_value=0.0)
        self.pdet = np.clip(pdet_z(self.z), 0.0, None)
        self._cache = {}

    def selected_nz(self, tau_nl, sigma_nl):
        tau_nl = float(tau_nl)
        sigma_nl = float(sigma_nl)
        key = (round(tau_nl, 5), round(sigma_nl, 5))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        p_tau = P_lognormal_delay(self.tau, tau_nl, sigma_nl)
        rate = trapezoid(self.R_sfh_form * p_tau[None, :], self.tau, axis=1)
        nz = np.clip(self.vol * rate * self.pdet, 0.0, None)
        norm = float(trapezoid(nz, self.z))
        if norm > 0.0:
            nz = nz / norm
        if len(self._cache) >= 256:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = nz
        return nz


class SFRSelectedNz:
    """SFR x P_det with no delay convolution."""

    def __init__(self, cosmo, z):
        self.delay_model = "sfh"
        self.z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + np.clip(self.z, 0.0, None))
        d_c = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float)
        h_z = np.asarray(ccl.h_over_h0(cosmo, a), dtype=float)
        self.vol = np.where(self.z > 0.0, d_c ** 2 / (h_z * (1.0 + self.z)), 0.0)
        data = np.load(pdet_path())
        pdet_z = interp1d(data["z"], data["p_raw"], bounds_error=False,
                          fill_value=0.0)
        self.pdet = np.clip(pdet_z(self.z), 0.0, None)
        self._nz = None

    def selected_nz(self, *delay_params):
        if self._nz is not None:
            return self._nz
        nz = np.clip(self.vol * R_SFH(self.z) * self.pdet, 0.0, None)
        norm = float(trapezoid(nz, self.z))
        if norm > 0.0:
            nz = nz / norm
        self._nz = nz
        return nz


class SFRDelaySelectedNz:
    """Selected n(z) for Gaussian / power-law delay, formation grid to z=100."""

    def __init__(self, cosmo, z, delay_model, n_tau=None):
        self.delay_model = normalize_delay_model(delay_model)
        if self.delay_model not in ("gauss", "powerlaw"):
            raise ValueError(
                "SFRDelaySelectedNz is for gauss/powerlaw, got %s"
                % self.delay_model)
        self.z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + np.clip(self.z, 0.0, None))
        self.t_L = np.asarray(ccl.lookback_time(cosmo, a), dtype=float)

        z_obs_max = max(float(self.z.max()), 5.0)
        z_form = np.unique(np.concatenate([
            np.array([0.0]),
            self.z[self.z > 0.0],
            np.geomspace(z_obs_max, Z_FORMATION_MAX, N_FORMATION_TAIL),
        ]))
        t_map = np.asarray(
            ccl.lookback_time(cosmo, 1.0 / (1.0 + z_form)), dtype=float)
        order = np.argsort(t_map)
        t_map = t_map[order]
        z_form = z_form[order]
        self.t_max = float(t_map[-1])
        self._t2z = PchipInterpolator(t_map, z_form, extrapolate=False)
        self._t_map_lo = float(t_map[0])
        self._t_map_hi = float(t_map[-1])

        if self.delay_model == "gauss":
            n_tau = N_TAU_GAUSS if n_tau is None else int(n_tau)
            self.tau = np.linspace(TAU_MIN_GYR, self.t_max, n_tau)
        else:
            n_tau = N_TAU_POWERLAW if n_tau is None else int(n_tau)
            self.tau = np.logspace(
                np.log10(TAU_PL_MIN), np.log10(self.t_max), n_tau)

        t_form = self.t_L[:, None] + self.tau[None, :]
        valid = t_form <= self.t_max
        t_clip = np.clip(t_form, self._t_map_lo, self._t_map_hi)
        z_at_form = self._t2z(t_clip)
        self.R_sfh_form = np.where(valid, R_SFH(z_at_form), 0.0)

        d_c = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float)
        h_z = np.asarray(ccl.h_over_h0(cosmo, a), dtype=float)
        self.vol = np.where(self.z > 0.0, d_c ** 2 / (h_z * (1.0 + self.z)), 0.0)

        data = np.load(pdet_path())
        pdet_z = interp1d(data["z"], data["p_raw"], bounds_error=False,
                          fill_value=0.0)
        self.pdet = np.clip(pdet_z(self.z), 0.0, None)
        self._cache = {}

    def selected_nz(self, *delay_params):
        if self.delay_model == "gauss":
            tau_0, sigma = float(delay_params[0]), float(delay_params[1])
            key = ("g", round(tau_0, 5), round(sigma, 5))
            p_tau = P_gaussian_delay(self.tau, tau_0, sigma)
        else:
            alpha = float(delay_params[0])
            key = ("pl", round(alpha, 5))
            p_tau = P_powerlaw_delay(self.tau, alpha, tau_max=self.t_max)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rate = trapezoid(self.R_sfh_form * p_tau[None, :], self.tau, axis=1)
        nz = np.clip(self.vol * rate * self.pdet, 0.0, None)
        norm = float(trapezoid(nz, self.z))
        if norm > 0.0:
            nz = nz / norm
        if len(self._cache) >= 256:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = nz
        return nz


_ENGINE = None
_ENGINES = {}


def _engine_cache_key(z, delay_model, year):
    z = np.asarray(z, dtype=float)
    return (delay_model, int(year), z.shape, z.tobytes())


def get_engine(cosmo, z, delay_model=None):
    """Return a selected-n(z) engine on this redshift grid.

    Joint fits keep two grids (uniform 500 for C_ell, piecewise 147 for
    DM–z). Engines are cached so switching grids does not rebuild the
    formation table every likelihood call. Do not interpolate p_sel
    between grids; call this with the native z of each integral.
    """
    global _ENGINE
    delay_model = normalize_delay_model(delay_model)
    z = np.asarray(z, dtype=float)
    year = sfh_year()
    key = _engine_cache_key(z, delay_model, year)
    engine = _ENGINES.get(key)
    if engine is None:
        if delay_model == "lognormal":
            engine = SFRLognormalSelectedNz(cosmo, z)
            engine.delay_model = "lognormal"
        elif delay_model == "sfh":
            engine = SFRSelectedNz(cosmo, z)
        else:
            engine = SFRDelaySelectedNz(cosmo, z, delay_model=delay_model)
        engine.sfh_year = year
        _ENGINES[key] = engine
    _ENGINE = engine
    return engine


def unit_norm(z, p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    norm = float(trapezoid(p, z))
    return p / norm if norm > 0.0 else p


def _delay_pdf(engine, params):
    model = getattr(engine, "delay_model", "lognormal")
    if model == "gauss":
        return P_gaussian_delay(engine.tau, *params)
    if model == "powerlaw":
        return P_powerlaw_delay(engine.tau, *params, tau_max=engine.t_max)
    return P_lognormal_delay(engine.tau, *params)


def engine_intrinsic_p(engine, *delay_params):
    """Unit-normalised intrinsic p(z) (no P_det) on the engine grid."""
    model = getattr(engine, "delay_model", "lognormal")
    if model == "sfh":
        return unit_norm(engine.z, engine.vol * R_SFH(engine.z))
    p_tau = _delay_pdf(engine, delay_params)
    rate = trapezoid(engine.R_sfh_form * p_tau[None, :], engine.tau, axis=1)
    return unit_norm(engine.z, engine.vol * rate)


def selected_p(cosmo, z, *delay_params):
    """Unit-normalised selected p(z) on the requested redshift grid.

    Always native on ``z`` (cached engine per grid). Do not interpolate
    a coarse p_sel onto a denser low-z mesh.
    """
    z = np.asarray(z, dtype=float)
    engine = get_engine(cosmo, z)
    if engine.z.shape == z.shape and np.allclose(engine.z, z):
        return engine.selected_nz(*delay_params)
    p = np.interp(z, engine.z, engine.selected_nz(*delay_params),
                  left=0.0, right=0.0)
    return unit_norm(z, p)


def intrinsic_p(cosmo, z, *delay_params):
    """Unit-normalised intrinsic p(z) (no P_det) on the requested grid."""
    z = np.asarray(z, dtype=float)
    engine = get_engine(cosmo, z)
    p = engine_intrinsic_p(engine, *delay_params)
    if engine.z.shape == z.shape and np.allclose(engine.z, z):
        return p
    p = np.interp(z, engine.z, p, left=0.0, right=0.0)
    return unit_norm(z, p)
