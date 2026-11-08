# -*- coding: utf-8 -*-
"""FRB redshift MCMC: C_ell (galaxy×DM + DM×DM), DM–z, or the joint fit.

Modes (MCMC_MODE or --mode):
  cl     — C_ell cross + auto; free DM_host
  dmz    — P_IGM ⊗ host; default Macquart p_Δ with free mu_host, sigma_host, F
  joint  — both; C_ell host mean from (mu_host, sigma_host)

IGM_MODEL selects the IGM PDF for dmz/joint:
  macquart  — Macquart p_Δ (default); free F; lognormal host
  lognormal — lognormal IGM, A_IGM=230 fixed; free mu_host, sigma_host
  gauss     — truncated-Gaussian IGM ⊗ truncated-Gaussian host (analytic); free dm_host, sigma_dm_host

Production dmz/joint keep DM >= 5 on FFT 2000×4000 and make_z_grid()
(50+20+20+60). Macquart adds the one DM < 5 source on
linspace(0.005, 0.008, 61) with a per-source [0, DM_i] trapezoid.
C_ell keeps linspace(0, 5, 500).

DELAY_MODEL selects lognormal, gauss, power-law delay, or SFH.
Cl theory (cl/joint, emcee and dynesty, every delay model) uses NaMaster
bandpower windows. MCMC_CL_BANDPOWER=0 falls back to ell_eff.
Default SFRD is Madau & Fragos 2017.
"""

import os
import re
import hashlib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import numpy as np
import pyccl as ccl
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cl_dm_catalog as clcat
import dm_z as dmz
from frb_redshift_model import (
    DELAY_MODEL, REF_PZ_ALPHA, REF_PZ_BETA, TAU_PL_MIN,
    analytic_reference_pz, delay_fiducial, delay_names, delay_param_spec,
    format_delay_values, get_engine, intrinsic_p,
    normalize_delay_model, pdet_path, selected_p, sfh_label,
)
from mymcmc import MCMC_class, MCplot, set_base_dir
from mymcmc.mymcmc import read_burn_in_record

F_IGM = clcat.F_IGM
X_E = clcat.X_E
LMAX = clcat.LMAX
PLOT_ELL_MIN = clcat.PLOT_ELL_MIN
CROSS_SCALE_POWER = clcat.CROSS_SCALE_POWER
DM_HOST_SOURCE = clcat.DM_HOST_SOURCE
DM_MW_RESIDUAL = clcat.DM_MW_RESIDUAL

def _env_flag(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _env_optional_flag(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in ("1", "true", "yes")


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    return float(value)


_MODE_ALIASES = {
    "cl": "cl", "cross": "cl", "c_ell": "cl", "cell": "cl",
    "dmz": "dmz", "dm-z": "dmz", "dm_z": "dmz",
    "joint": "joint", "combined": "joint",
}


def normalize_mcmc_mode(name=None):
    if name is None or str(name).strip() == "":
        name = os.getenv("MCMC_MODE", "cl")
    key = str(name).strip().lower().replace("_", "-")
    if key not in _MODE_ALIASES:
        raise ValueError("unknown MCMC_MODE %r" % name)
    return _MODE_ALIASES[key]


def delay_model_dirname(delay_model=None):
    model = normalize_delay_model(delay_model)
    return {
        "lognormal": "lognormal", "gauss": "gauss",
        "powerlaw": "power-law", "sfh": "SFH",
    }[model]


def results_root():
    """MCMC output tree. The angular-spectrum input is configured separately with CL_DATA_FILE."""
    raw = os.getenv("RUN_ROOT", "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return ROOT / "runs"


def igm_run_dirname(delay_model=None, igm_model=None):
    """Disk folder under the output root/{mode}/. Non-macquart IGM uses dedicated names."""
    delay_dir = delay_model_dirname(delay_model)
    igm = dmz.normalize_igm_model(igm_model)
    if igm == "macquart":
        return delay_dir
    if delay_dir == "lognormal" and igm == "lognormal":
        return "lognormal_igm_ln"
    if delay_dir == "lognormal" and igm == "gauss":
        return "lognormal_igm_gauss"
    return "%s_igm_%s" % (delay_dir, igm)


def chain_stem(mode=None):
    if mode is None or str(mode).strip() == "":
        mode = MCMC_MODE
    return {"cl": "frb_redshift", "dmz": "frb_dmz", "joint": "frb_joint"}[
        normalize_mcmc_mode(mode)]


FIT_ELL_LOC = _env_flag("MCMC_FIT_ELL_LOC", False)
FIT_K_CUT = _env_flag("MCMC_FIT_K_CUT", True)
FIT_N_ELL = False
FIT_DM_HOST = _env_flag("MCMC_FIT_DM_HOST", True)
FIT_B_FRB = _env_flag("MCMC_FIT_B_FRB", True)
FIT_BETA_FRB = _env_flag("MCMC_FIT_BETA_FRB", False)
FIT_B_E = _env_flag("MCMC_FIT_B_E", False)
FIT_HOST_LN = False
FIT_F = False
FIT_HOST_GAUSS = False
IGM_MODEL = "macquart"
USE_CL_BANDPOWER = False
CL_BANDPOWER_NELL = int(os.getenv("MCMC_CL_BANDPOWER_NELL", "24") or 24)
CL_BANDPOWER_GRID = os.getenv("MCMC_CL_BANDPOWER_GRID", "log").strip().lower()
USE_CL = True
USE_DMZ = False
MCMC_MODE = "cl"
ELL_LOC_PRIOR = (100, 2000)
K_CUT_PRIOR = (0.4, 10.0)
N_ELL_PRIOR = (0.0, 5.0)
N_ELL_START = 1.0
DM_HOST_PRIOR = (50.0, 250.0)  # cl / gauss-host; lower bound 50 (not 40)
B_FRB_PRIOR = (0.5, 3.0)
B_E_PRIOR = (0.5, 2.0)
B_FRB_START = 1.0
B_E_START = 1.0
BETA_FRB_PRIOR = (-5.0, 5.0)
BETA_FRB_START = 0.0
BETA_FRB_FIXED = 0.0
MU_HOST_PRIOR = (3.0, 6.0)
SIGMA_HOST_PRIOR = (0.2, 1.5)
MU_HOST_START = 4.5
SIGMA_HOST_START = 1.0
SIGMA_DM_HOST_PRIOR = (10.0, 150.0)
SIGMA_DM_HOST_START = 50.0
F_PRIOR = (0.03, 0.5)
F_START = 0.3
DELAY_PARAMS, DELAY_PLOT_NAMES = delay_param_spec()
N_DELAY = len(DELAY_PARAMS)
DELAY_FID = delay_fiducial()
MCMC_STEPS = int(os.getenv("MCMC_STEPS", "50000"))
CHAIN_BURN_IN_FRAC = float(os.getenv("MCMC_BURN_IN_FRAC", "0.3") or 0.3)
N_PZ_DRAW = 1000

COSMO = PEG = PEE = ELL_THEORY = ELL_SPARSE = Z = Z_DM = DDM_DZ = B_ELECTRON = None
COSMIC_DM = COSMIC_DM_DMZ = MODEL_ENTRIES = DATA_VECTOR = INV_COV_JOINT = None
ELL_LOC_FIXED = K_CUT_FIXED = N_ELL_FIXED = D_BAR = None
DM_HOST_FIXED = B_FRB_FIXED = B_E_FIXED = None
MU_HOST_FIXED = MU_HOST_START
SIGMA_HOST_FIXED = SIGMA_HOST_START
F_FIXED = _env_float("MCMC_F_FIXED", F_START)
DM_OBS = None
DM_TINY = None
Z_TINY_COMP = None
COSMIC_TINY_COMP = None
N_FRB = 0
DM_GRID = None
_PEG_CACHE = {}
_PEE_CACHE = {}


def resolve_cl_bandpower():
    """Windowed Cl theory for every cl/joint run.

    MCMC_CL_BANDPOWER=1/0 overrides. No Cl in dmz.
    """
    forced = _env_optional_flag("MCMC_CL_BANDPOWER")
    if forced is not None:
        return bool(forced) and USE_CL
    return bool(USE_CL)


def configure_mode(mode=None):
    global MCMC_MODE, USE_CL, USE_DMZ, FIT_HOST_LN, FIT_F, FIT_HOST_GAUSS
    global FIT_DM_HOST, FIT_B_FRB, FIT_BETA_FRB, FIT_K_CUT, FIT_N_ELL
    global FIT_ELL_LOC, FIT_B_E
    global USE_CL_BANDPOWER, IGM_MODEL
    MCMC_MODE = normalize_mcmc_mode(mode)
    USE_CL = MCMC_MODE in ("cl", "joint")
    USE_DMZ = MCMC_MODE in ("dmz", "joint")
    IGM_MODEL = dmz.normalize_igm_model()
    FIT_HOST_LN = USE_DMZ
    FIT_F = _env_flag("MCMC_FIT_F", USE_DMZ)
    FIT_HOST_GAUSS = False
    FIT_N_ELL = False
    if MCMC_MODE == "cl":
        FIT_DM_HOST = _env_flag("MCMC_FIT_DM_HOST", True)
        FIT_B_FRB = _env_flag("MCMC_FIT_B_FRB", True)
        FIT_BETA_FRB = _env_flag("MCMC_FIT_BETA_FRB", False)
        FIT_K_CUT = _env_flag("MCMC_FIT_K_CUT", True)
        FIT_ELL_LOC = _env_flag("MCMC_FIT_ELL_LOC", False)
        FIT_B_E = _env_flag("MCMC_FIT_B_E", False)
        IGM_MODEL = "macquart"
    elif MCMC_MODE == "dmz":
        FIT_DM_HOST = False
        FIT_B_FRB = False
        FIT_BETA_FRB = False
        FIT_K_CUT = False
        FIT_N_ELL = False
        FIT_ELL_LOC = False
        FIT_B_E = False
    else:
        FIT_DM_HOST = False
        FIT_B_FRB = _env_flag("MCMC_FIT_B_FRB", True)
        FIT_BETA_FRB = _env_flag("MCMC_FIT_BETA_FRB", False)
        FIT_K_CUT = _env_flag("MCMC_FIT_K_CUT", True)
        FIT_ELL_LOC = _env_flag("MCMC_FIT_ELL_LOC", False)
        FIT_B_E = _env_flag("MCMC_FIT_B_E", False)
    if USE_DMZ and IGM_MODEL == "lognormal":
        FIT_HOST_LN = True
        FIT_F = False
        FIT_HOST_GAUSS = False
    elif USE_DMZ and IGM_MODEL == "gauss":
        FIT_HOST_LN = False
        FIT_F = False
        FIT_HOST_GAUSS = True
        FIT_DM_HOST = False
    USE_CL_BANDPOWER = resolve_cl_bandpower()
    return MCMC_MODE


configure_mode()


def frb_source_p(*delay):
    return selected_p(COSMO, Z, *delay)


def cl_dm_host_value(mu_host, sigma_host, dm_host):
    if FIT_HOST_LN:
        return dmz.dm_host_mean(mu_host, sigma_host)
    if FIT_HOST_GAUSS:
        return dmz.dm_host_gauss_mean(dm_host, sigma_host)
    return float(dm_host)


def dm_tracer(delay=None, p=None, b_e=None):
    if p is None:
        if delay is None:
            delay = ()
        p = frb_source_p(*delay)
    else:
        p = np.clip(np.asarray(p, dtype=float), 0.0, None)
        norm = float(trapezoid(p, Z))
        p = p / norm if norm > 0.0 else p
    if b_e is None:
        b_e = B_E_FIXED if B_E_FIXED is not None else B_ELECTRON
    tail = -cumulative_trapezoid(p[::-1], Z[::-1], initial=0.0)[::-1]
    kernel = DDM_DZ * tail
    kernel_norm = trapezoid(kernel, Z)
    tracer = ccl.NumberCountsTracer(
        COSMO, dndz=(Z, kernel),
        bias=(Z, np.full_like(Z, float(b_e) * kernel_norm)), has_rsd=False)
    return tracer, p


def frb_dm_model(dm_host):
    return DM_MW_RESIDUAL + COSMIC_DM + float(dm_host) / (1.0 + Z)


def frb_source_tracer(p, b_frb, dm_host=None, beta_frb=None):
    if dm_host is None:
        dm_host = DM_HOST_FIXED
    if beta_frb is None:
        beta_frb = BETA_FRB_FIXED
    amp = float(b_frb) * np.power(1.0 + Z, float(beta_frb))
    return ccl.NumberCountsTracer(
        COSMO, dndz=(Z, p),
        bias=(Z, amp * (frb_dm_model(dm_host) - D_BAR)), has_rsd=False)


def unpack_theta(theta):
    delay = tuple(float(x) for x in theta[:N_DELAY])
    i = N_DELAY
    mu_host, sigma_host = MU_HOST_FIXED, SIGMA_HOST_FIXED
    F = F_FIXED
    b_frb = B_FRB_FIXED
    beta_frb = BETA_FRB_FIXED
    b_e = B_E_FIXED
    ell_loc, k_cut = ELL_LOC_FIXED, K_CUT_FIXED
    n_ell = N_ELL_FIXED
    dm_host = DM_HOST_FIXED
    if FIT_HOST_LN:
        mu_host = float(theta[i])
        sigma_host = float(theta[i + 1])
        i += 2
    if FIT_HOST_GAUSS:
        dm_host = float(theta[i])
        sigma_host = float(theta[i + 1])
        i += 2
    if FIT_F:
        F = float(theta[i])
        i += 1
    if FIT_B_FRB:
        b_frb = float(theta[i])
        i += 1
    if FIT_BETA_FRB:
        beta_frb = float(theta[i])
        i += 1
    if FIT_B_E:
        b_e = float(theta[i])
        i += 1
    if FIT_ELL_LOC:
        ell_loc = float(theta[i])
        i += 1
    if FIT_K_CUT:
        k_cut = float(theta[i])
        i += 1
    if FIT_N_ELL:
        n_ell = float(theta[i])
        i += 1
    if FIT_DM_HOST:
        dm_host = float(theta[i])
    return (
        delay, mu_host, sigma_host, F, b_frb, beta_frb, b_e, ell_loc, k_cut,
        n_ell, dm_host)


def build_theta(delay, b_frb=None, ell_loc=None, k_cut=None,
                n_ell=None, dm_host=None, b_e=None,
                mu_host=None, sigma_host=None, F=None, sigma_dm_host=None,
                beta_frb=None):
    theta = [float(x) for x in delay]
    if FIT_HOST_LN:
        theta.append(MU_HOST_FIXED if mu_host is None else float(mu_host))
        theta.append(SIGMA_HOST_FIXED if sigma_host is None else float(sigma_host))
    if FIT_HOST_GAUSS:
        theta.append(DM_HOST_FIXED if dm_host is None else float(dm_host))
        theta.append(SIGMA_DM_HOST_START if sigma_dm_host is None else float(sigma_dm_host))
    if FIT_F:
        theta.append(F_FIXED if F is None else float(F))
    if FIT_B_FRB:
        theta.append(B_FRB_FIXED if b_frb is None else float(b_frb))
    if FIT_BETA_FRB:
        theta.append(BETA_FRB_FIXED if beta_frb is None else float(beta_frb))
    if FIT_B_E:
        theta.append(B_E_FIXED if b_e is None else float(b_e))
    if FIT_ELL_LOC:
        theta.append(ELL_LOC_FIXED if ell_loc is None else float(ell_loc))
    if FIT_K_CUT:
        theta.append(K_CUT_FIXED if k_cut is None else float(k_cut))
    if FIT_N_ELL:
        theta.append(N_ELL_START if n_ell is None else float(n_ell))
    if FIT_DM_HOST:
        theta.append(DM_HOST_FIXED if dm_host is None else float(dm_host))
    return theta


def peg_power(k_cut):
    if (not FIT_K_CUT) or abs(k_cut - K_CUT_FIXED) < 1e-12:
        return PEG
    key = round(float(k_cut), 4)
    pk = _PEG_CACHE.get(key)
    if pk is None:
        pk = clcat.make_peg_power(COSMO, key)
        if len(_PEG_CACHE) >= 32:
            _PEG_CACHE.pop(next(iter(_PEG_CACHE)))
        _PEG_CACHE[key] = pk
    return pk


def pee_power(k_cut):
    if (not FIT_K_CUT) or abs(k_cut - K_CUT_FIXED) < 1e-12:
        return PEE
    key = round(float(k_cut), 4)
    pk = _PEE_CACHE.get(key)
    if pk is None:
        pk = clcat.make_pee_power(COSMO, key)
        if len(_PEE_CACHE) >= 32:
            _PEE_CACHE.pop(next(iter(_PEE_CACHE)))
        _PEE_CACHE[key] = pk
    return pk


def cl_bandpower_ell_sparse(ell_theory):
    """Sparse ell grid for CCL; results are interpolated back to ell_theory."""
    ell_max = float(np.asarray(ell_theory, dtype=float)[-1])
    n = max(8, int(CL_BANDPOWER_NELL))
    if CL_BANDPOWER_GRID == "linear":
        return np.linspace(2.0, ell_max, n)
    return np.geomspace(2.0, ell_max, n)


def upsample_cl(ell_sparse, cl_sparse, ell_full):
    """PCHIP in ell; Cl=0 below the first sparse ell (Limber starts at 2)."""
    ell_sparse = np.asarray(ell_sparse, dtype=float)
    cl_sparse = np.asarray(cl_sparse, dtype=float)
    ell_full = np.asarray(ell_full, dtype=float)
    out = np.zeros_like(ell_full, dtype=float)
    lo = float(ell_sparse[0])
    ok = ell_full >= lo
    out[ok] = PchipInterpolator(ell_sparse, cl_sparse, extrapolate=False)(
        ell_full[ok])
    return out


def model_vector(theta, p=None):
    """Return theory matching the data vector.

    Default: evaluate at each band's ell_eff. With MCMC_CL_BANDPOWER=1,
    evaluate on a sparse ell grid, PCHIP to ell_theory, then apply windows.
    """
    (delay, mu_host, sigma_host, F, b_frb, beta_frb, b_e, ell_loc, k_cut,
     n_ell, dm_host) = unpack_theta(theta)
    dm_host = cl_dm_host_value(mu_host, sigma_host, dm_host)
    dm, p = dm_tracer(delay, p=p, b_e=b_e)
    frb = frb_source_tracer(p, b_frb, dm_host, beta_frb=beta_frb)
    peg = peg_power(k_cut)
    pee = pee_power(k_cut) if FIT_N_ELL else None
    model = []
    for entry in MODEL_ENTRIES:
        if USE_CL_BANDPOWER:
            ell = ELL_SPARSE
        else:
            ell = entry["ell"]
        beam = clcat.localization_beam(ell, ell_loc)
        if entry.get("kind") == "dm_auto":
            cl_ee = ccl.angular_cl(COSMO, dm, dm, ell, p_of_k_a=pee)
            cl_ef = ccl.angular_cl(COSMO, dm, frb, ell, p_of_k_a=peg)
            cl_ff = ccl.angular_cl(COSMO, frb, frb, ell)
            dense = (cl_ee + 2.0 * cl_ef + cl_ff) * beam ** 2
            if USE_CL_BANDPOWER:
                dense = upsample_cl(ell, dense, ELL_THEORY)
                model.append(entry["window"] @ dense + n_ell)
            else:
                model.append(dense + n_ell)
        else:
            dense = (
                ccl.angular_cl(COSMO, entry["tracer"], dm, ell, p_of_k_a=peg)
                + ccl.angular_cl(COSMO, entry["tracer"], frb, ell)
            ) * beam
            if USE_CL_BANDPOWER:
                dense = upsample_cl(ell, dense, ELL_THEORY)
                model.append(entry["window"] @ dense)
            else:
                model.append(dense)
    return np.concatenate(model)


def chi2_cl(theta, p=None):
    diff = DATA_VECTOR - model_vector(theta, p=p)
    return float(diff @ INV_COV_JOINT @ diff)


def _dmz_z_cosmic():
    """Native DM–z quadrature grid and 〈DM_IGM〉. Not the C_ell Z."""
    z_use = Z_DM if Z_DM is not None else Z
    cosmic = COSMIC_DM_DMZ if COSMIC_DM_DMZ is not None else COSMIC_DM
    return z_use, cosmic


def chi2_dmz(theta, p=None):
    delay, mu_host, sigma_host, F, *_rest = unpack_theta(theta)
    dm_host = _rest[-1]
    z_use, cosmic = _dmz_z_cosmic()
    p_use = selected_p(COSMO, z_use, *delay)
    if IGM_MODEL == "gauss":
        log_l, _ = dmz.log_marginal_dm_likelihood(
            z_use, p_use, DM_OBS, cosmic, dm_host, sigma_host, F,
            dm_mw=DM_MW_RESIDUAL, igm_model="gauss")
    else:
        log_l, _ = dmz.log_marginal_dm_likelihood(
            z_use, p_use, DM_OBS, cosmic, mu_host, sigma_host, F,
            dm_mw=DM_MW_RESIDUAL, method="fft", dm_grid=DM_GRID,
            igm_model=IGM_MODEL)
        if IGM_MODEL == "macquart" and DM_TINY is not None and DM_TINY.size:
            p_comp = selected_p(COSMO, Z_TINY_COMP, *delay)
            log_l += dmz.log_tiny_macquart_likelihood(
                Z_TINY_COMP, p_comp, DM_TINY, COSMIC_TINY_COMP,
                F, mu_host, sigma_host)
    return -2.0 * log_l


def chi2_parts(theta, p=None):
    delay = unpack_theta(theta)[0]
    if USE_CL:
        if p is None:
            p = frb_source_p(*delay)
        chi_cl = chi2_cl(theta, p=p)
    else:
        chi_cl = 0.0
    chi_dm = chi2_dmz(theta) if USE_DMZ else 0.0
    return chi_cl, chi_dm, chi_cl + chi_dm


def chi2(theta, p=None):
    return chi2_parts(theta, p=p)[2]


def applied_burn_in(out_dir):
    """Return (frac, n_burn, n_steps) actually used to flatten the chain."""
    rec = read_burn_in_record(str(out_dir))
    if rec:
        return (
            float(rec.get("burn_in_frac", CHAIN_BURN_IN_FRAC)),
            int(rec.get("burn_in_steps_per_walker", 0) or 0),
            int(rec.get("n_steps", 0) or 0),
        )
    return float(CHAIN_BURN_IN_FRAC), 0, 0


def last_run_text(text):
    matches = list(re.finditer(r"^={10,}\nRun: ", text, re.M))
    return text[matches[-1].start():] if matches else text


def parse_param(text, name, default=None):
    hits = re.findall(r"^\s*%s\s*=\s*([0-9.]+)" % re.escape(name), text, re.M)
    if hits:
        return float(hits[-1])
    if default is None:
        raise KeyError("parameter %s not found in results.txt" % name)
    return float(default)


def bin_z_range(row):
    """Return (zmin, zmax). LRG rows are (pz_bin, zmin, zmax)."""
    row = np.asarray(row, dtype=float).ravel()
    if row.size >= 3:
        return float(row[-2]), float(row[-1])
    return float(row[0]), float(row[1])


def format_bin_title(sample, z0, z1, closing="]"):
    if sample == "DM":
        return r"FRB ${\rm DM}\times{\rm DM}$"
    if sample == "LRG":
        return rf"{sample}: $z \in [{z0:.2f}, {z1:.2f}{closing}$"
    return rf"{sample}: $z \in [{z0:.1f}, {z1:.1f}{closing}$"


def format_bin_short(sample, z0, z1):
    if sample == "DM":
        return r"DM$\times$DM"
    if sample == "LRG":
        return rf"{sample} ${z0:.2f}$–${z1:.2f}$"
    return rf"{sample} ${z0:.1f}$–${z1:.1f}$"


def catalog_npz_path():
    path = Path(os.getenv("CL_DATA_FILE", "data/angular power spectrum/cl_dm_catalog.npz"))
    return path if path.is_absolute() else ROOT / path


def _sha256_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path).resolve()
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_manifest_payload(data_file, joint, parameter_names):
    """Identity of the data/model/selection represented by an MCMC chain."""
    import P_IGM
    model_files = (
        Path(__file__),
        Path(clcat.__file__),
        Path(selected_p.__code__.co_filename),
        Path(dmz.__file__),
        Path(P_IGM.__file__),
        ROOT / "cosmology.py",
    )
    payload = {
        "model_sha256": np.array(_sha256_files(model_files)),
        "pdet_sha256": np.array(_sha256_files([pdet_path()])),
        "pigm_sha256": np.array(_sha256_files([P_IGM.NPZ_PATH])),
        "frb_fits_sha256": np.array(_sha256_files([dmz.FRB_DATA_FILE])),
        "f_igm": np.array(float(F_IGM)),
        "likelihood_mode": np.array(MCMC_MODE),
        "mcmc_mode": np.array(MCMC_MODE),
        "delay_model": np.array(DELAY_MODEL),
        "burn_in_frac": np.array(CHAIN_BURN_IN_FRAC),
        "parameter_names": np.asarray(parameter_names, dtype="U32"),
        "fit_ell_loc": np.array(int(FIT_ELL_LOC)),
        "fit_k_cut": np.array(int(FIT_K_CUT)),
        "fit_n_ell": np.array(int(FIT_N_ELL)),
    }
    if DELAY_MODEL == "powerlaw":
        payload["tau_pl_min"] = np.array(float(TAU_PL_MIN))
    if FIT_BETA_FRB:
        payload["fit_beta_frb"] = np.array(1)
        payload["beta_frb_prior"] = np.asarray(BETA_FRB_PRIOR, dtype=float)
    if IGM_MODEL != "macquart":
        payload["igm_model"] = np.array(IGM_MODEL)
        payload["a_igm"] = np.array(float(dmz.A_IGM))
    elif USE_DMZ:
        n_bulk = int(dmz.load_extragalactic_dm()[0].size)
        n_tiny = int(dmz.load_tiny_extragalactic_dm()[0].size)
        payload["n_frb"] = np.array(n_bulk + n_tiny)
        payload["dm_tiny_grid"] = np.array(
            [dmz.TINY_Z_LO, dmz.TINY_Z_HI,
             float(dmz.N_TINY_Z), float(dmz.N_TINY_DM)])
    if USE_CL and data_file is not None and joint is not None:
        payload["npz_sha256"] = np.array(_sha256_files([data_file]))
        payload["selected_spectrum_indices"] = np.asarray(
            joint["spectrum_indices"], dtype=int)
        payload["selected_band_indices"] = np.asarray(
            joint["band_indices"], dtype=int)
        payload["likelihood_ell"] = np.asarray(joint["ell"], dtype=float)
    return payload


def ensure_run_manifest(out_dir, payload, create=True):
    """Refuse to mix an existing chain with different data or model settings."""
    out_dir = Path(out_dir)
    path = out_dir / "run_manifest.npz"
    backend = out_dir / ("temple/%s/backend.h5" % chain_stem())
    if not path.is_file():
        if backend.is_file():
            raise RuntimeError(
                f"{backend} exists without {path}; remove the old MCMC "
                "directory before resuming")
        if create:
            np.savez(path, **payload)
            return path
        raise FileNotFoundError(f"required MCMC run manifest not found: {path}")
    with np.load(path, allow_pickle=False) as stored:
        missing = sorted(set(payload) - set(stored.files))
        mismatched = [
            key for key, value in payload.items()
            if key in stored.files
            and not np.array_equal(np.asarray(stored[key]), np.asarray(value))
        ]
    if missing or mismatched:
        details = ", ".join(missing + mismatched)
        raise RuntimeError(
            "existing MCMC run manifest does not match the current "
            f"data/model/selection ({details}); start a fresh output directory")
    return path


def _keep_indices(n_bins, spec):
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_bins))
    if ":" in spec:
        start_s, _, end_s = spec.partition(":")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else n_bins
        idx = list(range(start, end))
    else:
        idx = [int(x) for x in spec.split(",") if x.strip() != ""]
    for i in idx:
        if i < 0 or i >= n_bins:
            raise ValueError("bin index %d out of range 0..%d" % (i, n_bins - 1))
    return idx


def _data_cut_specs():
    """Return per-sample tomographic selections from the MCMC environment."""
    return {
        "BGS": os.getenv("MCMC_BGS_KEEP", "0,1,2,3"),
        "LRG": os.getenv("MCMC_LRG_KEEP", ""),
    }


def load_cross_samples(data):
    """Load and synchronously cut spectra, windows, tracers, and joint covariance."""
    names = set(data.files)
    required = {
        "npz_schema_version", "ell", "ell_theory", "lmax",
        "cross_cl_joint", "cross_cov_joint", "cross_bandpower_windows",
        "cross_band_indices", "cross_band_ell", "cross_order_sample",
        "cross_order_sample_bin", "cross_order_z_min", "cross_order_z_max",
        "cross_order_label", "k_cut_mpc_inv",
    }
    missing = sorted(required - names)
    if missing:
        raise KeyError(
            "not a revised joint-covariance NPZ; missing: " + ", ".join(missing))
    schema = int(np.asarray(data["npz_schema_version"]).flat[0])
    if schema < 2:
        raise ValueError(f"NPZ schema {schema} is obsolete; expected >=2")

    ell_theory = np.asarray(data["ell_theory"], dtype=float)
    lmax = int(np.asarray(data["lmax"]).flat[0])
    if ell_theory.shape != (lmax + 1,) or not np.array_equal(
            ell_theory, np.arange(lmax + 1, dtype=float)):
        raise ValueError("ell_theory must be the dense grid 0..LMAX")

    order_sample = np.asarray(data["cross_order_sample"]).astype(str)
    order_bin = np.asarray(data["cross_order_sample_bin"], dtype=int)
    z_min = np.asarray(data["cross_order_z_min"], dtype=float)
    z_max = np.asarray(data["cross_order_z_max"], dtype=float)
    labels = np.asarray(data["cross_order_label"]).astype(str)
    band_indices = np.asarray(data["cross_band_indices"], dtype=int)
    band_ell = np.asarray(data["cross_band_ell"], dtype=float)
    windows = np.asarray(data["cross_bandpower_windows"], dtype=float)
    data_full = np.asarray(data["cross_cl_joint"], dtype=float)
    covariance_full = np.asarray(data["cross_cov_joint"], dtype=float)
    n_spectra = order_sample.size
    if n_spectra == 0:
        raise ValueError("revised NPZ contains no joint spectra")
    if not (
        order_bin.shape == z_min.shape == z_max.shape == labels.shape
        == (n_spectra,)
    ):
        raise ValueError("joint spectrum-order metadata has inconsistent shapes")
    if band_indices.ndim != 2:
        raise ValueError("cross_band_indices must have shape (spectrum, band)")
    n_band = band_indices.shape[1]
    if band_indices.shape[0] != n_spectra:
        raise ValueError("cross_band_indices does not match spectrum order")
    if windows.shape != (n_spectra, n_band, lmax + 1):
        raise ValueError("cross_bandpower_windows has an inconsistent shape")
    if band_ell.shape != (n_spectra, n_band):
        raise ValueError("cross_band_ell has an inconsistent shape")
    n_full = data_full.size
    if covariance_full.shape != (n_full, n_full):
        raise ValueError("cross_cov_joint has an inconsistent shape")
    if np.any(band_indices < 0) or np.any(band_indices >= n_full):
        raise ValueError("cross_band_indices contains out-of-range entries")

    keep_specs = _data_cut_specs()
    selected_spectra = []
    for sample in dict.fromkeys(order_sample.tolist()):
        positions = np.flatnonzero(order_sample == sample)
        local_keep = _keep_indices(len(positions), keep_specs.get(sample, ""))
        if not local_keep:
            print(f"{sample}: dropped all bins")
            continue
        chosen = positions[np.asarray(local_keep, dtype=int)]
        selected_spectra.extend(chosen.tolist())
        kept_ranges = ", ".join(
            "%.2f–%.2f" % (z_min[i], z_max[i]) for i in chosen)
        print(f"{sample}: keep bins {local_keep}  ({kept_ranges})")
    if not selected_spectra:
        raise ValueError("data cut removed every joint spectrum")
    selected_spectra = np.asarray(selected_spectra, dtype=int)
    selected_bands = band_indices[selected_spectra].reshape(-1)

    data_vector = data_full[selected_bands]
    covariance = covariance_full[np.ix_(selected_bands, selected_bands)]
    if not np.all(np.isfinite(covariance)):
        raise ValueError("selected joint covariance contains non-finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
        raise ValueError("selected joint covariance is not symmetric")
    if np.any(np.diag(covariance) <= 0.0):
        raise ValueError("selected joint covariance has a non-positive diagonal")
    inverse_covariance = np.linalg.inv(covariance)

    sample_info = {
        "BGS": {
            "prefix": "bgs", "bias": "b_bgs", "color": "mediumpurple",
            "fname": "cl_BGS_DM_bestfit.pdf",
        },
        "LRG": {
            "prefix": "lrg", "bias": "b_lrg", "color": "darkorange",
            "fname": "cl_LRG_DM_bestfit.pdf",
        },
        "DM": {
            "prefix": None, "bias": None, "color": "crimson",
            "fname": "cl_DM_DM_bestfit.pdf",
        },
    }
    entries = []
    for selected_index, original_index in enumerate(selected_spectra):
        sample = order_sample[original_index]
        if sample not in sample_info:
            raise ValueError(f"unsupported joint sample {sample!r}")
        info = sample_info[sample]
        sample_bin = int(order_bin[original_index])
        local_slice = slice(selected_index * n_band, (selected_index + 1) * n_band)
        original_bands = band_indices[original_index]
        entry = {
            "name": sample,
            "kind": "dm_auto" if sample == "DM" else "galaxy_dm",
            "sample_bin": sample_bin,
            "original_index": int(original_index),
            "joint_slice": local_slice,
            "band_indices": original_bands,
            "ell": band_ell[original_index],
            "window": windows[original_index],
            "cl": data_full[original_bands],
            "cov": covariance[local_slice, local_slice],
            "z_range": (float(z_min[original_index]), float(z_max[original_index])),
            "label": labels[original_index],
        }
        if sample == "DM":
            entry.update({
                "nz_z": None,
                "nz": None,
                "bias": np.nan,
                "bin": (0.0, 5.0),
                "tracer": None,
            })
        else:
            prefix = info["prefix"]
            required_sample = {
                f"{prefix}_nz_z", f"{prefix}_nz", f"{prefix}_bins", info["bias"],
            }
            missing_sample = sorted(required_sample - names)
            if missing_sample:
                raise KeyError(
                    f"{sample} tracer metadata missing: {', '.join(missing_sample)}")
            biases = np.atleast_1d(np.asarray(data[info["bias"]], dtype=float))
            bias = float(biases[0] if biases.size == 1 else biases[sample_bin])
            entry.update({
                "nz_z": np.asarray(data[f"{prefix}_nz_z"][sample_bin], dtype=float),
                "nz": np.asarray(data[f"{prefix}_nz"][sample_bin], dtype=float),
                "bias": bias,
                "bin": tuple(np.asarray(data[f"{prefix}_bins"][sample_bin], dtype=float)),
            })
        entry["err"] = np.sqrt(np.clip(np.diag(entry["cov"]), 0.0, None))
        entries.append(entry)

    samples = []
    for sample in dict.fromkeys(entry["name"] for entry in entries):
        selected = [entry for entry in entries if entry["name"] == sample]
        info = sample_info[sample]
        if sample == "DM":
            samples.append({
                "name": sample,
                "kind": "dm_auto",
                "entries": selected,
                "cl": np.asarray([entry["cl"] for entry in selected]),
                "cov": np.asarray([entry["cov"] for entry in selected]),
                "err": np.asarray([entry["err"] for entry in selected]),
                "inv": np.asarray([np.linalg.inv(entry["cov"]) for entry in selected]),
                "windows": np.asarray([entry["window"] for entry in selected]),
                "nz_z": None,
                "nz": None,
                "bins": [entry["bin"] for entry in selected],
                "z_ranges": [entry["z_range"] for entry in selected],
                "bias": np.array([np.nan]),
                "color": info["color"],
                "fname": info["fname"],
                "scale_power": 0,
                "multiply_ell": False,
            })
            continue
        scale_key = f"scale_power_{sample}_DM"
        scale_values = (
            np.atleast_1d(np.asarray(data[scale_key], dtype=int))
            if scale_key in names
            else np.array([int(data["cross_scale_power"])], dtype=int)
        )
        samples.append({
            "name": sample,
            "entries": selected,
            "cl": np.asarray([entry["cl"] for entry in selected]),
            "cov": np.asarray([entry["cov"] for entry in selected]),
            "err": np.asarray([entry["err"] for entry in selected]),
            "inv": np.asarray([np.linalg.inv(entry["cov"]) for entry in selected]),
            "windows": np.asarray([entry["window"] for entry in selected]),
            "nz_z": np.asarray([entry["nz_z"] for entry in selected]),
            "nz": np.asarray([entry["nz"] for entry in selected]),
            "bins": [entry["bin"] for entry in selected],
            "z_ranges": [entry["z_range"] for entry in selected],
            "bias": np.asarray([entry["bias"] for entry in selected]),
            "color": info["color"],
            "fname": info["fname"],
            "scale_power": int(scale_values.flat[0]),
            "multiply_ell": True,
        })

    joint = {
        "entries": entries,
        "spectrum_indices": selected_spectra,
        "band_indices": selected_bands,
        "data": data_vector,
        "cov": covariance,
        "inv": inverse_covariance,
        "windows": windows[selected_spectra],
        "ell": band_ell[selected_spectra],
        "ell_theory": ell_theory,
    }
    return samples, joint


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FRB redshift MCMC")
    parser.add_argument("--mode", default=None, help="cl, dmz, or joint")
    args, _unknown = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    configure_mode(args.mode if args.mode else None)
    stem = chain_stem()

    out_raw = os.getenv("MCMC_OUT_DIR", "")
    OUT_DIR = (
        Path(out_raw).expanduser()
        if out_raw
        else results_root() / MCMC_MODE / igm_run_dirname()
    )
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_base_dir(OUT_DIR)

    samples = joint = DATA_FILE = None
    n_cl = 0
    COSMO = clcat.make_cosmology()
    Z = np.linspace(0.0, 5.0, 500)
    Z_DM = None
    COSMIC_DM_DMZ = None
    get_engine(COSMO, Z)
    print("P_det:", pdet_path())
    print("p(z)/C_ell z grid: %d (uniform [0, 5])" % int(Z.size))
    print("MCMC_MODE =", MCMC_MODE)
    print("DELAY_MODEL =", DELAY_MODEL)
    print("IGM_MODEL =", IGM_MODEL)
    print("OUT_DIR:", OUT_DIR)
    print("R_SFH:", sfh_label())

    if USE_CL:
        DATA_FILE = catalog_npz_path()
        data = np.load(DATA_FILE, allow_pickle=False)
        samples, joint = load_cross_samples(data)
        ELL_THEORY = joint["ell_theory"]
        ELL_SPARSE = cl_bandpower_ell_sparse(ELL_THEORY) if USE_CL_BANDPOWER else None
        DATA_VECTOR = joint["data"]
        INV_COV_JOINT = joint["inv"]
        n_cl = DATA_VECTOR.size
        LMAX = int(np.asarray(data["lmax"]).flat[0])
        print("enabled samples:", ", ".join(item["name"] for item in samples))
        print("npz:", DATA_FILE)
        print("selected joint covariance shape:", joint["cov"].shape)
        B_ELECTRON = float(data["b_electron"])
        B_E_FIXED = B_ELECTRON
        K_CUT_FIXED = float(data["k_cut_mpc_inv"])
        ELL_LOC_FIXED = float(data["ell_loc"])
        DM_HOST_SOURCE = float(data["dm_host_source"])
        DM_HOST_FIXED = DM_HOST_SOURCE
        DM_MW_RESIDUAL = float(data["dm_mw_residual"])
        CROSS_SCALE_POWER = int(data["cross_scale_power"])
        b_frb_fid = float(data["b_frb_source"])
        B_FRB_FIXED = b_frb_fid
        D_BAR = float(data["frb_dm_mean"])
        FIT_N_ELL = any(
            entry.get("kind") == "dm_auto" for entry in joint["entries"])
        N_ELL_FIXED = 0.0
        PEG = clcat.make_peg_power(COSMO, K_CUT_FIXED)
        PEE = clcat.make_pee_power(COSMO, K_CUT_FIXED) if FIT_N_ELL else None
        DDM_DZ = clcat.mean_igm_dm_derivative(COSMO, Z)
        COSMIC_DM = clcat.mean_cosmic_dm(COSMO, Z)
        for item in samples:
            if item["name"] == "DM":
                for entry in item["entries"]:
                    entry["kind"] = "dm_auto"
                    entry["tracer"] = None
                item["tracers"] = [None] * len(item["entries"])
                print("DM x DM  N_ell is a free additive white-noise term")
                continue
            tracers = []
            for entry in item["entries"]:
                z_g, n_g = entry["nz_z"], entry["nz"]
                bias = float(entry["bias"])
                tracer = ccl.NumberCountsTracer(
                    COSMO, dndz=(z_g, n_g),
                    bias=(z_g, np.full_like(z_g, bias)), has_rsd=False)
                entry["tracer"] = tracer
                tracers.append(tracer)
                z0, z1 = entry["z_range"]
                print("%s  b=%.4f" % (format_bin_short(item["name"], z0, z1), bias))
            item["tracers"] = tracers
        MODEL_ENTRIES = joint["entries"]
    else:
        B_ELECTRON = 1.0
        B_E_FIXED = 1.0
        K_CUT_FIXED = clcat.K_CUT_MPC_INV
        ELL_LOC_FIXED = clcat.ELL_LOC
        DM_HOST_FIXED = clcat.DM_HOST_SOURCE
        b_frb_fid = float(clcat.B_FRB_SOURCE)
        B_FRB_FIXED = b_frb_fid
        D_BAR = 0.0
        N_ELL_FIXED = 0.0
        DDM_DZ = dmz.mean_igm_dm_derivative(COSMO, Z)
        COSMIC_DM = dmz.mean_cosmic_dm(COSMO, Z)

    if USE_DMZ:
        Z_DM = dmz.make_z_grid()
        COSMIC_DM_DMZ = dmz.mean_cosmic_dm(COSMO, Z_DM)
        get_engine(COSMO, Z_DM)
        DM_OBS, dm_mean = dmz.load_extragalactic_dm()
        DM_GRID = dmz.make_dm_grid(DM_OBS, igm_model=IGM_MODEL)
        if IGM_MODEL == "macquart":
            DM_TINY, _ = dmz.load_tiny_extragalactic_dm()
            Z_TINY_COMP = dmz.make_tiny_composite_z(Z_DM)
            COSMIC_TINY_COMP = dmz.mean_cosmic_dm(COSMO, Z_TINY_COMP)
            get_engine(COSMO, Z_TINY_COMP)
            N_FRB = int(DM_OBS.size + DM_TINY.size)
        else:
            DM_TINY = np.zeros(0, dtype=float)
            Z_TINY_COMP = None
            COSMIC_TINY_COMP = None
            N_FRB = int(DM_OBS.size)
        if not USE_CL:
            D_BAR = dm_mean
        print("DM-z z grid: %d nodes (50+20+20+60 on [0, 0.2, 0.5, 1, 5])"
              % int(Z_DM.size))
        print("N_FRB = %d  bulk=%d  DM_EG min=%.4g  mean=%.4g  "
              "(catalog DMMEAN=%.4g)"
              % (N_FRB, int(DM_OBS.size), float(np.min(DM_OBS)),
                 float(np.mean(DM_OBS)), dm_mean))
        if DM_TINY is not None and DM_TINY.size:
            print("DM<5: %d source, DM=%s, z=linspace(%.3f, %.3f, %d), "
                  "n_dm=%d"
                  % (int(DM_TINY.size),
                     np.array2string(DM_TINY, precision=6, separator=","),
                     dmz.TINY_Z_LO, dmz.TINY_Z_HI, dmz.N_TINY_Z,
                     dmz.N_TINY_DM))

    print("FIT_ELL_LOC =", FIT_ELL_LOC, "FIT_K_CUT =", FIT_K_CUT,
          "FIT_N_ELL =", FIT_N_ELL, "FIT_DM_HOST =", FIT_DM_HOST,
          "FIT_B_FRB =", FIT_B_FRB, "FIT_BETA_FRB =", FIT_BETA_FRB,
          "FIT_HOST_LN =", FIT_HOST_LN,
          "FIT_HOST_GAUSS =", FIT_HOST_GAUSS, "FIT_F =", FIT_F)
    if USE_CL:
        if USE_CL_BANDPOWER:
            print("Cl theory: NaMaster bandpower windows (%s ell n=%d, PCHIP)"
                  % (CL_BANDPOWER_GRID, ELL_SPARSE.size))
        else:
            print("Cl theory: ell_eff")
    print("F_IGM = %.4g (mean DM, fixed)" % F_IGM)
    if FIT_DM_HOST:
        print("DM_host prior = [%.3g, %.3g]  start = %.3g"
              % (DM_HOST_PRIOR[0], DM_HOST_PRIOR[1], DM_HOST_FIXED))
    if FIT_B_FRB:
        print("b_frb prior = [%.3g, %.3g]  start = %.3g"
              % (B_FRB_PRIOR[0], B_FRB_PRIOR[1], B_FRB_START))
    if FIT_BETA_FRB:
        print("beta_frb prior = [%.3g, %.3g]  start = %.3g"
              % (BETA_FRB_PRIOR[0], BETA_FRB_PRIOR[1], BETA_FRB_START))
    if FIT_K_CUT:
        print("k_cut prior = [%.3g, %.3g] Mpc^-1  start = %.3g"
              % (K_CUT_PRIOR[0], K_CUT_PRIOR[1], K_CUT_FIXED))
    if FIT_HOST_GAUSS:
        print("DM_host prior = [%.3g, %.3g]  start = %.3g"
              % (DM_HOST_PRIOR[0], DM_HOST_PRIOR[1], DM_HOST_FIXED))
        print("sigma_DM_host prior = [%.3g, %.3g]  start = %.3g"
              % (SIGMA_DM_HOST_PRIOR[0], SIGMA_DM_HOST_PRIOR[1],
                 SIGMA_DM_HOST_START))
    if FIT_F:
        print("F prior = [%.3g, %.3g]  start = %.3g"
              % (F_PRIOR[0], F_PRIOR[1], F_START))
    elif USE_DMZ and IGM_MODEL == "macquart":
        print("F = %.4g (scatter, fixed)" % F_FIXED)
    elif USE_DMZ:
        print("A_IGM = %.4g (scatter, fixed)" % dmz.A_IGM)
    print("delay parameters:",
          ", ".join(p[0] for p in DELAY_PARAMS) or "none (SFH)")

    params = [list(row) for row in DELAY_PARAMS]
    plot_names = list(DELAY_PLOT_NAMES)
    if FIT_HOST_LN:
        params.extend([
            ["mu_host", MU_HOST_START, MU_HOST_PRIOR[0], MU_HOST_PRIOR[1]],
            ["sigma_host", SIGMA_HOST_START,
             SIGMA_HOST_PRIOR[0], SIGMA_HOST_PRIOR[1]],
        ])
        plot_names.extend([r"\mu_{\rm host}", r"\sigma_{\rm host}"])
    if FIT_HOST_GAUSS:
        params.extend([
            ["dm_host", DM_HOST_FIXED, DM_HOST_PRIOR[0], DM_HOST_PRIOR[1]],
            ["sigma_dm_host", SIGMA_DM_HOST_START,
             SIGMA_DM_HOST_PRIOR[0], SIGMA_DM_HOST_PRIOR[1]],
        ])
        plot_names.extend([r"{\rm DM}_{\rm host}", r"\sigma_{\rm DM,host}"])
    if FIT_F:
        params.append(["F", F_START, F_PRIOR[0], F_PRIOR[1]])
        plot_names.append(r"F")
    if FIT_B_FRB:
        params.append(["b_frb", B_FRB_START, B_FRB_PRIOR[0], B_FRB_PRIOR[1]])
        plot_names.append(r"b_{\rm FRB}")
    if FIT_BETA_FRB:
        params.append(
            ["beta_frb", BETA_FRB_START, BETA_FRB_PRIOR[0], BETA_FRB_PRIOR[1]])
        plot_names.append(r"\beta_{\rm FRB}")
    if FIT_B_E:
        params.append(["b_e", B_E_START, B_E_PRIOR[0], B_E_PRIOR[1]])
        plot_names.append(r"b_{e}")
    if FIT_ELL_LOC:
        params.append(["ell_loc", ELL_LOC_FIXED, ELL_LOC_PRIOR[0], ELL_LOC_PRIOR[1]])
        plot_names.append(r"\ell_{\rm loc}")
    if FIT_K_CUT:
        params.append(["k_cut", K_CUT_FIXED, K_CUT_PRIOR[0], K_CUT_PRIOR[1]])
        plot_names.append(r"k_{\rm cut}\,[{\rm Mpc}^{-1}]")
    if FIT_N_ELL:
        params.append(["N_ell", N_ELL_START, N_ELL_PRIOR[0], N_ELL_PRIOR[1]])
        plot_names.append(r"N_\ell")
    if FIT_DM_HOST:
        params.append(["dm_host", DM_HOST_FIXED, DM_HOST_PRIOR[0], DM_HOST_PRIOR[1]])
        plot_names.append(r"{\rm DM}_{\rm host}")
    n_data = n_cl + (N_FRB if USE_DMZ else 0)
    manifest_payload = run_manifest_payload(
        DATA_FILE, joint, [parameter[0] for parameter in params])
    try:
        manifest_path = ensure_run_manifest(OUT_DIR, manifest_payload)
    except RuntimeError as exc:
        if os.environ.get("SKIP_MCMC") != "1":
            raise
        print("WARNING:", exc, "; plotting anyway")
        manifest_path = OUT_DIR / "run_manifest.npz"
    print("run manifest:", manifest_path)
    print("free parameters:", ", ".join(p[0] for p in params))
    print("data points:", n_data)
    print("chain burn-in: %.0f%% of the steps actually run"
          % (100.0 * CHAIN_BURN_IN_FRAC))
    theta0 = [p[1] for p in params]
    t0 = time.perf_counter()
    chi_cl0, chi_dm0, chi_tot0 = chi2_parts(theta0)
    dt_ms = 1e3 * (time.perf_counter() - t0)
    print("chi2 at start: Cl=%.3f  DM=%.3f  tot=%.3f  [%.1f ms]"
          % (chi_cl0, chi_dm0, chi_tot0, dt_ms))
    if USE_CL:
        chi2_null = float(DATA_VECTOR @ INV_COV_JOINT @ DATA_VECTOR)
        print("chi2 at null (model C_ell = 0): %.3f" % chi2_null)
        print("delta chi2_Cl (null - start): %.3f" % (chi2_null - chi_cl0))

    if os.environ.get("RUN_DYNESTY", "").strip().lower() in ("1", "true", "yes"):
        dlogz = float(os.environ.get("DYNESTY_DLOGZ", "0.01") or 0.01)
        maxcall = int(os.environ.get("DYNESTY_MAXCALL", "600000") or 600000)
        ncores = int(os.environ.get("MYMCMC_NCORES", "1") or 1)
        progress_log = OUT_DIR / "dynesty_progress.log"
        os.environ["MYMCMC_PROGRESS_LOG"] = str(progress_log)
        evidence_path = OUT_DIR / "dynesty_evidence.json"
        print("RUN_DYNESTY=1  dlogz_init=%g  maxcall=%d  ncores=%d"
              % (dlogz, maxcall, ncores))
        with open(progress_log, "w", encoding="utf-8") as fh:
            fh.write("[%s] start dynesty  OUT_DIR=%s  MODE=%s  DELAY_MODEL=%s\n"
                     % (time.strftime("%Y-%m-%d %H:%M:%S"), OUT_DIR,
                        MCMC_MODE, DELAY_MODEL))
        mc = MCMC_class(
            params, chi2, stem, data_num=n_data,
            nwalkers=32, use_nelder=False)
        mc.run_dynesty(
            dlogz_init=dlogz, maxcall=maxcall, ncores=ncores,
            evidence_path=str(evidence_path),
            extra_meta={
                "dataset": "result/%s" % MCMC_MODE,
                "delay_model": DELAY_MODEL,
                "model": os.environ.get("DYNESTY_MODEL_LABEL", DELAY_MODEL),
                "mcmc_mode": MCMC_MODE,
                "cl_bandpower": bool(USE_CL_BANDPOWER),
            },
        )
        print("Dynesty complete:", evidence_path)
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # 4. Run MCMC
    # ------------------------------------------------------------------
    progress_log = OUT_DIR / "mcmc_progress.log"
    os.environ["MYMCMC_PROGRESS_LOG"] = str(progress_log)
    print("progress log:", progress_log)
    if os.environ.get("SKIP_MCMC") != "1":
        with open(progress_log, "w", encoding="utf-8") as fh:
            fh.write("[%s] start MCMC  OUT_DIR=%s  MODE=%s  DELAY_MODEL=%s  "
                     "IGM_MODEL=%s  "
                     "FIT_ELL_LOC=%s  FIT_K_CUT=%s  "
                     "FIT_N_ELL=%s  FIT_DM_HOST=%s  FIT_B_FRB=%s  "
                     "FIT_BETA_FRB=%s  "
                     "FIT_HOST_LN=%s  FIT_HOST_GAUSS=%s  FIT_F=%s\n"
                     % (time.strftime("%Y-%m-%d %H:%M:%S"), OUT_DIR,
                        MCMC_MODE, DELAY_MODEL, IGM_MODEL, FIT_ELL_LOC, FIT_K_CUT,
                        FIT_N_ELL, FIT_DM_HOST, FIT_B_FRB, FIT_BETA_FRB,
                        FIT_HOST_LN, FIT_HOST_GAUSS, FIT_F))
        mc = MCMC_class(params, chi2, stem, data_num=n_data, nwalkers=32,
                        use_nelder=_env_flag("MCMC_OPTIMIZE", True))
        mc.run(steps=MCMC_STEPS, add_steps=int(os.getenv("MCMC_ADD_STEPS", "10000")),
               max_steps=int(os.getenv("MCMC_MAX_STEPS", "100000")))
        with open(progress_log, "a", encoding="utf-8") as fh:
            fh.write("[%s] MCMC finished; writing plots and best-fit files\n"
                     % time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("SKIP_MCMC=1; reusing existing chain")

    # Numerical posterior summary; publication figures use picture.py.
    pl = MCplot([["%s_converged" % stem, MCMC_MODE]],
                new_name=plot_names, ignore_rows=0.0)
    pl.results2
