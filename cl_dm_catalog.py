"""Angular-spectrum theory for fitting the supplied measured bandpowers.

Raw BGS/LRG catalogs are not required or included. NaMaster bandpower windows
are read from the supplied archive rather than measured again here.
"""

from cosmology import production_cosmology
import os
from pathlib import Path
import numpy as np
import pyccl as ccl
from astropy.constants import G, c, m_p, pc
from astropy.io import fits
from scipy.integrate import cumulative_trapezoid, trapezoid
ROOT = Path(__file__).resolve().parent
FRB_DATA_FILE = ROOT / "data/frb/chimefrbcat2_process.fits"
# Skip unused spectra: False turns off Field construction, Cl, covariance, plots, and npz keys.
COMPUTE_DM_DM = True          # FRB DM×DM auto, included in the joint covariance
COMPUTE_GALAXY_AUTO = False   # BGS×BGS if BGS is loaded; LRG×LRG for LRG_BINS[0] only
COMPUTE_BGS_DM = True         # BGS×DM
COMPUTE_LRG_DM = True         # LRG×DM for every entry in LRG_BINS

# Every interval is [z_min, z_max), except the final interval, which includes z_max.
BGS_BINS = [(0, 0.1), (0.1,  0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
AUTO_Z_RANGE = (BGS_BINS[0][0], BGS_BINS[-1][1])
# Delete a row to drop that tomographic bin. Column 0 is pz_bin / lrg_binN_clean.fits.
# zmin, zmax are labels + bias midpoints only (catalogs are already split by pz_bin).
LRG_BINS = [
    (1, 0.40, 0.55),
    (2, 0.55, 0.72),
    (3, 0.72, 0.86),
    (4, 0.86, 1.02),
]

# 10^x scaling. Change these numbers directly; nothing is chosen automatically.
CROSS_SCALE_POWER = 1          # galaxy × DM: 10^x ell C_ell
GALAXY_AUTO_SCALE_POWER = 4    # galaxy auto: 10^x ell C_ell
# DM-DM is plotted as C_ell with no 10^x and no ell prefactor.
BGS_DM_SCALE_POWERS = [CROSS_SCALE_POWER] * len(BGS_BINS)
LRG_DM_SCALE_POWERS = [CROSS_SCALE_POWER] * len(LRG_BINS)

def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


# Override with CL_LMAX / CL_N_ELL_BINS for the sensitivity grid. LMAX_MASK stays 3000.
LMAX = _env_int("CL_LMAX", 1000)
LMAX_MASK = 3000
N_ELL_BINS = _env_int("CL_N_ELL_BINS", 6)
ELL_LOC = 1e3

ELL_MIN = 40
PLOT_ELL_MIN = 6  # figure and theory curve; data bandpowers still start at ELL_MIN
N_Z_BINS = 30

B_ELECTRON = 1.0
B_FRB_SOURCE = 1
# DESI fiducial: b(z) D(z) = const, D(0)=1. Bin bias uses the bin midpoint.
B_BGS_D = 1.34
B_LRG_D = 1.7

DM_HOST_SOURCE = 150.0
DM_MW_RESIDUAL = 0.0
K_CUT_INVERSE_MPC = 0.9  # Fixed inverse cutoff scale [Mpc].
K_CUT_MPC_INV = 1.0 / K_CUT_INVERSE_MPC  # Physical k_cut [Mpc^-1].
# Fiducial FRB p(z) ∝ z^2 exp(-6.7 z). MCMC still fits tau_NL, sigma_NL.
F_IGM = _env_float("CL_F_IGM", 0.93)
X_E = 0.875




def read_frb_dm_mean(path=None):
    """Catalog extra-galactic DM mean stored in FITS header DMMEAN."""
    if path is None:
        path = FRB_DATA_FILE
    with fits.open(path, memmap=True) as hdul:
        return float(hdul[1].header["DMMEAN"])


def make_cosmology():
    return production_cosmology(Omega_b=0.04897, Omega_c=0.3111 - 0.04897,
                         h=0.6766, sigma8=0.81, n_s=0.96)


def desi_bias(cosmo, z, b_d):
    """Galaxy bias b(z) = b_d / D(z) with D(0)=1."""
    z = np.asarray(z, dtype=float)
    return b_d / ccl.growth_factor(cosmo, 1.0 / (1.0 + z))


def frb_source_distribution(z, tau_nl=None, sigma_nl=None, cosmo=None):
    """Fiducial FRB p(z) ∝ z^2 exp(-6.7 z), unit integral."""
    from frb_redshift_model import analytic_reference_pz
    return analytic_reference_pz(z)


def mean_igm_dm_derivative(cosmo, z):
    """Mean IGM dDM/dz in pc cm^-3 for the configured baryon model."""
    return (3.0 * c.value * cosmo["Omega_b"] * cosmo["h"] * 100.0
            / (8.0 * np.pi * G.value * m_p.value) * (1.0 + z) * F_IGM * X_E
            / ccl.h_over_h0(cosmo, 1.0 / (1.0 + z))
            * (1e3 / (pc.value * 1e6)) * (1e-6 / pc.value))


def make_dm_tracer(cosmo, source_nz=None, z=None):
    """DM NumberCountsTracer.

    CCL renormalises ``dndz`` to unit integral, so the bias is
    ``B_ELECTRON * ∫ kernel dz`` to restore the physical DM window.
    """
    if z is None:
        z = np.linspace(0.0, 5.0, 500)
    if source_nz is None:
        source_nz = frb_source_distribution(z, cosmo=cosmo)
    source_nz = np.clip(np.asarray(source_nz, dtype=float), 0.0, None)
    source_nz = source_nz / trapezoid(source_nz, z)
    source_tail = -cumulative_trapezoid(source_nz[::-1], z[::-1], initial=0.0)[::-1]
    kernel = mean_igm_dm_derivative(cosmo, z) * source_tail
    kernel_norm = trapezoid(kernel, z)
    tracer = ccl.NumberCountsTracer(
        cosmo, dndz=(z, kernel),
        bias=(z, np.full_like(z, B_ELECTRON * kernel_norm)), has_rsd=False)
    return tracer, z, source_nz, kernel


def make_peg_power(cosmo, k_cut_mpc_inv=K_CUT_MPC_INV):
    """Return P_eg=P_mm^NL exp(-k/k_cut), with k_cut in Mpc^-1."""
    k_cut_mpc_inv = float(k_cut_mpc_inv)
    if not np.isfinite(k_cut_mpc_inv) or k_cut_mpc_inv <= 0.0:
        raise ValueError("k_cut_mpc_inv must be finite and positive [Mpc^-1]")
    nonlin_power = cosmo.get_nonlin_power()
    return ccl.Pk2D.from_function(
        lambda k, a: np.log(nonlin_power(k, a, cosmo)) - k / k_cut_mpc_inv,
        is_logp=True, spline_params=cosmo.cosmo.spline_params)


def make_pee_power(cosmo, k_cut_mpc_inv=K_CUT_MPC_INV):
    """Return P_ee=P_mm^NL exp(-2 k/k_cut), with k_cut in Mpc^-1."""
    k_cut_mpc_inv = float(k_cut_mpc_inv)
    if not np.isfinite(k_cut_mpc_inv) or k_cut_mpc_inv <= 0.0:
        raise ValueError("k_cut_mpc_inv must be finite and positive [Mpc^-1]")
    nonlin_power = cosmo.get_nonlin_power()
    return ccl.Pk2D.from_function(
        lambda k, a: np.log(nonlin_power(k, a, cosmo)) - 2.0 * k / k_cut_mpc_inv,
        is_logp=True, spline_params=cosmo.cosmo.spline_params)


def localization_beam(ell, ell_loc=None):
    if ell_loc is None:
        ell_loc = ELL_LOC
    return np.exp(-0.5 * (np.asarray(ell, dtype=float) / ell_loc) ** 2)


def mean_cosmic_dm(cosmo, z):
    z_dense = np.linspace(0.0, max(5.0, 1.02 * np.max(z)), 4000)
    dm_dense = cumulative_trapezoid(
        mean_igm_dm_derivative(cosmo, z_dense), z_dense, initial=0.0)
    return np.interp(z, z_dense, dm_dense)


def make_frb_source_tracer(cosmo, source_nz=None, z=None, b_frb=None, d_bar=None):
    """NumberCountsTracer for toy FRB-source clustering C_ell^fg.

    CCL renormalises ``dndz`` to unit integral, so the residual DM is put
    in the bias: ``b(z) = b_FRB [DM(z) - d_bar]``.  ``d_bar`` is the catalog
    sample mean of extra-galactic DM (FITS header DMMEAN), not <DM_model>_p(z).
    """
    if z is None:
        z = np.linspace(0.0, 5.0, 500)
    if source_nz is None:
        source_nz = frb_source_distribution(z, cosmo=cosmo)
    if b_frb is None:
        b_frb = B_FRB_SOURCE
    if d_bar is None:
        d_bar = read_frb_dm_mean()
    source_nz = np.clip(np.asarray(source_nz, dtype=float), 0.0, None)
    source_nz = source_nz / trapezoid(source_nz, z)
    dm = DM_MW_RESIDUAL + mean_cosmic_dm(cosmo, z) + DM_HOST_SOURCE / (1.0 + z)
    return ccl.NumberCountsTracer(
        cosmo, dndz=(z, source_nz),
        bias=(z, b_frb * (dm - d_bar)), has_rsd=False)
