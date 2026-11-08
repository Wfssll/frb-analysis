# -*- coding: utf-8 -*-
"""
mymcmc — Lightweight MCMC module extracted and simplified from qcosmc.

Provides:
  MCMC_class  — MCMC runner with two modes: runMC (pickle-based) and run (HDF5-based)
  MCplot      — getdist-based plotting (1D, 2D, triangle) with the same colour scheme

Usage (minimal)::

    from mymcmc import MCMC_class, MCplot

    def chi2(theta):
        return np.sum(((y - model(theta)) / sigma)**2)

    params = [['a', 0.0, -5, 5], ['b', 0.0, -5, 5]]
    mc = MCMC_class(params, chi2, 'test_chain', data_num=len(y), nwalkers=32)
    mc.runMC(steps=500, add_steps=200, max_steps=5000)
    pl = MCplot([['test_chain_converged', 'test_chain']])
    pl.plot3D([1, 2])
"""

import numpy as np
import emcee
import os
import sys
import json
import warnings
import datetime
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from getdist import MCSamples, plots
from time import perf_counter as _clock
import pickle as _pickle
import dynesty

# Module-level dynesty callables (must be pickle/fork friendly).
_DYNESTY_CHI2 = None
_DYNESTY_PARAMS = None


def _dynesty_prior_transform(u):
    return np.array(
        [low + u[i] * (high - low)
         for i, (_, _, low, high) in enumerate(_DYNESTY_PARAMS)],
        dtype=float,
    )


def _dynesty_loglike(theta):
    return -0.5 * _DYNESTY_CHI2(theta)

# ---- configurable base directory for all output ---------------------------
_BASE_DIR = os.getcwd()


def set_base_dir(path):
    """Set the root directory for chains/, temple/, results/ output."""
    global _BASE_DIR
    _BASE_DIR = os.path.abspath(path)
    os.makedirs(_BASE_DIR, exist_ok=True)


def _path(*parts):
    return os.path.join(_BASE_DIR, *parts)


BURN_IN_FRAC_DEFAULT = 0.3


def burn_in_frac_default():
    raw = os.environ.get("MCMC_BURN_IN_FRAC", "").strip()
    if raw:
        return float(raw)
    return BURN_IN_FRAC_DEFAULT


def burn_in_steps(n_steps, frac=None):
    """Number of leading steps discarded per walker (``frac`` of total)."""
    n_steps = int(n_steps)
    if frac is None:
        frac = burn_in_frac_default()
    frac = float(frac)
    if n_steps <= 1:
        return 0
    n_burn = int(frac * n_steps)
    return min(max(n_burn, 0), n_steps - 1)


def write_burn_in_record(n_steps, n_burn, frac=None, source="backend"):
    """Persist the burn-in actually applied when flattening the chain."""
    if frac is None:
        frac = burn_in_frac_default()
    rec_dir = _path("chains")
    os.makedirs(rec_dir, exist_ok=True)
    rec = {
        "n_steps": int(n_steps),
        "burn_in_frac": float(frac),
        "burn_in_steps_per_walker": int(n_burn),
        "source": str(source),
    }
    path = os.path.join(rec_dir, "burn_in.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rec, handle, indent=2)
        handle.write("\n")
    return rec


def read_burn_in_record(out_dir=None):
    path = os.path.join(out_dir or _BASE_DIR, "chains", "burn_in.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _template_npy(out_dir, stem):
    conv = os.path.join(out_dir, "chains", stem + "_converged.npy")
    temp = os.path.join(out_dir, "temple", stem, stem + "_temp.npy")
    for path in (conv, temp):
        if os.path.isfile(path):
            return np.load(path, allow_pickle=True)
    return None


def load_walker_chain(out_dir, stem, nwalkers=32):
    """Load (nwalkers, nsteps, ndim) chain and log-prob.

    Preference: HDF5 backend, then unburned ``*_temp.npy``, then the
    already-flattened ``*_converged.npy`` (no further burn-in).
    """
    backend_file = os.path.join(out_dir, "temple", stem, "backend.h5")
    if os.path.isfile(backend_file):
        backend = emcee.backends.HDFBackend(backend_file, read_only=True)
        chain = backend.get_chain().transpose(1, 0, 2)
        log_prob = backend.get_log_prob().T
        return chain, log_prob, "backend"
    temp_npy = os.path.join(out_dir, "temple", stem, stem + "_temp.npy")
    if os.path.isfile(temp_npy):
        blob = np.load(temp_npy, allow_pickle=True)
        samples = np.asarray(blob[0], dtype=float)
        log_prob = np.asarray(blob[1], dtype=float)
        if samples.shape[0] % int(nwalkers) != 0:
            raise ValueError(
                "%s length %d is not divisible by nwalkers=%d"
                % (temp_npy, samples.shape[0], nwalkers))
        nsteps = samples.shape[0] // int(nwalkers)
        return (
            samples.reshape(int(nwalkers), nsteps, -1),
            log_prob.reshape(int(nwalkers), nsteps),
            "temp",
        )
    conv = os.path.join(out_dir, "chains", stem + "_converged.npy")
    if os.path.isfile(conv):
        blob = np.load(conv, allow_pickle=True)
        samples = np.asarray(blob[0], dtype=float)
        log_prob = np.asarray(blob[1], dtype=float)
        if samples.shape[0] % int(nwalkers) != 0:
            raise ValueError(
                "%s length %d is not divisible by nwalkers=%d"
                % (conv, samples.shape[0], nwalkers))
        n_kept = samples.shape[0] // int(nwalkers)
        return (
            samples.reshape(int(nwalkers), n_kept, -1),
            log_prob.reshape(int(nwalkers), n_kept),
            "converged",
        )
    return None, None, None


def write_converged_npy(stem, chain, log_prob, n_burn, param_names, theta_fit,
                        ranges, data_num):
    """Flatten after per-walker burn-in and write ``*_converged.npy``."""
    samples, flat_log = MCMC_class._burn_and_flatten(chain, log_prob, n_burn)
    pct = np.percentile(samples, [16, 50, 84], axis=0)
    theta_fact = [
        (float(pct[1, i]), float(pct[2, i] - pct[1, i]),
         float(pct[1, i] - pct[0, i]))
        for i in range(len(pct[0]))
    ]
    minkaf = float(-2 * np.max(flat_log))
    chains_dir = _path("chains")
    os.makedirs(chains_dir, exist_ok=True)
    savefile_name = os.path.join(chains_dir, stem + "_converged.npy")
    data_to_save = np.array([
        samples, flat_log, list(param_names), list(theta_fit),
        theta_fact, minkaf, int(data_num), ranges,
    ], dtype=object)
    np.save(savefile_name, data_to_save, allow_pickle=True)
    n_steps = int(np.asarray(chain).shape[1])
    print("Results saved to: %s" % savefile_name)
    print("Burn-in: %d steps per walker (%.0f%% of %d)"
          % (int(n_burn), 100.0 * float(n_burn) / max(n_steps, 1), n_steps))
    print("Total samples: %d" % len(samples))
    return samples, flat_log, minkaf


def reprocess_directory(out_dir, stem, frac=None, nwalkers=32):
    """Rebuild flattened chain using 30% of the saved total steps.

    Complete HDF5 / ``*_temp.npy`` files are burned as ``frac * n_steps``.
    If only ``*_converged.npy`` remains, keep those already-unburned samples.
    """
    out_dir = os.path.abspath(out_dir)
    set_base_dir(out_dir)
    if frac is None:
        frac = burn_in_frac_default()
    chain, log_prob, source = load_walker_chain(out_dir, stem, nwalkers=nwalkers)
    if chain is None:
        raise FileNotFoundError("no chain files under %s stem=%s" % (out_dir, stem))
    n_steps = int(chain.shape[1])
    if source == "converged":
        n_burn = 0
        print("%s: no complete chain; keeping %d already-unburned steps/walker"
              % (out_dir, n_steps))
    else:
        n_burn = burn_in_steps(n_steps, frac)
        print("%s: source=%s  n_steps=%d  burn-in=%d (%.0f%%)"
              % (out_dir, source, n_steps, n_burn, 100.0 * frac))
    blob = _template_npy(out_dir, stem)
    if blob is None:
        raise FileNotFoundError("missing template npy for names/ranges in %s" % out_dir)
    param_names = [str(x) for x in blob[2]]
    theta_fit = [float(x) for x in blob[3]]
    data_num = int(blob[6])
    ranges = blob[7]
    samples, flat_log, minkaf = write_converged_npy(
        stem, chain, log_prob, n_burn, param_names, theta_fit, ranges, data_num)
    n_params = samples.shape[1]
    constraints = []
    for i in range(n_params):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        constraints.append((float(med), float(med - lo), float(hi - med)))
    chi2_min = float(minkaf)
    dof = int(data_num) - n_params
    chi2_red = chi2_min / dof if dof > 0 else float("nan")
    aic = chi2_min + 2.0 * n_params
    bic = chi2_min + n_params * np.log(max(int(data_num), 1))
    _write_results_txt(
        stem, n_params, constraints, param_names,
        chi2_min, dof, chi2_red, aic, bic,
        data_num=int(data_num), burn_in=n_burn, n_steps=n_steps)
    rec = write_burn_in_record(n_steps, n_burn, frac, source)
    return rec

# ============================================================================
# Style setup — replicates qstyle(0) from qcosmc.FigStyle
# ============================================================================

_style_rc = {
    'axes.axisbelow': True,
    'axes.edgecolor': '.0',
    'axes.titlesize': 18,
    'axes.facecolor': '1',
    'axes.grid': False,
    'axes.labelcolor': '.0',
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'figure.dpi': 100,
    'lines.linewidth': 1,
    'font.family': ['sans-serif'],
    'font.sans-serif': ['Arial Unicode MS', 'Arial', 'DejaVu Sans',
                        'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'grid.color': '.8',
    'grid.linestyle': '-',
    'image.cmap': 'rocket',
    'legend.scatterpoints': 1,
    'lines.solid_capstyle': 'round',
    'text.color': '0',
    'axes.labelsize': 'large',
    'xtick.color': '0.',
    'xtick.direction': 'in',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'xtick.labelsize': 11,
    'ytick.color': '.0',
    'ytick.direction': 'in',
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'ytick.labelsize': 11,
    'legend.fancybox': False,
    'legend.numpoints': 1,
    'legend.fontsize': 16,
    'legend.frameon': False,
    'axes.unicode_minus': False,
}


def _setup_style(tex=False, rc=None):
    """Replicate qcosmc.FigStyle.qstyle(tex, rc)."""
    color_palette = ['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm', 'r', "#0DA2E7", "#0260CA"]
    mpl.rc('text', usetex=tex)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    merged = dict(_style_rc)
    if rc:
        merged.update(rc)
    sns.set_style(merged)
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['axes.labelsize'] = 16
    return color_palette


_setup_style(tex=False, rc={'figure.figsize': (8, 6)})

# Global colour / line-style tables (kept identical to qcosmc.MCMC)
colors = ['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm', 'r', "#0DA2E7", "#0260CA"]
lss = ['-', '--', '-.', ':']

# ============================================================================
# timer decorator (inlined from qcosmc.Decorator)
# ============================================================================


def timer(func):
    """Print elapsed wall-clock time for *func*."""
    def wrapper(*args, **kwargs):
        start = _clock()
        results = func(*args, **kwargs)
        finish = _clock()
        hours, sech = divmod(finish - start, 3600)
        minute, secm = divmod(sech, 60)
        print('=' * 55)
        print(" %s spends time: %d hours %d minutes %s seconds" %
              (func.__name__, hours, minute, round(secm, 1)))
        print('=' * 55)
        return results
    return wrapper


# ============================================================================
# MCMC_class
# ============================================================================

class MCMC_class:
    """MCMC runner with convergence diagnostics, resume, and intermediate saves.

    Parameters
    ----------
    params : list of [name, fit, lower, upper]
    chi2 : callable  theta -> chi^2
    Chains_name : str  base name for output files
    data_num : int  number of data points (for dof reporting)
    nwalkers : int  number of walkers (default 32)
    use_nelder : bool  True → Nelder-Mead best-fit start (default);
                       False → fiducial-value start.
    """

    def __init__(self, params, chi2, Chains_name, data_num=0, nwalkers=32,
                 use_nelder=True):
        self.chi2 = chi2
        self.Chains_name = Chains_name
        self.data_num = data_num
        self.use_nelder = use_nelder
        self.params_all = {
            'name': [p[0] for p in params],
            'fit': np.asarray([p[1] for p in params], dtype=object),
            'lower': np.asarray([p[2] for p in params], dtype=object),
            'upper': np.asarray([p[3] for p in params], dtype=object),
        }
        self.n = len(self.params_all['name'])
        self.theta_fit = np.zeros(self.n)
        self.theta_fact = np.zeros(self.n)
        self.nwalkers = nwalkers
        self._tau_history = []
        self.pos = self.optimize()

    # -- probability helpers -------------------------------------------------

    def _lnprior(self, x):
        for i in range(self.n):
            if not (self.params_all['lower'][i] <= x[i] <= self.params_all['upper'][i]):
                return -np.inf
        return 0.0

    def _lik(self, theta):
        return np.exp(-self.chi2(theta) / 2.)

    def _lnprob(self, theta):
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - self.chi2(theta) / 2.0

    def _ff(self, theta):
        return -2.0 * self._lnprob(theta)

    # -- initialisation ------------------------------------------------------

    def optimize(self):
        """Build initial walker positions.

        Two modes (controlled by ``self.use_nelder``):
          - True  (default): use Nelder-Mead to find best fit, scatter walkers around it.
          - False:            scatter walkers around the user-given fiducial values.
        """
        nc = 1e-4

        if self.use_nelder:
            _progress_log("Nelder-Mead optimisation starting")
            print("\nOptimising with Nelder-Mead ...")
            result = opt.minimize(self._ff, self.params_all['fit'],
                                  method='Nelder-Mead')
            self.theta_fit = result['x']
            for i in range(self.n):
                print("  Best-fit {0} = {1}".format(
                    self.params_all['name'][i], self.theta_fit[i]))
            print("  Minimum chi2 = %s" % self.chi2(self.theta_fit))
            _progress_log("Nelder-Mead done  chi2_min=%.6g" %
                          float(self.chi2(self.theta_fit)))
            centre = self.theta_fit
        else:
            print("\nUsing fiducial values as MCMC starting centre ...")
            self.theta_fit = np.array(self.params_all['fit'], dtype=np.float64)
            for i in range(self.n):
                print("  {0} = {1}".format(
                    self.params_all['name'][i], self.theta_fit[i]))
            print("  chi2 at fiducial = %s" % self.chi2(self.theta_fit))
            centre = self.theta_fit

        pos = np.zeros((self.nwalkers, self.n))
        noise = np.random.randn(self.nwalkers, self.n)
        for i in range(self.n):
            val = centre[i]
            if np.abs(val) > 1e-4 or val == 0:
                pos[:, i] = val + nc * noise[:, i]
            else:
                pos[:, i] = val + nc * val * noise[:, i]
        print("Initial positions generated.\n")
        _progress_log("walker positions ready  nwalkers=%d  ndim=%d"
                      % (self.nwalkers, self.n))
        return pos

    # -- one-shot MCMC (simple, non-resumable) --------------------------------

    @timer
    def MCMC(self, steps=50000, tol=0.01, **kwargs):
        """Run one-shot MCMC with convergence check (no resume)."""
        ndim, nwalkers = self.n, self.nwalkers
        print("Running MCMC ...")
        pool = _get_pool()
        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob,
                                            pool=pool, **kwargs)
            sampler.run_mcmc(self.pos, steps, progress=True, store=True)
        finally:
            _close_pool(pool)
        self.sampler = sampler
        print("Sampling done. Checking convergence ...")
        try:
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(tau - tau) / tau < 0.01)
            print("Converged." if converged else "May not be fully converged.")
            print("Autocorrelation times:", tau)
        except emcee.autocorr.AutocorrError:
            print("Autocorrelation time could not be estimated (too few steps).")
        self.savefile(sampler)
        return sampler

    # -- save helpers --------------------------------------------------------

    @staticmethod
    def _burn_and_flatten(chain, log_prob, burn_steps):
        """Discard the first ``burn_steps`` of every walker, then flatten.

        ``chain`` is ``(nwalkers, nsteps, ndim)`` and ``log_prob`` is
        ``(nwalkers, nsteps)``.  Flattening first (C-order) is walker-major,
        so slicing the flat array would drop early steps of walker 0 instead
        of a common burn-in.
        """
        chain = np.asarray(chain)
        log_prob = np.asarray(log_prob)
        n_burn = int(burn_steps)
        if 0 < n_burn < chain.shape[1]:
            chain = chain[:, n_burn:, :]
            log_prob = log_prob[:, n_burn:]
        samples = chain.reshape(-1, chain.shape[-1])
        return samples, log_prob.reshape(-1)

    def savefile(self, sampler, burn_in=None, append_mode=False):
        """Save MCMC results to a .npy file (converged chain).

        ``burn_in`` is the number of steps discarded from every walker
        before flattening.  ``None`` means ``frac * n_steps`` with
        ``frac`` from ``MCMC_BURN_IN_FRAC`` (default 0.3).
        """
        chain = np.asarray(sampler.chain)
        log_prob = np.asarray(sampler.lnprobability)
        n_steps = int(chain.shape[1])
        if burn_in is None:
            burn_in = burn_in_steps(n_steps)
        theta_name = list(self.params_all['name'])
        theta_fit = [float(x) for x in self.params_all['fit']]
        ranges = [[float(self.params_all['lower'][i]),
                   float(self.params_all['upper'][i])]
                  for i in range(len(self.params_all['lower']))]
        write_converged_npy(
            self.Chains_name, chain, log_prob, burn_in,
            theta_name, theta_fit, ranges, int(self.data_num))
        print('Chains name is "%s".  data number: %s'
              % (self.Chains_name, self.data_num))
        if append_mode:
            print("Warning: append_mode is ignored; the flattened chain "
                  "is rewritten from the complete walker traces.")

    def _save_intermediate_complete_results(self, all_chains, all_log_prob,
                                            total_steps):
        """Save intermediate .npy + trace plots."""
        try:
            temp_dir = _path('temple', self.Chains_name)
            os.makedirs(temp_dir, exist_ok=True)
            base = self.Chains_name + "_temp"

            samples = all_chains.reshape((-1, all_chains.shape[-1]))
            log_prob_flat = all_log_prob.flatten()

            theta_fit = [float(x) for x in self.params_all['fit']]
            theta_name = list(self.params_all['name'])
            ranges = [[float(self.params_all['lower'][i]),
                       float(self.params_all['upper'][i])]
                      for i in range(len(self.params_all['lower']))]

            pct = np.percentile(samples, [16, 50, 84], axis=0)
            theta_fact = [(float(pct[1, i]),
                           float(pct[2, i] - pct[1, i]),
                           float(pct[1, i] - pct[0, i])) for i in range(len(pct[0]))]
            minkaf = float(-2 * np.max(log_prob_flat))

            data_to_save = np.array([
                samples, log_prob_flat, theta_name, theta_fit,
                theta_fact, minkaf, int(self.data_num), ranges,
            ], dtype=object)
            save_path = os.path.join(temp_dir, base + ".npy")
            np.save(save_path, data_to_save, allow_pickle=True)
            print("Intermediate complete results saved: %s.npy" % base)

            self._generate_trace_plot(all_chains, all_log_prob, total_steps,
                                      temp_dir, base)
            self._generate_tau_evolution_plot(temp_dir, base)
        except Exception as e:
            print("Warning: Could not save intermediate results: %s" % e)

    # -- plotting helpers ----------------------------------------------------

    def _generate_trace_plot(self, all_chains, all_log_prob, total_steps,
                             save_dir, base_name):
        """Trace plot for every parameter + log-prob trace."""
        try:
            nwalkers, n_steps, n_params = all_chains.shape
            param_names = self.params_all['name']

            if n_params <= 3:
                ncols, nrows = n_params, 1
            elif n_params <= 6:
                ncols, nrows = 3, 2
            else:
                ncols = 3
                nrows = int(np.ceil(n_params / 3))

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(max(12, ncols * 4),
                                              max(6, nrows * 3)))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = axes.reshape(1, -1)
            elif ncols == 1:
                axes = axes.reshape(-1, 1)

            for i in range(n_params):
                ax = axes.flat[i]
                for w in range(nwalkers):
                    ax.plot(all_chains[w, :, i], alpha=0.3, linewidth=0.5)
                ax.set_title('$%s$' % param_names[i], fontsize=12)
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)

                flat = all_chains[:, :, i].reshape(-1)
                mean_i = float(np.mean(flat))
                std_i = float(np.std(flat, ddof=1))
                ax.axhline(mean_i, color='k', linestyle='--', alpha=0.6)
                ax.text(0.98, 0.02, 'mean=%.4g\nstd=%.4g' % (mean_i, std_i),
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
                try:
                    series = all_chains[:, :, i].T
                    tau_i = emcee.autocorr.integrated_time(series, quiet=True)
                    tau_val = float(np.mean(tau_i))
                    ax.text(0.03, 0.97, r'$\tau\approx$%.1f' % tau_val,
                            transform=ax.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception:
                    pass

            for i in range(n_params, nrows * ncols):
                axes.flat[i].set_visible(False)

            fig.suptitle('MCMC Trace Plot — %s (Step %d)' % (base_name, total_steps),
                         fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, base_name + "_trace.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("Trace plot saved: %s_trace.png" % base_name)

            # log-prob trace
            self._generate_logprob_trace(all_log_prob, total_steps, save_dir, base_name)
        except Exception as e:
            print("Warning: Could not generate trace plot: %s" % e)

    def _generate_logprob_trace(self, all_log_prob, total_steps, save_dir, base_name):
        """Standalone log-probability trace."""
        try:
            nwalkers, n_steps = all_log_prob.shape
            plt.figure(figsize=(12, 4))
            for w in range(nwalkers):
                plt.plot(all_log_prob[w, :], alpha=0.3, linewidth=0.5)
            mean_lp = np.mean(all_log_prob, axis=0)
            plt.plot(mean_lp, 'r-', linewidth=2, label='Mean')
            plt.title('Log Probability Trace — Step %d' % total_steps, fontsize=14)
            plt.xlabel('Step')
            plt.ylabel('Log Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            max_lp = np.max(all_log_prob)
            plt.text(0.02, 0.98, 'Max: %.2f\nChi2_min: %.2f' % (max_lp, -2*max_lp),
                     transform=plt.gca().transAxes, va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, base_name + "_logprob_trace.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("Log probability trace saved: %s_logprob_trace.png" % base_name)
        except Exception as e:
            print("Warning: Could not generate log prob trace: %s" % e)

    def _generate_tau_evolution_plot(self, save_dir, base_name):
        """Convergence diagnostic: max(tau)*50 vs N_step."""
        try:
            if not hasattr(self, '_tau_history') or len(self._tau_history) < 1:
                return
            steps_list = np.array([e[0] for e in self._tau_history])
            tau_max_list = np.array([e[1] for e in self._tau_history])
            max_tau50 = tau_max_list * 50

            fig, ax = plt.subplots(figsize=(10, 6))

            # Reference line: y = x  (i.e. max(tau)*50 == N_step)
            lim = max(steps_list[-1], max_tau50[-1]) * 1.15
            ref = np.linspace(0, lim, 100)
            ax.plot(ref, ref, '--k', linewidth=2, alpha=0.7,
                    label=r'$50\hat{\tau}_{\max} = N_{\rm step}$')

            # max(tau)*50 曲线（1个点时仅画点，≥2个点才连线）
            marker = 'o'
            if len(steps_list) >= 2:
                ax.plot(steps_list, max_tau50, 'o-', color='#348ABD',
                        linewidth=2.5, markersize=8, markerfacecolor='white',
                        label=r'$50 \times \max(\hat{\tau})$')
            else:
                ax.plot(steps_list, max_tau50, 'o', color='#348ABD',
                        markersize=10, markerfacecolor='white',
                        label=r'$50 \times \max(\hat{\tau})$')

            # 收敛判断
            cross_idx = None
            for j in range(len(steps_list)):
                if max_tau50[j] < steps_list[j]:
                    cross_idx = j
                    break
            if cross_idx is not None:
                ax.plot(steps_list[cross_idx], max_tau50[cross_idx], 'o',
                        color='#A60628', markersize=16, zorder=5,
                        markeredgewidth=2, markerfacecolor='none',
                        label=r'Converged at step %d' % steps_list[cross_idx])
            else:
                ax.plot(steps_list[-1], max_tau50[-1], 's',
                        color='#A60628', markersize=14, zorder=5,
                        label='Not yet converged')

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel('Total Steps', fontsize=14, fontweight='bold')
            ax.set_ylabel(r'$50 \times \max(\hat{\tau})$', fontsize=14, fontweight='bold')
            ax.set_title(r'MCMC Convergence: $50\hat{\tau}_{\max} < N_{\rm step}$?',
                         fontsize=15, fontweight='bold', pad=12)
            ax.legend(fontsize=12, loc='upper left', frameon=True,
                      fancybox=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=12)

            # 标注两个区域
            ax.fill_between(ref, 0, ref, alpha=0.08, color='green',
                            label='_nolegend_')
            ax.fill_between(ref, ref, lim, alpha=0.08, color='red',
                            label='_nolegend_')
            ax.text(lim * 0.6, lim * 0.3, 'Converged', fontsize=16,
                    color='green', fontweight='bold', alpha=0.4)
            ax.text(lim * 0.15, lim * 0.8, 'Not converged', fontsize=16,
                    color='red', fontweight='bold', alpha=0.4)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, base_name + "_convergence.png"),
                        dpi=200, bbox_inches='tight')
            plt.close()
            print("Convergence plot saved: %s_convergence.png" % base_name)
        except Exception as e:
            print("Warning: Could not generate convergence plot: %s" % e)

    # -- checkpoint for pickle-based resume (runMC) --------------------------

    def _save_checkpoint(self, all_chains, all_log_prob, total_steps, pos):
        """Write pickle checkpoint for runMC."""
        temple_dir = _path('temple', self.Chains_name)
        os.makedirs(temple_dir, exist_ok=True)
        pos_file = os.path.join(temple_dir, "sampler_pos.npy")
        chain_file = os.path.join(temple_dir, "sampler_chain_tmp.pkl")
        np.save(pos_file, pos)
        with open(chain_file, 'wb') as f:
            _pickle.dump({
                'chains': all_chains,
                'log_prob': all_log_prob,
                'total_steps': total_steps,
                'tau_history': getattr(self, '_tau_history', []),
            }, f)

    def _load_checkpoint(self):
        """Load pickle checkpoint for runMC. Returns (pos, chains, log_prob, total_steps) or None."""
        temple_dir = _path('temple', self.Chains_name)
        pos_file = os.path.join(temple_dir, "sampler_pos.npy")
        chain_file = os.path.join(temple_dir, "sampler_chain_tmp.pkl")

        pos = np.load(pos_file) if os.path.exists(pos_file) else None

        if os.path.exists(chain_file):
            try:
                with open(chain_file, 'rb') as f:
                    saved = _pickle.load(f)
                # restore tau history so convergence plot continues across resumes
                if 'tau_history' in saved:
                    self._tau_history = list(saved['tau_history'])
                return pos, saved['chains'], saved['log_prob'], saved['total_steps']
            except Exception as e:
                print("Error loading checkpoint: %s. Starting fresh." % e)
        return pos, None, None, 0

    def _cleanup_checkpoint(self):
        """Remove temporary checkpoint files."""
        temple_dir = _path('temple', self.Chains_name)
        for suffix in ['sampler_pos.npy', 'sampler_chain_tmp.pkl']:
            p = os.path.join(temple_dir, suffix)
            if os.path.exists(p):
                os.remove(p)

    # ========================================================================
    # runMC — segmented MCMC with pickle-based resume (original interface)
    # ========================================================================

    @timer
    def runMC(self, steps=50000, add_steps=2000, max_steps=200000,
              save_intermediate=True, **kwargs):
        """Segmented MCMC with automatic resume via pickle checkpoints.

        Convergence criterion: max(autocorr_time) * 50 < total_steps.

        Parameters
        ----------
        steps : int  initial number of steps per segment
        add_steps : int  steps added when not yet converged
        max_steps : int  hard stop after this many total steps
        save_intermediate : bool  save .npy + plots after each segment
        """
        ndim, nwalkers = self.n, self.nwalkers

        # -- resume or start fresh --
        saved_pos, all_chains, all_log_prob, total_steps = self._load_checkpoint()
        if saved_pos is not None:
            pos = saved_pos
            print("Resuming from checkpoint (step %d) ..." % total_steps)
        else:
            pos = self.pos
            print("Starting fresh MCMC from fiducial positions ...")

        if all_chains is None:
            total_steps = 0

        converged = False
        pool = _get_pool()

        try:
            while total_steps < max_steps:
                print("\nRunning MCMC for %d steps (total so far: %d) ..." %
                      (steps, total_steps))

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob,
                                                pool=pool, **kwargs)
                sampler.run_mcmc(pos, steps, progress=True, store=True,
                                 progress_kwargs={"file": sys.stdout})

                current_chain = sampler.get_chain().transpose(1, 0, 2)
                current_log_prob = sampler.get_log_prob().T

                if all_chains is None:
                    all_chains = current_chain
                    all_log_prob = current_log_prob
                else:
                    all_chains = np.concatenate([all_chains, current_chain], axis=1)
                    all_log_prob = np.concatenate([all_log_prob, current_log_prob], axis=1)

                total_steps += steps
                pos = sampler.get_last_sample().coords
                self._save_checkpoint(all_chains, all_log_prob, total_steps, pos)

                # -- convergence check (only: max(tau)*50 < total_steps) --
                try:
                    tau = emcee.autocorr.integrated_time(
                        all_chains.transpose(1, 0, 2), quiet=True)
                    self._tau_history.append((total_steps, np.max(tau)))
                    max_tau = np.max(tau)
                    print("Autocorrelation times: %s" % tau)
                    print("Max tau: %.2f" % max_tau)

                    converged = max_tau * 50 < total_steps
                    print("Convergence check: max(tau)*50 = %.2f < %d ? %s" %
                          (max_tau * 50, total_steps, converged))
                    if converged:
                        print("Chains converged at total iteration %d." % total_steps)
                except emcee.autocorr.AutocorrError:
                    converged = False
                    print("Autocorrelation time not yet estimable.")

                if save_intermediate:
                    self._save_intermediate_complete_results(all_chains, all_log_prob,
                                                             total_steps)

                if converged:
                    break

                print("Chains not yet converged, adding %d steps ..." % add_steps)
                steps = add_steps

                if total_steps >= max_steps:
                    print("Reached maximum steps (%d), stopping." % max_steps)
                    break
        finally:
            # On macOS spawn context, the pool reference inside the EnsembleSampler
            # must not outlive our pool context. Ensure sampler is deleted first.
            try:
                del sampler
            except NameError:
                pass
            _close_pool(pool)

        # -- save final results --
        class _FakeSampler:
            def __init__(s, chain, lnprob):
                s.chain = chain
                s.lnprobability = lnprob

        final_sampler = _FakeSampler(all_chains, all_log_prob)
        self.sampler = final_sampler
        n_burn = burn_in_steps(total_steps)
        self.savefile(final_sampler, burn_in=n_burn)
        self._save_final_results(all_chains, all_log_prob, burn_in=n_burn)
        write_burn_in_record(total_steps, n_burn, source="pickle")

        if converged:
            self._cleanup_checkpoint()

        print("\nAll MCMC runs completed. Total steps: %d" % total_steps)
        return final_sampler

    # ========================================================================
    # run — HDF5-based MCMC (tutorial-style, cleaner resume)
    # ========================================================================

    @timer
    def run(self, steps=50000, add_steps=2000, max_steps=200000,
            save_intermediate=True, burn_in=None, burn_in_frac=None, **kwargs):
        """HDF5-backed segmented MCMC with convergence checks.

        Convergence criterion: max(autocorr_time) * 50 < total_steps.

        Compared to ``runMC``, this method uses an emcee HDF5 backend
        (inspired by the emcee tutorial) for cleaner state persistence.

        Parameters
        ----------
        steps : int  initial steps per segment
        add_steps : int  extra steps per segment when not yet converged
        max_steps : int  hard stop after this many total steps
        save_intermediate : bool  save .npy + plots after each segment
        burn_in : int or None  steps discarded from every walker.  ``None``
            (default) means ``burn_in_frac`` of the steps actually run
        burn_in_frac : float or None  fraction of total steps to discard
            (default 0.3, overridable by ``MCMC_BURN_IN_FRAC``)
        """
        if burn_in_frac is None:
            burn_in_frac = burn_in_frac_default()
        if os.environ.get("MCMC_STEPS", "").strip():
            steps = int(os.environ["MCMC_STEPS"])
        if os.environ.get("MCMC_ADD_STEPS", "").strip():
            add_steps = int(os.environ["MCMC_ADD_STEPS"])
        if os.environ.get("MCMC_MAX_STEPS", "").strip():
            max_steps = int(os.environ["MCMC_MAX_STEPS"])
        ndim, nwalkers = self.n, self.nwalkers
        temple_dir = _path('temple', self.Chains_name)
        os.makedirs(temple_dir, exist_ok=True)
        backend_file = os.path.join(temple_dir, "backend.h5")
        tau_history_file = os.path.join(temple_dir, "tau_history.npy")

        # -- initialise or resume --
        backend = emcee.backends.HDFBackend(backend_file)
        resume = False
        old_shape = None
        if backend.initialized:
            try:
                old_shape = tuple(backend.shape)
                resume = backend.iteration > 0 and old_shape == (nwalkers, ndim)
            except (OSError, IOError, KeyError):
                resume = False
        if resume:
            print("HDF5 backend found. Resuming from iteration %d." % backend.iteration)
            pos = backend.get_last_sample().coords
            # restore tau_history so convergence plot continues across resumes
            if os.path.exists(tau_history_file):
                try:
                    self._tau_history = list(np.load(tau_history_file, allow_pickle=True))
                except Exception:
                    self._tau_history = []
        else:
            if old_shape is not None:
                print("HDF5 backend shape %s != (%d, %d); starting a new chain."
                      % (old_shape, nwalkers, ndim))
            else:
                print("Starting fresh MCMC with HDF5 backend.")
            backend.reset(nwalkers, ndim)
            pos = self.pos
            self._tau_history = []
            if os.path.exists(tau_history_file):
                try:
                    os.remove(tau_history_file)
                except OSError:
                    pass

        converged = False
        total_steps = backend.iteration

        pool = _get_pool()
        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob,
                                            backend=backend, pool=pool, **kwargs)

            while sampler.iteration < max_steps:
                seg = min(steps, max_steps - sampler.iteration)
                print("\nRunning MCMC for %d steps (total so far: %d) ..." %
                      (seg, sampler.iteration))
                sys.stdout.flush()
                _progress_log(
                    "MCMC segment start  done=%d  next=%d  max=%d"
                    % (sampler.iteration, seg, max_steps))
                sampler.run_mcmc(pos, seg, progress=True, store=True,
                                 progress_kwargs={"file": sys.stdout})
                pos = sampler.get_last_sample().coords
                total_steps = sampler.iteration
                _progress_log(
                    "MCMC progress  iteration=%d/%d (%.1f%%)"
                    % (sampler.iteration, max_steps,
                       100.0 * sampler.iteration / float(max_steps)))

                # Full chain as (nwalkers, nsteps, ndim)
                chain = sampler.get_chain().transpose(1, 0, 2)
                log_prob = sampler.get_log_prob().T

                # -- convergence check (only: max(tau)*50 < total_steps) --
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                    self._tau_history.append((total_steps, np.max(tau)))
                    # persist tau_history for resume support
                    try:
                        np.save(tau_history_file, self._tau_history, allow_pickle=True)
                    except Exception:
                        pass
                    max_tau = np.max(tau)
                    print("Autocorrelation times: %s" % tau)
                    print("Max tau: %.2f" % max_tau)

                    converged = max_tau * 50 < total_steps
                    print("Convergence check: max(tau)*50 = %.2f < %d ? %s" %
                          (max_tau * 50, total_steps, converged))
                    _progress_log(
                        "convergence check  iteration=%d  max_tau=%.2f  "
                        "need<%.0f  converged=%s"
                        % (total_steps, max_tau, max_tau * 50, converged))
                    if converged:
                        print("Chains converged at iteration %d." % total_steps)
                        _progress_log("chains converged at iteration %d"
                                      % total_steps)
                except emcee.autocorr.AutocorrError:
                    converged = False
                    print("Autocorrelation time not yet estimable.")
                    _progress_log(
                        "autocorr not yet estimable at iteration %d"
                        % total_steps)

                if save_intermediate:
                    self._save_intermediate_complete_results(chain, log_prob, total_steps)

                if converged:
                    break

                if sampler.iteration >= max_steps:
                    print("Reached maximum steps (%d), stopping." % max_steps)
                    _progress_log("reached max_steps=%d" % max_steps)
                    break

                print("Chains not yet converged, adding %d steps ..." % add_steps)
                _progress_log("adding %d more steps" % add_steps)
                steps = add_steps
        finally:
            try:
                del sampler
            except NameError:
                pass
            _close_pool(pool)

        # -- build a fake-sampler for savefile compatibility --
        chain_data = backend.get_chain().transpose(1, 0, 2)  # (nwalkers, nsteps, ndim)
        log_prob_data = backend.get_log_prob().T             # (nwalkers, nsteps)

        class _FakeSampler:
            def __init__(s, c, lp):
                s.chain = c
                s.lnprobability = lp

        final_sampler = _FakeSampler(chain_data, log_prob_data)
        self.sampler = final_sampler
        n_burn = int(burn_in) if burn_in is not None else burn_in_steps(
            total_steps, burn_in_frac)
        self.savefile(final_sampler, burn_in=n_burn)
        self._save_final_results(chain_data, log_prob_data, burn_in=n_burn)
        write_burn_in_record(total_steps, n_burn, burn_in_frac, source="backend")

        if converged:
            try:
                # keep HDF5 for possible later inspection
                pass
            except Exception:
                pass

        print("\nAll MCMC runs completed. Total steps: %d" % total_steps)
        _progress_log("MCMC complete  total_steps=%d  burn_in=%d"
                      % (total_steps, n_burn))
        return final_sampler

    # -- final results --------------------------------------------------

    def _save_final_results(self, samples, log_prob, burn_in=None):
        """Write results.txt after a completed run."""
        try:
            n_steps = int(np.asarray(samples).shape[1])
            if burn_in is None:
                burn_in = burn_in_steps(n_steps)
            n_params = self.n
            param_names = [str(x) for x in self.params_all['name']]
            flat_samples, flat_log_prob = self._burn_and_flatten(
                samples, log_prob, burn_in)

            constraints = []
            for i in range(n_params):
                lo, med, hi = np.percentile(flat_samples[:, i], [16, 50, 84])
                constraints.append((float(med), float(med - lo), float(hi - med)))

            chi2_min = float(-2 * np.max(flat_log_prob))
            dof = int(self.data_num) - n_params
            chi2_red = chi2_min / dof if dof > 0 else float('nan')
            aic = chi2_min + 2.0 * n_params
            bic = chi2_min + n_params * np.log(max(int(self.data_num), 1))

            _write_results_txt(
                self.Chains_name, n_params, constraints, param_names,
                chi2_min, dof, chi2_red, aic, bic,
                data_num=int(self.data_num),
                burn_in=burn_in,
                n_steps=n_steps,
            )
        except Exception as e:
            print("Warning: Could not save final results: %s" % e)

    # -- convenience ---------------------------------------------

    def run_dynesty(self, dlogz_init=0.01, maxcall=600000, bound='multi',
                    sample='rwalk', ncores=None, evidence_path=None,
                    extra_meta=None):
        """Run dynesty nested sampling using this object's chi2 and priors."""
        params = [
            [self.params_all['name'][i],
             float(self.params_all['fit'][i]),
             float(self.params_all['lower'][i]),
             float(self.params_all['upper'][i])]
            for i in range(self.n)
        ]
        return run_dynesty(
            params, self.chi2, name=self.Chains_name,
            data_num=int(self.data_num), dlogz_init=dlogz_init,
            maxcall=maxcall, bound=bound, sample=sample, ncores=ncores,
            evidence_path=evidence_path, extra_meta=extra_meta,
        )

    def check(self, *arg):
        """Print chi^2 at fiducial (or at *arg* if given)."""
        if arg:
            print('%s = %s' % (self.chi2.__name__, self.chi2(arg)))
        else:
            print('%s = %s' % (self.chi2.__name__, self.chi2(self.params_all['fit'])))


# ============================================================================
# Multiprocessing helper
# ============================================================================

def _progress_log(msg):
    """Print a timestamped progress line and optionally append to a log file.

    Set ``MYMCMC_PROGRESS_LOG`` to a path to mirror status into a small
    watchable file alongside the noisy tqdm output.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (stamp, msg)
    print(line, flush=True)
    path = os.environ.get("MYMCMC_PROGRESS_LOG", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print("Warning: could not write progress log %s: %s" % (path, exc),
              flush=True)


def _progress_chunk_size(default=100):
    raw = os.environ.get("MYMCMC_PROGRESS_CHUNK", "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(1, value)


def _limit_threads():
    """Keep BLAS/OpenMP single-threaded in forked MCMC workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _close_pool(pool):
    if pool is None:
        return
    pool.close()
    pool.terminate()
    pool.join()


def _get_pool():
    """Return an emcee worker pool for Linux SLURM / local runs.

    On Linux, ``fork`` is the correct start method for emcee: walkers call a
    bound method (``self._lnprob``), which ``spawn`` cannot pickle. macOS
    defaults to ``spawn``; we still request ``fork`` there for the same reason.

    ``MYMCMC_NCORES`` must equal the allocated cores (``SLURM_CPUS_PER_TASK``).
    Set ``MYMCMC_SERIAL=1`` to disable the pool. Set ``OMP_NUM_THREADS=1``
    before importing numpy/pyccl so forked workers do not deadlock.
    """
    if os.environ.get("MYMCMC_SERIAL", "").lower() in ("1", "true", "yes"):
        print("MCMC pool: serial (MYMCMC_SERIAL=1)")
        return None
    try:
        nproc = int(os.environ.get("MYMCMC_NCORES", "0") or 0)
        if nproc <= 0:
            nproc = os.cpu_count() or 1
        ctx = multiprocessing.get_context("fork")
        print("MCMC pool: Linux fork, %d worker processes (emcee EnsembleSampler)"
              % nproc)
        sys.stdout.flush()
        return ctx.Pool(nproc, initializer=_limit_threads)
    except (ValueError, AttributeError):
        warnings.warn("'fork' unavailable; running MCMC without parallel pool.")
        return None


def _write_results_txt(chain_name, n_params, constraints, pnames,
                       chi2_min, dof, chi2_red, aic, bic,
                       logZ=None, logZ_err=None, data_num=None,
                       burn_in=None, n_steps=None):
    """Append MCMC results to results/results.txt (module-level helper)."""
    results_dir = _path('results')
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, 'results.txt')

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('Run: %s\n' % chain_name)
        f.write('Time: %s\n' % now)
        f.write('=' * 80 + '\n')
        if data_num is not None:
            f.write('Data points: %d\n' % int(data_num))
        if n_steps is not None:
            f.write('Steps per walker: %d\n' % int(n_steps))
        if burn_in is not None:
            f.write('Burn-in: %d steps per walker\n' % int(burn_in))
            if n_steps:
                f.write('Burn-in fraction: %.3f\n'
                        % (float(burn_in) / float(n_steps)))
        f.write('Parameters (%d):\n' % n_params)

        for i in range(n_params):
            med, lo, hi = constraints[i]
            pname = pnames[i]
            f.write('  %s = %.3f  [-%.3f, +%.3f]\n' % (pname, med, lo, hi))

        f.write('\nFit statistics:\n')
        f.write('  chi2_min    = %.3f\n' % chi2_min)
        f.write('  chi2/dof    = %.3f  (dof=%d)\n' % (chi2_red, dof))
        f.write('  AIC         = %.3f\n' % aic)
        f.write('  BIC         = %.3f\n' % bic)
        if logZ is not None:
            f.write('  ln(Z)       = %.3f +/- %.3f\n' % (logZ, logZ_err))

    print("Results appended to: %s" % txt_path)


def _save_results_pdf(chain_name, n_params, constraints, pnames,
                      chi2_min, dof, chi2_red, aic, bic,
                      logZ=None, logZ_err=None, data_num=None):
    """Save results PDF — two-column layout, large fonts."""
    try:
        results_dir = _path('results')
        os.makedirs(results_dir, exist_ok=True)
        pdf_path = os.path.join(results_dir, chain_name + '_constraint_reslu.pdf')

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nrows = int(np.ceil(n_params / 2.0))

        # Height budget
        fig_h = max(5.5, 2.5 + nrows * 0.1)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        fy = 0.95

        # ---- Title ----
        ax.text(0.5, fy, 'MCMC Results: %s' % chain_name.replace('_', ' '),
                fontsize=22, fontweight='bold', ha='center', va='top')
        fy -= 0.10

        # ---- Subtitle ----
        subtitle_lines = ['Time: %s' % now]
        if data_num is not None:
            subtitle_lines.append('Data points: %d' % int(data_num))
        for line in subtitle_lines:
            ax.text(0.5, fy, line, fontsize=16, ha='center', va='top', color='gray')
            fy -= 0.065
        fy -= 0.015

        # ---- Parameters header ----
        ax.text(0.5, fy, 'Parameters (%d)' % n_params,
                fontsize=20, fontweight='bold', ha='center', va='top')
        fy -= 0.085

        # ---- Two-column parameters ----
        col_x = [0.16, 0.66]
        param_font = 20
        row_h = 0.1
        for i in range(n_params):
            med, lo, hi = constraints[i]
            pname = pnames[i]
            col = i % 2
            row = i // 2
            x = col_x[col]
            y = fy - row * row_h
            txt = r'$\mathrm{%s} = %.3f_{-\,%.3f}^{+\,%.3f}$' % (pname, med, lo, hi)
            ax.text(x, y, txt, fontsize=param_font, va='top')

        fy -= nrows * row_h + 0.05

        # ---- Separator ----
        ax.axhline(y=fy + 0.02, xmin=0.1, xmax=0.9, color='gray', linewidth=1, alpha=0.5)
        fy -= 0.04

        # ---- Fit Statistics header ----
        ax.text(0.5, fy, 'Fit Statistics',
                fontsize=20, fontweight='bold', ha='center', va='top')
        fy -= 0.085

        # ---- Fit Statistics — two columns ----
        stat_font = 20
        stat_row_h = 0.1
        left_stats = [
            r'$\chi^2_{\rm min}$ = %.3f' % chi2_min,
            r'$\chi^2_{\rm min}\,/\,{\rm dof}$ = %.3f  (dof = %d)' % (chi2_red, dof),
        ]
        right_stats = [
            r'$\mathrm{AIC}$ = %.3f' % aic,
            r'$\mathrm{BIC}$ = %.3f' % bic,
        ]
        if logZ is not None:
            left_stats.append(r'$\ln(Z)$ = %.3f $\pm$ %.3f' % (logZ, logZ_err))

        n_stat_rows = max(len(left_stats), len(right_stats))
        for row in range(n_stat_rows):
            if row < len(left_stats):
                ax.text(col_x[0], fy - row * stat_row_h, left_stats[row],
                        fontsize=stat_font, va='top')
            if row < len(right_stats):
                ax.text(col_x[1], fy - row * stat_row_h, right_stats[row],
                        fontsize=stat_font, va='top')

        plt.tight_layout(pad=0.5)
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Results PDF saved: %s" % pdf_path)
    except Exception as e:
        print("Warning: Could not save results PDF: %s" % e)


def _dynesty_equal_samples(res):
    samples = np.asarray(res.samples, dtype=float)
    try:
        logwt = np.asarray(res.logwt, dtype=float)
        weights = np.exp(logwt - float(res.logz[-1]))
        wsum = float(np.sum(weights))
        if wsum <= 0.0 or not np.isfinite(wsum):
            return samples
        from dynesty.utils import resample_equal
        return resample_equal(samples, weights / wsum)
    except Exception:
        return samples


def run_dynesty(params, chi2, name="dynesty", data_num=0, dlogz_init=0.01,
                maxcall=600000, bound='multi', sample='rwalk', ncores=None,
                chain_samples=None, pnames=None, evidence_path=None,
                extra_meta=None):
    """Run dynesty DynamicNestedSampler and return Bayesian evidence.

    ``params`` uses the MCMC_class format ``[name, fit, low, high]``.
    ``chi2`` is theta -> chi^2, with ln L = -chi^2 / 2.
    """
    global _DYNESTY_CHI2, _DYNESTY_PARAMS
    _DYNESTY_CHI2 = chi2
    _DYNESTY_PARAMS = [list(row) for row in params]
    ndim = len(_DYNESTY_PARAMS)
    if ncores is None:
        try:
            ncores = int(os.environ.get("MYMCMC_NCORES", "0") or 0)
        except ValueError:
            ncores = 0
        if ncores <= 0:
            ncores = max(1, multiprocessing.cpu_count() - 1)
    ncores = max(1, int(ncores))

    print("\n=== Running Dynesty for %s ===" % name)
    print("  ndim=%d, ncores=%d, maxcall=%d, dlogz_init=%g"
          % (ndim, ncores, maxcall, dlogz_init))

    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(ncores, initializer=_limit_threads)
    try:
        sampler = dynesty.DynamicNestedSampler(
            _dynesty_loglike, _dynesty_prior_transform, ndim=ndim,
            bound=bound, sample=sample,
            pool=pool, queue_size=ncores,
        )
        sampler.run_nested(dlogz_init=dlogz_init, maxcall=maxcall,
                           print_progress=True)
        res = sampler.results
    finally:
        _close_pool(pool)

    logZ = float(res.logz[-1])
    logZ_err = float(res.logzerr[-1])
    logl_max = float(np.max(res.logl))
    chi2_min = -2.0 * logl_max
    data_n = int(data_num)
    dof = data_n - ndim
    chi2_red = chi2_min / dof if dof > 0 else float('nan')
    aic = chi2_min + 2.0 * ndim
    bic = chi2_min + ndim * np.log(max(data_n, 1))
    names = [str(row[0]) for row in _DYNESTY_PARAMS]
    if pnames is None:
        pnames = names

    print("\n  %s dynesty results:" % name)
    print("    logZ       = %.3f +/- %.3f" % (logZ, logZ_err))
    print("    chi2_min   = %.3f" % chi2_min)
    print("    chi2/dof   = %.3f  (dof=%d)" % (chi2_red, dof))
    print("    AIC        = %.3f" % aic)
    print("    BIC        = %.3f" % bic)

    if chain_samples is not None:
        post = np.asarray(chain_samples, dtype=float)
    else:
        post = _dynesty_equal_samples(res)
    constraints = []
    for i in range(ndim):
        lo, med, hi = np.percentile(post[:, i], [16, 50, 84])
        constraints.append((float(med), float(med - lo), float(hi - med)))

    results_dir = _path('results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(_path('chains'), exist_ok=True)
    _write_results_txt(
        name, ndim, constraints, pnames,
        chi2_min, dof, chi2_red, aic, bic,
        logZ=logZ, logZ_err=logZ_err, data_num=data_n,
    )
    _save_results_pdf(
        name, ndim, constraints, pnames,
        chi2_min, dof, chi2_red, aic, bic,
        logZ=logZ, logZ_err=logZ_err, data_num=data_n,
    )
    np.savez(
        _path('chains', name + '_dynesty.npz'),
        samples=np.asarray(res.samples, dtype=float),
        logwt=np.asarray(res.logwt, dtype=float),
        logz=np.asarray(res.logz, dtype=float),
        logzerr=np.asarray(res.logzerr, dtype=float),
        logl=np.asarray(res.logl, dtype=float),
        param_names=np.array(names),
    )

    payload = {
        'name': name,
        'n_params': ndim,
        'param_names': names,
        'logZ': logZ,
        'logZ_err': logZ_err,
        'chi2_min': chi2_min,
        'chi2_red': chi2_red,
        'dof': dof,
        'AIC': aic,
        'BIC': bic,
        'data_num': data_n,
        'dlogz_init': float(dlogz_init),
        'maxcall': int(maxcall),
        'constraints': {
            names[i]: {
                'median': constraints[i][0],
                'minus': constraints[i][1],
                'plus': constraints[i][2],
            }
            for i in range(ndim)
        },
    }
    if extra_meta:
        payload.update(extra_meta)
    if evidence_path is None:
        evidence_path = os.path.join(results_dir, 'dynesty_evidence.json')
    os.makedirs(os.path.dirname(os.path.abspath(evidence_path)) or '.',
                exist_ok=True)
    with open(evidence_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
        fh.write('\n')
    print("Saved evidence JSON:", evidence_path)
    return payload


# ============================================================================
# MCplot — getdist-based plotting (same colour scheme as qcosmc)
# ============================================================================

class MCplot:
    """Load and plot MCMC chains with getdist.

    Parameters
    ----------
    Chains : list of [root, legend] pairs
    new_name : list of str or None  override parameter names
    ignore_rows : float  fraction of rows to ignore in getdist
    ccolor : list or None  custom contour colours
    """

    def __init__(self, Chains, new_name=None, ignore_rows=0.0, ccolor=None):
        self.root = [i[0] for i in Chains]
        self.lengend = [i[1] for i in Chains]
        self._n = len(Chains)
        self.minkaf = np.zeros(self._n)
        self.data_num = np.zeros(self._n)
        self.ccolor = ccolor
        self.Samp = []

        for i in range(self._n):
            savefile_name = _path('chains',
                                         self.root[i] + ".npy")
            data = np.load(savefile_name, allow_pickle=True)
            self.samples = data[0]
            self.log_prob = data[1]
            self.theta_name = data[2]
            self.theta_fit = data[3]
            self.theta_fact = data[4]
            self.minkaf[i] = data[5]
            self.data_num[i] = data[6]
            ranges = data[7]

            self.theta_name = [str(x) for x in data[2]]
            # new_name is display LaTeX only. Getdist splits `names` on spaces, so
            # using r"\sigma_{\rm NL}" as a name collides with r"\sigma_{\rm host}".
            if new_name:
                self.label_name = [str(x) for x in new_name]
            else:
                self.label_name = list(self.theta_name)
            self.Samp.append(MCSamples(
                samples=self.samples,
                names=self.theta_name,
                labels=self.label_name,
                ranges=ranges,
                settings={'ignore_rows': ignore_rows},
            ))

        self.param_names = [na.name for na in self.Samp[0].getParamNames().names]

    # -- results tables -------------------------------------------------

    @property
    def results(self):
        """Alias for results2."""
        return self.results2

    @property
    def results_app(self):
        """Alias for results2."""
        return self.results2

    @property
    def results2(self):
        """Print 1-sigma constraints, save clean PDF."""
        all_results = []
        for k in range(self._n):
            n_params = len(self.Samp[k].getParamNames().names)
            pnames = self.label_name
            samples_k = self.Samp[k].samples
            chain_name = self.root[k]

            constraints = []
            for i in range(n_params):
                lo, med, hi = np.percentile(samples_k[:, i], [16, 50, 84])
                constraints.append((float(med), float(med - lo), float(hi - med)))

            chi2_min = float(self.minkaf[k])
            dof = int(self.data_num[k]) - n_params
            chi2_red = chi2_min / dof if dof > 0 else float('nan')
            aic = chi2_min + 2.0 * n_params
            bic = chi2_min + n_params * np.log(max(self.data_num[k], 1))

            # stdout
            result_lines = []
            for i in range(n_params):
                med, lo, hi = constraints[i]
                pname = pnames[i]
                if np.isclose(lo, hi):
                    result_lines.append(
                        r"$\mathrm{%s} = %.3f \pm %.3f}$" % (pname, med, lo))
                else:
                    result_lines.append(
                        r"$\mathrm{%s} = %.3f_{-\,%.3f}^{+\,%.3f}$" % (pname, med, lo, hi))
            all_results.append(',  '.join(result_lines))

            print("\n--- Results for dataset: %s ---" % chain_name)
            print(',  '.join(result_lines))
            print(r'$\chi^2_{\rm min}$ = %.3f' % chi2_min)
            print(r'$\chi^2_{\rm min}\,/\,{\rm dof}$ = %.3f  (dof = %d)' % (chi2_red, dof))
            print(r'$\mathrm{AIC}$ = %.3f' % aic)
            print(r'$\mathrm{BIC}$ = %.3f' % bic)

            # PDF (for loading pre-existing chains where runner wasn't used)
            _save_results_pdf(chain_name, n_params, constraints, pnames,
                              chi2_min, dof, chi2_red, aic, bic,
                              data_num=int(self.data_num[k]))

        return all_results

    # -- dynesty nested sampling ------------------------------------------

    def run_dynesty(self, params, chi2, dlogz_init=0.01, maxcall=600000,
                    bound='multi', sample='rwalk', ncores=None,
                    evidence_path=None, extra_meta=None):
        """Run dynesty DynamicNestedSampler and return Bayesian evidence."""
        samples = self.Samp[0].samples if self.Samp else None
        pnames = (self.label_name if hasattr(self, 'label_name')
                  else [p[0] for p in params])
        data_n = int(self.data_num[0]) if len(self.data_num) > 0 else 0
        return run_dynesty(
            params, chi2, name=self.root[0], data_num=data_n,
            dlogz_init=dlogz_init, maxcall=maxcall, bound=bound,
            sample=sample, ncores=ncores, chain_samples=samples,
            pnames=pnames, evidence_path=evidence_path, extra_meta=extra_meta,
        )

    # -- 1D / 2D / 3D (triangle) plots -----------------------------------

    def plot1D(self, n, colorn=0, width_inch=8, **kwargs):
        """1D marginalised posterior for the *n*-th parameter (1-indexed)."""
        g = plots.getSinglePlotter(width_inch=width_inch)
        g.plot_1d(self.Samp, self.param_names[n - 1], ls=lss,
                  colors=colors[colorn:colorn + self._n],
                  lws=[1.5] * self._n, **kwargs)
        ax = plt.gca()
        if all(self.lengend):
            leg = ax.legend(self.lengend, loc=1, fontsize=16, frameon=False)
            for line, text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'], lw=1.5)
        if 'xaxis' in kwargs:
            ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['xaxis']))
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        results_dir = _path('results')
        os.makedirs(results_dir, exist_ok=True)
        if 'name' in kwargs:
            g.export(os.path.join(results_dir, '%s.pdf' % kwargs['name']))
        else:
            g.export(os.path.join(results_dir,
                      ''.join(self.root) + self.param_names[n - 1].replace('\\', '')
                      + '_contour_map.pdf'))
        return g

    def plot2D(self, pp, colorn=0, contour_num=2, width_inch=8, **kwargs):
        """2D joint posterior for parameter indices *pp* = [i, j] (1-indexed)."""
        g = plots.getSinglePlotter(width_inch=width_inch, ratio=1)
        if self.ccolor:
            g.settings.solid_colors = self.ccolor
        g.settings.num_plot_contours = contour_num
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 18
        g.settings.legend_frame = False

        g.plot_2d(self.Samp,
                  self.param_names[pp[0] - 1],
                  self.param_names[pp[1] - 1],
                  filled=True, **kwargs)

        ax = g.get_axes()
        if 'x_locator' in kwargs:
            ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['x_locator']))
        if 'y_locator' in kwargs:
            ax.yaxis.set_major_locator(plt.MultipleLocator(kwargs['y_locator']))
        if 'lims' in kwargs:
            xmin, xmax, ymin, ymax = kwargs['lims']
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'], lw=1.5)
        if hasattr(self, 'lengend') and self.lengend:
            g.add_legend(self.lengend, fontsize=16)

        results_dir = _path('results')
        os.makedirs(results_dir, exist_ok=True)
        if 'name' in kwargs:
            g.export(os.path.join(results_dir, "%s.pdf" % kwargs['name']))
        else:
            g.export(os.path.join(results_dir,
                                  ''.join(self.root) + '_contour_map.pdf'))
        return g

    def plot3D(self, pp, colorn=0, contour_num=2, legend_fontsize=18,
               axes_fontsize=14, lab_fontsize=18, text_annotations=None,
               **kwargs):
        """Triangle (corner) plot.

        Parameters
        ----------
        pp : list  parameter indices (1-indexed), or 0 for all
        colorn : int  offset into the global colour list
        contour_num : int  number of contour levels
        text_annotations : list of dict or None  e.g. [{'text':'...','x':0.5,'y':0.5}]
        """
        if pp == 0:
            t_name = self.param_names
        else:
            t_name = [self.param_names[i - 1] for i in pp]

        g = plots.get_subplot_plotter(width_inch=9)
        if self.ccolor:
            g.settings.solid_colors = self.ccolor
        g.settings.num_plot_contours = contour_num
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axes_fontsize = axes_fontsize
        g.settings.lab_fontsize = lab_fontsize
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add = 0.8

        if all(self.lengend):
            g.triangle_plot(self.Samp, t_name, filled_compare=True,
                            contour_lws=2, legend_labels=self.lengend,
                            contour_colors=colors[colorn:colorn + self._n],
                            legend_loc='upper right', **kwargs)
        else:
            g.triangle_plot(self.Samp, t_name, filled_compare=True,
                            contour_lws=2,
                            contour_colors=colors[colorn:colorn + self._n],
                            **kwargs)

        if 'xlim' in kwargs:
            for xi in kwargs['xlim']:
                for ax in g.subplots[:, xi[0] - 1]:
                    if ax is not None:
                        ax.set_xlim(xi[1][0], xi[1][1])
        if 'tline' in kwargs:
            for ax in g.subplots[:, 0]:
                ax.axvline(kwargs['tline'], color='k', ls='--', alpha=0.5)
        if 'ax_range' in kwargs:
            for axi in kwargs['ax_range']:
                g.subplots[-1, axi[0] - 1].xaxis.set_major_locator(
                    plt.MultipleLocator(axi[1]))
                if axi[0] - 1 > 0:
                    g.subplots[axi[0] - 1, 0].yaxis.set_major_locator(
                        plt.MultipleLocator(axi[1]))
        if text_annotations:
            for ann in text_annotations:
                txt_kw = {k: v for k, v in ann.items()
                          if k not in ('text', 'x', 'y', 'fontsize', 'color')}
                g.fig.text(ann.get('x', 0.5), ann.get('y', 0.5),
                           ann.get('text', ''),
                           fontsize=ann.get('fontsize', 12),
                           color=ann.get('color', 'black'),
                           **txt_kw)

        results_dir = _path('results')
        os.makedirs(results_dir, exist_ok=True)
        if 'name' in kwargs:
            g.export(os.path.join(results_dir, "%s.pdf" % kwargs['name']))
        else:
            g.export(os.path.join(results_dir,
                                  ''.join(self.root) + '_contour_map.pdf'))
        return g

    def rename(self, new_name):
        """Rename parameters in the saved chain files."""
        for i in range(self._n):
            savefile_name = _path('chains',
                                         self.root[i] + ".npy")
            data = np.load(savefile_name, allow_pickle=True)
            np.save(savefile_name,
                    (data[0], new_name, data[3], data[4], data[5], data[6], data[7]))
        self.param_names = [na.name for na in self.Samp[0].getParamNames().names]
