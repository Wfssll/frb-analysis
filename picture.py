"""Render the fifteen adopted publication figures; see --help."""

from cosmology import production_cosmology
_COVARIANCE_ONLY = _SELECTION_ONLY = _INTRINSIC_ONLY = False
import json


import os


import sys


os.environ["DELAY_MODEL"] = "gauss"


os.environ["MCMC_MODE"] = "cl"


os.environ["OMP_NUM_THREADS"] = "1"


os.environ["OPENBLAS_NUM_THREADS"] = "1"


os.environ["MKL_NUM_THREADS"] = "1"


os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


os.environ["NUMEXPR_NUM_THREADS"] = "1"


from pathlib import Path


from concurrent.futures import ThreadPoolExecutor


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


import numpy as np


from matplotlib import font_manager


from matplotlib.patches import Patch


from matplotlib.ticker import (
    AutoMinorLocator, FixedLocator, FormatStrFormatter, LogLocator,
    MultipleLocator, NullFormatter,
)


if not (_COVARIANCE_ONLY or _SELECTION_ONLY or _INTRINSIC_ONLY):
    import pyccl as ccl
    from getdist import plots
    from scipy.integrate import cumulative_trapezoid, trapezoid
    from scipy.interpolate import interp1d


if not (_COVARIANCE_ONLY or _SELECTION_ONLY or _INTRINSIC_ONLY):
    import cl_dm_catalog as clcat
    import mcmc_frb_redshift as mc
    import dm_z as dmz
    from frb_redshift_model import get_engine
    from plot_result import (
        COLOR_WEAK, _chain, _getdist_settings, _optional_chain, load_chain, mcsamples,
    )


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(os.getenv("RESULT_ROOT", str(ROOT / "result"))).resolve()


RESULTS_TXT = RESULT_ROOT / "cl" / "gauss" / "results" / "results.txt"


OUT_DIR = ROOT / "picture"


PNG_DIR = OUT_DIR / "png"


OUT_PDF = OUT_DIR / "data.pdf"


XMIN = 30.0


PANEL_ORDER = ("BGS", "LRG", "DM")


FIGURE_OUTPUT_NAMES = {
    "frb_bias_gauss_cl_joint": "bias",
    "frb_posterior_gauss_cl_joint": "contour_map_gauss",
    "frb_posterior_lognormal_cl_joint": "contour_map_lognormal",
    "frb_posterior_power-law_cl_joint": "contour_map_powerlaw",
    "cl_dm_correlation": "cov_matrix",
    "cl_data_style_test": "data",
    "cl_data": "data",
    "frb_dmz_gauss_sfh_joint": "dmz",
    "bgs_lrg_nz_combined_test": "nz_galaxy",
    "frb_intrinsic_gauss_cl_joint_mean_std_1sigma": "pz",
    "frb_intrinsic_gauss_cl_joint_lightblue": "pz_goldblue",
    "frb_intrinsic_gauss_cl_joint": "pz_median",
    "frb_intrinsic_gauss_cl_joint_median_1sigma_2sigma": "pz_median_2sigma",
    "bias_three_model": "test_bias",
    "robust_constant_combined": "test_constant",
    "dmz_three_model_divide": "test_dmz",
    "cl_null_dm_test": "test_null",
    "cz_pz_three_model": "test_pz",
    "frb_intrinsic_observed_selection": "P_det",
    "frb_gaussian_delay_selection": "P_det",
    "P_det_test2_panels": "P_det_test2",
}


def figure_output_stem(stem):
    """Use current names while accepting legacy output-stem arguments."""
    return FIGURE_OUTPUT_NAMES.get(str(stem), str(stem))


def figure_output_path(path):
    path = Path(path)
    return path.with_name(figure_output_stem(path.stem) + path.suffix)


def load_intrinsic_summary(model, n_draw=10000):
    """Reuse moments and quantiles from the SAME CL/Joint posterior draws."""
    key = "powerlaw" if model == "power-law" else model
    cache = picture_asset(f"frb_intrinsic_{key}_cl_joint_interval_summary_data.npz")
    if cache.is_file() and not RECOMPUTE:
        with np.load(cache, allow_pickle=False) as saved:
            return dict(saved)
    result = intrinsic_posterior_data(n_draw=n_draw, delay_model=key, extended_summary=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PNG_DIR / cache.name, **result)
    return result


def plot_three_model_mean(n_draw=10000):
    result = {}
    for model in ("gauss", "lognormal", "power-law"):
        data = load_intrinsic_summary(model, n_draw=n_draw)
        data.pop("quantiles", None)
        data.update(summary_method="mean_std", std_ddof=0,
                    summary="Pointwise posterior mean +/- 1 population standard deviation for BOTH CL and Joint",
                    display_bounds="PDF clipped to nonnegative values; CDF clipped to [0,1]; raw means/std retained")
        for mode in ("cl", "joint"):
            for quantity in ("pdf", "cdf"):
                prefix = mode + "_" + quantity
                mean, std = data[prefix + "_mean"], data[prefix + "_std"]
                upper = 1. if quantity == "cdf" else np.inf
                data[prefix] = np.stack([np.clip(mean - std, 0., upper), mean,
                                         np.clip(mean + std, 0., upper)])
        result[model] = data
    return plot_three_model_pz(result=result, paired_only=True, output_stem="test_pz_mean")


def plot_constant_mean_clean():
    """Restyle saved mean/std predictions and unchanged bias profiles."""
    import robust

    source = picture_asset("test_constant_mean_data.npz")
    with np.load(source, allow_pickle=False) as saved:
        data = dict(saved)
    curves = {case["name"] + suffix: data["bias_" + case["name"] + suffix]
              for case in robust.PZ_CASES_CONSTANT for suffix in ("_x", "_p")}
    style = robust.plot_robust_constant_combined_clean(
        pz_data=data, curves=curves, output=OUT_DIR / "test_constant_mean_clean.pdf")
    np.savez_compressed(PNG_DIR / "test_constant_mean_clean_data.npz", **data)
    return style


def picture_asset_names(name):
    """Current asset name first, then legacy names for existing caches."""
    name = Path(name)
    suffixes = ("_data.npz", "_quantiles.csv", "_methods.md", ".png", ".svg", ".json", ".pdf")
    for old, current in sorted(FIGURE_OUTPUT_NAMES.items(), key=lambda item: len(item[0]), reverse=True):
        for suffix in suffixes:
            if name.name in (old + suffix, current + suffix):
                canonical = name.with_name(current + suffix)
                aliases = [name.with_name(alias + suffix) for alias, target in FIGURE_OUTPUT_NAMES.items()
                           if target == current]
                return list(dict.fromkeys([canonical, name, *aliases]))
    return [name]


def picture_asset(name):
    """Resolve current or legacy cache/table names without recomputation."""
    name = Path(name)
    if name.name == "localized_hosts_dm_eg.csv":
        return ROOT / "data/frb/localized_hosts_dm_eg.csv"
    if name.is_absolute():
        return name
    names = picture_asset_names(name)
    for candidate in names:
        for folder in (OUT_DIR, PNG_DIR, ROOT / "picture", ROOT / "picture" / "png"):
            path = folder / candidate
            if path.is_file():
                return path
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    return PNG_DIR / names[0]


def load_center(path):
    """Posterior center written by the C_ell Gaussian-delay run."""
    text = mc.last_run_text(path.read_text())
    return {
        "tau": mc.parse_param(text, "tau_gauss"),
        "sigma": mc.parse_param(text, "sigma_gauss"),
        "b_frb": mc.parse_param(text, "b_frb"),
        "k_cut": mc.parse_param(text, "k_cut"),
        "n_ell": mc.parse_param(text, "N_ell"),
        "dm_host": mc.parse_param(text, "dm_host"),
    }


def prepare(data):
    """Load the same spectra the C_ell chain used, and the CCL tracers."""
    mc.configure_mode("cl")
    mc.COSMO = clcat.make_cosmology()
    mc.Z = np.linspace(0.0, 5.0, 500)
    mc.get_engine(mc.COSMO, mc.Z)
    samples, joint = mc.load_cross_samples(data)
    mc.ELL_THEORY = joint["ell_theory"]
    mc.B_ELECTRON = float(data["b_electron"])
    mc.B_E_FIXED = mc.B_ELECTRON
    mc.K_CUT_FIXED = float(data["k_cut_mpc_inv"])
    mc.ELL_LOC_FIXED = float(data["ell_loc"])
    mc.DM_HOST_FIXED = float(data["dm_host_source"])
    mc.DM_MW_RESIDUAL = float(data["dm_mw_residual"])
    mc.B_FRB_FIXED = float(data["b_frb_source"])
    mc.D_BAR = float(data["frb_dm_mean"])
    mc.FIT_N_ELL = any(item["name"] == "DM" for item in samples)
    mc.N_ELL_FIXED = 0.0
    mc.PEG = clcat.make_peg_power(mc.COSMO, mc.K_CUT_FIXED)
    mc.PEE = clcat.make_pee_power(mc.COSMO, mc.K_CUT_FIXED)
    mc.DDM_DZ = clcat.mean_igm_dm_derivative(mc.COSMO, mc.Z)
    mc.COSMIC_DM = clcat.mean_cosmic_dm(mc.COSMO, mc.Z)
    for item in samples:
        if item["name"] == "DM":
            item["tracers"] = [None] * len(item["entries"])
            continue
        tracers = []
        for entry in item["entries"]:
            z_g, n_g = entry["nz_z"], entry["nz"]
            bias = float(entry["bias"])
            tracer = ccl.NumberCountsTracer(
                mc.COSMO, dndz=(z_g, n_g),
                bias=(z_g, np.full_like(z_g, bias)), has_rsd=False)
            tracers.append(tracer)
        item["tracers"] = tracers
    return samples


def theory_cl(ell, galaxy_tracer, dm, frb, k_cut, ell_loc, n_ell, kind):
    """Same C_ell as the best-fit panels in mcmc_frb_redshift.py."""
    beam = clcat.localization_beam(ell, ell_loc)
    peg = mc.peg_power(k_cut)
    if kind == "dm_auto":
        pee = mc.pee_power(k_cut)
        cl_ee = ccl.angular_cl(mc.COSMO, dm, dm, ell, p_of_k_a=pee)
        cl_ef = ccl.angular_cl(mc.COSMO, dm, frb, ell, p_of_k_a=peg)
        cl_ff = ccl.angular_cl(mc.COSMO, frb, frb, ell)
        return (cl_ee + 2.0 * cl_ef + cl_ff) * beam ** 2 + n_ell
    cl_eg = ccl.angular_cl(mc.COSMO, galaxy_tracer, dm, ell, p_of_k_a=peg)
    cl_fg = ccl.angular_cl(mc.COSMO, galaxy_tracer, frb, ell)
    return (cl_eg + cl_fg) * beam


def ylabel_for(kind, scale_power):
    """Spectrum label with the DM units used by the catalog.

    The unit belongs to the density field, so the galaxy-DM cross spectrum is
    in pc cm^-3 and its auto spectrum in (pc cm^-3)^2. Arial has no Unicode
    superscripts here, so the labels stay in mathtext.
    """
    if kind == "dm_auto":
        return (r"$C_{\ell}^{\mathrm{DD}}\ "
                r"[(\mathrm{pc}\,\mathrm{cm}^{-3})^{2}]$")
    base = r"C_{\ell}^{\mathrm{Dg}}"
    if scale_power:
        return (rf"$10^{{{scale_power}}}\ell\,{base}\ "
                r"[\mathrm{pc}\,\mathrm{cm}^{-3}]$")
    return rf"$\ell\,{base}\ [\mathrm{{pc}}\,\mathrm{{cm}}^{{-3}}]$"


def clear_upper_right(ell, y_line, ell_data, y_lo_data, y_hi_data,
                      x_frac=0.45, y_frac=0.75, margin=0.05):
    """y limits that leave the upper-right corner free for an in-panel label.

    Autoscaling alone let the theory curve or a tall error bar run through the
    corner, which is why two panels previously had to move their bin label to
    the bottom right. Everything right of ``x_frac`` of the log-x axis (curve
    and error-bar caps alike) is now required to stay below ``y_frac`` of the
    panel height, which the one-line label occupies above ~0.80. The upper
    limit is raised only when that fails, so panels that already have room
    keep their autoscale.
    """
    ell = np.asarray(ell, dtype=float)
    y_line = np.asarray(y_line, dtype=float)
    ell_data = np.asarray(ell_data, dtype=float)
    lows = [np.min(y_lo_data)]
    highs = [np.max(y_hi_data)]
    # y_line is empty for the null test, which has no model curve.
    if y_line.size:
        lows.append(float(y_line.min()))
        highs.append(float(y_line.max()))
    lo = float(min(lows))
    hi = float(max(highs))
    span = hi - lo
    if span <= 0.0:
        span = 1.0
    lo -= margin * span
    hi += margin * span
    log_lo, log_hi = np.log10(XMIN), np.log10(np.max(ell))
    cut = 10.0 ** (log_lo + x_frac * (log_hi - log_lo))
    right = []
    if y_line.size and np.any(ell >= cut):
        right.append(float(np.max(y_line[ell >= cut])))
    far = ell_data >= cut
    if np.any(far):
        right.append(float(np.max(np.asarray(y_hi_data)[far])))
    if right:
        right_max = max(right)
        if right_max > lo + y_frac * (hi - lo):
            hi = lo + (right_max - lo) / y_frac
    return lo, hi


def style_x_axis(ax, ell_max):
    """Left edge at 30. Major labels are only 10^2 and 10^3."""
    ax.set_xscale("log")
    ax.set_xlim(XMIN, ell_max)
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e3]))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _pos: r"$10^{%d}$" % int(round(np.log10(value)))))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def collect_panels(samples):
    by_name = {item["name"]: item for item in samples}
    panels = []
    for name in PANEL_ORDER:
        item = by_name[name]
        n_bin = len(item["entries"])
        for i, entry in enumerate(item["entries"]):
            closing = "]" if i == n_bin - 1 else ")"
            panels.append((item, i, entry, closing))
    if len(panels) != 9:
        raise RuntimeError("expected 9 spectra, found %d" % len(panels))
    return panels


COLOR_JOINT = "#7796AD"


POSTERIOR_PROBES = (
    {"mode": "cl", "stem": "frb_redshift", "color": "#EC8FB4", "label": r"$C_\ell$"},
    {"mode": "joint", "stem": "frb_joint", "color": "#66B8E8", "label": r"$\mathrm{joint}$"},
)


NUISANCE_PARAMS = (
    ("b_frb", r"b_f"),
    ("beta_frb", r"\beta_{\mathrm{f}}"),
    ("k_cut", r"k_{\rm cut}"),
    ("N_ell", r"N_\ell"),
    ("F", r"F"),
)


POSTERIOR_SPECS = (
    {
        "key": "gauss",
        "delay_model": "gauss",
        "params": ("tau_gauss", "sigma_gauss"),
        "labels": (r"\tau_{\rm G}", r"\sigma_{\rm G}"),
    },
    {
        "key": "lognormal",
        "delay_model": "lognormal",
        "params": ("tau_NL", "sigma_NL"),
        "labels": (r"\tau_{\rm LN}", r"\sigma_{\rm LN}"),
    },
    {
        "key": "power-law",
        "delay_model": "powerlaw",
        "params": ("alpha",),
        "labels": (r"\alpha_{\tau}",),
    },
)


def _samples_available(path, param_names, labels):
    """Keep parameters that this chain actually sampled."""
    _samples, names, _ranges = load_chain(path)
    keep_names = []
    keep_labels = []
    for name, lab in zip(param_names, labels):
        if name in names:
            keep_names.append(name)
            keep_labels.append(lab)
    if not keep_names:
        return None, set()
    return mcsamples(path, keep_names, keep_labels), set(keep_names)


def _posterior_series(delay_model):
    series = []
    for probe in POSTERIOR_PROBES:
        path = _optional_chain(_chain(probe["mode"], delay_model, probe["stem"]))
        if path is None:
            print("missing", probe["mode"], delay_model)
            continue
        series.append((path, probe["color"], probe["label"]))
    return series


def plot_posterior_triangle(series, params, labels, out_pdf):
    """C_ell vs joint triangle. Legend sits in the empty upper-right corner."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "axes.titlesize": 20,
        "figure.figsize": (12, 12),
    })
    loaded = []
    for path, color, lab in series:
        samp, have = _samples_available(path, params, labels)
        if samp is None:
            continue
        loaded.append((samp, color, lab, have))
    present = []
    for name, _lab in zip(params, labels):
        if any(name in item[3] for item in loaded):
            present.append(name)
    if not loaded or not present:
        print("skip posterior: no overlapping parameters for", out_pdf)
        return
    n = len(present)
    width = 12 if n >= 5 else 8
    g = plots.get_subplot_plotter(width_inch=width)
    _getdist_settings(g)
    g.settings.alpha_filled_add = 0.75
    g.settings.solid_contour_palefactor = 0.60
    g.settings.lw_contour = 1.4
    g.settings.fig_width_inch = width
    g.make_figure(
        nx=n, ny=n,
        sharex=g.settings.no_triangle_axis_labels,
        sharey=g.settings.no_triangle_axis_labels,
    )
    bottom = n - 1
    lims = {}
    for i, param in enumerate(present):
        for i2 in range(bottom, i, -1):
            g._subplot(
                i, i2, pars=(param, present[i2]),
                sharex=g.subplots[bottom, i] if i2 != bottom else None,
                sharey=g.subplots[i2, 0] if i > 0 else None,
            )
        ax = g._subplot(
            i, i, pars=(param,),
            sharex=g.subplots[bottom, i] if i != bottom else None,
        )
        roots_i = [item[0] for item in loaded if param in item[3]]
        colors_i = [item[1] for item in loaded if param in item[3]]
        g._inner_ticks(ax, False)
        xlim = g.plot_1d(
            roots_i, param,
            do_xlabel=i == n - 1,
            no_label_no_numbers=g.settings.no_triangle_axis_labels,
            label_right=True, no_zero=True, no_ylabel=True, no_ytick=True,
            colors=colors_i, lws=[1.8] * len(roots_i),
            ax=ax, _ret_range=True,
        )
        lims[i] = xlim

    for i, param in enumerate(present):
        for i2 in range(i + 1, n):
            param2 = present[i2]
            ax = g.subplots[i2, i]
            pair = [
                item for item in loaded
                if param in item[3] and param2 in item[3]
            ]
            if not pair:
                if ax is not None:
                    ax.set_visible(False)
                continue
            g.plot_2d(
                [item[0] for item in pair],
                param_pair=[param, param2],
                filled=True,
                do_xlabel=i2 == n - 1,
                do_ylabel=i == 0,
                no_label_no_numbers=g.settings.no_triangle_axis_labels,
                add_legend_proxy=False,
                colors=[item[1] for item in pair],
                lws=[1.4] * len(pair),
                ax=ax, _no_finish=True,
            )
            g._inner_ticks(ax)
            if lims.get(i2) is not None:
                ax.set_ylim(lims[i2])
            if i2 == bottom and lims.get(i) is not None:
                ax.set_xlim(lims[i])

    # Display crop only: retain every sample and the full KDE normalization.
    # All six chains retain at least 99.47% of their probability below 2.5.
    if "b_frb" in present:
        bias_index = present.index("b_frb")
        for row in range(n):
            for col in range(row + 1):
                ax = g.subplots[row, col]
                if ax is None:
                    continue
                if col == bias_index:
                    ax.set_xlim(0.5, 2.5)
                if row == bias_index and col < row:
                    ax.set_ylim(0.5, 2.5)

    g._subplots_adjust()
    handles = [
        Patch(facecolor=item[1], edgecolor=item[1], alpha=0.8)
        for item in loaded
    ]
    legend_labels = [item[2] for item in loaded]
    corner = g.subplots[0, n - 1]
    if corner is None:
        g.fig.legend(
            handles, legend_labels,
            loc="upper right", frameon=False,
            fontsize=g.settings.legend_fontsize,
        )
    else:
        corner.set_axis_off()
        corner.legend(
            handles, legend_labels,
            loc="upper right", frameon=False,
            fontsize=g.settings.legend_fontsize,
        )
    out_pdf = figure_output_path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    g.export(str(out_pdf))
    g.export(str(out_pdf.with_suffix(".png")), dpi=220)
    plt.close(g.fig)
    print("Saved:", out_pdf, flush=True)


def plot_posteriors():
    """Three C_ell / joint triangles: Gaussian, log-normal, power-law."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in POSTERIOR_SPECS:
        params = list(spec["params"]) + [row[0] for row in NUISANCE_PARAMS]
        labels = list(spec["labels"]) + [row[1] for row in NUISANCE_PARAMS]
        plot_posterior_triangle(
            _posterior_series(spec["delay_model"]),
            params, labels,
            figure_output_path(OUT_DIR / ("frb_posterior_%s_cl_joint.pdf" % spec["key"])),
        )


DMZ_QUANTILES = np.array([0.02275, 0.15865, 0.5, 0.84135, 0.97725])


def _conditional_dm_pdf(theta, z, cosmic_dm, dm, resolution=1.0):
    """Native Macquart IGM + lognormal host convolution; no redshift prior.

    theta is (mu_host, sigma_host, F). Gaussian *delay* does not select a
    Gaussian IGM distribution. Resolve the narrow low-z IGM and high-z host
    kernels on finer convolution grids than the exported DM grid.
    """
    mu, sigma, feedback = theta
    pdf = np.empty((dm.size, z.size))
    zero = np.flatnonzero(z == 0)
    if zero.size:
        pdf[:, zero] = dmz.p_host_obs(dm[:, None], 0.0, mu, sigma)
    for mask, base_step in (
            ((z > 0) & (z < 0.03), 0.125),
            ((z >= 0.03) & (z < 1.5), 0.5),
            (z >= 1.5, 0.25)):
        use = np.flatnonzero(mask)
        step = base_step * resolution
        grid = np.arange(int(np.ceil(dm[-1] / step)) + 2) * step
        for start in range(0, len(use), 12):
            sl = use[start:start + 12]
            pdf[:, sl] = dmz.p_dm_given_z_fft(
                dm, z[sl], cosmic_dm[sl], feedback, mu, sigma,
                dm_grid=grid, igm_model="macquart", dm_mw=0.0)
    return pdf


def compare_dmz_prediction(n_draw=5120, seed=20261016, zmax=1.5,
                           dm_max=16000.0, dm_step=1.0, workers=2, models=None):
    """Marginalize conditional PDFs over retained joint posterior rows.

    Average PDFs first, then invert the averaged CDF. Predictive intervals
    include sightline/host scatter and posterior parameter uncertainty.
    Neither p(z), the detection function, nor a second likelihood weight is
    applied at fixed z. No individual-source inferred redshifts are plotted.
    Finite-DM tail probability is retained as missing mass, not renormalized.
    """
    import hashlib
    if not 0.03 <= zmax <= 5.0:
        raise ValueError("zmax must lie between 0.03 and 5")
    z = np.unique(np.r_[0.0, 0.005, 0.01, 0.02, min(0.3, zmax),
                        np.linspace(0.03, zmax, 125)])
    dm = np.linspace(0, dm_max, int(round(dm_max / dm_step)) + 1)
    cosmo = dmz.make_cosmology()
    cosmic = dmz.mean_cosmic_dm(cosmo, z)
    names_used = ["mu_host", "sigma_host", "F"]
    result = dict(z=z, dm=dm, quantile_levels=DMZ_QUANTILES,
                  cosmic_mean_dm=cosmic, parameter_names=np.array(names_used),
                  seed=seed, zmax=zmax, dm_step=dm_step, dm_max=dm_max,
                  posterior_mode="joint", igm_model="macquart",
                  method="mean p(DM_eg | z, theta) over equal-weight joint "
                         "posterior rows; no p(z) or selection reweighting",
                  interval_meaning="equal-tailed posterior predictive "
                                   "68.27% and 95.45%; not mean-curve errors")

    def summarize(pdf, grid):
        cdf = cumulative_trapezoid(pdf, grid, axis=0, initial=0.0)
        mass = cdf[-1]
        if (not np.all(np.isfinite(pdf)) or np.min(pdf) < 0 or
                mass.min() <= DMZ_QUANTILES[-1] or mass.max() > 1.005):
            raise ValueError(f"DM grid inadequate: probability {mass.min()}..{mass.max()}")
        q = np.array([np.interp(DMZ_QUANTILES, c, grid) for c in cdf.T])
        if not np.all(np.diff(q, axis=1) > 0):
            raise ValueError("Unordered predictive quantiles")
        return q, mass

    if models is None:
        models = (("gauss", "gauss"), ("sfh", "SFH"))
    for model_index, (key, folder) in enumerate(models):
        path = RESULT_ROOT / "joint" / folder / "chains/frb_joint_converged.npy"
        samples, names, _ = load_chain(path)
        count = min(int(n_draw), len(samples))
        if count < 8:
            raise ValueError("Use at least eight posterior draws")
        indices = np.random.default_rng(seed + model_index).choice(
            len(samples), count, replace=False)
        theta = samples[indices][:, [names.index(name) for name in names_used]]
        result[key + "_n_samples"] = len(samples)
        result[key + "_parameter_quantiles"] = np.quantile(
            samples[:, [names.index(name) for name in names_used]],
            [0.16, 0.5, 0.84], axis=0)
        del samples
        print(f"{folder}: marginalizing {count} posterior draws", flush=True)
        total = np.zeros((dm.size, z.size))
        half_q = None

        def one_draw(row):
            return _conditional_dm_pdf(row, z, cosmic, dm)

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            # Bounded batches avoid retaining a PDF for every posterior draw.
            for start in range(0, count, max(1, int(workers))):
                stop = min(start + max(1, int(workers)), count)
                for i, pdf in enumerate(pool.map(one_draw, theta[start:stop]), start + 1):
                    total += pdf
                    if i == count // 2:
                        half_q, _ = summarize(total / i, dm)
                    if i == 1 or i % 32 == 0 or i == count:
                        print(f"  {folder}: {i}/{count}", flush=True)
        mean_pdf = total / count
        q, mass = summarize(mean_pdf, dm)
        # Independent quadrature refinement at low, intermediate and high z.
        iz = np.unique(np.searchsorted(z, [0, 0.005, 0.03, 0.5, 1, zmax]))
        test_rows = theta[np.linspace(0, count - 1, 8).astype(int)]
        fine_dm = np.linspace(0, dm_max, 2 * (dm.size - 1) + 1)
        coarse = np.zeros((dm.size, len(iz)))
        fine = np.zeros((fine_dm.size, len(iz)))
        for row in test_rows:
            coarse += _conditional_dm_pdf(row, z[iz], cosmic[iz], dm)
            fine += _conditional_dm_pdf(row, z[iz], cosmic[iz], fine_dm, 0.5)
        cq, _ = summarize(coarse / len(test_rows), dm)
        fq, _ = summarize(fine / len(test_rows), fine_dm)
        grid_error = float(np.max(np.abs(cq - fq)))
        if grid_error > 2.0:
            raise ValueError(f"DM-grid refinement changes quantiles by {grid_error:.3f}; refine")
        mc_error = float(np.max(np.abs(q - half_q)))
        print(f"{folder}: integrated mass {mass.min():.6f}..{mass.max():.6f}; "
              f"grid refinement {grid_error:.3f}, half/full draws {mc_error:.3f} pc cm^-3",
              flush=True)
        result.update({key + "_pdf": mean_pdf.astype(np.float32),
                       key + "_dm_quantiles": q, key + "_half_quantiles": half_q,
                       key + "_integrated_mass": mass,
                       key + "_chain_indices": indices, key + "_theta": theta,
                       key + "_chain_path": os.path.relpath(path, ROOT),
                       key + "_n_draw": count, key + "_grid_error": grid_error,
                       key + "_half_full_error": mc_error,
                       key + "_chain_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def plot_dmz_three_divide():
    """Three joint delay models, each drawn as the single-model DM-z figure.

    The conditional density, quantile lines and localized sources use the
    blue palette of frb_dmz_gauss_sfh_joint.pdf. Gaussian keeps its 0-16000
    grid; log-normal and power-law keep the cached 0-6000 grid. No new
    posterior draws are taken.
    """
    import csv
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import (FuncFormatter, MultipleLocator, AutoMinorLocator,
                                   NullFormatter)
    from scipy.interpolate import RegularGridInterpolator

    cache = picture_asset("dmz_three_model_data.npz")
    if not cache.is_file():
        raise FileNotFoundError("Missing %s; run --dmz-three-only first" % cache)
    with np.load(cache, allow_pickle=False) as saved:
        z = np.array(saved["z"], dtype=float)
        gauss_dm = np.array(saved["dm"], dtype=float)
        overlay_dm = np.array(saved["overlay_dm"], dtype=float)
        pdfs = {key: np.array(saved[key + "_pdf"], dtype=float)
                for key in ("gauss", "lognormal", "powerlaw")}
        quantiles = {key: np.array(saved[key + "_dm_quantiles"], dtype=float)
                     for key in ("gauss", "lognormal", "powerlaw")}
    host_path = picture_asset("localized_hosts_dm_eg.csv")
    with host_path.open(newline="", encoding="utf-8") as stream:
        hosts = list(csv.DictReader(stream))
    models = (("gauss", gauss_dm, "Gaussian"),
              ("lognormal", overlay_dm, "Log-normal"),
              ("powerlaw", overlay_dm, "Power-law"))
    density_colors = ["#FFFFFF", "#E9F3FA", "#BFDDF0", "#7CB5D6", "#347FA9", "#163F65"]
    median_color, interval_color = "#163F65", "#245D80"
    point_styles = (("CHIME_confirmed", "#83B5A1", "#315F50", "CHIME"),
                    ("CHIME_not_confirmed", "#EAB1AD", "#B66E6B", "Others"))
    excluded_sources = ("FRB20190417A", "FRB20190520B", "FRB20220831A")
    display_zmin, display_zmax = 0.0, 1.5
    ymin, ymax = 0.0, 1800.0
    for key, grid, _label in models:
        q = quantiles[key]
        pdf = pdfs[key]
        if q.shape != (len(z), 5) or pdf.shape != (len(grid), len(z)):
            raise ValueError("Cached %s prediction has an unexpected shape" % key)
        if not np.isfinite(q).all() or np.any(np.diff(q, axis=1) <= 0):
            raise ValueError("Invalid predictive quantiles for %s" % key)
        if grid[0] > ymin or grid[-1] < ymax or z[0] > display_zmin or z[-1] < display_zmax:
            raise ValueError("Cached %s grid does not cover the display window" % key)
    rc = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
          "mathtext.fontset": "dejavusans", "font.size": 12,
          "axes.labelsize": 22, "axes.linewidth": 0.85,
          "xtick.labelsize": 16.5, "ytick.labelsize": 16.5,
          "xtick.major.width": 0.9, "ytick.major.width": 0.9,
          "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
          "xtick.major.size": 5, "ytick.major.size": 5,
          "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
          "pdf.fonttype": 42, "ps.fonttype": 42,
          "axes.edgecolor": "#303030", "text.color": "#222222",
          "axes.labelcolor": "#222222", "figure.facecolor": "white"}
    host_z = np.array([float(row["z_spec"]) for row in hosts])
    host_dm = np.array([float(row["DM_eg_pc_cm3"]) for row in hosts])
    host_groups = np.array([row["plot_group"] for row in hosts])
    excluded = np.array([row["FRB_source"] in excluded_sources for row in hosts])
    on_plot = ((host_z >= display_zmin) & (host_z <= display_zmax)
               & (host_dm >= ymin) & (host_dm <= ymax) & ~excluded)
    z_edges = np.linspace(display_zmin, display_zmax, 601)
    dm_edges = np.linspace(ymin, ymax, 601)
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    dm_centers = 0.5 * (dm_edges[1:] + dm_edges[:-1])
    zz, dd = np.meshgrid(z_centers, dm_centers)
    cmap = LinearSegmentedColormap.from_list("frb_density", density_colors)
    norm = Normalize(-5.0, -1.5, clip=True)
    stem = OUT_DIR / "test_dmz"
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(18.4, 6.0), sharex=True, sharey=True)
        # Leave 0.2 in before the rotated cm^{-3} superscript, the leftmost ink.
        fig.subplots_adjust(left=0.067, right=0.90, bottom=0.14, top=0.96, wspace=0.08)
        heatmap = None
        for ax, (key, grid, label) in zip(axes, models):
            interpolate = RegularGridInterpolator((grid, z), pdfs[key], bounds_error=True)
            density = interpolate(np.stack((dd, zz), axis=-1))
            log_density = np.log10(np.maximum(density, 1e-300))
            heatmap = ax.pcolormesh(z_edges, dm_edges, log_density, cmap=cmap, norm=norm,
                                   shading="flat", rasterized=True, zorder=0)
            q = quantiles[key]
            for column in (1, 3):
                ax.plot(z, q[:, column], color=interval_color, lw=0.95, ls="--",
                        alpha=0.9, zorder=2)
            for column in (0, 4):
                ax.plot(z, q[:, column], color=interval_color, lw=0.9, ls="-.",
                        alpha=0.85, zorder=2)
            ax.plot(z, q[:, 2], color=median_color, lw=1.4, ls="-", zorder=4)
            for group, face, edge, _point_label in reversed(point_styles):
                select = (host_groups == group) & on_plot
                ax.scatter(host_z[select], host_dm[select], s=23, facecolor=face,
                           edgecolor=edge, linewidth=0.5,
                           zorder=7 if group == "CHIME_confirmed" else 6)
            ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top",
                    fontsize=18, color="#222222", zorder=8)
            ax.set(xscale="linear", yscale="linear", xlim=(display_zmin, display_zmax),
                   ylim=(ymin, ymax), xlabel=r"$z$")
            ax.xaxis.set_major_locator(MultipleLocator(0.3))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_major_locator(MultipleLocator(300))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: "" if np.isclose(value, 0) else f"{value:g}"))
            ax.xaxis.set_minor_locator(AutoMinorLocator(3))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(AutoMinorLocator(3))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(which="both", direction="in", top=True, right=True, pad=6)
            point_handles = [Line2D([], [], ls="none", marker="o", markersize=8.5,
                                   markerfacecolor=face, markeredgecolor=edge,
                                   markeredgewidth=0.5)
                             for _, face, edge, _point_label in point_styles]
            ax.legend(point_handles, [style[3] for style in point_styles],
                      loc="lower right", bbox_to_anchor=(0.96, 0.065),
                      borderaxespad=0, frameon=False, ncol=1, fontsize=16,
                      handlelength=1.6, labelspacing=0.7, handletextpad=0.7)
        axes[0].set_ylabel(r"$\mathrm{DM}_{\rm eg}\ [\mathrm{pc}\,\mathrm{cm}^{-3}]$")
        cax = fig.add_axes([0.915, 0.14, 0.012, 0.82])
        colorbar = fig.colorbar(heatmap, cax=cax, ticks=[-5, -4, -3, -2, -1.5])
        colorbar.set_label(r"$\log_{10}\,p(\mathrm{DM}_{\rm eg}\mid z)$",
                           fontsize=19.5, labelpad=11)
        colorbar.ax.tick_params(labelsize=15.5, direction="out", length=4.5, width=0.8)
        colorbar.outline.set_linewidth(0.7)
        for suffix in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + suffix), dpi=450)
        plt.close(fig)
    print("Hosts drawn: %d/%d" % (int(on_plot.sum()), len(hosts)), flush=True)
    print("Saved:", stem.with_suffix(".pdf"), flush=True)


def plot_dmz_comparison(n_draw=5120, seed=20261016, zmax=1.5, result=None,
                        hosts=None, color_theme="blue", three_models=False):
    """Gaussian-delay probability-density map and quantiles on linear z and DM axes."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import (FuncFormatter, MultipleLocator, AutoMinorLocator,
                                   NullFormatter)
    from scipy.interpolate import RegularGridInterpolator
    if color_theme not in ("blue", "gray"):
        raise ValueError("color_theme must be 'blue' or 'gray'")
    gray = color_theme == "gray"
    density_colors = (["#FFFFFF", "#F5F5F5", "#E3E3E3", "#C9C9C9", "#ABABAB", "#898989"]
                      if gray else
                      ["#FFFFFF", "#E9F3FA", "#BFDDF0", "#7CB5D6", "#347FA9", "#163F65"])
    median_color, interval_color = ("#565656", "#777777") if gray else ("#163F65", "#245D80")
    point_styles = (("CHIME_confirmed", "#C7B1DE", "#8B70A6", "CHIME"),
                    ("CHIME_not_confirmed", "#A9CEE8", "#648FAF", "Others")) if gray else (
                   ("CHIME_confirmed", "#83B5A1", "#315F50", "CHIME"),
                   ("CHIME_not_confirmed", "#EAB1AD", "#B66E6B", "Others"))
    if result is None:
        result = compare_dmz_prediction(n_draw=n_draw, seed=seed, zmax=zmax)
    result = dict(result)
    fixed = str(result.get("prediction_kind", "posterior_predictive")) == "fixed_parameters"
    display_zmax = min(float(zmax), float(result["z"][-1]))
    display_zmin = max(0.0, float(result["z"][0]))
    result["display_zmin"] = display_zmin
    result["display_zmax"] = display_zmax
    result["display_z_scale"] = "linear"
    result["display_dm_min"] = 0.0
    result["display_dm_max"] = 1800.0
    result["display_dm_scale"] = "linear"
    result["density_color_log10_limits"] = np.array([-5.0, -1.5])
    result["display_color_theme"] = color_theme
    result["display_legend_fontsize"] = 16.0 if three_models else 20.0
    excluded_sources = ("FRB20190417A", "FRB20190520B", "FRB20220831A")
    result["display_excluded_sources"] = np.array(excluded_sources)
    result["density_color_meaning"] = "log10 of density per pc cm^-3; no peak normalization; no log-DM Jacobian"
    result.pop("inset_z", None)
    result["figure_size_inches"] = np.array([9.0, 6.6] if three_models else [8.0, 6.0])
    result["display_interval_probabilities"] = np.array([0.6827, 0.9545])
    result["display_models"] = np.array(
        ["gauss", "lognormal", "powerlaw"] if three_models else ["gauss"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / ("frb_dmz_gauss_sfh_joint_central" if fixed
                      else "dmz")
    if gray:
        # Gray alternatives were archived, rather than renamed to dmz_gray.
        stem = OUT_DIR / ("frb_dmz_gauss_sfh_joint_central_gray" if fixed
                          else "frb_dmz_gauss_sfh_joint_gray")
    styles = (("gauss", "#80BCE3", "#387FAB", "-", "Gaussian delay"),)
    if three_models:
        stem = OUT_DIR / "dmz_three_model"
        styles = (("gauss", "#80BCE3", "#163F65", "-", "Gaussian"),
                  ("lognormal", None, "#AA6B13", "-", "Log-normal"),
                  ("powerlaw", None, "#984375", "-", "Power-law"))
        result["display_density_model"] = "gauss"
        result["display_model_line_colors"] = np.array([s[2] for s in styles])
    np.savez_compressed(str(stem) + "_data.npz", **result)
    rc = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
          "mathtext.fontset": "dejavusans", "font.size": 12,
          "axes.labelsize": 22, "axes.linewidth": 0.85,
          "xtick.labelsize": 16.5, "ytick.labelsize": 16.5,
          "xtick.major.width": 0.9, "ytick.major.width": 0.9,
          "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
          "xtick.major.size": 5, "ytick.major.size": 5,
          "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
          "pdf.fonttype": 42, "ps.fonttype": 42,
          "axes.edgecolor": "#303030", "text.color": "#222222",
          "axes.labelcolor": "#222222", "figure.facecolor": "white"}
    z = result["z"]
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=tuple(result["figure_size_inches"]))
        # Wider margins than the 16 pt layout: a 22 pt y label and a 17.5 pt
        # colorbar label need the room or they clip at the canvas edge.
        if three_models:
            fig.subplots_adjust(left=0.15, right=0.805, bottom=0.15, top=0.84)
        else:
            fig.subplots_adjust(left=0.175, right=0.79, bottom=0.165, top=0.965)
        z_edges = np.linspace(display_zmin, display_zmax, 601)
        dm_edges = np.linspace(0.0, 1800.0, 601)
        z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
        dm_centers = 0.5 * (dm_edges[1:] + dm_edges[:-1])
        zz, dd = np.meshgrid(z_centers, dm_centers)
        interpolate = RegularGridInterpolator((result["dm"], z), result["gauss_pdf"],
                                             bounds_error=True)
        density = interpolate(np.stack((dd, zz), axis=-1))
        log_density = np.log10(np.maximum(density, 1e-300))
        cmap = LinearSegmentedColormap.from_list("frb_density", density_colors)
        heatmap = ax.pcolormesh(z_edges, dm_edges, log_density, cmap=cmap,
                               norm=Normalize(-5.0, -1.5, clip=True),
                               shading="flat", rasterized=True, zorder=0)
        cax = fig.add_axes([0.831, 0.15, 0.024, 0.69] if three_models else
                           [0.822, 0.165, 0.026, 0.80])
        colorbar = fig.colorbar(heatmap, cax=cax, ticks=[-5, -4, -3, -2, -1.5])
        colorbar.set_label(r"$\log_{10}\,p(\mathrm{DM}_{\rm eg}\mid z)$", fontsize=19.5, labelpad=11)
        colorbar.ax.tick_params(labelsize=15.5, direction="out", length=4.5, width=0.8)
        colorbar.outline.set_linewidth(0.7)
        if three_models:
            colorbar.ax.set_title("Gaussian", fontsize=12, pad=9)
        for key, fill, line, ls, label in reversed(styles):
            q = result[key + "_dm_quantiles"]
            if q.shape != (len(z), 5) or not np.isfinite(q).all() or np.any(np.diff(q, axis=1) <= 0):
                raise ValueError(f"Invalid predictive quantiles for {key}")
            band_color = line if three_models else interval_color
            center_color = line if three_models else median_color
            for column in (1, 3):
                ax.plot(z, q[:, column], color=band_color, lw=1.05 if three_models else 0.95,
                        ls="--", alpha=0.9, zorder=2)
            for column in (0, 4):
                ax.plot(z, q[:, column], color=band_color, lw=1.0 if three_models else 0.9,
                        ls="-.", alpha=0.85, zorder=2)
            ax.plot(z, q[:, 2], color=center_color, lw=1.55 if three_models else 1.4,
                    ls=ls, label=label, zorder=4)
        ymax = 1800.0
        ymin = 0.0
        if hosts is not None:
            host_z = np.array([float(row["z_spec"]) for row in hosts])
            host_dm = np.array([float(row["DM_eg_pc_cm3"]) for row in hosts])
            visible = (host_z >= display_zmin) & (host_z <= display_zmax)
            in_dm_range = (host_dm >= ymin) & (host_dm <= ymax)
            excluded = np.array([row["FRB_source"] in excluded_sources for row in hosts])
            on_plot = visible & in_dm_range & ~excluded
            host_groups = np.array([row["plot_group"] for row in hosts])
            for group, face, edge, label in reversed(point_styles):
                select = host_groups == group
                ax.scatter(host_z[select & on_plot], host_dm[select & on_plot], s=23, facecolor=face,
                           edgecolor=edge, linewidth=0.5,
                           zorder=7 if group == "CHIME_confirmed" else 6)
                print(f"{label}: {select.sum()} total, {(select & on_plot).sum()} visible", flush=True)
            print(f"Hosts visible: {on_plot.sum()}/{len(hosts)}; "
                  f"DM_eg range {host_dm.min():.3f}–{host_dm.max():.3f}", flush=True)
        ax.set(xscale="linear", yscale="linear", xlim=(display_zmin, display_zmax),
               ylim=(ymin, ymax), xlabel=r"$z$",
               ylabel=r"$\mathrm{DM}_{\rm eg}\ [\mathrm{pc}\,\mathrm{cm}^{-3}]$")
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_major_locator(MultipleLocator(300))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "" if np.isclose(value, 0) else f"{value:g}"))
        ax.xaxis.set_minor_locator(AutoMinorLocator(3))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", direction="in", top=True, right=True, pad=6)
        handles, labels = [], []
        if hosts is not None:
            point_handles = [Line2D([], [], ls="none", marker="o", markersize=8.5,
                                   markerfacecolor=face, markeredgecolor=edge,
                                   markeredgewidth=0.5)
                             for _, face, edge, _ in point_styles]
            handles.extend(point_handles)
            labels.extend(style[3] for style in point_styles)
            ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.065),
                      borderaxespad=0, frameon=False, ncol=1,
                      fontsize=16 if three_models else 20, handlelength=1.6,
                      labelspacing=0.9, handletextpad=0.7)

        if three_models:
            # The two legend rows distinguish model color from interval style
            # without placing six extra labels over the curves and sources.
            model_handles = [Line2D([], [], color=s[2], lw=1.8) for s in styles]
            level_handles = [Line2D([], [], color="#404040", lw=1.5, ls=ls)
                             for ls in ("-", "--", "-.")]
            paired_handles = [h for pair in zip(model_handles, level_handles) for h in pair]
            paired_labels = [label for pair in zip([s[4] for s in styles],
                                                    ["Median", r"$1\sigma$", r"$2\sigma$"])
                             for label in pair]
            fig.legend(paired_handles, paired_labels, loc="upper center",
                       bbox_to_anchor=(0.48, 0.995), ncol=3, frameon=False,
                       fontsize=14, handlelength=2.5, handletextpad=0.6,
                       columnspacing=1.6, labelspacing=0.45, borderaxespad=0)

        for suffix in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + suffix), dpi=450)
        plt.close(fig)
    if three_models:
        model_keys = [s[0] for s in styles]
        table = np.column_stack([z] + [result[key + "_dm_quantiles"] for key in model_keys])
        np.savetxt(str(stem) + "_quantiles.csv", table, delimiter=",",
                   header="z," + ",".join(key + "_" + q for key in model_keys
                       for q in ("q02275", "q15865", "median", "q84135", "q97725")), comments="")
        notes = [
            "# Three delay-model joint DM–z predictions\n",
            "Gaussian retains the supplied figure's cached conditional density and quantiles. "
            "Log-normal and Power-law use result/joint/lognormal and result/joint/power-law "
            "converged equal-weight chains, sampling complete posterior rows.\n",
            "All models average p(DM_eg | z, theta) before inverting the CDF; "
            "Macquart IGM plus rest-frame log-normal host, fixed cosmology and f_IGM=0.93. "
            "No redshift prior, selection efficiency, or additional likelihood weights "
            "are applied at fixed z. The center is the predictive median.\n",
            "1 sigma and 2 sigma denote equal-tailed 68.27% and 95.45% predictive "
            "intervals, including IGM/host scatter and posterior parameter uncertainty. "
            "They are not uncertainty intervals on a mean curve.\n",
            "Only the Gaussian density is filled. Other models have median, 1 sigma, "
            "and 2 sigma lines. Model color and line-style legends apply to all three. "
            "The density color scale and source exclusions match the supplied figure.\n",
            "The additional models use a 0–6000 pc cm^-3 integration grid at unit spacing. "
            "The finite upper tail is not renormalized; integrated mass must exceed "
            "0.97725 at every z. Causal convolution leaves densities below the grid "
            "limit unchanged by truncation of the upper DM support.\n",
            "| Model | Posterior rows | Draws | Integrated mass | Grid refinement error | "
            "Half/full draw error |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for key in model_keys:
            mass = result[key + "_integrated_mass"]
            notes.append(f"| {key} | {result[key + '_n_samples']} | {result[key + '_n_draw']} | "
                         f"{mass.min():.6f}–{mass.max():.6f} | "
                         f"{result[key + '_grid_error']:.3f} | "
                         f"{result[key + '_half_full_error']:.3f} |")
        for key in model_keys:
            notes.append(f"\n{key} chain: `{result[key + '_chain_path']}`; "
                         f"SHA256: `{result[key + '_chain_sha256']}`.\n")
        notes.extend([
            "\nErrors are maximum absolute quantile changes in pc cm^-3. "
            "The NPZ preserves posterior draw indices, parameters and all numerical checks.\n",
            "Sources: localized_hosts_dm_eg.csv, using existing MW corrections, groups "
            "and exclusions: FRB20190417A, FRB20190520B, FRB20220831A. "
            f"Shown: {on_plot.sum()} of {len(hosts)} localized sources.\n",
            "Reproduce: `python picture.py --dmz-three-only --dmz-three-draws "
            f"{int(result['overlay_requested_draws'])}`. Cached overlay predictions are reused "
            "only when chain hashes, draw count, seed and redshift grid agree.\n",
        ])
        Path(str(stem) + "_methods.md").write_text("\n".join(notes), encoding="utf-8")
        print("Saved:", stem.with_suffix(".pdf"), flush=True)
        return
    table = np.column_stack([z, result["gauss_dm_quantiles"], result["sfh_dm_quantiles"]])
    np.savetxt(str(stem) + "_quantiles.csv", table, delimiter=",",
               header="z," + ",".join(key + "_" + q for key in ("gauss", "sfh")
                                       for q in ("q02275", "q15865", "median", "q84135", "q97725")),
               comments="")
    notes = [
        "# Gaussian delay 的条件 DM 后验预测\n",
        "当前图仅显示 Gaussian-delay 模型；数值缓存与 CSV 中保留先前的 SFH 计算结果。\n",
        ("固定参数版：使用 result/joint 下全部保留后验的各参数边缘中位数。"
         "不是最大似然点，也不进行后验平均；概率分布仅包含固定参数下的 IGM/宿主散布。\n"
         if fixed else
         "使用 result/joint 下各模型已去除 burn-in 的等权 MCMC 后验。"
         "整行抽样保留参数相关性，不重复施加似然权重。\n"),
        (r"$$p(\mathrm{DM}_{\rm eg}\mid z,\theta_{\rm med},\mathcal M).$$" + "\n"
         if fixed else
         r"$$p(\mathrm{DM}_{\rm eg}\mid z,\mathcal D,\mathcal M)"
        r"=\int p(\mathrm{DM}_{\rm eg}\mid z,\theta,\mathcal M)"
        r"p(\theta\mid\mathcal D,\mathcal M)\,d\theta.$$" + "\n"),
        "固定红移的原生条件分布来自 dm_z.p_dm_given_z_fft：Macquart IGM "
        "+ 静止系 lognormal 宿主；DM_eg = DM_IGM + DM_host,rest/(1+z)。"
        "Gaussian 指延迟分布，不是 IGM 分布。\n",
        "固定 z 时不乘 p(z) 或探测效率；两种延迟模型通过各自的 "
        "(mu_host, sigma_host, F) 联合后验影响预测。宇宙学和 f_IGM=0.93 固定。\n",
        "主图渐变为条件概率密度的 log10 值，密度以 (pc cm⁻³)⁻¹ 为单位，"
        "色标范围 −5 至 −1.5；范围外的颜色饱和，不改变 PDF 本身。"
        "使用缓存的原始 PDF 在线性 DM、z 坐标中插值并显示，"
        "不按每个红移的峰值归一化，也不变换为每单位 log DM 的密度。"
        "1σ（15.865–84.135%）与 2σ（2.275–97.725%）等尾预测区间仅画边界线。"
        "中位数为实线，1σ 边界为虚线，2σ 边界为点划线；"
        "图内右下角图例仅列观测点类别，字号 20 pt；轴标题 22 pt，刻度标签 16.5 pt。"
        f"主图显示红移 {display_zmin:g}–{display_zmax:g}，尺寸 8×6 英寸，无小图。"
        "横纵坐标均为线性刻度；纵轴范围 0–1800 pc cm⁻³，原点仅保留横轴的 0 标签。"
        "范围外的预测区间和观测点仅在显示时裁切，数值结果保留。"
        "这是单个 FRB 的预测散布，"
        "不是平均关系的参数误差带，也不是二维联合可信轮廓。\n",
        "积分在有限 DM 网格上完成，保留网格外的尾部概率，"
        "没有把有限网格内的 PDF 再归一化为 1。\n",
        "| 模型 | 后验样本总数 | 计算参数组数 | 网格积分概率范围 | "
        "步长减半的最大分位数变化 | 前半/全部抽样最大分位数变化 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for key, *_ in styles:
        mass = result[key + "_integrated_mass"]
        sampling_error = "—" if fixed else f"{result[key + '_half_full_error']:.3f}"
        notes.append(f"| {key} | {result[key + '_n_samples']} | "
                     f"{result[key + '_n_draw']} | {mass.min():.6f}–{mass.max():.6f} | "
                     f"{result[key + '_grid_error']:.3f} | "
                     f"{sampling_error} |")
    if fixed:
        notes.extend(["\n中心参数（全部保留后验的边缘中位数）：\n",
                      "| 模型 | mu_host | sigma_host | F |",
                      "|---|---:|---:|---:|"])
        for key, *_ in styles:
            mu, sigma, feedback = result[key + "_theta"][0]
            notes.append(f"| {key} | {mu:.8f} | {sigma:.8f} | {feedback:.8f} |")
    notes.extend(["\n分位数变化的单位为 pc cm⁻³。NPZ 保留完整条件 PDF、"
                  "参数、链路径及 SHA256（后验平均版另存抽样行号）；CSV 保留各红移的分位数。\n",
                  ("复算：`python picture.py --dmz-central-only`。" if fixed else
                   f"复算：`python picture.py --dmz-compare-only --dmz-draws {int(result['gauss_n_draw'])}`。") +
                  "可用 `--dmz-zmax` 改变红移上限，`--dmz-theme gray` 生成灰色渐变版。"])
    notes.append("\n配色：" + ("浅灰概率密度渐变、浅紫 CHIME 点、浅蓝 Others 点。"
                              if gray else "蓝色概率密度渐变、绿色 CHIME 点、浅红 Others 点。"))
    if hosts is not None:
        notes.append(f"\n定位表保留全部 {len(hosts)} 个独立源，当前红移范围内 "
                     f"{visible.sum()} 个，应用显示范围与指定源隐藏条件后绘制 {on_plot.sum()} 个。"
                     "数据及扣除明细见 `localized_hosts_dm_eg.csv`，"
                     "方法见 `localized_hosts_dm_eg_methods.md`。CHIME 为确认探测，"
                     "Others 为未确认探测（含候选对应），不等于确认未探测。点不带误差棒，"
                     "银河系模型系统误差未传播。复算两幅叠加图："
                     "`python picture.py --dmz-hosts-only`。")
        notes.append("\n按用户指定仅在绘图时隐藏：" + "、".join(excluded_sources) +
                     "。原始处理表和模型后验预测均保留不变，此隐藏不代表模型拟合或样本质量筛选。")
        outside = [f"{row['FRB_source']} (z={row['z_spec']})"
                   for row, shown in zip(hosts, visible) if not shown]
        if outside:
            notes.append("\n保留在处理表但超出主图红移范围的源：" + "；".join(outside) + "。")
        off_y = [f"{row['FRB_source']} ({float(row['DM_eg_pc_cm3']):.3f} pc cm⁻³)"
                 for row, in_z, in_dm in zip(hosts, visible, in_dm_range) if in_z and not in_dm]
        if off_y:
            notes.append("\n红移范围内但超出纵轴显示范围的源：" + "；".join(off_y) + "。")
        negative = [f"{row['FRB_source']} ({float(row['DM_eg_pc_cm3']):.3f} pc cm⁻³)"
                    for row in hosts if float(row["DM_eg_pc_cm3"]) < 0]
        if negative:
            notes.append("\n处理表原样保留、当前纵坐标范围内不显示的负 DM_eg：" + "；".join(negative) +
                         "。这些是采用指定银河系模型扣除后的残差，不是负的物理电子柱密度。")
    Path(str(stem) + "_methods.md").write_text("\n".join(notes), encoding="utf-8")
    print("Saved:", stem.with_suffix(".pdf"), flush=True)


def cl_data_curves():
    """Collect every quantity the 3x3 C_ell figure draws.

    theory_cl is expensive (~3 min for the nine panels), so this returns
    only plain arrays and per-panel metadata. plot_cl_data() draws straight
    from it, and the same dict is cached in cl_data_curves.npz so style
    edits never pay for the theory again.
    """
    center = load_center(RESULTS_TXT)
    data = np.load(mc.catalog_npz_path(), allow_pickle=False)
    samples = prepare(data)
    ell_loc = mc.ELL_LOC_FIXED
    b_e = mc.B_E_FIXED
    delay = (center["tau"], center["sigma"])
    dm, p_fit = mc.dm_tracer(delay, b_e=b_e)
    frb = mc.frb_source_tracer(p_fit, center["b_frb"], center["dm_host"])
    ell_theory = np.asarray(mc.ELL_THEORY, dtype=float)
    ell_line = ell_theory[ell_theory >= 2.0]
    ell_max = float(ell_theory[-1])
    print(
        "gauss C_ell center: tau=%.4f sigma=%.4f b_frb=%.4f k_cut=%.4f "
        "N_ell=%.4f dm_host=%.4f ell_loc=%.4f"
        % (center["tau"], center["sigma"], center["b_frb"], center["k_cut"],
           center["n_ell"], center["dm_host"], ell_loc)
    )

    panels = []
    for index, (item, i, entry, closing) in enumerate(collect_panels(samples)):
        kind = item.get("kind", "galaxy_dm")
        print("theory %s bin %d" % (item["name"], i), flush=True)
        scale_power = int(item["scale_power"])
        scale = 10.0 ** scale_power
        multiply_ell = bool(item.get("multiply_ell", True))
        n_ell = center["n_ell"] if kind == "dm_auto" else 0.0
        galaxy = item["tracers"][i]
        ell_data = np.asarray(entry["ell"], dtype=float)
        ell_query = np.concatenate([ell_line, ell_data])
        ell_unique, inverse = np.unique(ell_query, return_inverse=True)
        theory_unique = theory_cl(
            ell_unique, galaxy, dm, frb, center["k_cut"], ell_loc, n_ell, kind)
        theory_query = theory_unique[inverse]
        theory_line = theory_query[:ell_line.size]
        line_ell = ell_line if multiply_ell else np.ones_like(ell_line)
        data_ell = ell_data if multiply_ell else np.ones_like(ell_data)
        z0, z1 = entry["z_range"]
        panels.append({
            "name": item["name"], "index": index, "kind": kind,
            "bin": i, "closing": closing,
            "z0": float(z0), "z1": float(z1),
            "scale_power": scale_power,
            "multiply_ell": multiply_ell,
            # Already scaled exactly as drawn, so the plot code only styles.
            "ell_line": ell_line, "model_line": scale * line_ell * theory_line,
            "ell_data": ell_data,
            "data": scale * data_ell * entry["cl"],
            "err": scale * data_ell * entry["err"],
        })
    return {
        "panels": panels,
        "ell_max": ell_max,
        "ell_loc": float(ell_loc),
        "xmin": float(XMIN),
        "center": np.array(list(center.values()), dtype=float),
        "center_keys": np.array(list(center.keys()), dtype=object),
        "b_e": float(b_e),
    }


def save_cl_data_curves(path=None, curves=None):
    """Write cl_data_curves.npz so later redraws skip the theory."""
    if curves is None:
        curves = cl_data_curves()
    if path is None:
        path = picture_asset("cl_data_curves.npz")
    path = Path(path)
    flat = {}
    for panel in curves["panels"]:
        tag = "p%02d_" % panel["index"]
        for key, value in panel.items():
            if key in ("name", "kind", "closing"):
                flat[tag + key] = np.array(value, dtype=object)
            elif key == "multiply_ell":
                flat[tag + key] = np.array(value)
            else:
                flat[tag + key] = np.asarray(value)
    flat["n_panels"] = np.array(len(curves["panels"]))
    for key in ("ell_max", "ell_loc", "xmin", "b_e"):
        flat[key] = np.array(curves[key])
    flat["center"] = np.asarray(curves["center"])
    flat["center_keys"] = np.asarray(curves["center_keys"], dtype=object)
    np.savez_compressed(str(path), **flat)
    print("Saved:", path, flush=True)
    return path


def load_cl_data_curves(path=None):
    """Rebuild the plot_cl_data() payload from cl_data_curves.npz."""
    if path is None:
        path = picture_asset("cl_data_curves.npz")
    path = Path(path)
    with np.load(path, allow_pickle=True) as store:
        flat = {key: store[key] for key in store.files}
    panels = []
    for index in range(int(flat["n_panels"])):
        tag = "p%02d_" % index
        panel = {"index": index}
        for key in ("name", "kind", "closing"):
            panel[key] = str(flat[tag + key])
        for key in ("z0", "z1", "scale_power"):
            panel[key] = int(flat[tag + key]) if key == "scale_power" \
                else float(flat[tag + key])
        panel["multiply_ell"] = bool(flat[tag + "multiply_ell"])
        panel["bin"] = int(flat[tag + "bin"])
        for key in ("ell_line", "model_line", "ell_data", "data", "err"):
            panel[key] = np.asarray(flat[tag + key], dtype=float)
        panels.append(panel)
    return {
        "panels": panels,
        "ell_max": float(flat["ell_max"]),
        "ell_loc": float(flat["ell_loc"]),
        "xmin": float(flat["xmin"]),
        "center": np.asarray(flat["center"], dtype=float),
        "center_keys": [str(key) for key in flat["center_keys"]],
        "b_e": float(flat["b_e"]),
    }


CL_DATA_STYLE = {
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.bfit": "Arial:bold:italic",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "mathtext.cal": "Arial",
    "mathtext.fallback": None,
    "font.size": 19,
    "axes.labelsize": 24,
    "axes.titlesize": 19,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 21,
    "axes.labelpad": 8,
    "axes.titlepad": 7,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#404040",
    "text.color": "#303030",
    "axes.labelcolor": "#303030",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


CL_DATA_GRID = dict(figsize=(16.2, 12.6), left=0.092, right=0.985,
                    bottom=0.072, top=0.930, wspace=0.28, hspace=0.20)


CL_NULL_GRID = dict(CL_DATA_GRID, wspace=0.28)


CL_YLABEL_SIZE = 19


CL_YLABEL_PAD = 5


CL_PANEL_COLORS = {"BGS": "#4D97CF", "LRG": "#D96B6B", "DM": "#9370B7"}


CL_SPECTRUM_LABELS = {
    "BGS": r"BGS $\times$ DM",
    "LRG": r"LRG $\times$ DM",
    "DM": r"DM $\times$ DM",
}


def cl_bin_label(sample, z0, z1, closing):
    """In-panel bin label, matching the axes-block style.

    Arial contains <= but not the set-membership glyph, so the bin edges stay
    as explicit <=/< comparisons in mathtext.
    """
    if sample == "DM":
        return CL_SPECTRUM_LABELS["DM"]
    digits = 2 if sample == "LRG" else 1
    upper = r"\leq" if closing == "]" else "<"
    return rf"${z0:.{digits}f} \leq z {upper} {z1:.{digits}f}$"


def cl_grid_figure(grid=None):
    """3 x 3 axes with the shared spectra style and the requested geometry."""
    grid = CL_DATA_GRID if grid is None else grid
    fig, axes = plt.subplots(3, 3, layout="none", figsize=grid["figsize"])
    fig.subplots_adjust(**{key: grid[key] for key in
                           ("left", "right", "bottom", "top", "wspace", "hspace")})
    return fig, axes


def cl_panel_label(ax, text):
    """Bin label anchored inside the upper-right corner of a panel."""
    ax.legend([Patch(facecolor="none", edgecolor="none")], [text],
              loc="upper right", frameon=False, fontsize=19,
              handlelength=0, handletextpad=0, borderaxespad=0.6)


def cl_y_label(ax, kind, scale_power, column):
    """Rotated y label: first column always, DM auto spectrum as well."""
    if column == 0 or kind == "dm_auto":
        ax.set_ylabel(ylabel_for(kind, scale_power), fontsize=CL_YLABEL_SIZE)
        ax.yaxis.labelpad = CL_YLABEL_PAD
        return True
    return False


def cl_figure_legend(fig, handles, anchor_y=0.945, anchor_x=0.5066, ncol=4):
    """Shared top legend, centred on the tight-cropped page.

    savefig uses bbox_inches="tight", so the page is cropped to the ink box.
    Centring on figure fraction 0.5 lands the legend right of the page centre,
    because the rotated y labels extend the ink further left than right, and
    the legend's own bbox carries leading handle padding that its visible ink
    does not. anchor_x is the measured correction; it is passed in so the
    measured figures and the null test can each be verified against their own
    rendered page. Values were obtained by scanning the saved raster.
    """
    return fig.legend(list(handles.values()), list(handles),
                      loc="lower center", bbox_to_anchor=(anchor_x, anchor_y),
                      ncol=ncol, frameon=False, handlelength=2.0,
                      columnspacing=2.2, handletextpad=0.7,
                      borderaxespad=0.3)


def plot_cl_data(out_pdf=OUT_PDF, curves=None):
    """Render the spectra with Arial, pastel data and one shared legend."""
    if curves is None:
        curves = cl_data_curves()
    ell_max = curves["ell_max"]

    plt.rcParams.update(CL_DATA_STYLE)
    colors = CL_PANEL_COLORS
    spectrum_labels = CL_SPECTRUM_LABELS
    fig, axes = cl_grid_figure()
    legend_handles = {}
    for index, (ax, panel) in enumerate(zip(axes.ravel(), curves["panels"])):
        col = index % 3
        kind = panel["kind"]
        scale_power = panel["scale_power"]
        ell_line = panel["ell_line"]
        shown = ell_line >= XMIN
        z0, z1 = panel["z0"], panel["z1"]
        closing = panel["closing"]
        ax.plot(
            ell_line[shown], panel["model_line"][shown],
            color="#303030", lw=1.8, zorder=2)
        observed = ax.errorbar(
            panel["ell_data"], panel["data"], yerr=panel["err"],
            fmt="o", ms=8, capsize=4.5, elinewidth=1.5,
            capthick=1.5, markeredgecolor="black", markeredgewidth=0.9,
            color=colors[panel["name"]], label=spectrum_labels[panel["name"]],
            zorder=3)
        legend_handles.setdefault(spectrum_labels[panel["name"]], observed)
        ax.axhline(0.0, color="#D0D0D0", lw=0.8, zorder=0)
        style_x_axis(ax, ell_max)
        # Every panel carries its own 10^2 / 10^3 tick labels: the grid is read
        # panel by panel, so a shared bottom row meant looking down a column to
        # identify the scale. Only the $\ell$ axis title stays on the bottom
        # row, where it labels the column instead of repeating nine times.
        # Every panel label now sits in the upper right, so the y range has to
        # guarantee that corner is empty instead of moving the label instead.
        y_lo, y_hi = clear_upper_right(
            ell_line[shown], panel["model_line"][shown],
            panel["ell_data"],
            panel["data"] - panel["err"], panel["data"] + panel["err"])
        ax.set_ylim(y_lo, y_hi)
        cl_y_label(ax, kind, scale_power, col)
        cl_panel_label(ax, cl_bin_label(panel["name"], z0, z1, closing))
        ax.grid(False, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\ell$")
    cl_figure_legend(fig, legend_handles, anchor_x=0.5081, ncol=3)
    out_pdf = figure_output_path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved:", out_pdf, flush=True)


PZ_TYPOGRAPHY_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 14,
    "axes.labelsize": 17, "xtick.labelsize": 13, "ytick.labelsize": 13,
}


PZ_LEGEND_STYLE = {
    "loc": "center right", "bbox_to_anchor": (0.975, 0.47),
    "handlelength": 3.0, "handleheight": 1.1, "handletextpad": 0.85,
    "labelspacing": 0.8, "fontsize": 15,
}


PZ_MEAN_CL_COLOR = "#D66F56"


PZ_MEAN_JOINT_COLOR = "#6BAED6"


SELECTION_PALETTE = ("#D49B32", "#398EBA", "#BE6895")


def plot_selection_panels_test(gaussian_sigma=1.5, output_stem="P_det_test2"):
    """Compare three Gaussian delays with a grey, no-delay SFR baseline."""
    from scipy.integrate import trapezoid
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    with np.load(picture_asset("P_det_data.npz"), allow_pickle=False) as saved:
        result = dict(saved)
    info = json.loads(picture_asset("P_det.json").read_text())
    if (info.get("delay_model") != "gauss" or info.get("sigma_G_Gyr") != 2.5
            or info.get("tau_G_values_Gyr") != [1., 3., 5.]):
        raise ValueError("The two-panel illustration requires Gaussian tau=1,3,5, sigma=2.5")
    z = result["z"]
    gaussian_sigma = float(gaussian_sigma)
    if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
        raise ValueError("Gaussian delay width must be positive")
    if gaussian_sigma != info["sigma_G_Gyr"]:
        import pyccl as ccl
        from frb_redshift_model import engine_intrinsic_p, get_engine, sfh_year
        if sfh_year() != info["sfh_year"]:
            raise ValueError("Use the SFH_YEAR of the baseline Gaussian selection curves")
        cosmo = production_cosmology(Omega_b=0.04897, Omega_c=0.3111 - 0.04897,
                              h=0.6766, sigma8=0.81, n_s=0.96)
        engine = get_engine(cosmo, z, delay_model="gauss")
        for tau in info["tau_G_values_Gyr"]:
            key = f"tau_{tau:g}"
            intrinsic = engine_intrinsic_p(engine, tau, gaussian_sigma)
            observed = engine.selected_nz(tau, gaussian_sigma)
            product = intrinsic * result["pdet_relative"]
            weight = float(trapezoid(product, z))
            np.testing.assert_allclose(trapezoid(intrinsic, z), 1., atol=1e-12)
            np.testing.assert_allclose(trapezoid(observed, z), 1., atol=1e-12)
            np.testing.assert_allclose(observed, product / weight, rtol=2e-12, atol=1e-13)
            result.update({key + "_intrinsic": intrinsic,
                           key + "_observed_unit": observed,
                           key + "_observed_product": product,
                           key + "_parameters": np.array([tau, gaussian_sigma])})
            info["models"][key].update({"sigma_G_Gyr": gaussian_sigma,
                                        "role": "Illustrative delay parameters, not a posterior estimate",
                                        "intrinsic_integral": float(trapezoid(intrinsic, z)),
                                        "observed_integral": float(trapezoid(observed, z)),
                                        "relative_detection_weight_integral": weight})
            print(f"Gaussian tau_G={tau:g}, sigma_G={gaussian_sigma:g} Gyr: unit integrals verified", flush=True)
        info["sigma_G_Gyr"] = gaussian_sigma
        info["prediction_kind"] = "Illustrative Gaussian delays at a fixed width; these are not joint-posterior central predictions"
    # Reuse the native no-delay reference from the existing selection plot.
    sfr = None
    cache, metadata_path = picture_asset("P_det_test_data.npz"), picture_asset("P_det_test.json")
    if cache.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        with np.load(cache, allow_pickle=False) as saved:
            if (metadata.get("sfh_year") == info["sfh_year"]
                    and metadata.get("selection_sha256") == info["selection_sha256"]
                    and "sfr_intrinsic" in saved and np.array_equal(saved["z"], z)
                    and np.array_equal(saved["pdet_relative"], result["pdet_relative"])):
                sfr = saved["sfr_intrinsic"].copy()
    if sfr is None:
        import pyccl as ccl
        from frb_redshift_model import engine_intrinsic_p, get_engine, sfh_year
        if sfh_year() != info["sfh_year"]:
            raise ValueError("Use the SFH_YEAR of the saved Gaussian selection curves")
        cosmo = production_cosmology(Omega_b=0.04897, Omega_c=0.3111 - 0.04897,
                              h=0.6766, sigma8=0.81, n_s=0.96)
        sfr = engine_intrinsic_p(get_engine(cosmo, z, delay_model="sfh"))
    sfr_product = sfr * result["pdet_relative"]
    sfr_weight = float(trapezoid(sfr_product, z))
    if not np.isfinite(sfr_weight) or sfr_weight <= 0:
        raise ValueError("Invalid SFR selection weight")
    result.update(sfr_intrinsic=sfr, sfr_observed_unit=sfr_product / sfr_weight,
                  sfr_observed_product=sfr_product,
                  sfr_relative_detection_weight_integral=sfr_weight)
    np.testing.assert_allclose(trapezoid(sfr, z), 1., atol=1e-12)
    np.testing.assert_allclose(trapezoid(result["sfr_observed_unit"], z), 1., atol=1e-12)
    sfr_color = "#737373"
    sfr_outline_color, sfr_outline_width = "#666666", 0.85
    sfr_alpha = {"intrinsic": 0.32, "observed_unit": 0.24}
    series = (("tau_1", 1., "#E69F00"),
              ("tau_3", 3., "#0072B2"),
              ("tau_5", 5., "#8B65A6"))
    rc = {**PZ_TYPOGRAPHY_RC, "axes.linewidth": 0.9,
          "xtick.major.size": 4., "ytick.major.size": 4.,
          "xtick.major.width": 0.8, "ytick.major.width": 0.8,
          "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
          "figure.facecolor": "white", "axes.facecolor": "white",
          "savefig.facecolor": "white", "axes.grid": False,
          "text.color": "#222222", "axes.labelcolor": "#222222"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharex=True)
        fig.subplots_adjust(left=0.078, right=0.90, bottom=0.16, top=0.80, wspace=0.34)
        for ax, quantity, alpha, ylabel in (
                (axes[0], "intrinsic", 0.65, r"$n_{\rm int}(z)$"),
                (axes[1], "observed_unit", 0.45, r"$n_f(z)$")):
            sfr_curve = result["sfr_" + quantity]
            ax.fill_between(z, 0., sfr_curve, facecolor=sfr_color, edgecolor="none",
                            linewidth=0, alpha=sfr_alpha[quantity], zorder=0.5)
            ymax = float(sfr_curve[(z >= 0.) & (z <= 3.)].max())
            for key, tau, color in reversed(series):
                curve = result[key + "_" + quantity]
                ax.fill_between(z, 0., curve, facecolor=color, edgecolor="none",
                                linewidth=0, alpha=alpha)
                ymax = max(ymax, float(curve[(z >= 0.) & (z <= 3.)].max()))
            # Keep the reference visible through overlapping coloured fills.
            ax.plot(z, sfr_curve, color=sfr_outline_color, lw=sfr_outline_width,
                    solid_capstyle="round", zorder=4.5)
            ax.set(xlim=(0., 3.), ylim=(0., ymax * 1.10), xlabel=r"$z$", ylabel=ylabel)
            ax.xaxis.set_major_locator(MultipleLocator(1.))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda value, _: "" if abs(value) < 1e-12 else f"{value:g}"))
            ax.tick_params(which="both", direction="in", top=True, right=False, pad=5)
            ax.xaxis.labelpad = 5
            ax.yaxis.labelpad = 8
        right = axes[1].twinx()
        mask = result["pdet_plot_mask"]
        right.plot(z[mask], result["pdet_relative"][mask], color="#DB746B", lw=2.,
                   ls=(0, (6., 2.4, 1.5, 2.4)))
        right.set_yscale("log")
        right.set_ylim(1e-6, 1.18)
        right.set_ylabel(r"$P_{\rm det}(z)$", labelpad=10)
        right.yaxis.set_major_locator(FixedLocator([1., 1e-2, 1e-4, 1e-6]))
        right.yaxis.set_minor_locator(FixedLocator([]))
        right.tick_params(which="both", direction="in", left=False, right=True, pad=5)
        label_z = 1.95
        right.annotate(r"$P_{\rm det}(z)$",
                       xy=(label_z, np.interp(label_z, z, result["pdet_relative"])),
                       xytext=(0, 10), textcoords="offset points", fontsize=12,
                       ha="center", va="bottom")
        labels = [rf"$\tau_{{\rm G}}={tau:g}\,\mathrm{{Gyr}}$" for _, tau, _ in series]
        handles = [Patch(facecolor=color, alpha=0.65, edgecolor="none") for _, _, color in series]
        handles.append(Patch(facecolor=sfr_color, alpha=sfr_alpha["intrinsic"], edgecolor="none"))
        labels.append("SFR")
        fig.legend(handles, labels,
                   loc="upper center", bbox_to_anchor=(0.49, 0.99), ncol=4,
                   frameon=False, fontsize=13, handlelength=2.0,
                   handletextpad=0.6, columnspacing=1.8)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUT_DIR / figure_output_stem(output_stem)
        for suffix in ("pdf", "png"):
            output = stem.with_suffix(".pdf") if suffix == "pdf" else PNG_DIR / (stem.name + "." + suffix)
            fig.savefig(output, dpi=450)
            print("Saved:", output, flush=True)
        plt.close(fig)
    info.update({"source_figure": "P_det.pdf", "layout": "Two panels: intrinsic, selected",
                 "colors": {**{key: color for key, _, color in series}, "sfr": sfr_color},
                 "fill_alpha": {"intrinsic": 0.65, "observed": 0.45},
                 "sfr_fill_alpha": {"intrinsic": sfr_alpha["intrinsic"], "observed": sfr_alpha["observed_unit"]},
                 "sfr_outline": {"color": sfr_outline_color, "linewidth_pt": sfr_outline_width,
                                 "zorder": 4.5, "style": "solid"},
                 "includes_sfr_reference": True,
                 "sfr_reference": {"model": "Native no-delay SFR redshift density, including volume and observer-time factors",
                                   "sfh_year": info["sfh_year"], "color": sfr_color,
                                   "normalization_interval": info["redshift_interval"],
                                   "intrinsic_integral": float(trapezoid(sfr, z)),
                                   "observed_integral": float(trapezoid(result["sfr_observed_unit"], z))},
                 "palette_rule": "Orange, blue and purple Gaussian filled areas; light grey SFR with a thin foreground grey outline",
                 "density_outlines": False, "legend_labels": labels,
                 "panel_letters": False,
                 "line_styles": {"intrinsic": "none", "observed": "none", "sfr": "solid", "selection": "dash-dot"},
                 "line_widths_pt": {"intrinsic": 0., "observed": 0., "sfr": sfr_outline_width, "selection": 2.},
                 "selection_label": "Direct label on the right-panel curve",
                 "selection_color": "#DB746B", "selection_linewidth_pt": 2.,
                 "display_range": [0., 3.], "axis_label_fontsize_pt": 17.,
                 "tick_fontsize_pt": 13., "legend_fontsize_pt": 13.,
                 "width_inches": 10.6, "height_inches": 4.4,
                 "fill_definition": "Area below each density; not uncertainty intervals"})
    np.savez_compressed(PNG_DIR / (stem.name + "_data.npz"), **result)
    (PNG_DIR / (stem.name + ".json")).write_text(json.dumps(info, indent=2) + "\n")
    return result, info


def intrinsic_posterior_data(n_draw=10000, delay_model="gauss", extended_summary=False):
    """Reconstruct intrinsic PDFs and integrate EACH posterior draw.

    Use the original 500-point grid, cleaned chains and random seeds from
    plot_compare.py. No selection function, smoothing or truncation at the
    visible x limit is applied. The matrix quadrature below is algebraically
    identical to engine_intrinsic_p, and is checked against it for each chain.
    Pointwise median PDFs need not themselves have exactly unit integral;
    the CDF quantiles therefore come from the normalized individual draws.
    """
    import pyccl as ccl
    from scipy.integrate import cumulative_trapezoid, trapezoid
    from frb_redshift_model import (
        _delay_pdf, delay_names, engine_intrinsic_p, get_engine,
        normalize_delay_model, sfh_year,
    )

    if n_draw < 1:
        raise ValueError("--pz-draws must be positive")
    delay_model = normalize_delay_model(delay_model)
    folder = {"gauss": "gauss", "lognormal": "lognormal",
              "powerlaw": "power-law"}[delay_model]
    z = np.linspace(0.0, 5.0, 500)
    cosmo = production_cosmology(Omega_b=0.04897, Omega_c=0.3111 - 0.04897,
                          h=0.6766, sigma8=0.81, n_s=0.96)
    engine = get_engine(cosmo, z, delay_model=delay_model)
    sfh = engine_intrinsic_p(get_engine(cosmo, z, delay_model="sfh"))
    # Trapezoidal weights along the delay-time axis.
    dt = np.diff(engine.tau)
    weights = np.concatenate(([dt[0] / 2], (dt[:-1] + dt[1:]) / 2,
                              [dt[-1] / 2]))
    kernel = (engine.R_sfh_form * weights).T
    result = {
        "z": z, "quantiles": np.array([0.16, 0.50, 0.84]),
        "sfh_pdf": sfh,
        "sfh_cdf": cumulative_trapezoid(sfh, z, initial=0),
        "sfh_year": sfh_year(), "normalization_range": np.array([0., 5.]),
        "pdf_display_range": np.array([0., 5.]),
        "cdf_display_range": np.array([0., 5.]),
        "summary": "pointwise posterior median and equal-tailed 68% interval",
        "delay_model": delay_model,
    }
    if extended_summary:
        from scipy.special import ndtr
        result.update({
            "sigma_quantile_levels": ndtr(np.array([-2., -1., 0., 1., 2.])),
            "sigma_coverage": ndtr(np.array([1., 2.])) - ndtr(np.array([-1., -2.])),
            "sigma_multipliers": np.array([1., 2.]),
            "std_ddof": 0,
            "delay_parameter_names": np.asarray(delay_names(delay_model)),
            "extended_summary_method": "Pointwise quantiles and arithmetic mean/std of the SAME unit-normalized posterior PDF/CDF draws; no standard-error scaling",
            "display_bound_rule": "PDF bounds clipped to >=0; CDF bounds clipped to [0,1] ONLY when drawing; unmodified mean +/- k*std saved",
        })
    for mode, stem, seed in (("cl", "frb_redshift", 1),
                             ("joint", "frb_joint", 2)):
        # Explicit paths: RESULT_ROOT overrides must not select another dataset.
        path = RESULT_ROOT / mode / folder / "chains" / (stem + "_converged.npy")
        # Same trusted bundle layout as plot_result.load_chain, without
        # importing the unrelated posterior-plotting and likelihood modules.
        blob = np.load(path, allow_pickle=True)
        samples = np.asarray(blob[0], dtype=float)
        names = [str(name) for name in blob[2]]
        columns = [names.index(p) for p in delay_names(delay_model)]
        count = min(n_draw, len(samples))
        if count < 1:
            raise ValueError(f"Empty posterior chain: {path}")
        indices = np.random.default_rng(seed).choice(len(samples), count, replace=False)
        params = samples[indices][:, columns]
        del samples, blob
        if not np.all(np.isfinite(params)):
            raise ValueError(f"Nonfinite delay parameters in {path}")
        if delay_model != "powerlaw" and np.any(params[:, 1] <= 0):
            raise ValueError(f"Nonpositive delay width in {path}")
        if delay_model == "lognormal" and np.any(params[:, 0] <= 0):
            raise ValueError(f"Nonpositive log-normal delay scale in {path}")
        pdf = np.empty((count, len(z)))
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            delay_pdf = np.stack([_delay_pdf(engine, p)
                                  for p in params[start:stop]])
            block = np.clip((delay_pdf @ kernel) * engine.vol, 0.0, None)
            norm = trapezoid(block, z, axis=1)
            if np.any(norm <= 0) or not np.all(np.isfinite(block)):
                raise ValueError(f"Invalid intrinsic distribution in {path}")
            pdf[start:stop] = block / norm[:, None]
        for i in (0, count // 2, count - 1):
            np.testing.assert_allclose(pdf[i], engine_intrinsic_p(engine, *params[i]),
                                       rtol=1e-11, atol=1e-13)
        cdf = cumulative_trapezoid(pdf, z, axis=1, initial=0)
        np.testing.assert_allclose(trapezoid(pdf, z, axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(cdf[:, -1], 1., atol=1e-12)
        if np.any(np.diff(cdf, axis=1) < -1e-14):
            raise ValueError("Cumulative distributions must be monotonic")
        result[mode + "_pdf"] = np.quantile(pdf, result["quantiles"], axis=0)
        result[mode + "_cdf"] = np.quantile(cdf, result["quantiles"], axis=0)
        result[mode + "_source"] = os.path.relpath(path, ROOT)
        result[mode + "_draw_indices"] = indices
        result[mode + "_seed"] = seed
        if extended_summary:
            import hashlib
            result[mode + "_delay_parameters"] = params
            result[mode + "_chain_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            for quantity, values in (("pdf", pdf), ("cdf", cdf)):
                prefix = mode + "_" + quantity
                mean = values.mean(axis=0)
                std = values.std(axis=0, ddof=0)
                result[prefix + "_sigma_quantiles"] = np.quantile(
                    values, result["sigma_quantile_levels"], axis=0)
                result[prefix + "_mean"] = mean
                result[prefix + "_std"] = std
                result[prefix + "_mean_std_bounds"] = np.stack([
                    np.stack([mean - k * std, mean + k * std])
                    for k in result["sigma_multipliers"]])
                # Independent second-moment check catches accidental SEM use.
                np.testing.assert_allclose(std ** 2, np.mean((values - mean) ** 2, axis=0),
                                           rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(trapezoid(result[mode + "_pdf_mean"], z), 1., atol=1e-12)
            np.testing.assert_allclose(
                cumulative_trapezoid(result[mode + "_pdf_mean"], z, initial=0),
                result[mode + "_cdf_mean"], atol=1e-12)
        print(f"{folder}/{mode}: {count} draws from {os.path.relpath(path, ROOT)}; "
              "normalization and cumulative checks passed", flush=True)
    return result


def plot_three_model_pz(n_draw=10000, result=None, paired_only=False,
                        output_stem="test_pz"):
    """Separate intrinsic PDF/CDF panels, with six posterior comparisons.

    Color and hatch identify the delay model; light dashed C_ell and dark
    solid Joint curves identify the data. Integrate each normalized draw
    before taking CDF quantiles, never the pointwise median PDF. All models
    use the same 0 <= z <= 5 normalization and display grid. The green
    dash-dot SFR curve is the no-delay reference, without a posterior band.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

    models = (
        ("gauss", "Gaussian", "#D7A94D", "#9D6A13", "/"),
        ("lognormal", "Log-normal", "#79B7D8", "#206894", "\\"),
        ("power-law", "Power-law", "#D59AB4", "#9D446D", "."),
    )
    if result is None:
        result = {key: intrinsic_posterior_data(n_draw, delay_model=key)
                  for key, *_ in models}
    z = result["gauss"]["z"]
    for key, *_ in models:
        np.testing.assert_array_equal(result[key]["z"], z)
        if str(result[key].get("summary_method", "median")) != "mean_std":
            np.testing.assert_array_equal(result[key]["quantiles"], [0.16, 0.5, 0.84])
        for quantity in ("pdf", "cdf"):
            np.testing.assert_allclose(result[key]["sfh_" + quantity],
                                       result["gauss"]["sfh_" + quantity])
    sfr_color = "#668269"
    sfr_ls = (0, (6, 2, 1.5, 2))

    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 15,
        "axes.labelsize": 21, "axes.linewidth": 1.1,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.axisbelow": True,
        "axes.grid": False, "grid.color": "#8A8A8A",
        "grid.alpha": 0.16, "grid.linewidth": 0.7,
        "hatch.linewidth": 0.4, "legend.frameon": False,
    }
    def draw_panel(ax, quantity, show_xlabels=True):
        """Draw one PDF/CDF panel and return its shared legend handles."""
        handles, labels = [], []
        # Paint all bands first so no band's fill obscures a median.
        for key, label, light, dark, hatch in models:
            for mode, color, ls, alpha in (
                    ("cl", light, (0, (5, 2.5)), 0.16),
                    ("joint", dark, "-", 0.14)):
                lo, median, hi = result[key][mode + "_" + quantity]
                if (not np.all(np.isfinite([lo, median, hi]))
                        or np.any(lo > median) or np.any(median > hi)
                        or np.any(lo < -1e-13)):
                    raise ValueError(f"Invalid {key}/{mode} {quantity} quantiles")
                ax.fill_between(z, lo, hi, facecolor=color, alpha=alpha,
                                linewidth=0, zorder=2)
                if mode == "joint":
                    ax.fill_between(z, lo, hi, facecolor="none", edgecolor=color,
                                    hatch=hatch, alpha=0.35, linewidth=0, zorder=2.2)
                if mode == "cl":
                    handles.append(Patch(facecolor=color, alpha=0.55,
                                         edgecolor=dark, linewidth=1.0))
        for key, _, light, dark, _ in models:
            for mode, color, ls in (("cl", light, (0, (5, 2.5))),
                                     ("joint", dark, "-")):
                ax.plot(z, result[key][mode + "_" + quantity][1],
                        color=color, ls=ls, lw=2.0, zorder=4)
        ax.plot(z, result["gauss"]["sfh_" + quantity],
                color=sfr_color, ls=sfr_ls, lw=2.0, zorder=4)
        # Colour identifies the delay model, line style the probe, so the
        # legend needs three swatches plus three style keys instead of one
        # long entry per model/probe pair.
        handles += [Line2D([], [], color="#404040", ls=(0, (5, 2.5)), lw=2.0),
                    Line2D([], [], color="#404040", ls="-", lw=2.0),
                    Line2D([], [], color=sfr_color, ls=sfr_ls, lw=2.0)]
        labels = ([label for _k, label, _l, _d, _h in models]
                  + [r"$C_\ell$", "joint", "SFR"])
        ax.set_xlim(0, 5)
        if quantity == "pdf":
            ymax = max(result[key][mode + "_pdf"][2].max()
                       for key, *_ in models for mode in ("cl", "joint"))
            ymax = max(ymax, result["gauss"]["sfh_pdf"].max())
            ax.set_ylim(0, 1.06 * ymax)
            ax.set_ylabel(r"$p(z)$", labelpad=8)
        else:
            ax.set_ylim(0, 1.04)
            ax.set_ylabel(r"$p(<z)$", labelpad=8)
        ax.set_xlabel(r"$z$", labelpad=5)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if not show_xlabels:
            # Stacked panels share the x axis, so the lower one carries the
            # tick labels and the shared axis title; no duplicate numbers.
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        if quantity == "cdf":
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        # Only the x tick keeps a zero, so the corner shows one 0, not two.
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "" if abs(v) < 1e-12 else f"{v:g}"))
        ax.tick_params(which="both", direction="in", top=True, right=True, pad=6)
        ax.grid(True, axis="x", which="major")
        for spine in ax.spines.values():
            spine.set_color("#2B2B2B")
        return handles, labels

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(rc):
        for quantity, stem in (() if paired_only else (("pdf", "pz_three_model"), ("cdf", "cz_three_model"))):
            fig, ax = plt.subplots(figsize=(8.0, 5.3))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.965)
            handles, labels = draw_panel(ax, quantity)
            ax.legend(handles, labels, ncol=1, fontsize=17, handlelength=2.6,
                      loc="upper right" if quantity == "pdf" else "lower right",
                      labelspacing=0.95, borderaxespad=0.5, handletextpad=0.8)
            path = OUT_DIR / (stem + ".pdf")
            fig.savefig(path)
            # PNG companion uses the same canvas for convenient visual checks.
            fig.savefig(path.with_suffix(".png"), dpi=300)
            plt.close(fig)
            print("Saved:", path, flush=True)

        # PDF and CDF side by side. Both panels use the same colour and
        # linestyle keys, so each carries an identical legend: upper right on
        # the PDF, lower right on the CDF where that corner is empty. The
        # canvas keeps each axes at the single-panel size (6.75 x 4.37 in) so
        # fonts, line widths and the legend keep their original scale.
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4))
        fig.subplots_adjust(left=0.066, right=0.988, bottom=0.155, top=0.965,
                            wspace=0.155)
        handles, labels = draw_panel(axes[0], "pdf")
        draw_panel(axes[1], "cdf")
        axes[1].set_ylabel(r"$P(z)$", labelpad=8)
        legend_kw = dict(ncol=1, fontsize=17, handlelength=2.6,
                         labelspacing=0.95, borderaxespad=0.5,
                         handletextpad=0.8)
        axes[0].legend(handles, labels, loc="upper right", **legend_kw)
        axes[1].legend(handles, labels, loc="lower right", **legend_kw)
        path = OUT_DIR / (output_stem + ".pdf")
        fig.savefig(path)
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_DIR / (output_stem + ".png"), dpi=300)
        plt.close(fig)
        print("Saved:", path, flush=True)

        if paired_only:
            packed = {"z": z}
            for key, *_ in models:
                packed.update({key + "_" + name: value for name, value in result[key].items()
                               if name != "z"})
            np.savez_compressed(PNG_DIR / (output_stem + "_data.npz"), **packed)
            return result

        # Same two panels stacked. Each axes works out at ~6.8 x 4.45 in,
        # i.e. almost the 8.0 x 5.3 in single-panel geometry, so the curve
        # shapes keep their proportions instead of reading flatter. The
        # shared x axis drops the duplicate tick labels and one legend still
        # covers both panels.
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 10.4), sharex=True)
        fig.subplots_adjust(left=0.125, right=0.975, bottom=0.075, top=0.98,
                            hspace=0.115)
        handles, labels = draw_panel(axes[0], "pdf", show_xlabels=False)
        draw_panel(axes[1], "cdf")
        axes[0].legend(handles, labels, ncol=1, fontsize=17, handlelength=2.6,
                       loc="upper right", labelspacing=0.95,
                       borderaxespad=0.5, handletextpad=0.8)
        path = OUT_DIR / "cz_pz_three_model_updown.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)
        print("Saved:", path, flush=True)
    packed = {"z": z}
    for key, *_ in models:
        packed.update({key + "_" + name: value for name, value in result[key].items()
                       if name != "z"})
    np.savez_compressed(OUT_DIR / "three_model_pz_data.npz", **packed)
    return result


def plot_intrinsic_pz(n_draw=10000, result=None, joint_color=None,
                      output_stem="pz_median",
                      sfh_label="SFR", cl_color=None, delay_model="gauss",
                      interval_summary=None, mean_sigma_levels=(1,)):
    """One panel with PDF/CDF on separate left/right axes.

    Notebook cells 7, 10 and 11 motivate the boxed axes, inward ticks, subtle
    grid. Color identifies the probe; the no-delay SFR reference is grey.
    Solid/dashed lines identify PDF/CDF. The legend occupies the open
    middle-right region. The lightblue variant explicitly uses gold C_ell.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

    output_stem = figure_output_stem(output_stem)
    if cl_color is None:
        cl_color = PZ_MEAN_CL_COLOR if output_stem == "pz" else "#E58262"
    if joint_color is None:
        joint_color = PZ_MEAN_JOINT_COLOR if output_stem == "pz" else "#6BAED6"
    if result is None:
        result = intrinsic_posterior_data(n_draw=n_draw, delay_model=delay_model,
                                         extended_summary=interval_summary is not None)
    if interval_summary not in (None, "median", "mean_std"):
        raise ValueError("Unknown intrinsic interval summary")
    if tuple(mean_sigma_levels) not in ((1,), (1, 2)):
        raise ValueError("mean_sigma_levels must be (1,) or (1, 2)")
    two_levels = interval_summary == "median" or (interval_summary == "mean_std" and 2 in mean_sigma_levels)
    result = dict(result)
    result.pop("display_range", None)  # Upgrade older cached figure metadata.
    result["pdf_display_range"] = np.array([0., 5.])
    result["cdf_display_range"] = np.array([0., 5.])
    z = result["z"]
    styles = (
        ("cl", cl_color, r"$C_\ell$"),
        ("joint", joint_color, "Joint"),
    )
    sfh_color = "#777777"
    result.update({"plot_cl_color": cl_color, "plot_joint_color": joint_color,
                   "plot_sfr_color": sfh_color, "plot_sfr_label": sfh_label})
    if interval_summary:
        result["plot_interval_summary"] = interval_summary
        result["plot_band_alpha"] = np.array([0.30, 0.12] if two_levels else [0.30])
        result["plot_sigma_levels"] = np.array([1., 2.] if two_levels else [1.])
        result["summary"] = (
            "Pointwise posterior median with central 68.2689% and 95.4500% credible intervals"
            if interval_summary == "median" else
            f"Pointwise arithmetic posterior mean +/- {'1 and 2' if two_levels else '1'} population standard deviations (ddof=0); not central credible intervals")
    cdf_ls = (0, (4.5, 2.2))
    rc = {
        **PZ_TYPOGRAPHY_RC, "axes.linewidth": 1.1,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.width": 0.65, "ytick.minor.width": 0.65,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": True,
        "axes.axisbelow": True, "grid.color": "#8A8A8A",
        "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False,
    }
    with plt.rc_context(rc):
        fig, pdf_ax = plt.subplots(figsize=(7.2, 4.8))
        cdf_ax = pdf_ax.twinx()
        fig.subplots_adjust(left=0.12, right=0.875, bottom=0.15, top=0.965)
        for ax, quantity, ls in ((pdf_ax, "pdf", "-"),
                                 (cdf_ax, "cdf", cdf_ls)):
            centers = []
            for mode, color, _ in styles:
                prefix = mode + "_" + quantity
                if interval_summary == "median":
                    q = result[prefix + "_sigma_quantiles"]
                    center, bounds = q[2], ((q[0], q[4]), (q[1], q[3]))
                elif interval_summary == "mean_std":
                    center = result[prefix + "_mean"]
                    b = result[prefix + "_mean_std_bounds"]
                    bounds = (b[1], b[0]) if two_levels else (b[0],)
                else:
                    lo, center, hi = result[prefix]
                    bounds = ((lo, hi),)
                for (lo, hi), alpha in zip(bounds, (0.12, 0.30) if two_levels else (0.30,)):
                    lo = np.clip(lo, 0., 1. if quantity == "cdf" else np.inf)
                    hi = np.clip(hi, 0., 1. if quantity == "cdf" else np.inf)
                    ax.fill_between(z, lo, hi, facecolor=color, alpha=alpha,
                                    linewidth=0, zorder=2)
                centers.append((center, color))
            for center, color in centers:
                ax.plot(z, center, color=color, ls=ls, lw=1.85, zorder=4)
            ax.plot(z, result["sfh_" + quantity], color=sfh_color,
                    ls=ls, lw=1.8, zorder=3)
            ax.set_xlim(*result[quantity + "_display_range"])
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            # Show one zero at the origin, on the horizontal axis.
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _: f"{value:g}"))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda value, _: "" if np.isclose(value, 0) else f"{value:.1f}"))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#2B2B2B")
            ax.tick_params(which="both", direction="in", top=True,
                           left=(quantity == "pdf"), right=(quantity == "cdf"),
                           pad=5)
            ax.grid(False)
        # A single frame and x-grid avoid implying that the two y scales agree.
        pdf_ax.grid(True, axis="x", which="major")
        for spine in cdf_ax.spines.values():
            spine.set_visible(False)
        if interval_summary == "median":
            density_max = max(result[mode + "_pdf_sigma_quantiles"][4].max()
                              for mode, *_ in styles)
            median_peak = max(result[mode + "_pdf_sigma_quantiles"][2].max()
                              for mode, *_ in styles)
        elif interval_summary == "mean_std":
            density_max = max(result[mode + "_pdf_mean_std_bounds"][1 if two_levels else 0, 1].max()
                              for mode, *_ in styles)
            median_peak = max(result[mode + "_pdf_mean"].max()
                              for mode, *_ in styles)
        else:
            density_max = max(result[mode + "_pdf"][2].max()
                              for mode, *_ in styles)
            median_peak = max(result[mode + "_pdf"][1].max()
                              for mode, *_ in styles)
        median_peak = max(median_peak, result["sfh_pdf"].max())
        # Keep the highest PDF median near CDF=1 in display coordinates.
        # Add 2.5% extra PDF-axis headroom above the aligned layout so the
        # credible band has breathing room; leave the CDF scale unchanged.
        # Only the axis scales change; all densities and CDFs stay untouched.
        cdf_limit = 1.08 if interval_summary else max(1.08, 1.01 * density_max / median_peak)
        pdf_limit = max(density_max, median_peak) * 1.06 if interval_summary else median_peak * cdf_limit * 1.025
        pdf_ax.set_ylim(0, pdf_limit)
        pdf_ax.set_ylabel(r"$p(z)$", labelpad=8)
        pdf_ax.set_xlabel(r"$z$", labelpad=5)
        pdf_ax.yaxis.set_major_locator(MultipleLocator(0.2 if pdf_limit > 1.15 else 0.1))
        cdf_ax.set_ylim(0, cdf_limit)
        cdf_ax.set_ylabel(r"$P(z)$", labelpad=10)
        cdf_ax.yaxis.set_major_locator(MultipleLocator(0.2))
        handles = [Patch(facecolor=color, alpha=0.30, edgecolor="none")
                   for _, color, _ in styles]
        handles.extend([
            Line2D([], [], color=sfh_color, lw=1.8),
            Line2D([], [], color="#000000", lw=1.85, ls="-"),
            Line2D([], [], color="#000000", lw=1.85, ls=cdf_ls),
        ])
        labels = [s[2] for s in styles] + [sfh_label, "PDF", "CDF"]
        result["plot_legend_labels"] = np.asarray(labels)
        cdf_ax.legend(handles, labels, **PZ_LEGEND_STYLE)
        result.update({"plot_typography_reference": "pz_goldblue.pdf",
                       "plot_axis_label_fontsize_pt": 17., "plot_tick_fontsize_pt": 13.,
                       "plot_legend_fontsize_pt": 15., "plot_legend_labelspacing": 0.8,
                       "plot_legend_handlelength": 3., "plot_legend_handleheight": 1.1,
                       "plot_legend_handletextpad": 0.85})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUT_DIR / output_stem
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = stem.with_suffix(".pdf") if extension == "pdf" else PNG_DIR / (output_stem + ".png")
            fig.savefig(path, dpi=450)
            print("Saved:", path, flush=True)
        plt.close(fig)
    np.savez_compressed(PNG_DIR / (output_stem + "_data.npz"), **result)


def gaussian_bias_posterior_data():
    """Same GetDist estimator and all cleaned samples as plot_compare.py."""
    result = {
        "delay_model": "gauss",
        "parameter": "b_frb",
        "normalization": "Each marginalized GetDist posterior divided by its peak",
        "fill_meaning": "Full posterior profile, not a credible-interval band",
    }
    for mode, stem in (("cl", "frb_redshift"), ("joint", "frb_joint")):
        path = RESULT_ROOT / mode / "gauss" / "chains" / (stem + "_converged.npy")
        samples = mcsamples(path, ["b_frb"], [r"b_{\rm FRB}"])
        density = samples.get1DDensity("b_frb")
        x = np.asarray(density.x, dtype=float)
        p = np.clip(np.asarray(density.P, dtype=float), 0.0, None)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)) or p.max() <= 0:
            raise ValueError(f"Invalid bias posterior: {path}")
        result[mode + "_x"] = x
        result[mode + "_density"] = p / p.max()
        result[mode + "_source"] = os.path.relpath(path, ROOT)
        result[mode + "_n_samples"] = samples.samples.shape[0]
        print(f"Gaussian delay / {mode}: {samples.samples.shape[0]:,} samples; "
              f"b_FRB posterior mode = {x[np.argmax(p)]:.4f}", flush=True)
    return result


def plot_gaussian_bias(result=None):
    """FRB bias posteriors with Gaussian galaxy-bias reference constraints.

    Galaxy means and 1-sigma errors are supplied by the user for 0.2<z<0.7,
    mean z=0.51. They are Gaussian approximations to measurement constraints,
    not the population scatter of individual galaxies. All curves retain
    the original figure's peak normalization for shape/position comparison.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

    if result is None:
        result = gaussian_bias_posterior_data()
    result = dict(result)
    styles = (
        ("cl", "#F1E2C6", "#D5BC8E", "-.", r"$C_\ell$"),
        ("joint", "#DFC18D", "#B7955B", "-", "Joint"),
    )
    references = (
        ("star_forming", 1.23, 0.08, "#B8D9EE", "#7BB3D8",
         (0, (4.5, 2.2)), "Star-forming"),
        ("quiescent", 1.75, 0.04, "#F0BFC2", "#DF949B",
         (0, (4.5, 2.2)), "Quiescent"),
    )
    reference_x = np.linspace(0.5, 2.5, 4001)
    for key, mean, sigma, *_ in references:
        result[key + "_x"] = reference_x
        result[key + "_density"] = np.exp(-0.5 * ((reference_x - mean) / sigma) ** 2)
        result[key + "_mean"] = mean
        result[key + "_sigma"] = sigma
    result["galaxy_reference_z_range"] = np.array([0.2, 0.7])
    result["galaxy_reference_mean_z"] = 0.51
    result["galaxy_reference_source"] = "User-provided galaxy bias comparison table"
    result["galaxy_reference_method"] = "Gaussian approximations to reported bias " \
        "constraints; quoted errors treated as 1 sigma; peak-normalized; " \
        "not reweighted to the FRB redshift kernel"
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 12,
        "axes.labelsize": 22, "axes.linewidth": 0.85,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.labelsize": 16.5, "ytick.labelsize": 16.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.width": 0.55, "ytick.minor.width": 0.55,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": False,
        "axes.axisbelow": True, "grid.color": "#8A8A8A",
        "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False,
    }
    with plt.rc_context(rc):
        # Same 8 in width as frb_dmz_gauss_sfh_joint so identical point sizes
        # also land identically once both panels are scaled to one column.
        fig, ax = plt.subplots(figsize=(8.0, 5.3))
        # Legend stays inside the panel; the margins only keep the 22 pt labels
        # and the 16.5 pt end ticks clear of the canvas edge.
        fig.subplots_adjust(left=0.17, right=0.955, bottom=0.20, top=0.965)
        for mode, fill, line, ls, _ in styles:
            x, p = result[mode + "_x"], result[mode + "_density"]
            ax.fill_between(x, 0., p, color=fill, alpha=0.25,
                            linewidth=0, zorder=1)
            ax.plot(x, p, color=line, lw=1.65, ls=ls, zorder=2)
        for key, mean, sigma, fill, line, ls, label in references:
            density = result[key + "_density"]
            ax.fill_between(reference_x, 0, density, color=fill, alpha=0.34,
                            linewidth=0, zorder=3)
            ax.plot(reference_x, density, color=line, lw=1.55, ls=ls, zorder=5)
        ax.set_xlim(min(result[s[0] + "_x"][0] for s in styles),
                    2.5)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(r"$b_f$", labelpad=5)
        ax.set_ylabel(r"$P(b_f)$", labelpad=8)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: "0" if np.isclose(value, 0) else f"{value:.1f}"))
        ax.tick_params(which="both", direction="in", top=True, right=True, pad=6)
        # The end labels are centred on the spines, so at 16.5 pt the first one
        # runs into the y axis and the last one past the frame; anchor them.
        for label in ax.get_xticklabels():
            if np.isclose(float(label.get_text() or "nan"), ax.get_xlim()[0]):
                label.set_horizontalalignment("left")
            elif np.isclose(float(label.get_text() or "nan"), ax.get_xlim()[1]):
                label.set_horizontalalignment("right")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#2B2B2B")
        handles = [Patch(facecolor=fill, alpha=0.40, edgecolor="none")
                   for _, fill, _, _, _ in styles]
        handles.extend(Line2D([], [], color=line, ls=ls, lw=1.55)
                       for _, _, _, _, line, ls, _ in references)
        ax.legend(handles, [s[4] for s in styles] + [r[-1] for r in references],
                  loc="upper right", bbox_to_anchor=(0.99, 0.93),
                  frameon=False, fontsize=13.5, borderaxespad=0.3,
                  handlelength=1.9, handleheight=0.9, handletextpad=0.6,
                  labelspacing=0.7)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUT_DIR / "bias"
        for suffix in ("pdf", "png"):
            path = stem.with_suffix("." + suffix)
            fig.savefig(path, dpi=450)
            print("Saved:", path, flush=True)
        plt.close(fig)
    np.savez_compressed(str(stem) + "_data.npz", **result)
    Path(str(stem) + "_methods.md").write_text(
        "# FRB 与星系 bias 参考分布\n\n"
        "FRB 的 C_ell 和 Joint 曲线来自 astra Gaussian-delay 后验，"
        "沿用原图的 GetDist 估计和峰值归一化。\n\n"
        "星系参考值来自用户提供的表格：0.2 < z < 0.7，平均红移 0.51。"
        "Star-forming galaxy：1.23 ± 0.08；Quiescent galaxy：1.75 ± 0.04。"
        "将误差作为 1σ，以高斯函数近似测量约束；不是逐个星系 bias 的总体散布。\n\n"
        "所有曲线的峰值均设为 1，因此纵轴表示相对概率密度，"
        "曲线面积不用于比较模型概率。阴影显示完整曲线形状，不是可信区间。"
        "星系参考没有按 FRB 的红移核重新加权，仅供位置和宽度对照。\n\n"
        "横轴显示 0.5–2.5；完整 FRB 后验数组和星系参考参数保存在同名 NPZ 中。\n",
        encoding="utf-8")


BIAS_MODELS = (
    {"dir": "gauss", "key": "gauss", "label": "Gaussian delay", "color": "#D95F02"},
    {"dir": "lognormal", "key": "lognormal", "label": "Log-normal delay",
     "color": "#1B9E77"},
    {"dir": "power-law", "key": "powerlaw", "label": "Power-law delay",
     "color": "#7570B3"},
)


def three_model_bias_data():
    """Peak-normalised b_frb posteriors: cl and joint for each delay model."""
    result = {"parameter": "b_frb",
              "normalization": "Each marginalized GetDist posterior divided by its peak",
              "fill_meaning": "Full posterior profile, not a credible-interval band",
              "models": np.array([m["key"] for m in BIAS_MODELS])}
    for model in BIAS_MODELS:
        for mode, stem in (("cl", "frb_redshift"), ("joint", "frb_joint")):
            path = (RESULT_ROOT / mode / model["dir"] / "chains"
                    / (stem + "_converged.npy"))
            if not path.is_file():
                print("missing", mode, model["dir"], flush=True)
                continue
            samples = mcsamples(path, ["b_frb"], [r"b_{\rm FRB}"])
            density = samples.get1DDensity("b_frb")
            x = np.asarray(density.x, dtype=float)
            p = np.clip(np.asarray(density.P, dtype=float), 0.0, None)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)) or p.max() <= 0:
                raise ValueError(f"Invalid bias posterior: {path}")
            tag = "%s_%s" % (model["key"], mode)
            result[tag + "_x"] = x
            result[tag + "_density"] = p / p.max()
            result[tag + "_source"] = os.path.relpath(path, ROOT)
            result[tag + "_n_samples"] = samples.samples.shape[0]
            print("%s / %s: %s samples; b_FRB mode = %.4f"
                  % (model["label"], mode, f"{samples.samples.shape[0]:,}",
                     x[np.argmax(p)]), flush=True)
    return result


def plot_three_model_bias(result=None):
    """b_FRB posteriors: colour = delay model, dash/solid = C_ell / joint."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

    if result is None:
        result = three_model_bias_data()
    result = dict(result)
    # Model palette shared with picture/pz_three_model: pale wash for C_ell,
    # saturated wash plus hatching for joint, so the two probes stay separable.
    models = (
        ("gauss", "Gaussian", "#D7A94D", "#9D6A13", "/"),
        ("lognormal", "Log-normal", "#79B7D8", "#206894", "\\"),
        ("powerlaw", "Power-law", "#D59AB4", "#9D446D", "."),
    )
    # Reference lines use pale sage green and pale lavender: the old pale
    # blue/pink read as the same family as the log-normal blue and power-law
    # mauve. Both are kept low-chroma (C* ~ 27-30) so the model curves stay
    # the visual focus.
    references = (("star_forming", 1.23, 0.08, "#82B78F", "Star-forming"),
                  ("quiescent", 1.75, 0.04, "#AFA1D6", "Quiescent"))
    reference_x = np.linspace(0.5, 2.5, 4001)
    for key, mean, sigma, _color, _label in references:
        result[key + "_x"] = reference_x
        result[key + "_density"] = np.exp(-0.5 * ((reference_x - mean) / sigma) ** 2)
        result[key + "_mean"] = mean
        result[key + "_sigma"] = sigma
    x_lo = min(result[k + "_cl_x"][0] for k, *_ in models if k + "_cl_x" in result)
    cl_ls = (0, (5, 2.5))
    ref_ls = (0, (5, 1.8, 1.2, 1.8))
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 12,
        "axes.labelsize": 22, "axes.linewidth": 0.85,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.labelsize": 16.5, "ytick.labelsize": 16.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.width": 0.55, "ytick.minor.width": 0.55,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": False, "axes.axisbelow": True,
        "hatch.linewidth": 0.4, "legend.frameon": False,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8.0, 5.3))
        fig.subplots_adjust(left=0.15, right=0.975, bottom=0.175, top=0.965)
        # Galaxy constraints first, so no model wash sits on top of them.
        for _key, mean, sigma, color, _label in references:
            density = np.exp(-0.5 * ((reference_x - mean) / sigma) ** 2)
            ax.plot(reference_x, density, color=color, lw=1.6, ls=ref_ls, zorder=1)
        for key, _label, light, dark, hatch in models:
            tag = key + "_cl"
            if tag + "_x" in result:
                ax.fill_between(result[tag + "_x"], 0.0, result[tag + "_density"],
                                color=light, alpha=0.13, linewidth=0, zorder=2)
            # Joint keeps no wash: three stacked fills greyed the panel out,
            # so only the C_ell bands stay shaded.
        for key, _label, light, dark, hatch in models:
            # Both medians use the saturated colour: the pale tint was
            # invisible on top of the shared wash, so the dash pattern alone
            # separates C_ell from joint.
            if key + "_cl_x" in result:
                ax.plot(result[key + "_cl_x"], result[key + "_cl_density"],
                        color=dark, ls=cl_ls, lw=2.3, zorder=4)
            if key + "_joint_x" in result:
                ax.plot(result[key + "_joint_x"], result[key + "_joint_density"],
                        color=dark, ls="-", lw=1.9, zorder=5)
        ax.set_xlim(x_lo, 2.5)
        ax.set_ylim(0.0, 1.10)
        ax.set_xlabel(r"$b_f$", labelpad=5)
        ax.set_ylabel(r"$P(b_f)$", labelpad=8)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "" if abs(v) < 1e-12 else f"{v:.1f}"))
        ax.tick_params(which="both", direction="in", top=True, right=True, pad=6)
        for label in ax.get_xticklabels():
            text = label.get_text()
            if text and np.isclose(float(text), ax.get_xlim()[0]):
                label.set_horizontalalignment("left")
            elif text and np.isclose(float(text), ax.get_xlim()[1]):
                label.set_horizontalalignment("right")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#2B2B2B")
        handles = [Patch(facecolor=light, alpha=0.75, edgecolor=dark, linewidth=1.0)
                   for _k, _l, light, dark, hatch in models]
        labels = [label for _k, label, _l, _d, _h in models]
        handles += [Line2D([], [], color="#404040", ls=cl_ls, lw=2.0),
                    Line2D([], [], color="#404040", ls="-", lw=2.0)]
        labels += [r"$C_\ell$", "joint"]
        handles += [Line2D([], [], color=color, ls=ref_ls, lw=1.6)
                    for _k, _m, _s, color, _l in references]
        labels += [label for _k, _m, _s, _c, label in references]
        # One column inside the panel; a white backing keeps the tail of the
        # Joint posteriors from running through the text.
        ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.992, 0.985),
                  ncol=1, fontsize=14, handlelength=2.1, handleheight=0.95,
                  handletextpad=0.6, labelspacing=0.42,
                  borderaxespad=0.25,
                  frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.90)
        stem = OUT_DIR / "test_bias"
        for suffix in ("pdf", "png"):
            path = stem.with_suffix("." + suffix)
            fig.savefig(path, dpi=450)
            print("Saved:", path, flush=True)
        plt.close(fig)
    np.savez_compressed(str(stem) + "_data.npz", **result)
    return result


NZ_PRINT_SCALE = 406.4 / 183.0


NZ_BIN_LINEWIDTH = 0.8 * NZ_PRINT_SCALE


NZ_TOTAL_LINEWIDTH = 1.0 * NZ_PRINT_SCALE


NZ_BIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


NZ_BIN_LABELS = {
    "BGS": ["0.0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4"],
    "LRG": ["0.40–0.55", "0.55–0.72", "0.72–0.86", "0.86–1.02"],
}


NZ_STYLE = {
    "axes.linewidth": 0.5 * NZ_PRINT_SCALE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "legend.fontsize": 16, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.bfit": "Arial:bold:italic",
    "mathtext.sf": "Arial", "mathtext.tt": "Arial", "mathtext.cal": "Arial",
    "font.size": 22, "axes.labelsize": 22,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "text.color": "#282D33", "axes.labelcolor": "#282D33",
    "xtick.color": "#41464D", "ytick.color": "#41464D",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
}


BGS_NZ_PATH = ROOT / "data" / "galaxy" / "bgs_pz_dndz_gaussian_dz_0.01.txt"


LRG_NZ_PATH = ROOT / "data" / "galaxy" / "main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"


NZ_NATURE_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.bfit": "Arial:bold:italic",
    "mathtext.sf": "Arial", "mathtext.tt": "Arial", "mathtext.cal": "Arial",
    "font.size": 7.0, "axes.labelsize": 14.0,
    "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
}


NZ_NATURE_FIGSIZE = (183.0 / 25.4, 76.0 / 25.4)


NZ_NATURE_BIN_LW = 1.05


NZ_NATURE_TOTAL_LW = 0.85


NZ_NATURE_INK = "#222222"


NZ_NATURE_PANEL_SIZE = 10.0


NZ_NATURE_TOTAL_COLOR = "#4B4B4B"


NZ_NATURE_BIN_COLORS = ("#0072B2", "#D55E00", "#009E73", "#9B51A5")


def ensure_arial():
    """Use a font bundled with Matplotlib on all supported systems."""
    return "DejaVu Sans"


def nz_bin_colors(sample):
    """Distinct bin colors, in increasing redshift order for either sample."""
    if sample not in NZ_BIN_LABELS:
        raise ValueError(f"Unknown redshift sample: {sample}")
    return list(NZ_NATURE_BIN_COLORS)


def load_redshift_distribution(path, sample):
    """Read reviewed source bins, preserving their original distribution model."""
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 7 or not np.all(np.isfinite(raw)):
        raise ValueError(f"Invalid redshift table: {path}")
    widths = raw[:, 1] - raw[:, 0]
    if np.any(widths <= 0) or not np.allclose(raw[:-1, 1], raw[1:, 0]):
        raise ValueError(f"Redshift intervals must be ordered and contiguous: {path}")
    edges = np.r_[raw[:, 0], raw[-1, 1]]
    bins = raw[:, 3:7].T.copy()
    if np.any(bins < 0):
        raise ValueError(f"Negative redshift distribution: {path}")
    source_weight = float(bins.sum())
    if source_weight <= 0:
        raise ValueError(f"The four selected bins have zero total weight: {path}")
    info = {"path": str(path.resolve()), "selected_bin_columns": [3, 4, 5, 6],
            "selected_bins": [1, 2, 3, 4]}
    if sample == "BGS":
        count_line = next((line for line in path.read_text().splitlines()
                           if line.startswith("# n_gal ")), None)
        if count_line is None:
            raise ValueError(f"BGS table is missing its n_gal header: {path}")
        counts = [int(value) for value in count_line.split()[2:6]]
        if len(counts) != 4 or any(count <= 0 for count in counts):
            raise ValueError("BGS needs four positive source bin counts")
        # Source values are expected counts per interval, NOT dN/dz.
        # Use their retained weight to make the full tabulated integral exactly
        # one (the finite source range omits a small Gaussian tail).
        info.update({"bin_counts": counts, "normalization_galaxies": sum(counts),
                     "source_expected_counts": source_weight,
                     "bin_selection_z_median": "[0,0.1), [0.1,0.2), [0.2,0.3), [0.3,0.4)",
                     "total_definition": "Sum of the same four plotted Gaussian photo-z bin distributions",
                     "excluded_bins": [5]})
    else:
        info.update({"total_definition": "Sum of the same four plotted combined North+South LRG bin distributions",
                     "source_units": "Galaxies per square degree in each redshift interval (Delta z = 0.02 for the default source)",
                     "unit_reference": "https://arxiv.org/html/2309.06443v2#S3.SS1",
                     "source_surface_density_deg_minus2": source_weight})
    # Both source tables are interval weights. Use one common denominator for
    # all four bins: each bin retains its sample fraction, and their sum is a
    # probability density with unit integral over the full source range.
    bins /= source_weight * widths[None, :]
    total = bins.sum(axis=0)
    info.update({"normalization": "Interval weight / (sum of interval weights for bins1–4 * interval width)",
                 "units": "Probability density per unit redshift",
                 "normalization_range": [float(edges[0]), float(edges[-1])],
                 "total_integral_full_source": float(np.sum(total * widths)),
                 "bin_integrals_full_source": np.sum(bins * widths[None, :], axis=1).tolist()})
    if np.any(total < 0):
        raise ValueError("Negative total redshift distribution")
    return edges, bins, total, info


def draw_nz_smooth_test(bgs_path, lrg_path, stem, dpi=450, smoothing_bins=1.0):
    """Smooth filled-bin display with the original bin probability fractions."""
    import hashlib
    from scipy.integrate import trapezoid
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    if not np.isfinite(smoothing_bins) or smoothing_bins <= 0:
        raise ValueError("The display smoothing width must be positive")
    stem = figure_output_path(stem)
    ensure_arial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    data, panels = {}, {}
    fill_alpha = 0.60
    with plt.rc_context(NZ_NATURE_RC):
        fig = plt.figure(figsize=NZ_NATURE_FIGSIZE)
        for sample, path, rect in [
                ("BGS", bgs_path, [0.078, 0.175, 0.420, 0.797]),
                ("LRG", lrg_path, [0.550, 0.175, 0.420, 0.797])]:
            edges, bins, total, info = load_redshift_distribution(path, sample)
            widths = np.diff(edges)
            centers = (edges[:-1] + edges[1:]) / 2.
            # Resolve the original table before applying a gentle, positive
            # Gaussian display kernel. Keep each source bin's area intact.
            knots = np.r_[edges[0], centers, edges[-1]]
            z = np.r_[(edges[:-1, None] + widths[:, None] * np.arange(32) / 32.).ravel(), edges[-1]]
            grid_step = float(np.median(np.diff(z)))
            np.testing.assert_allclose(np.diff(z), grid_step, rtol=1e-10, atol=1e-12)
            smoothing_sigma_z = float(smoothing_bins * np.median(widths))
            targets = np.sum(bins * widths[None, :], axis=1)
            curves, scales, before_smoothing, smoothing_scales = [], [], [], []
            for values, target in zip(bins, targets):
                interpolant = PchipInterpolator(knots, np.r_[values[0], values, values[-1]],
                                               extrapolate=False)
                integral = float(interpolant.integrate(edges[0], edges[-1]))
                if integral <= 0 or not np.isfinite(integral):
                    raise ValueError(f"Invalid smooth {sample} bin integral")
                scale = float(target / integral)
                density = interpolant(z) * scale
                if not np.all(np.isfinite(density)) or np.min(density) < -1e-12:
                    raise ValueError(f"Invalid smooth {sample} density")
                density = np.maximum(density, 0.)
                np.testing.assert_allclose(trapezoid(density, z), target, rtol=2e-6, atol=1e-10)
                before_smoothing.append(density.copy())
                density = gaussian_filter1d(density, smoothing_sigma_z / grid_step,
                                            mode="reflect", truncate=4.)
                smoothing_scale = float(target / trapezoid(density, z))
                density *= smoothing_scale
                np.testing.assert_allclose(trapezoid(density, z), target, rtol=1e-12, atol=1e-12)
                curves.append(density)
                scales.append(scale)
                smoothing_scales.append(smoothing_scale)
            curves = np.asarray(curves)
            before_smoothing = np.asarray(before_smoothing)
            colors = nz_bin_colors(sample)
            ax = fig.add_axes(rect)
            for density, color in zip(curves, colors):
                ax.fill_between(z, 0., density, facecolor=color, alpha=fill_alpha,
                                edgecolor="none", linewidth=0)
            xlim = (0., 0.55) if sample == "BGS" else (0.25, 1.25)
            # Match the relative peak heights of the two displayed samples.
            ymax, ystep = (4., 1.) if sample == "BGS" else (2.15, 0.5)
            xstep = 0.1 if sample == "BGS" else 0.25
            ax.set(xlim=xlim, ylim=(0., ymax), xlabel=r"$z$")
            ax.xaxis.labelpad = 3.
            ax.xaxis.set_major_locator(MultipleLocator(xstep, offset=xlim[0]))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(ystep))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f" if sample == "BGS" else "%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(which="major", pad=3.)
            if sample == "BGS":
                ax.set_ylabel(r"$n_g(z)$", labelpad=4.)
            ax.text(0.035, 0.955, sample, transform=ax.transAxes, ha="left", va="top",
                    fontsize=NZ_NATURE_PANEL_SIZE, fontweight="bold", color=NZ_NATURE_INK)
            ax.legend([Patch(facecolor=color, alpha=fill_alpha, edgecolor="none") for color in colors],
                      NZ_BIN_LABELS[sample], loc="upper right", bbox_to_anchor=(0.975, 0.975),
                      frameon=False, prop={"family": "Arial", "size": NZ_NATURE_RC["legend.fontsize"]},
                      handlelength=1.6, handletextpad=0.5, labelspacing=0.32, borderaxespad=0.)
            data.update({sample + "_z": z, sample + "_density": curves,
                         sample + "_source_edges": edges, sample + "_source_density": bins,
                         sample + "_source_total": total, sample + "_bin_fractions": targets,
                         sample + "_interpolation_scale": np.asarray(scales),
                         sample + "_density_before_smoothing": before_smoothing,
                         sample + "_smoothing_scale": np.asarray(smoothing_scales),
                         sample + "_smoothing_sigma_z": smoothing_sigma_z,
                         sample + "_smoothing_sigma_native_bins": smoothing_bins,
                         sample + "_bin_labels": np.asarray(NZ_BIN_LABELS[sample])})
            panels[sample] = {**info, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "xlim": list(xlim), "ylim": [0., ymax], "bin_colors": colors,
                              "interpolation_area_scale": scales,
                              "smoothing_sigma_z": smoothing_sigma_z,
                              "smoothing_sigma_native_bins": smoothing_bins,
                              "smoothing_area_scale": smoothing_scales,
                              "peak_density_change_fraction": (curves.max(axis=1) / before_smoothing.max(axis=1) - 1.).tolist(),
                              "rendered_bin_integrals": trapezoid(curves, z, axis=1).tolist()}
        for suffix in ("pdf", "png"):
            output = stem.with_suffix(".pdf") if suffix == "pdf" else PNG_DIR / (stem.name + "." + suffix)
            fig.savefig(output, dpi=dpi)
            print("Saved:", output, flush=True)
        plt.close(fig)
    metadata = {"source_figure": "nz_galaxy.pdf", "panel_order": ["BGS", "LRG"], "panels": panels,
                "total_plotted": False, "density_outlines": False, "fill_alpha": fill_alpha,
                "interpolation": "C1 shape-preserving PCHIP through interval centers; boundary values held at adjacent centers",
                "smoothing": "Gaussian convolution on the fine display grid; reflected boundaries; sigma equals one native interval by default",
                "normalization": "Each continuous bin is rescaled after smoothing to retain its original probability fraction over the full source interval",
                "font": "DejaVu Sans", "math_font": "Arial", "width_mm": 183., "height_mm": 76.,
                "font_sizes_pt": {"axis_label": NZ_NATURE_RC["axes.labelsize"],
                                  "ticks": NZ_NATURE_RC["xtick.labelsize"],
                                  "legend": NZ_NATURE_RC["legend.fontsize"]}}
    np.savez_compressed(PNG_DIR / (stem.name + "_data.npz"), **data)
    (PNG_DIR / (stem.name + ".json")).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def load_cl_correlation(path):
    """Normalize selected catalog covariance, using its explicit band ordering.

    Omit BGS [0.4, 0.5] and put DD first, followed by the retained BGS/LRG
    cross spectra. Select a principal submatrix, preserving all covariance
    terms between the retained bandpowers.
    The returned matrix is signed; taking its absolute value is exclusively
    a display choice, recorded by the renderer in its figure metadata.
    """
    import hashlib

    path = Path(path)
    required = {"cross_cov_joint", "cross_order_sample", "cross_band_indices",
                "cross_band_ell", "cross_order_label", "cross_order_z_min",
                "cross_order_z_max"}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError("Missing joint-covariance metadata: " + ", ".join(missing))
        covariance = np.asarray(data["cross_cov_joint"], dtype=float)
        samples = np.asarray(data["cross_order_sample"]).astype(str)
        bands = np.asarray(data["cross_band_indices"], dtype=int)
        ell = np.asarray(data["cross_band_ell"], dtype=float)
        labels = np.asarray(data["cross_order_label"]).astype(str)
        z_min = np.asarray(data["cross_order_z_min"], dtype=float)
        z_max = np.asarray(data["cross_order_z_max"], dtype=float)

    if samples.ndim != 1 or not (labels.shape == z_min.shape == z_max.shape == samples.shape):
        raise ValueError("Inconsistent spectrum-order metadata")
    if bands.ndim != 2 or bands.shape[0] != samples.size or ell.shape != bands.shape:
        raise ValueError("Band indices and effective multipoles must match the spectra")
    n_points = bands.size
    if covariance.shape != (n_points, n_points) or n_points == 0:
        raise ValueError("Joint covariance shape does not match the band ordering")
    if not np.array_equal(np.sort(bands.ravel()), np.arange(n_points)):
        raise ValueError("Band indices must cover the full covariance exactly once")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(ell)):
        raise ValueError("Non-finite covariance or effective multipoles")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
        raise ValueError("Joint covariance is not symmetric")
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("Covariance variances must be positive")
    if not set(samples.tolist()) <= {"DM", "BGS", "LRG"}:
        raise ValueError("Unknown spectrum type in the joint covariance")
    excluded = ((samples == "BGS") & np.isclose(z_min, 0.4, rtol=0, atol=1e-8)
                & np.isclose(z_max, 0.5, rtol=0, atol=1e-8))
    dd = np.flatnonzero(samples == "DM")
    dg = np.flatnonzero((samples != "DM") & ~excluded)
    if dd.size == 0 or dg.size == 0:
        raise ValueError("Both DD and Dg spectra are required")
    spectrum_order = np.r_[dd, dg]
    point_order = bands[spectrum_order].ravel()
    n_points = point_order.size
    ordered_covariance = covariance[np.ix_(point_order, point_order)]
    std = np.sqrt(np.diag(ordered_covariance))
    correlation = ordered_covariance / std[:, None] / std[None, :]
    if np.max(np.abs(correlation)) > 1.0 + 1e-10:
        raise ValueError("Normalized covariance contains correlations outside [-1, 1]")
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    n_dd = int(bands[dd].size)
    info = {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_covariance_key": "cross_cov_joint",
        "selection": "Exclude BGS [0.4, 0.5]; retain all other spectra and bandpowers",
        "source_point_count": int(bands.size),
        "excluded_spectrum_source_indices": np.flatnonzero(excluded).tolist(),
        "excluded_spectrum_labels": labels[excluded].tolist(),
        "normalization": "r_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)",
        "shape": list(correlation.shape),
        "bands_per_spectrum": int(bands.shape[1]),
        "group_boundaries": [0, n_dd, n_points],
        "group_labels": ["C_ell^{DD}", "C_ell^{Dg}"],
        "spectrum_source_indices": spectrum_order.tolist(),
        "point_source_indices": point_order.tolist(),
        "spectrum_labels": labels[spectrum_order].tolist(),
        "effective_ell": ell[spectrum_order].tolist(),
        "sample_spectrum_counts": {name: int(np.count_nonzero(samples[spectrum_order] == name))
                                   for name in ("DM", "BGS", "LRG")},
        "signed_correlation_min": float(correlation.min()),
        "signed_correlation_max": float(correlation.max()),
        "negative_entries": int(np.count_nonzero(correlation < 0)),
        "minimum_correlation_eigenvalue": float(np.linalg.eigvalsh(correlation).min()),
    }
    return correlation, info


def plot_cl_correlation(path=None, stem=None, dpi=600):
    """Selected absolute-correlation heatmap, with DD and Dg group labels."""
    from matplotlib.colors import LinearSegmentedColormap

    path = ROOT / "data" / "angular power spectrum" / "cl_dm_catalog.npz" if path is None else Path(path)
    stem = OUT_DIR / "cov_matrix" if stem is None else figure_output_path(stem)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    correlation, info = load_cl_correlation(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    ensure_arial()
    # Use linear normalization with an evenly spaced 0--1 colourbar.
    # Stronger light and mid blues reveal more off-diagonal structure while
    # keeping the light-blue colour of the companion intrinsic-p(z) figure.
    palette = [(0.0, "#FFFFFF"), (0.08, "#C6DBEF"), (0.25, "#6BAED6"),
               (0.50, "#3182BD"), (0.75, "#08519C"), (1.0, "#08306B")]
    cmap = LinearSegmentedColormap.from_list("frb_correlation_blue", palette, N=256)
    boundaries = info["group_boundaries"]
    n_dd, n_points = boundaries[1:]
    centers = [n_dd / 2.0, (n_dd + n_points) / 2.0]
    group_labels = [r"$C_{\ell}^{\mathrm{DD}}$", r"$C_{\ell}^{\mathrm{Dg}}$"]
    # Match the companion light-blue p(z) PDF at the same 7.2-inch width:
    # 17 pt mathematical labels and 13 pt Arial tick numbers.
    label_fontsize, tick_fontsize = 17.0, 13.0
    width_mm, height_mm = 182.88, 165.735

    with plt.rc_context({**NZ_NATURE_RC, "font.size": 14.0,
                         "mathtext.fontset": "dejavusans",
                         "axes.labelsize": label_fontsize,
                         "xtick.labelsize": tick_fontsize,
                         "ytick.labelsize": tick_fontsize, "xtick.top": False,
                         "ytick.right": False}):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        # Equal physical width and height preserve square covariance cells.
        ax = fig.add_axes([0.155, 0.11, 0.69, 0.69 * width_mm / height_mm])
        im = ax.imshow(np.abs(correlation), origin="upper", interpolation="nearest",
                       extent=(0, n_points, n_points, 0), aspect="equal",
                       cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xlim(0, n_points)
        ax.set_ylim(n_points, 0)
        ax.set_xticks([0, n_points])
        ax.set_xticklabels(["0", str(n_points)])
        ax.set_yticks(centers)
        ax.set_yticklabels(group_labels, fontsize=label_fontsize)
        ax.tick_params(axis="x", direction="out", pad=4.0, length=3.0)
        ax.tick_params(axis="y", length=0, pad=8.0)
        ax.minorticks_off()
        ax.grid(False)
        # Only the requested two scientific groups receive a dividing rule.
        ax.axvline(n_dd, color="#9EAFBB", lw=0.55)
        ax.axhline(n_dd, color="#9EAFBB", lw=0.55)
        for center, label in zip(centers, group_labels):
            ax.text(center / n_points, 1.025, label, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=label_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#404850")

        cax = fig.add_axes([0.88, 0.11, 0.026, 0.69 * width_mm / height_mm])
        colorbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0.0, 1.0, 6))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        colorbar.ax.tick_params(direction="out", length=2.8, pad=3.0)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.set_title(r"$|r_{ij}|$", fontsize=label_fontsize, pad=9.0)
        for suffix in ("pdf", "png"):
            output = stem.with_suffix("." + suffix)
            fig.savefig(output, dpi=dpi)
            print("Saved:", output, flush=True)
        plt.close(fig)

    info.update({"display": "Absolute correlation |r_ij|; signs are not encoded",
                 "color_limits": [0.0, 1.0], "color_scale": "linear",
                 "colorbar_values": "Actual absolute correlations",
                 "palette": [[value, color] for value, color in palette],
                 "width_mm": width_mm, "height_mm": height_mm,
                 "font": "DejaVu Sans", "math_font": "DejaVu Sans",
                 "font_sizes_pt": {"axis_group_labels": label_fontsize,
                                   "axis_ticks": tick_fontsize,
                                   "colorbar_label": label_fontsize,
                                   "colorbar_ticks": tick_fontsize},
                 "typography_reference": "pz_goldblue.pdf",
                 "interpolation": "nearest"})
    stem.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"Selected matrix: {n_points} points; DD={n_dd}, Dg={n_points - n_dd}; "
          f"signed r_min={info['signed_correlation_min']:.6f}", flush=True)
    return info


NULL_NPZ = ROOT / "null" / "null_test.npz"


NULL_STYLE = {
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.labelsize": 21,
    "axes.titlesize": 18,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 17,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def plot_dm_null_style(npz_path=NULL_NPZ, stem=None):
    """Same DM-permutation null test, drawn in the measured-spectra style.

    Reuses CL_DATA_STYLE and the shared 3 x 3 geometry so the null test can be
    placed beside the measured figure: Arial, panel labels inside the upper
    right corner instead of above the frame, every panel carrying its own
    log-x tick labels, and one centred legend on top. There is still no theory
    curve and no Gaussian covariance: points are the sample mean, bars the
    sample standard deviation of the 1000 shuffles.

    Unlike the measured figure, the y range cannot be auto-extended for the
    corner label: the null bandpowers are consistent with zero and any extra
    headroom would exaggerate the size of the error bars. The label is instead
    placed by the same clear_upper_right test, and when that test cannot find
    room within a bounded expansion the error-bar caps are what the label is
    measured against.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        ell = np.asarray(data["ell"], dtype=float)
        bgs_mean = np.asarray(data["cl_bgs_mean"], dtype=float)
        bgs_std = np.asarray(data["cl_bgs_std"], dtype=float)
        lrg_mean = np.asarray(data["cl_lrg_mean"], dtype=float)
        lrg_std = np.asarray(data["cl_lrg_std"], dtype=float)
        dm_mean = np.asarray(data["cl_dm_mean"], dtype=float)
        dm_std = np.asarray(data["cl_dm_std"], dtype=float)
        bgs_ranges = np.asarray(data["bgs_z_ranges"], dtype=float)
        lrg_ranges = np.asarray(data["lrg_z_ranges"], dtype=float)
        bgs_plot = np.asarray(data["bgs_plot"], dtype=int)
        scale_power = int(data["cross_scale_power"])
        ell_max = float(data["lmax"])
    if not np.all(np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError(f"Invalid ell bandpowers in {path}")

    scale = 10.0 ** scale_power
    panels = []
    for index, bin_index in enumerate(bgs_plot):
        z0, z1 = bgs_ranges[bin_index]
        panels.append({
            "mean": bgs_mean[bin_index], "std": bgs_std[bin_index],
            "name": "BGS", "kind": "galaxy_dm", "z": (z0, z1),
            "closing": "]" if index == len(bgs_plot) - 1 else ")",
        })
    for index, (z0, z1) in enumerate(lrg_ranges):
        panels.append({
            "mean": lrg_mean[index], "std": lrg_std[index],
            "name": "LRG", "kind": "galaxy_dm", "z": (z0, z1),
            "closing": "]" if index == len(lrg_ranges) - 1 else ")",
        })
    panels.append({
        "mean": dm_mean, "std": dm_std,
        "name": "DM", "kind": "dm_auto", "z": (0.0, 0.0), "closing": "]",
    })
    if len(panels) != 9:
        raise RuntimeError("expected 9 panels, got %d" % len(panels))

    if stem is None:
        stem = OUT_DIR / "test_null"
    stem = figure_output_path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(CL_DATA_STYLE)
    fig, axes = cl_grid_figure(CL_NULL_GRID)
    legend_handles = {}
    for index, (ax, panel) in enumerate(zip(axes.ravel(), panels)):
        col = index % 3
        kind = panel["kind"]
        multiply_ell = kind != "dm_auto"
        ell_factor = ell if multiply_ell else np.ones_like(ell)
        panel_scale = scale if multiply_ell else 1.0
        values = panel_scale * ell_factor * panel["mean"]
        errors = panel_scale * ell_factor * panel["std"]
        observed = ax.errorbar(
            ell, values, yerr=errors, fmt="o", ms=8, capsize=4.5,
            elinewidth=1.5, capthick=1.5,
            markeredgecolor="black", markeredgewidth=0.9,
            color=CL_PANEL_COLORS[panel["name"]],
            label=CL_SPECTRUM_LABELS[panel["name"]], zorder=3)
        legend_handles.setdefault(CL_SPECTRUM_LABELS[panel["name"]], observed)
        ax.axhline(0.0, color="#D0D0D0", lw=0.8, zorder=0)
        style_x_axis(ax, ell_max)
        z0, z1 = panel["z"]
        # No theory curve in a null test, so the line array is empty and only
        # the error-bar caps constrain the y range.
        y_lo, y_hi = clear_upper_right(
            ell, np.array([], dtype=float), ell, values - errors, values + errors)
        ax.set_ylim(y_lo, y_hi)
        cl_y_label(ax, kind, scale_power, col)
        cl_panel_label(ax, cl_bin_label(panel["name"], z0, z1, panel["closing"]))
        ax.grid(False, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\ell$")
    cl_figure_legend(fig, legend_handles, anchor_x=0.5128, ncol=3)

    for suffix in ("pdf", "png"):
        out = stem.with_suffix("." + suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("Saved:", out.resolve(), flush=True)
    plt.close(fig)


FINAL_FIGURES = (
    "bias", "contour_map_gauss", "contour_map_lognormal", "contour_map_powerlaw",
    "cov_matrix", "data", "dmz", "nz_galaxy_test", "pz", "P_det_test2",
    "test_bias", "test_dmz", "test_null", "test_pz_mean", "test_constant_mean_clean",
)
RECOMPUTE = False


def three_model_dmz_prediction(n_draw=1024, seed=20261013, zmax=1.5,
                             reference_draws=5120):
    """Build all three conditional predictive maps from their joint chains."""
    import hashlib
    reference_path = picture_asset("frb_dmz_gauss_sfh_joint_data.npz")
    with np.load(reference_path, allow_pickle=False) as saved:
        result = dict(saved)
    gauss_path = RESULT_ROOT / "joint/gauss/chains/frb_joint_converged.npy"
    gauss_hash = hashlib.sha256(gauss_path.read_bytes()).hexdigest()
    fresh_reference = reference_path.parent in (OUT_DIR, PNG_DIR)
    if (str(result["gauss_chain_sha256"]) != gauss_hash
            or (RECOMPUTE and not fresh_reference)):
        result = compare_dmz_prediction(n_draw=reference_draws, seed=seed, zmax=zmax)
        np.savez_compressed(PNG_DIR / "dmz_data.npz", **result)
    if not np.isclose(float(result["z"][-1]), zmax):
        raise ValueError("The Gaussian reference must cover the requested redshift range")
    overlay = compare_dmz_prediction(n_draw=n_draw, seed=seed, zmax=zmax,
        dm_max=6000., dm_step=1.,
        models=(("lognormal", "lognormal"), ("powerlaw", "power-law")))
    np.testing.assert_array_equal(overlay["z"], result["z"])
    np.testing.assert_allclose(overlay["cosmic_mean_dm"], result["cosmic_mean_dm"])
    result.update({key: value for key, value in overlay.items()
                   if key.startswith(("lognormal_", "powerlaw_"))})
    result.update(overlay_dm=overlay["dm"], overlay_requested_draws=n_draw,
                  overlay_seed=seed, overlay_dm_max=overlay["dm_max"],
                  overlay_dm_step=overlay["dm_step"])
    np.savez_compressed(PNG_DIR / "dmz_three_model_data.npz", **result)
    return result


def constant_mean_from_chains():
    """Rebuild the retained mean/std robustness panels from full posterior rows."""
    import robust
    result = robust.robust_pz_data(cases=robust.PZ_CASES_CONSTANT,
                                  n_draw=10000, seed=1, extended_summary=True)
    result.pop("quantiles", None)
    result["case_names"] = np.array([case["name"] for case in robust.PZ_CASES_CONSTANT])
    for case in robust.PZ_CASES_CONSTANT:
        for quantity in ("pdf", "cdf"):
            prefix = case["name"] + "_" + quantity
            result[prefix + "_original_quantiles"] = result[prefix]
            mean, std = result[prefix + "_mean"], result[prefix + "_std"]
            result[prefix] = np.stack([mean - std, mean, mean + std])
        x, density = robust.bias_posterior(case["path"])
        result["bias_" + case["name"] + "_x"] = x
        result["bias_" + case["name"] + "_p"] = density
    np.savez_compressed(PNG_DIR / "test_constant_mean_data.npz", **result)

def render_figure(name, recompute=False):
    """Render one adopted figure, reusing numerical summaries unless requested."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    if name == "bias":
        plot_gaussian_bias()
    elif name.startswith("contour_map_"):
        key = {"contour_map_gauss": "gauss", "contour_map_lognormal": "lognormal",
               "contour_map_powerlaw": "powerlaw"}[name]
        spec=next(s for s in POSTERIOR_SPECS if s["delay_model"] == key)
        plot_posterior_triangle(_posterior_series(key),
            list(spec["params"])+[r[0] for r in NUISANCE_PARAMS],
            list(spec["labels"])+[r[1] for r in NUISANCE_PARAMS], OUT_DIR / (name+".pdf"))
    elif name == "cov_matrix":
        plot_cl_correlation(stem=OUT_DIR / name)
    elif name == "data":
        cache = picture_asset("cl_data_curves.npz")
        curves = None if recompute else load_cl_data_curves(cache)
        plot_cl_data(OUT_DIR / "data.pdf", curves=curves)
    elif name == "dmz":
        result = compare_dmz_prediction(n_draw=5120, seed=20261013, zmax=1.5) if recompute else dict(np.load(picture_asset("frb_dmz_gauss_sfh_joint_data.npz"), allow_pickle=False))
        plot_dmz_comparison(result=result, zmax=1.5, color_theme="blue")
    elif name == "nz_galaxy_test":
        draw_nz_smooth_test(BGS_NZ_PATH, LRG_NZ_PATH, OUT_DIR / name)
    elif name == "pz":
        result = intrinsic_posterior_data(n_draw=10000,delay_model="gauss",extended_summary=True) if recompute else load_intrinsic_summary("gauss")
        plot_intrinsic_pz(result=result, output_stem="pz", interval_summary="mean_std")
    elif name == "P_det_test2":
        plot_selection_panels_test()
    elif name == "test_bias":
        plot_three_model_bias()
    elif name == "test_dmz":
        if recompute:
            three_model_dmz_prediction(n_draw=1024,seed=20261013,zmax=1.5)
        plot_dmz_three_divide()
    elif name == "test_null":
        plot_dm_null_style()
    elif name == "test_pz_mean":
        plot_three_model_mean()
    elif name == "test_constant_mean_clean":
        if recompute:
            constant_mean_from_chains()
        plot_constant_mean_clean()
    else:
        raise ValueError(name)
    # Canonical filenames for all adopted PNG counterparts.
    for alias, canonical in FIGURE_OUTPUT_NAMES.items():
        if canonical == name:
            legacy=OUT_DIR / (alias + ".png")
            if legacy.exists():
                legacy.replace(PNG_DIR / (name + ".png"))
    for ext in ("png",):
        p=OUT_DIR / (name+"."+ext)
        if p.exists():p.replace(PNG_DIR / p.name)
    print("Rendered:", name, flush=True)

def cli():
    global OUT_DIR, PNG_DIR, RECOMPUTE
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", choices=("all", *FINAL_FIGURES), default="all")
    p.add_argument("--recompute", action="store_true", help="Recompute expensive posterior summaries where applicable")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    a=p.parse_args()
    RECOMPUTE = a.recompute
    OUT_DIR=a.output_dir.resolve();PNG_DIR=OUT_DIR / "png"
    for name in FINAL_FIGURES if a.figure == "all" else (a.figure,):
        render_figure(name, recompute=a.recompute)

if __name__ == "__main__":
    cli()
