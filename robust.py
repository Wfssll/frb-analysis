"""Scientific posterior summaries for the eight adopted robustness cases."""

from cosmology import production_cosmology
from pathlib import Path
from datetime import datetime
import hashlib, json, os, re
SUITE_ROOT = Path(__file__).resolve().parent
def suite_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
import numpy as np


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


import pyccl as ccl


from astropy.constants import G, c, m_p, pc


from scipy.integrate import cumulative_trapezoid, trapezoid


import scipy.integrate as _scipy_integrate


if not hasattr(_scipy_integrate, "simps"):
    _scipy_integrate.simps = _scipy_integrate.simpson


from frb_redshift_model import (
    N_FORMATION_TAIL, N_TAU_GAUSS, P_gaussian_delay, R_SFH,
    TAU_MIN_GYR, Z_FORMATION_MAX,
)


ROOT = Path(__file__).resolve().parent


OUT_DIR = Path(os.getenv("ROBUST_ROOT", str(ROOT / "robust")))


RESULTS_TXT = Path(os.getenv("RESULT_ROOT", str(ROOT / "result"))) / "cl" / "gauss" / "results" / "results.txt"


F_IGM = 0.93


X_E = 0.875


E_MIN = 1e38


E_MAX = 1e42


DELTA_NU = 400e6


Z_MIN = 0.0


Z_MAX = 5.0


Z_CAL = 0.1


NZ = 500


NE = 1000


NSKY = 1000


SEED = 42


MPC_TO_M = 3.08567758e22


JYMS_FROM_ERG_MPC2_HZ = 1e22 / (MPC_TO_M ** 2)


ALPHA_FID = 1.8


DM_HOST_FID = 100.0


E_CUT_FID = 3e41


ALPHAS = (1.7, 1.8, 1.9)


DM_HOSTS = (50.0, 100.0, 150.0)


E_CUTS = (1e41, 3e41, 5e41)


COSMO = production_cosmology(
    Omega_b=0.04897,
    Omega_c=0.3111 - 0.04897,
    h=0.6766,
    sigma8=0.81,
    n_s=0.96,
)


def log(msg: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def log(msg: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def _pz_case(label, name, path, facecolor, linecolor, alpha, hatch=None,
             edge=False, fill_order=0):
    """One p(z) case. fill_order sets the paint order only; the legend keeps
    the tuple order."""
    return {
        "label": label, "name": name, "path": Path(path),
        "facecolor": facecolor, "linecolor": linecolor,
        "alpha": alpha, "hatch": hatch, "edge": edge,
        "fill_order": fill_order,
    }


def _case_named(cases, name):
    return next(case for case in cases if case["name"] == name)


def _dm_host_constant():
    """Brackets, and a purple wash instead of a second hatch.

    ell_loc free already carries the slash texture. A second hatch read as
    clutter, so this case is told apart by colour.
    """
    case = dict(_case_named(PZ_CASES_SELECTION, "dm_host_150"))
    case["label"] = r"$\mathrm{DM}_{\mathrm{host}}=150\,\mathrm{pc}\,\mathrm{cm}^{-3}$"
    case["facecolor"] = "#D4C4E8"
    case["linecolor"] = "#6B4C9A"
    case["alpha"] = 0.28
    case["hatch"] = None
    case["edge"] = True
    return case


PZ_SHARED = _pz_case(
    "Baseline", "baseline",
    Path(os.getenv("RESULT_ROOT", str(ROOT / "result"))) / "cl" / "gauss" / "chains" / "frb_redshift_converged.npy",
    "#F6C36B", "#D9992B", 0.30, fill_order=9)


PZ_CASES = (
    PZ_SHARED,
    _pz_case(r"$f_{\mathrm{d}}=0.84$", "fd_084",
             OUT_DIR / "fd_084" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy", "#9A9A9A", "#4D4D4D", 0.34),
    _pz_case(r"$\ell_{\mathrm{loc}}$ free", "ell_loc_free",
             OUT_DIR / "ell_loc_free" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy", "none", "#000000", 1.0, "//"),
    _pz_case(r"$\ell_{\mathrm{loc}}=500$", "ell_loc_500",
             OUT_DIR / "ell_loc_500" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy", "#F2A0A0", "#C9524F", 0.34),
)


PZ_CASES_SELECTION = (
    PZ_SHARED,
    _pz_case(r"$\gamma=1.7$", "alpha_17",
             OUT_DIR / "alpha_17" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy",
             "#A8C8E8", "#0072B2", 0.20, edge=True, fill_order=1),
    _pz_case(r"$\gamma=1.9$", "alpha_19",
             OUT_DIR / "alpha_19" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy",
             "#CDE7F5", "#2F7FA8", 0.20, edge=True, fill_order=2),
    _pz_case(r"$E_{\mathrm{cut}}=1\times10^{41}\,\mathrm{erg}$", "ecut_1e41",
             OUT_DIR / "ecut_1e41" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy",
             "#A6D9C7", "#007A5A", 0.20, edge=True, fill_order=3),
    _pz_case(r"$E_{\mathrm{cut}}=5\times10^{41}\,\mathrm{erg}$", "ecut_5e41",
             OUT_DIR / "ecut_5e41" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy",
             "#EFC6DA", "#A2557F", 0.20, edge=True, fill_order=4),
    _pz_case(r"$\mathrm{DM}_{\mathrm{host}}=150$", "dm_host_150",
             OUT_DIR / "dm_host_150" / "cl" / "gauss" / "chains"
             / "frb_redshift_converged.npy",
             "none", "#5A5A5A", 1.0, hatch="//", edge=True, fill_order=5),
)


PZ_CASES_CONSTANT = (
    PZ_SHARED,
    _case_named(PZ_CASES, "fd_084"),
    _case_named(PZ_CASES, "ell_loc_free"),
    _case_named(PZ_CASES_SELECTION, "alpha_17"),
    _case_named(PZ_CASES_SELECTION, "alpha_19"),
    _case_named(PZ_CASES_SELECTION, "ecut_1e41"),
    _case_named(PZ_CASES_SELECTION, "ecut_5e41"),
    _dm_host_constant(),
)


PZ_SFH_COLOR = "#668269"


PZ_CDF_STYLE = (0, (4.5, 2.2))


def pz_chain_band(path, z, engine, kernel, n_draw=10000, seed=1,
                  extended_summary=False):
    """Pointwise quantiles, with optional means/std and selected draw metadata."""
    blob = np.load(path, allow_pickle=True)
    samples = np.asarray(blob[0], dtype=float)
    names = [str(x) for x in blob[2]]
    columns = [names.index(p) for p in ("tau_gauss", "sigma_gauss")]
    count = min(int(n_draw), samples.shape[0])
    if count < 1:
        raise ValueError("empty chain: %s" % path)
    indices = np.random.default_rng(seed).choice(
        samples.shape[0], count, replace=False)
    params = samples[indices][:, columns]
    if not np.all(np.isfinite(params)) or np.any(params[:, 1] <= 0):
        raise ValueError("invalid Gaussian-delay samples in %s" % path)
    pdf = np.empty((count, z.size), dtype=float)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        delay = np.stack([P_gaussian_delay(engine.tau, *p)
                          for p in params[start:stop]])
        block = np.clip((delay @ kernel) * engine.vol, 0.0, None)
        norm = trapezoid(block, z, axis=1)
        if np.any(norm <= 0.0) or not np.all(np.isfinite(block)):
            raise ValueError("invalid intrinsic distribution in %s" % path)
        pdf[start:stop] = block / norm[:, None]
    cdf = cumulative_trapezoid(pdf, z, axis=1, initial=0)
    if np.any(np.diff(cdf, axis=1) < -1e-14):
        raise ValueError("cumulative distribution must be monotonic")
    result = {
        "pdf": np.quantile(pdf, (0.16, 0.50, 0.84), axis=0),
        "cdf": np.quantile(cdf, (0.16, 0.50, 0.84), axis=0),
        "draws": count,
    }
    if extended_summary:
        np.testing.assert_allclose(trapezoid(pdf, z, axis=1), 1., atol=1e-12)
        result.update(draw_indices=indices, delay_parameters=params,
                      chain_path=str(Path(path).resolve()),
                      chain_sha256=suite_hash(path), seed=seed, std_ddof=0)
        for name, values in (("pdf", pdf), ("cdf", cdf)):
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
            result[name + "_mean"] = mean
            result[name + "_std"] = std
            result[name + "_mean_std_bounds"] = np.stack([mean - std, mean + std])
        np.testing.assert_allclose(trapezoid(result["pdf_mean"], z), 1., atol=1e-12)
        np.testing.assert_allclose(cumulative_trapezoid(result["pdf_mean"], z, initial=0),
                                   result["cdf_mean"], atol=1e-12)
    return result


def robust_pz_data(cases=PZ_CASES, n_draw=10000, seed=1, extended_summary=False):
    """Bands for every case, plus the SFH reference curve."""
    import cl_dm_catalog as clcat
    from frb_redshift_model import engine_intrinsic_p, get_engine

    z = np.linspace(0.0, 5.0, 500)
    cosmo = clcat.make_cosmology()
    engine = get_engine(cosmo, z, delay_model="gauss")
    sfh = engine_intrinsic_p(get_engine(cosmo, z, delay_model="sfh"))
    # Trapezoidal weights along the delay-time axis.
    dt = np.diff(engine.tau)
    weights = np.concatenate(
        ([dt[0] / 2], (dt[:-1] + dt[1:]) / 2, [dt[-1] / 2]))
    kernel = (engine.R_sfh_form * weights).T
    result = {"z": z, "sfh_pdf": sfh,
              "sfh_cdf": cumulative_trapezoid(sfh, z, initial=0),
              "quantiles": np.array([0.16, 0.50, 0.84])}
    for case in cases:
        path = case["path"]
        if not path.is_file():
            raise FileNotFoundError(
                "missing robustness chain: %s" % path)
        band = pz_chain_band(path, z, engine, kernel, n_draw=n_draw, seed=seed,
                             extended_summary=extended_summary)
        log("%s: %d draws from %s"
            % (case["name"], band["draws"], path.relative_to(ROOT)))
        result[case["name"] + "_pdf"] = band["pdf"]
        result[case["name"] + "_cdf"] = band["cdf"]
        if extended_summary:
            for key, value in band.items():
                if key not in ("pdf", "cdf"):
                    result[case["name"] + "_" + key] = value
    if extended_summary:
        result.update(summary_method="mean_std", n_draw_requested=n_draw,
                      seed=seed, std_ddof=0,
                      summary="Pointwise mean +/- 1 population standard deviation of unit-normalized intrinsic posterior draws")
    return result


def _robust_legend(ax, handles, labels, crowded, loc="center right",
                   anchor=(0.985, 0.50), fig=None, ncol=1, fontsize=None,
                   labelspacing=None, handlelength=None):
    """Case legend. The default sits on the right, centred on the panel."""
    target = fig if fig is not None else ax
    target.legend(
        handles, labels, loc=loc, bbox_to_anchor=anchor, ncol=ncol,
        handlelength=(2.6 if crowded else 3.0) if handlelength is None else handlelength,
        handleheight=1.0 if crowded else 1.1,
        handletextpad=0.55 if crowded else 0.85,
        labelspacing=(0.32 if crowded else 0.7) if labelspacing is None else labelspacing,
        columnspacing=1.15,
        fontsize=(10.6 if crowded else 14) if fontsize is None else fontsize,
        frameon=False, borderaxespad=0.15)


def _pz_legend_items(cases, reference_label):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=case["facecolor"],
              edgecolor="0.15" if case["hatch"] else "none",
              hatch=case["hatch"], linewidth=0.0,
              alpha=(1.0 if case["hatch"] else
                     max(case["alpha"], 0.45) if case["edge"]
                     else case["alpha"]))
        for case in cases
    ]
    handles.extend([
        Line2D([], [], color=PZ_SFH_COLOR, lw=1.8),
        Line2D([], [], color="#000000", lw=1.85, ls="-"),
        Line2D([], [], color="#000000", lw=1.45, ls=PZ_CDF_STYLE),
    ])
    labels = [case["label"] for case in cases] + [reference_label, "PDF", "CDF"]
    return handles, labels


def bias_posterior(path, stride=8):
    """Peak-normalised GetDist density of b_frb from one C_ell chain."""
    from getdist import MCSamples

    blob = np.load(path, allow_pickle=True)
    samples = np.asarray(blob[0], dtype=float)
    names = [str(item) for item in blob[2]]
    column = samples[::stride, names.index("b_frb")]
    column = column[np.isfinite(column)]
    if column.size < 100:
        raise ValueError("too few finite b_frb samples in %s" % path)
    mc = MCSamples(samples=column[:, None], names=["b_frb"],
                   labels=[r"b_f"], settings={"ignore_rows": 0.0})
    density = mc.get1DDensity("b_frb")
    x = np.asarray(density.x, dtype=float)
    p = np.clip(np.asarray(density.P, dtype=float), 0.0, None)
    if p.max() <= 0 or not np.all(np.isfinite(p)):
        raise ValueError("invalid bias density in %s" % path)
    return x, p / p.max()


def load_robust_bias():
    """C_ell b_frb curves for the same cases as the combined p(z) figure."""
    cache = SUITE_ROOT / "picture" / "png" / "robust_constant_bias_data.npz"
    names = [case["name"] for case in PZ_CASES_CONSTANT]
    if cache.is_file():
        with np.load(cache) as saved:
            if all(name + "_x" in saved.files for name in names):
                return {key: np.array(saved[key]) for key in saved.files}
    result = {}
    for case in PZ_CASES_CONSTANT:
        x, p = bias_posterior(case["path"])
        result[case["name"] + "_x"] = x
        result[case["name"] + "_p"] = p
        log("%s: b_frb mode %.4f" % (case["name"], x[np.argmax(p)]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **result)
    return result


BIAS_REFERENCES = (
    ("Star-forming", 1.23, 0.08, "#82B78F"),
    ("Quiescent", 1.75, 0.04, "#AFA1D6"),
)


BIAS_REFERENCE_LS = (0, (5, 1.8, 1.2, 1.8))


def _paint_bias(ax, cases, curves, references=True):
    """Peak-normalised b_frb posteriors, using the p(z) case colours."""
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

    paint_order = sorted(
        cases, key=lambda c: (c["hatch"] is not None, c["fill_order"]))
    for case in paint_order:
        x = curves[case["name"] + "_x"]
        p = curves[case["name"] + "_p"]
        if (case["facecolor"] != "none" or case["hatch"]) and case["alpha"] > 0:
            ax.fill_between(
                x, 0.0, p, facecolor=case["facecolor"],
                edgecolor="0.15" if case["hatch"] else "none",
                hatch=case["hatch"], alpha=case.get("bias_fill_alpha", case["alpha"]),
                linewidth=0.0, zorder=2)
        ax.plot(x, p, color=case["linecolor"], lw=case.get("bias_linewidth", 1.55),
                solid_capstyle="round", zorder=case.get("line_zorder", 4))
    if references:
        reference_x = np.linspace(0.5, 2.5, 4001)
        for _label, mean, sigma, color in BIAS_REFERENCES:
            density = np.exp(-0.5 * ((reference_x - mean) / sigma) ** 2)
            ax.plot(reference_x, density, color=color, lw=1.8,
                    ls=BIAS_REFERENCE_LS, zorder=6)
    shown = []
    for case in cases:
        x = curves[case["name"] + "_x"]
        p = curves[case["name"] + "_p"]
        keep = p > 0.02
        if np.any(keep):
            shown.extend((float(x[keep][0]), float(x[keep][-1])))
    ax.set_xlim(min(shown) - 0.05, max(shown) + 0.08)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"$b_f$", labelpad=5)
    ax.set_ylabel(r"$P(b_f)$", labelpad=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: "" if np.isclose(value, 0) else f"{value:.1f}"))
    ax.tick_params(which="both", direction="in", top=True, right=True, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2B2B2B")


def _robust_style():
    return {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 14,
        "axes.labelsize": 17, "axes.linewidth": 1.1,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.width": 0.65, "ytick.minor.width": 0.65,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
        "axes.grid": False, "axes.axisbelow": True,
        "grid.color": "#8A8A8A", "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "hatch.linewidth": 0.55,
    }


def _paint_constant_pz(pdf_ax, cdf_ax, cases, data, sfh_color=None):
    """Left-axis density and right-axis cumulative, without a legend."""
    from matplotlib.ticker import (AutoMinorLocator, FixedLocator,
                                   FuncFormatter, MultipleLocator)

    z = data["z"]
    sfh_color = PZ_SFH_COLOR if sfh_color is None else sfh_color
    paint_order = sorted(
        cases, key=lambda c: (c["hatch"] is not None, c["fill_order"]))
    for case in paint_order:
        for ax, quantity in ((pdf_ax, "pdf"), (cdf_ax, "cdf")):
            lo, _median, hi = data[case["name"] + "_" + quantity]
            if str(data.get("summary_method", "median")) == "mean_std":
                lo = np.clip(lo, 0., 1. if quantity == "cdf" else np.inf)
                hi = np.clip(hi, 0., 1. if quantity == "cdf" else np.inf)
            if (case["facecolor"] != "none" or case["hatch"]) and case["alpha"] > 0:
                fill_alpha = case["alpha"] * (case.get("cdf_fill_alpha_scale", 1.) if quantity == "cdf" else 1.)
                ax.fill_between(
                    z, lo, hi, facecolor=case["facecolor"],
                    edgecolor="0.15" if case["hatch"] else "none",
                    hatch=case["hatch"], alpha=fill_alpha,
                    linewidth=0.0, zorder=2)
            if case["edge"]:
                for bound in (lo, hi):
                    ax.plot(z, bound, color=case["linecolor"], lw=0.55,
                            alpha=0.65, zorder=2.5)
    for ax, quantity, ls, lw in ((pdf_ax, "pdf", "-", 1.55),
                                 (cdf_ax, "cdf", PZ_CDF_STYLE, 1.15)):
        for case in cases:
            _lo, median, _hi = data[case["name"] + "_" + quantity]
            ax.plot(z, median, color=case["linecolor"], ls=ls,
                    lw=case.get(quantity + "_linewidth", lw),
                    alpha=case.get(quantity + "_linealpha", 1.),
                    solid_capstyle="round", zorder=case.get("line_zorder", 4))
        ax.plot(z, data["sfh_" + quantity], color=sfh_color,
                ls=ls, lw=1.8, zorder=3)
    for ax in (pdf_ax, cdf_ax):
        ax.set_xlim(0.0, 5.0)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:g}"))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#2B2B2B")
        ax.grid(False)
    pdf_ax.yaxis.set_major_locator(MultipleLocator(0.1))
    pdf_ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: "" if np.isclose(value, 0) else f"{value:.1f}"))
    pdf_ax.tick_params(which="both", direction="in", top=True, left=True,
                       right=False, pad=5)
    pdf_ax.grid(True, axis="x", which="major")
    cdf_ax.tick_params(which="both", direction="in", top=True, left=False,
                       right=True, pad=5)
    density_max = max(data[c["name"] + "_pdf"][2].max() for c in cases)
    median_peak = max(
        max(data[c["name"] + "_pdf"][1].max() for c in cases),
        float(data["sfh_pdf"].max()))
    is_mean = str(data.get("summary_method", "median")) == "mean_std"
    cdf_limit = 1.08 if is_mean else max(1.08, 1.01 * density_max / median_peak)
    pdf_ax.set_ylim(0.0, max(density_max, median_peak) * 1.06 if is_mean else median_peak * cdf_limit * 1.025)
    cdf_ax.set_ylim(0.0, cdf_limit)
    cdf_ax.yaxis.set_major_locator(
        FixedLocator(np.round(np.arange(0.2, 1.01, 0.2), 10)))
    pdf_ax.set_ylabel(r"$p(z)$", labelpad=8)
    pdf_ax.set_xlabel(r"$z$", labelpad=5)
    cdf_ax.set_ylabel(r"$P(z)$", labelpad=10)


def plot_robust_constant_combined_clean(pz_data, curves, output):
    """Retain every case's mean/std band; remove hatching and lighten fills."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    cases = []
    for original in PZ_CASES_CONSTANT:
        case = dict(original)
        # Replace the previously transparent hatched case with a pale grey
        # wash, and reduce opacity uniformly for the other coloured bands.
        if case["facecolor"] == "none":
            case["facecolor"] = "#CED3DB"
        case.update(hatch=None, alpha=min(float(original["alpha"]) * 0.55, 0.18))
        cases.append(case)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / "png"
    assets.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_robust_style()):
        fig = plt.figure(figsize=(14.4, 4.8))
        pdf_ax = fig.add_axes([0.060, 0.15, 0.3775, 0.815])
        cdf_ax = pdf_ax.twinx()
        bias_ax = fig.add_axes([0.565, 0.15, 0.420, 0.815])
        _paint_constant_pz(pdf_ax, cdf_ax, cases, pz_data)
        _paint_bias(bias_ax, cases, curves, references=False)
        handles, labels = _pz_legend_items(cases, "SFR")
        case_handles = [Patch(facecolor=to_rgba(case["facecolor"], max(case["alpha"], 0.25)),
                              edgecolor=to_rgba(case["linecolor"], 0.75), linewidth=0.65)
                        for case in cases]
        _robust_legend(bias_ax, case_handles, labels[:-3], crowded=True,
                       loc="upper right", anchor=(0.985, 0.985),
                       fontsize=12.2, labelspacing=0.48, handlelength=2.7)
        _robust_legend(cdf_ax, handles[-3:], labels[-3:], crowded=False,
                       loc="center right", anchor=(0.985, 0.50),
                       fontsize=17, labelspacing=1.15, handlelength=3.4)
        for extension in ("pdf", "png"):
            path = output if extension == "pdf" else assets / (output.stem + "." + extension)
            fig.savefig(path, dpi=450)
            log("wrote %s" % path)
        plt.close(fig)
    style = {"source_figure": "test_constant_mean.pdf", "hatches": False,
             "displayed_intervals": "All eight cases: pointwise mean +/- 1 population std for both PDF and CDF",
             "all_case_intervals_retained_in_data": True,
             "all_case_intervals_displayed": True,
             "bias_panel": "Original peak-normalized posterior profiles with lighter fills",
             "case_colors": {case["name"]: case["linecolor"] for case in cases},
             "case_fill_colors": {case["name"]: case["facecolor"] for case in cases},
             "case_fill_alpha": {case["name"]: case["alpha"] for case in cases},
             "legend": "Original layout: case key inside right panel; SFR/PDF/CDF key inside left panel",
             "font": "DejaVu Sans", "axis_label_fontsize_pt": 17., "tick_fontsize_pt": 13.,
             "case_legend_fontsize_pt": 12.2, "width_inches": 14.4, "height_inches": 4.8}
    (assets / (output.stem + ".json")).write_text(json.dumps(style, indent=2) + "\n")
    return style
