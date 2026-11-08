"""Posterior loading and GetDist styling for the adopted figures."""
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
from getdist import MCSamples, plots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "picture"


def _result_root():
    raw = os.getenv("RESULT_ROOT", "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return ROOT / "result"


RESULTS = _result_root()

from frb_redshift_model import (  # noqa: E402
    REF_PZ_ALPHA, REF_PZ_BETA, R_SFH,
    analytic_reference_pz, delay_names, engine_intrinsic_p, get_engine,
    pdet_path, sfh_year,
)

SEED_DMZ = 0
SEED_CL = 1
SEED_JOINT = 2

BLUE_DRAW = "#9ecae1"
BLUE_MED = "#2171b5"
GREEN_DRAW = "#74c476"
GREEN_MED = "#006d2c"
GREEN_LEGEND = "#41ab5d"
PURPLE_DRAW = "#bcbddc"
PURPLE_MED = "#6a51a3"
COLOR_WEAK = "#E03424"
COLOR_STRONG = "#006FED"
COLOR_JOINT = "#7a3db8"

DELAY_DIR = {
    "lognormal": "lognormal",
    "gauss": "gauss",
    "powerlaw": "power-law",
    "sfh": "SFH",
}

_IDENTITY_HASHES = None


def _chain(mode, delay_model, stem):
    return (
        RESULTS / mode / DELAY_DIR[delay_model] / "chains" / ("%s_converged.npy" % stem)
    )


MODELS = (
    {
        "key": "lognormal",
        "title": "Log-normal delay",
        "delay_model": "lognormal",
        "params": ("tau_NL", "sigma_NL"),
        "labels": (r"\tau_{\rm NL}", r"\sigma_{\rm NL}"),
        "kind": "2d",
    },
    {
        "key": "power-law",
        "title": "Power-law delay",
        "delay_model": "powerlaw",
        "params": ("alpha",),
        "labels": (r"\alpha_{\tau}",),
        "kind": "1d",
    },
    {
        "key": "gauss",
        "title": "Gaussian delay",
        "delay_model": "gauss",
        "params": ("tau_gauss", "sigma_gauss"),
        "labels": (r"\tau_{\rm gauss}", r"\sigma_{\rm gauss}"),
        "kind": "2d",
    },
    {
        "key": "SFH",
        "title": "SFH (no delay)",
        "delay_model": "sfh",
        "params": (),
        "labels": (),
        "kind": "none",
    },
)


def load_chain(path):
    blob = np.load(path, allow_pickle=True)
    samples = np.asarray(blob[0], dtype=float)
    names = [str(x) for x in blob[2]]
    ranges_list = blob[7]
    return samples, names, ranges_list


def mcsamples(path, param_names, labels):
    samples, names, ranges_list = load_chain(path)
    index = [names.index(name) for name in param_names]
    ranges = {
        name: list(ranges_list[names.index(name)]) for name in param_names
    }
    return MCSamples(
        samples=samples[:, index],
        names=list(param_names),
        labels=list(labels),
        ranges=ranges,
        settings={"ignore_rows": 0.0},
    )


def _getdist_settings(g):
    g.settings.num_plot_contours = 2
    g.settings.legend_fontsize = 18
    g.settings.axes_fontsize = 16
    g.settings.lab_fontsize = 20
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.8
    g.settings.lw1 = 2.0
    g.settings.lw_contour = 2.0


def _optional_chain(path):
    path = Path(path)
    return path if path.is_file() else None
