import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.figsize'] = (8, 6)

HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / 'data' / 'frb' / 'P_IGM.npz'

ALPHA = 3.0
BETA = 3.0
SIGMA_MIN = 1e-3
SIGMA_MAX = 10.0
N_SIGMA = 200

_interp_A = None
_interp_C0 = None


def sigma_diff(F, z):
    F = np.asarray(F, dtype=float)
    z = np.asarray(z, dtype=float)
    return F * np.power(z, -0.5)


def _kernel_moments(C0, sigma, alpha=ALPHA, beta=BETA):
    """I0 = ∫ k dΔ, I1 = ∫ Δ k dΔ with u = Δ^{-α}."""
    w = alpha * max(float(sigma), 1e-8)
    u_hi = max(abs(C0) + 15.0 * w, 80.0)
    u_lo = 1e-16
    u_log = np.geomspace(u_lo, u_hi, 2500)
    u_c = np.linspace(max(u_lo, C0 - 12.0 * w), max(u_lo, C0 + 12.0 * w), 4000)
    u = np.unique(np.concatenate([u_log, u_c]))
    u = u[u > 0.0]
    arg = -((u - C0) ** 2) / (2.0 * alpha ** 2 * sigma ** 2)
    gauss = np.exp(np.clip(arg, -700.0, 0.0))
    p0 = beta / alpha - 1.0 / alpha - 1.0
    pref = gauss / alpha
    I0 = trapezoid(pref * np.power(u, p0), u)
    I1 = trapezoid(pref * np.power(u, p0 - 1.0 / alpha), u)
    return I0, I1


def _mean_minus_1(C0, sigma):
    I0, I1 = _kernel_moments(C0, sigma)
    return I1 / I0 - 1.0


def solve_A_C0(sigma, C0_guess=1.0, alpha=ALPHA, beta=BETA):
    """Solve ⟨Δ⟩ = 1 for C0, then A from ∫ p_Δ dΔ = 1."""
    def f(c):
        return _mean_minus_1(c, sigma)

    step = max(0.5, 2.0 * float(sigma))
    a, b = C0_guess - step, C0_guess + step
    fa, fb = f(a), f(b)
    for _ in range(40):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            break
        if not np.isfinite(fb) or (np.isfinite(fa) and abs(fa) <= abs(fb)):
            a -= step
            fa = f(a)
        else:
            b += step
            fb = f(b)
        step *= 1.4
    C0 = brentq(f, a, b)
    I0, _ = _kernel_moments(C0, sigma, alpha=alpha, beta=beta)
    return 1.0 / I0, C0


def build_table(sigma_grid=None, path=NPZ_PATH):
    if sigma_grid is None:
        sigma_grid = np.geomspace(SIGMA_MIN, SIGMA_MAX, N_SIGMA)
    A = np.empty_like(sigma_grid)
    C0 = np.empty_like(sigma_grid)
    guess = 1.0
    for i, sig in enumerate(sigma_grid):
        A[i], C0[i] = solve_A_C0(sig, C0_guess=guess)
        guess = C0[i]
    np.savez(path, sigma=sigma_grid, A=A, C0=C0)
    return sigma_grid, A, C0


def load_table(path=NPZ_PATH):
    global _interp_A, _interp_C0
    data = np.load(path)
    _interp_A = PchipInterpolator(data['sigma'], data['A'], extrapolate=True)
    _interp_C0 = PchipInterpolator(data['sigma'], data['C0'], extrapolate=True)
    return data


def get_A_C0(F, z):
    if _interp_A is None:
        load_table()
    sig = np.clip(sigma_diff(F, z), SIGMA_MIN, SIGMA_MAX)
    return _interp_A(sig), _interp_C0(sig)


def p_delta(delta, F, z, alpha=ALPHA, beta=BETA):
    A, C0 = get_A_C0(F, z)
    sig = np.clip(sigma_diff(F, z), SIGMA_MIN, SIGMA_MAX)
    delta = np.asarray(delta, dtype=float)
    out = np.zeros_like(delta)
    ok = delta > 0.0
    d = delta[ok]
    exponent = -((np.power(d, -alpha) - C0) ** 2) / (2.0 * alpha ** 2 * sig ** 2)
    out[ok] = A * np.power(d, -beta) * np.exp(exponent)
    return out
