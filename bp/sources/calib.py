"""Threshold choice with a confidence bound (BRAINSTORM #6).

Pick the gate threshold tau that answers the most questions while the one-sided Clopper-Pearson *lower* bound on precision
over the answered dev (out-of-fold) cases stays >= target. With target 0.90 at 95 % confidence, a claim of
"precision >= 90 %" holds for that dev sample even though the point estimate is higher. Thresholds are chosen on dev only.
"""
import contextvars
import numpy as np

TARGET = 0.90         # normal mode: one-sided 95 % Clopper-Pearson lower bound on dev OOF precision >= 90 %
import os
# strict mode (app toggle / jeval --strict): lower bound >= 95 %, i.e. dev point precision ~96.5-98 % (BRAINSTORM #6 "97-98 %").
# LB >= 97 % was tried on dev OOF: 6 of 9 gated kinds then never answer, so 95 % is the strict bound that still answers.
TARGET_STRICT = float(os.environ.get("BP_STRICT_TARGET", "0.95"))
STRICT_GRID = np.round(np.arange(0.5, 0.9995, 0.005), 3)  # finer near 1: strict gates live in the top scores
_STRICT = contextvars.ContextVar("bp_strict", default=False)


def set_strict(on):
    """Strict mode for this request/thread context; returns a token for reset_strict."""
    return _STRICT.set(bool(on))


def reset_strict(tok):
    _STRICT.reset(tok)


def strict():
    return _STRICT.get()


def tau_for(M, kind=None):
    """The gate threshold to use now: M['tau_strict'] in strict mode (falls back to 'never answer' if the model has no strict
    threshold), else M['tau']. kind=None for single-gate models (tau is a number)."""
    key = "tau_strict" if strict() else "tau"
    t = M.get(key)
    if t is None:
        return 1.01 if strict() else None
    return t if kind is None else t.get(kind, 1.01)


def mode_text():
    return f"strict mode: precision lower bound ≥ {TARGET_STRICT:.0%} on dev" if strict() else f"precision lower bound ≥ {TARGET:.0%} on dev"


def cp_lower(k, n, conf=0.95):
    """One-sided Clopper-Pearson lower bound on a binomial proportion (k successes of n)."""
    if n <= 0 or k <= 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(1 - conf, k, n - k + 1))


def choose_tau(p, y, n_asked, target=0.90, conf=0.95, grid=None, min_tau=0.5):
    """p: gate scores of the top candidate per question (OOF), y: 1 if that candidate is right, n_asked: questions asked
    (the coverage denominator; questions with no candidate count as abstentions).
    Returns dict(tau, answered, precision, lower, n) for the max-coverage tau with lower bound >= target,
    or tau=1.01 (never answer) when no threshold qualifies."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    grid = np.arange(min_tau, 0.995, 0.01) if grid is None else grid
    best = {"tau": 1.01, "answered": 0.0, "precision": 0.0, "lower": 0.0, "n": 0}
    for t in grid:
        sel = p >= t
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(y[sel].sum()); lb = cp_lower(k, n, conf)
        if lb >= target and n > best["n"]:
            best = {"tau": round(float(t), 3), "answered": n / max(1, n_asked), "precision": k / n, "lower": lb, "n": n}
    return best


def choose_strict(p, y, n_asked, tau_normal, min_tau=0.5):
    """Strict threshold: lower bound >= TARGET_STRICT on the fine grid, never below the normal tau (strict only removes answers)."""
    g = STRICT_GRID[STRICT_GRID >= max(min_tau, tau_normal) - 1e-9]
    c = choose_tau(p, y, n_asked, target=TARGET_STRICT, grid=g)
    c["tau"] = max(c["tau"], tau_normal)
    return c
