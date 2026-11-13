"""Certified reference roots (Arb via python-flint) and accuracy scoring.

The double coefficients are converted exactly to an integer polynomial; Arb's fmpz_poly.complex_roots
returns rigorous enclosure balls with exact multiplicities. Each root component is certified by rounding
both ends of its ball to double: if they agree, that double is provably the correctly rounded value.
Components whose ball still contains 0 at high precision are recorded as "tiny" (|t| <= bound).
"""
from fractions import Fraction

import flint
import numpy as np

TINY_REL = 2.0 ** -119  # MAS contract: components below 2^-119 |z| are resolved absolutely


def to_fmpz_poly(c):
    fr = [Fraction(float(x)) for x in c]
    den = max(f.denominator for f in fr)  # all powers of two
    return flint.fmpz_poly([int(f * den) for f in fr[::-1]])


def _frac(x):  # exact value of an arb midpoint/radius
    m, e = x.man_exp()
    m, e = int(m), int(e)
    return Fraction(m * 2 ** e) if e >= 0 else Fraction(m, 2 ** -e)


def _component(x):
    """(value, certified, bound): certified -> value is the correctly rounded double of the exact
    component; otherwise the ball contains 0 and |component| <= bound."""
    M, R = _frac(x.mid()), _frac(x.rad())
    if R == 0:
        return float(M), True, 0.0
    lo, hi = M - R, M + R
    if lo > 0 or hi < 0:
        a, b = float(lo), float(hi)
        if a == b:
            return a, True, 0.0
        return float(M), False, float(abs(M) + R)  # ambiguous rounding
    return float(M), False, float(abs(M) + R)  # contains 0


def reference(c, precs=(256, 1024, 4096)):
    """Returns dict of arrays: re, im (CR doubles or midpoints), re_ok, im_ok (certified), re_bnd, im_bnd,
    absz, mult, prec."""
    P = to_fmpz_poly(c)
    old = flint.ctx.prec
    try:
        for prec in precs:
            flint.ctx.prec = prec
            roots = P.complex_roots()
            out = {k: [] for k in ("re", "im", "re_ok", "im_ok", "re_bnd", "im_bnd", "absz", "mult")}
            done = True
            for z, m in roots:
                a = float(abs(z).mid())
                for part, x in (("re", z.real), ("im", z.imag)):
                    v, ok, bnd = _component(x)
                    # an uncertified component is acceptable once it is provably below the tiny scale
                    if not ok and not bnd <= 2.0 ** -200 * a:
                        done = False
                    out[part].append(v), out[part + "_ok"].append(ok), out[part + "_bnd"].append(bnd)
                out["absz"].append(a), out["mult"].append(int(m))
            if done:
                break
    finally:
        flint.ctx.prec = old
    res = {k: np.array(v) for k, v in out.items()}
    res["prec"] = prec
    res["complete"] = done
    return res


def _match(z, w):
    """Index array j with z[j[i]] matched to w[i] (bijection, nearest-neighbour or optimal assignment)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    Z = np.column_stack([z.real, z.imag])
    W = np.column_stack([w.real, w.imag])
    if np.all(np.isfinite(Z)):
        _, j = cKDTree(Z).query(W)
        if len(set(j.tolist())) == len(w):
            return j
    D = np.abs(w[:, None] - np.where(np.isfinite(z), z, 1e300)[None, :])
    r, j = linear_sum_assignment(D)
    return j[np.argsort(r)]


def score(z, ref):
    """Accuracy of approximations z (complex, multiplicities expanded) against a reference.
    Returns dict: count_ok, max_relerr, cr_frac (fraction of roots with both components correctly
    rounded, tiny components within 2^-119 |z|), n_cr_bad."""
    mult = ref["mult"]
    w = np.repeat(ref["re"] + 1j * ref["im"], mult)
    n = len(w)
    z = np.asarray(z, dtype=complex)
    if len(z) != n:
        return dict(count_ok=False, max_relerr=np.inf, cr_frac=0.0, n_cr_bad=n)
    j = _match(z, w)
    zz = z[j]
    a = np.repeat(ref["absz"], mult)
    rel = np.where(a > 0, np.abs(zz - w) / np.where(a > 0, a, 1), np.abs(zz))
    tiny = TINY_REL * a
    ok = np.ones(n, dtype=bool)
    for part, v in (("re", zz.real), ("im", zz.imag)):
        t = np.repeat(ref[part], mult)
        cert = np.repeat(ref[part + "_ok"], mult)
        bnd = np.repeat(ref[part + "_bnd"], mult)
        good = np.where(cert, (v == t) | (np.abs(v - t) <= tiny), np.abs(v) + bnd <= tiny)
        ok &= good
    return dict(count_ok=True, max_relerr=float(np.max(rel)) if n else 0.0, cr_frac=float(np.mean(ok)) if n else 1.0,
                n_cr_bad=int(n - ok.sum()))


def mult_ok(mu, ref):
    """MAS returns distinct roots with multiplicities: same multiset of multiplicities as the reference."""
    return sorted(int(m) for m in mu) == sorted(int(m) for m in ref["mult"])
