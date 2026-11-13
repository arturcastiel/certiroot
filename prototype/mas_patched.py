"""Patched version of the MAS 2D solver. Same architecture, numerical fixes:
  1. poly_divmod drops the eliminated leading terms instead of trusting them to be ~0.
  2. Relative (scale-invariant) tolerances + normalized remainders in GCD / Sturm.
  3. Fujiwara root bound instead of the very loose Cauchy bound.
  4. Real roots: bracketed (safeguarded) Newton/bisection inside each Sturm interval.
  5. Winding number: adaptive sampling with a Taylor-disk certificate per segment.
  6. Complex seeds: accepted only if the polished root lies in its own box; else subdivide.
  7. Final polish on the undeflated factor; count check against the degree.
"""
import cmath
import math
import warnings
from fractions import Fraction
from typing import List, Optional, Tuple

EPS = 2.220446049250313e-16


def norm(p) -> float:
    return max((abs(c) for c in p), default=0.0)


def poly_clean(p, rtol: float = 0.0, scale: Optional[float] = None):
    s = norm(p) if scale is None else scale
    i = 0
    while i < len(p) - 1 and abs(p[i]) <= rtol * s:
        i += 1
    return list(p[i:])


def poly_eval(p, z):
    res = 0j
    for c in p:
        res = res * z + c
    return res


def poly_deriv(p):
    n = len(p) - 1
    if n <= 0:
        return [0j]
    return [p[i] * (n - i) for i in range(n)]


def poly_divmod(num, den):
    """Exact-structure long division: the remainder never contains the eliminated leading slots."""
    den = poly_clean(den)
    if abs(den[0]) == 0:
        raise ZeroDivisionError("division by zero polynomial")
    if len(num) < len(den):
        return [0j], list(num)
    rem = list(num)
    deg_diff = len(num) - len(den)
    quot = [0j] * (deg_diff + 1)
    for i in range(deg_diff + 1):
        factor = rem[i] / den[0]
        quot[i] = factor
        rem[i] = 0j
        for j in range(1, len(den)):
            rem[i + j] -= factor * den[j]
    r = rem[deg_diff + 1:]
    return quot, (r if r else [0j])


def _normalize(p):
    s = norm(p)
    return [c / s for c in p] if s > 0 else list(p)


def poly_gcd(a, b, rtol: float = 1e-10):
    """Euclid on max-norm-normalized polynomials; remainder is 'zero' relative to its dividend."""
    a = _normalize(poly_clean(a))
    b = _normalize(poly_clean(b))
    if len(a) < len(b):
        a, b = b, a
    while not (len(b) == 1 and b[0] == 0):
        _, r = poly_divmod(a, b)
        r = poly_clean(r, rtol, scale=norm(a))
        if norm(r) <= rtol * norm(a):
            r = [0j]
        a, b = b, (_normalize(r) if r != [0j] else r)
    return [c / a[0] for c in a]


def fujiwara_bound(p) -> float:
    p = poly_clean(p)
    n = len(p) - 1
    if n <= 0:
        return 0.0
    a = [abs(c) / abs(p[0]) for c in p]
    terms = [a[k] ** (1.0 / k) for k in range(1, n)] + [(a[n] / 2.0) ** (1.0 / n)]
    return 2.0 * max(terms + [0.0]) * 1.001 + 1e-12


# ------------------------------------------------------------------ real roots
def _exact_rem(a, b):
    rem = list(a)
    for i in range(len(a) - len(b) + 1):
        f = rem[i] / b[0]
        for j in range(len(b)):
            rem[i + j] -= f * b[j]
    rem = rem[len(a) - len(b) + 1:]
    while len(rem) > 1 and rem[0] == 0:
        rem.pop(0)
    return rem


def build_sturm_chain(p):
    """Sturm chain in exact rational arithmetic (floats convert to Fractions exactly)."""
    P = [Fraction(c.real) for c in p]
    n = len(P) - 1
    chain = [P, [P[i] * (n - i) for i in range(n)] or [Fraction(0)]]
    while len(chain[-1]) > 1:
        rem = _exact_rem(chain[-2], chain[-1])
        if rem == [0]:
            break
        lead = abs(rem[0])
        chain.append([-c / lead for c in rem])  # positive scaling keeps Sturm signs
    return chain


def _exact_eval(p, x):
    res = Fraction(0)
    for c in p:
        res = res * x + c
    return res


def count_sign_changes(chain, x: float) -> int:
    X = Fraction(x)
    vals = [_exact_eval(f, X) for f in chain]
    nz = [v for v in vals if v != 0]
    return sum(1 for i in range(len(nz) - 1) if (nz[i] > 0) != (nz[i + 1] > 0))


def isolate_real_chambers(chain, a: float, b: float):
    out, stack = [], [(a, b, count_sign_changes(chain, a), count_sign_changes(chain, b))]
    while stack:
        a, b, va, vb = stack.pop()
        k = va - vb
        if k <= 0:
            continue
        if k == 1 or (b - a) <= 4 * EPS * max(1.0, abs(a), abs(b)):
            out.append((a, b, k))  # k > 1 only for an unresolvable cluster
            continue
        mid = 0.5 * (a + b)
        if _exact_eval(chain[0], Fraction(mid)) == 0:
            mid += 1e-3 * (b - a)
        vm = count_sign_changes(chain, mid)
        stack += [(a, mid, va, vm), (mid, b, vm, vb)]
    return sorted(out)


def real_root_in_bracket(p, a: float, b: float) -> float:
    """Safeguarded Newton with bisection fallback (root is simple and bracketed)."""
    pr = [c.real for c in p]
    dpr = [c.real for c in poly_deriv(p)]
    fa, fb = poly_eval(pr, a).real, poly_eval(pr, b).real
    if fb == 0.0:
        return b
    if fa == 0.0:
        return a
    if (fa > 0) == (fb > 0):  # numerically flat bracket: fall back to midpoint
        return 0.5 * (a + b)
    x = 0.5 * (a + b)
    for _ in range(200):
        fx = poly_eval(pr, x).real
        if fx == 0.0:
            return x
        if (fx > 0) == (fa > 0):
            a, fa = x, fx
        else:
            b = x
        d = poly_eval(dpr, x).real
        xn = x - fx / d if d != 0 else 0.5 * (a + b)
        if not (a < xn < b):
            xn = 0.5 * (a + b)
        if abs(xn - x) <= 2 * EPS * max(1.0, abs(x)) or (b - a) <= 4 * EPS * max(1.0, abs(x)):
            return xn
        x = xn
    return x


# --------------------------------------------------------------- complex roots
def taylor_coeffs(p, z0):
    """c[k] = p^(k)(z0)/k!  (repeated synthetic division)."""
    c, out = list(p), []
    while c:
        for i in range(1, len(c)):
            c[i] += c[i - 1] * z0
        out.append(c[-1])
        c = c[:-1]
    return out


def _segment_arg(p, za, zb, min_len):
    """Certified arg change of p along [za, zb]; None if a root is (numerically) on the segment."""
    total, todo = 0.0, [(za, zb)]
    while todo:
        a, b = todo.pop()
        m, r = 0.5 * (a + b), 0.5 * abs(b - a)
        t = taylor_coeffs(p, m)
        bound = sum(abs(t[k]) * r ** k for k in range(1, len(t)))
        if bound < 0.9 * abs(t[0]):
            # p(segment) lies in a disk avoiding 0 => principal phase is the true arg change
            total += cmath.phase(poly_eval(p, b) / poly_eval(p, a))
        elif r < min_len:
            return None
        else:
            todo += [(m, b), (a, m)]
    return total


def winding_number(p, x0, x1, y0, y1) -> Optional[int]:
    min_len = 1e-13 * max(1.0, abs(x0), abs(x1), abs(y0), abs(y1))
    corners = [complex(x0, y0), complex(x1, y0), complex(x1, y1), complex(x0, y1)]
    total = 0.0
    for i in range(4):
        d = _segment_arg(p, corners[i], corners[(i + 1) % 4], min_len)
        if d is None:
            return None
        total += d
    return round(total / (2.0 * math.pi))


def eval_error_bound(p, z) -> float:
    """Rounding-error bound for Horner evaluation of p at z."""
    az, acc = abs(z), 0.0
    for c in p:
        acc = acc * az + abs(c)
    return 8 * len(p) * EPS * acc


def newton_polish(p, z, iters: int = 60):
    """Returns (z, converged). Converged = step tiny OR residual at rounding-noise level."""
    dp = poly_deriv(p)
    for _ in range(iters):
        f, d = poly_eval(p, z), poly_eval(dp, z)
        if abs(f) <= eval_error_bound(p, z):
            if d != 0:
                z -= f / d  # one extra step: the bound is pessimistic
            return z, True
        if d == 0:
            return z, False
        step = f / d
        z -= step
        if abs(step) <= 1e-14 * max(1.0, abs(z)):
            return z, True
    return z, False


def _split(p, x0, x1, y0, y1):
    for fx, fy in ((0.5, 0.5), (0.4871, 0.5131), (0.5237, 0.4763), (0.4419, 0.5581)):
        xm, ym = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        kids = [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]
        counts = [winding_number(p, *k) for k in kids]
        if None not in counts:
            return list(zip(kids, counts))
    raise RuntimeError("could not find a root-free subdivision line")


def isolate_complex_roots(p, x0, x1, y0, y1, count, max_depth: int = 60):
    roots, stack = [], [((x0, x1, y0, y1), count, 0)]
    while stack:
        (bx0, bx1, by0, by1), k, depth = stack.pop()
        if k <= 0:
            continue
        if k == 1:
            z, ok = newton_polish(p, complex(0.5 * (bx0 + bx1), 0.5 * (by0 + by1)))
            sx, sy = 1e-9 * (bx1 - bx0), 1e-9 * (by1 - by0)
            if ok and bx0 - sx <= z.real <= bx1 + sx and by0 - sy <= z.imag <= by1 + sy:
                roots.append(z)
                continue
        tiny = 1e-12 * max(1.0, abs(bx0), abs(bx1), abs(by0), abs(by1))
        if depth >= max_depth or max(bx1 - bx0, by1 - by0) < tiny:  # unresolvable cluster
            roots += [complex(0.5 * (bx0 + bx1), 0.5 * (by0 + by1))] * k
            continue
        for kid, kc in _split(p, bx0, bx1, by0, by1):
            stack.append((kid, kc, depth + 1))
    return roots


# ------------------------------------------------------------------ main solver
def _exact_divmod(a, b):
    a = list(a)
    if len(a) < len(b):
        return [Fraction(0)], a
    q = []
    for i in range(len(a) - len(b) + 1):
        f = a[i] / b[0]
        q.append(f)
        for j in range(len(b)):
            a[i + j] -= f * b[j]
    r = a[len(q):]
    while len(r) > 1 and r[0] == 0:
        r.pop(0)
    return q, (r or [Fraction(0)])


def _exact_gcd(a, b):
    while b != [0]:
        a, b = b, _exact_divmod(a, b)[1]
    return [c / a[0] for c in a]


def squarefree_exact(P):
    """Musser/Yun square-free decomposition over Q: exact multiplicities of the given polynomial."""
    n = len(P) - 1
    dP = [P[i] * (n - i) for i in range(n)] or [Fraction(0)]
    g = _exact_gcd(P, dP)
    w = _exact_divmod(P, g)[0]
    factors, mult = [], 1
    while len(w) > 1:
        y = _exact_gcd(w, g)
        z = _exact_divmod(w, y)[0]
        if len(z) > 1:
            factors.append((z, mult))
        w, g = y, _exact_divmod(g, y)[0]
        mult += 1
    return factors


def solve_mas_2d(real_coeffs: List[float], exact: bool = True,
                 gcd_rtol: float = 1e-10) -> List[Tuple[complex, int]]:
    """exact=True: multiplicities are exact for the polynomial as given (floats are rationals).
    exact=False: numerical multiplicities (roots closer than ~sqrt(gcd_rtol) get merged)."""
    p = poly_clean([complex(c, 0.0) for c in real_coeffs])
    if not p or p[0] == 0:
        raise ValueError("zero polynomial has infinitely many roots")
    if exact:
        P = [Fraction(c.real) for c in p]
        P = [c / P[0] for c in P]
        factors = [([complex(float(c)) for c in z], m, z) for z, m in squarefree_exact(P)]
        return _solve_factors(factors)
    p = [c / p[0] for c in p]

    dp = poly_deriv(p)
    g = poly_gcd(p, dp, gcd_rtol)
    factors, mult = [], 1
    w, _ = poly_divmod(p, g)
    while len(w) > 1:
        y = poly_gcd(w, g, gcd_rtol)
        z, _ = poly_divmod(w, y)
        if len(z) > 1:
            factors.append(([c / z[0] for c in z], mult, None))
        w = y
        g, _ = poly_divmod(g, y)
        mult += 1
    return _solve_factors(factors)


def _solve_factors(factors):
    sols = []
    for factor, m, exact_factor in factors:
        R = fujiwara_bound(factor)
        chain = build_sturm_chain(exact_factor if exact_factor is not None else factor)
        reals = []
        for a, b, k in isolate_real_chambers(chain, -R, R):
            reals += [real_root_in_bracket(factor, a, b)] * k
        deflated = list(factor)
        for r in sorted(reals, key=abs):  # forward deflation is most stable small-first
            deflated, _ = poly_divmod(deflated, [1.0, -r])
        sols += [(complex(r, 0.0), m) for r in reals]

        n_cplx = len(deflated) - 1
        if n_cplx > 0:
            Rc = fujiwara_bound(deflated)
            y0 = 0.0
            k_top = winding_number(deflated, -Rc, Rc, y0, Rc)
            if k_top is None:  # residual root on the real axis: drop the bottom edge slightly
                y0 = -1e-9 * Rc
                k_top = winding_number(deflated, -Rc, Rc, y0, Rc)
            found = isolate_complex_roots(deflated, -Rc, Rc, y0, Rc, k_top) if k_top else []
            others = reals + found + [z.conjugate() for z in found]
            for z0 in found:
                # polish on the undeflated factor; guard against jumping to a neighbouring root
                z, ok = newton_polish(factor, z0, iters=20)
                sep = min((abs(z0 - o) for o in others if o is not z0), default=math.inf)
                z = z if ok and abs(z - z0) < 0.5 * sep else z0
                sols += [(complex(z.real, abs(z.imag)), m), (complex(z.real, -abs(z.imag)), m)]
            if 2 * len(found) != n_cplx:
                warnings.warn(f"found {2*len(found)} complex roots, expected {n_cplx}")
    return sorted(sols, key=lambda s: (round(s[0].real, 6), round(s[0].imag, 6)))
