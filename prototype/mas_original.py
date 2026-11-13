import cmath
import math
from typing import List, Tuple

def poly_clean(p: List[complex], tol: float = 1e-12) -> List[complex]:
    i = 0
    while i < len(p) - 1 and abs(p[i]) <= tol:
        i += 1
    return p[i:]

def poly_eval(p: List[complex], z: complex) -> complex:
    res = 0j
    for c in p:
        res = res * z + c
    return res

def poly_deriv(p: List[complex]) -> List[complex]:
    n = len(p) - 1
    if n <= 0:
        return [0j]
    return [p[i] * (n - i) for i in range(n)]

def poly_divmod(num, den):
    num = poly_clean(num)
    den = poly_clean(den)
    if len(num) < len(den):
        return [0j], num
    rem = list(num)
    deg_diff = len(num) - len(den)
    quot = [0j] * (deg_diff + 1)
    for i in range(deg_diff + 1):
        factor = rem[i] / den[0]
        quot[i] = factor
        for j in range(len(den)):
            rem[i + j] -= factor * den[j]
    return poly_clean(quot), poly_clean(rem)

def poly_gcd(a, b, tol: float = 1e-7):
    a = poly_clean(a, tol)
    b = poly_clean(b, tol)
    while len(b) > 1 or abs(b[0]) > tol:
        _, r = poly_divmod(a, b)
        a, b = b, poly_clean(r, tol)
    lead = a[0]
    return [c / lead for c in a]

def cauchy_bound(p) -> float:
    p = poly_clean(p)
    if len(p) <= 1:
        return 0.0
    lead = abs(p[0])
    return 1.0 + max(abs(c) / lead for c in p[1:])

def build_sturm_chain(p):
    chain = [p, [c.real for c in poly_deriv(p)]]
    while len(chain[-1]) > 1:
        _, rem = poly_divmod(chain[-2], chain[-1])
        rem_real = [c.real for c in rem]
        if len(rem_real) == 1 and abs(rem_real[0]) < 1e-12:
            break
        chain.append([-c for c in rem_real])
    return chain

def count_sign_changes(chain, x: float) -> int:
    vals = [poly_eval(f, x).real for f in chain]
    non_zero = [v for v in vals if abs(v) > 1e-12]
    return sum(1 for i in range(len(non_zero) - 1) if (non_zero[i] > 0) != (non_zero[i + 1] > 0))

def isolate_real_chambers(chain, a: float, b: float):
    count = count_sign_changes(chain, a) - count_sign_changes(chain, b)
    if count <= 0:
        return []
    if count == 1:
        return [(a, b)]
    mid = 0.5 * (a + b)
    if abs(poly_eval(chain[0], mid)) < 1e-12:
        mid += 1e-7 * (b - a)
    return isolate_real_chambers(chain, a, mid) + isolate_real_chambers(chain, mid, b)

def polish_complex_halley(p, z0: complex, tol: float = 1e-13) -> complex:
    dp = poly_deriv(p)
    ddp = poly_deriv(dp)
    z = z0
    for _ in range(100):
        fz = poly_eval(p, z)
        if abs(fz) < tol:
            return z
        dfz = poly_eval(dp, z)
        ddfz = poly_eval(ddp, z)
        denom = 2.0 * dfz * dfz - fz * ddfz
        if abs(denom) > 1e-15:
            z -= (2.0 * fz * dfz) / denom
        elif abs(dfz) > 1e-15:
            z -= fz / dfz
        else:
            z += 1e-5 + 1e-5j
    return z

def winding_number(p, x0, x1, y0, y1, steps_per_edge: int = 40) -> int:
    pts = []
    for k in range(steps_per_edge):
        t = k / steps_per_edge
        pts.append(complex(x0 + t * (x1 - x0), y0))
    for k in range(steps_per_edge):
        t = k / steps_per_edge
        pts.append(complex(x1, y0 + t * (y1 - y0)))
    for k in range(steps_per_edge):
        t = k / steps_per_edge
        pts.append(complex(x1 - t * (x1 - x0), y1))
    for k in range(steps_per_edge):
        t = k / steps_per_edge
        pts.append(complex(x0, y1 - t * (y1 - y0)))
    pts.append(pts[0])
    total_angle = 0.0
    prev_val = poly_eval(p, pts[0])
    if abs(prev_val) < 1e-11:
        return -1
    for z in pts[1:]:
        curr_val = poly_eval(p, z)
        if abs(curr_val) < 1e-11:
            return -1
        diff = cmath.phase(curr_val / prev_val)
        total_angle += diff
        prev_val = curr_val
    return round(total_angle / (2.0 * math.pi))

def isolate_complex_boxes(p, x0, x1, y0, y1, depth: int = 0):
    if depth > 25:
        return [complex(0.5 * (x0 + x1), 0.5 * (y0 + y1))]
    count = winding_number(p, x0, x1, y0, y1, steps_per_edge=30 + depth * 4)
    if count == 0:
        return []
    if count == 1:
        return [complex(0.5 * (x0 + x1), 0.5 * (y0 + y1))]
    xm = x0 + 0.5013 * (x1 - x0)
    ym = y0 + 0.5017 * (y1 - y0)
    return (
        isolate_complex_boxes(p, x0, xm, y0, ym, depth + 1) +
        isolate_complex_boxes(p, xm, x1, y0, ym, depth + 1) +
        isolate_complex_boxes(p, x0, xm, ym, y1, depth + 1) +
        isolate_complex_boxes(p, xm, x1, ym, y1, depth + 1)
    )

def solve_mas_2d(real_coeffs):
    p = [complex(c, 0.0) for c in poly_clean(real_coeffs)]
    lead = p[0]
    p = [c / lead for c in p]
    R = cauchy_bound(p)
    dp = poly_deriv(p)
    g = poly_gcd(p, dp)
    factors = []
    mult = 1
    w, _ = poly_divmod(p, g)
    while len(w) > 1:
        y = poly_gcd(w, g)
        z, _ = poly_divmod(w, y)
        if len(z) > 1:
            factors.append((z, mult))
        w = y
        g, _ = poly_divmod(g, y)
        mult += 1
    all_solutions = []
    for factor, m in factors:
        real_factor = [c.real for c in factor]
        chain = build_sturm_chain(real_factor)
        chambers = isolate_real_chambers(chain, -R, R)
        deflated = list(factor)
        for a, b in chambers:
            r = polish_complex_halley(deflated, complex(0.5 * (a + b), 0.0)).real
            all_solutions.append((complex(r, 0.0), m))
            deflated, _ = poly_divmod(deflated, [1.0, -complex(r, 0.0)])
        if len(deflated) > 1:
            seeds = isolate_complex_boxes(deflated, -R, R, 1e-6, R)
            for seed in seeds:
                z_root = polish_complex_halley(deflated, seed)
                if z_root.imag > 1e-7:
                    all_solutions.append((z_root, m))
                    all_solutions.append((z_root.conjugate(), m))
    return sorted(all_solutions, key=lambda item: (round(item[0].real, 6), round(item[0].imag, 6)))
