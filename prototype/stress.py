import signal, random, math, cmath, time, sys
import numpy as np
import importlib; solve_mas_2d = importlib.import_module(sys.argv[1] if len(sys.argv)>1 else "mas_original").solve_mas_2d

def poly_from_roots(roots):
    c = np.poly(np.array(roots, dtype=complex))
    return [float(x.real) for x in c]

class TO(Exception): pass
def handler(s, f): raise TO()
signal.signal(signal.SIGALRM, handler)

def match_err(found, true):
    """Greedy match found multiset to true multiset; return (count_ok, max_err)."""
    f = list(found); t = list(true)
    if len(f) != len(t):
        return False, float('nan')
    err = 0.0
    for r in sorted(t, key=lambda z: (z.real, z.imag)):
        j = min(range(len(f)), key=lambda k: abs(f[k] - r))
        err = max(err, abs(f[j] - r) / max(1.0, abs(r)))
        f.pop(j)
    return True, err

import mpmath
def ref_roots(coeffs, fallback):
    mpmath.mp.dps = 50
    try:
        rs = mpmath.polyroots([mpmath.mpf(c) for c in coeffs], maxsteps=2000, extraprec=400)
        return [complex(r) for r in (rs if isinstance(rs, list) else [rs])]
    except Exception:
        return fallback

def run(name, true_roots, coeffs=None, timeout=60):
    if coeffs is None:
        coeffs = poly_from_roots(true_roots)
    true_roots = ref_roots(coeffs, true_roots)
    signal.alarm(timeout)
    t0 = time.time()
    try:
        sol = solve_mas_2d(coeffs)
        signal.alarm(0)
    except TO:
        print(f"[TIMEOUT ] {name}"); return False
    except RecursionError:
        signal.alarm(0); print(f"[RECURSE ] {name}"); return False
    except Exception as e:
        signal.alarm(0); print(f"[EXCEPT  ] {name}: {type(e).__name__}: {e}"); return False
    dt = time.time() - t0
    found = [z for z, m in sol for _ in range(m)]
    ok, err = match_err(found, true_roots)
    np_found = list(np.roots(coeffs)) if len(coeffs) > 1 else []
    _, np_err = match_err(np_found, true_roots)
    good = ok and err < 1e-6
    tag = "OK      " if good else ("BADCOUNT" if not ok else "INACCUR ")
    print(f"[{tag}] {name:42s} deg={len(true_roots):3d} found={len(found):3d} relerr={err:.1e} numpy={np_err:.1e} t={dt:.2f}s")
    if not good and len(found) <= 12:
        print("           got:", [(complex(round(z.real,6), round(z.imag,6)), m) for z, m in sol])
    return good

cases = []
cases.append(("user example", [2,2,-3,1+2j,1-2j,2j,-2j]))
cases.append(("linear", [5.0]))
cases.append(("x^3 - x (root at 0)", [-1,0,1]))
cases.append(("(x-1)^3 (x+2)^2", [1,1,1,-2,-2]))
cases.append(("(x-1)^5", [1]*5))
cases.append(("(x-1)^8", [1]*8))
cases.append(("(x^2+1)^3", [1j,-1j]*3))
cases.append(("close: (x-1)(x-1.001)", [1, 1.001]))
cases.append(("close: (x-1)(x-1-1e-6)", [1, 1+1e-6]))
cases.append(("small imag: 1 +/- 1e-4 i", [1+1e-4j, 1-1e-4j]))
cases.append(("small imag: 1 +/- 1e-7 i", [1+1e-7j, 1-1e-7j]))
cases.append(("large roots 100..500", [100,200,300,400,500]))
cases.append(("large complex 1000 +/- 1000i", [1000+1000j, 1000-1000j, 3]))
cases.append(("tiny roots 1e-3 scale", [1e-3, 2e-3, -1e-3+1e-3j, -1e-3-1e-3j]))
for n in (8, 12, 16, 20):
    cases.append((f"Wilkinson W{n}", list(range(1, n+1))))
for n in (10, 20, 30):
    cases.append((f"roots of unity x^{n}-1", [cmath.exp(2j*math.pi*k/n) for k in range(n)]))
for n in (10, 15):
    cases.append((f"Chebyshev T{n}", [math.cos((2*k+1)*math.pi/(2*n)) for k in range(n)]))

results = []
for name, roots in cases:
    results.append(run(name, [complex(r) for r in roots]))

# scale invariance: same roots, coefficients scaled by 1e8
r = [complex(x) for x in [2,2,-3,1+2j,1-2j,2j,-2j]]
c = [x*1e8 for x in poly_from_roots(r)]
results.append(run("user example, coeffs * 1e8", r, coeffs=c))

# Random polynomials
random.seed(1)
rand_ok = 0; rand_n = 0
for trial in range(60):
    deg_real = random.randint(0, 6); n_pairs = random.randint(0, 5)
    roots = [complex(random.uniform(-5, 5)) for _ in range(deg_real)]
    for _ in range(n_pairs):
        z = complex(random.uniform(-5,5), random.uniform(0.05,5)); roots += [z, z.conjugate()]
    if not roots: continue
    rand_n += 1
    rand_ok += run(f"random #{trial}", roots)
print(f"\nNamed cases passed: {sum(results)}/{len(results)}; random passed: {rand_ok}/{rand_n}")
