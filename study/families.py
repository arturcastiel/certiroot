"""Test polynomial families for the MAS vs numpy study (coefficients highest degree first, float64).

Every case is a deterministic function of (family, degree, seed), so references and timings computed in
separate processes refer to bitwise the same polynomial.
"""
import numpy as np

P = np.polynomial.polynomial


def _desc(asc):
    return np.array(asc[::-1], dtype=float)


def kac(n, rng):  # i.i.d. N(0,1) coefficients
    return rng.standard_normal(n + 1)


def int9(n, rng):  # integers in [-9, 9]; nonzero leading/constant coefficients
    c = rng.integers(-9, 10, n + 1).astype(float)
    c[0] = c[0] or 1.0
    c[-1] = c[-1] or 1.0
    return c


def lognormal(n, rng):  # coefficient magnitudes spread over e^-20 .. e^20
    return rng.standard_normal(n + 1) * np.exp(rng.uniform(-20, 20, n + 1))


def randroots(n, rng):  # conjugate-symmetric random roots in the unit disc
    m = n // 2
    z = np.sqrt(rng.uniform(0, 1, m)) * np.exp(2j * np.pi * rng.uniform(0, 1, m))
    r = list(z) + list(np.conj(z)) + list(rng.uniform(-1, 1, n - 2 * m))
    return np.real(np.poly(r))


def unity(n, rng):  # x^n - 1
    c = np.zeros(n + 1)
    c[0], c[-1] = 1.0, -1.0
    return c


def wilkinson(n, rng):
    return np.poly(np.arange(1, n + 1)).astype(float)


def chebyshev(n, rng):
    return np.polynomial.chebyshev.cheb2poly([0] * n + [1])[::-1].astype(float)


def mignotte(n, rng, a=10.0, k=2, sign=-1.0):  # x^n + sign 2 (a x - 1)^k: k roots within ~a^(-n/k) of 1/a
    q = np.array([1.0])
    for _ in range(k):
        q = P.polymul(q, [-1.0, a])
    c = np.zeros(n + 1)
    c[n] = 1.0
    c[: len(q)] += sign * 2.0 * q
    return _desc(c)


def cluster(n, rng, k=3, delta=1e-8):  # k real roots 1, 1+delta, ... plus random conjugate pairs / reals
    r = [1.0 + j * delta for j in range(k)]
    m = (n - k) // 2
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    r += list(z) + list(np.conj(z)) + list(rng.standard_normal(n - k - 2 * m))
    return np.real(np.poly(r))


def multiple(n, rng):  # exact integer coefficients, roots of multiplicity up to 4 (+ Gaussian-integer pairs)
    while True:
        roots = []
        while len(roots) < n:
            r = int(rng.integers(-2, 3))
            roots += [complex(r)] * min(int(rng.integers(1, 5)), n - len(roots))
            if len(roots) + 2 <= n and rng.random() < 0.5:
                w = complex(int(rng.integers(-1, 2)), int(rng.integers(1, 3)))
                roots += [w, w.conjugate()]
        re, im = [1], [0]
        for r in roots:  # exact product of (x - r) over the Gaussian integers
            a, b = int(r.real), int(r.imag)
            nre, nim = re + [0], im + [0]
            for j in range(len(re)):
                nre[j + 1] -= re[j] * a - im[j] * b
                nim[j + 1] -= re[j] * b + im[j] * a
            re, im = nre, nim
        if max(abs(v) for v in re) < 2 ** 53:
            return np.array(re, dtype=float)


def imagaxis(n, rng):  # conjugate pairs on the imaginary axis (roots with a tiny or zero real part)
    npair = max(1, n // 4)
    sc = rng.uniform(0.2, 2.0)
    r = [1j * (j + 1) * sc * rng.uniform(0.9, 1.1) for j in range(npair)]
    r += [np.conj(v) for v in r]
    m = (n - 2 * npair) // 2
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    r += list(z) + list(np.conj(z)) + list(rng.standard_normal(n - 2 * npair - 2 * m))
    c = np.real(np.poly(r))
    return c / np.max(np.abs(c))


def spread(n, rng):  # real roots 10^k spanning many orders of magnitude
    h = n // 2
    return np.poly([10.0 ** (k * 12.0 / max(h, 1)) for k in range(-h, n - h)]).astype(float)


SCALING, HARD = "scaling", "structured"


def _rand(counts, default=20):
    return lambda n: counts.get(n, default)


# (family, generator(n, rng), degrees, instances per degree, group)
FAMILIES = [
    ("kac", kac, [10, 20, 40, 80, 160, 320, 640, 1280], _rand({160: 10, 320: 6, 640: 4, 1280: 2}), SCALING),
    ("int9", int9, [10, 20, 40, 80, 160, 320, 640, 1280], _rand({160: 10, 320: 6, 640: 4, 1280: 2}), SCALING),
    ("lognormal", lognormal, [10, 20, 40, 80, 160, 320, 640], _rand({160: 10, 320: 6, 640: 4}), SCALING),
    ("randroots", randroots, [10, 20, 40, 80, 160, 320], _rand({160: 10, 320: 6}), SCALING),
    ("unity", unity, [10, 20, 40, 80, 160, 320, 640, 1280], lambda n: 1, SCALING),
    ("wilkinson", wilkinson, [8, 10, 12, 15, 18, 20, 22, 25], lambda n: 1, HARD),
    ("chebyshev", chebyshev, [10, 20, 30, 40, 50, 60, 80, 100], lambda n: 1, HARD),
    ("mignotte-a10", lambda n, r: mignotte(n, r, 10.0), [10, 20, 40, 60, 80, 100], lambda n: 1, HARD),
    ("mignotte-a100", lambda n, r: mignotte(n, r, 100.0), [10, 20, 40, 60, 80, 100], lambda n: 1, HARD),
    ("mignotte3-a10", lambda n, r: mignotte(n, r, 10.0, k=3), [12, 20, 40, 60, 80, 100], lambda n: 1, HARD),
    ("cluster3-1e-6", lambda n, r: cluster(n, r, 3, 1e-6), [10, 20, 40, 60], lambda n: 5, HARD),
    ("cluster3-1e-8", lambda n, r: cluster(n, r, 3, 1e-8), [10, 20, 40, 60], lambda n: 5, HARD),
    ("cluster2-1e-12", lambda n, r: cluster(n, r, 2, 1e-12), [10, 20, 40, 60], lambda n: 5, HARD),
    ("multiple", multiple, [10, 20, 30, 40, 60], lambda n: 6, HARD),
    ("imagaxis", imagaxis, [10, 20, 40, 60], lambda n: 6, HARD),
    ("spread", spread, [13, 25], lambda n: 1, HARD),
]


def cases():
    """Yields (case_id, family, group, degree, seed, coefficients)."""
    for fi, (name, gen, degs, nseed, group) in enumerate(FAMILIES):
        for n in degs:
            for s in range(nseed(n)):
                rng = np.random.default_rng([fi, n, s])
                c = np.ascontiguousarray(gen(n, rng), dtype=float)
                yield f"{name}/{n}/{s}", name, group, n, s, c
