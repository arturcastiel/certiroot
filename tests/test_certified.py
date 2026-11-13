"""Checks the correct-rounding contract against Arb's certified root enclosures (python-flint).

The double coefficients are converted exactly to an integer polynomial. A component passes if it is
the correctly rounded double of the exact root, or lies within 2^-119 |z| of it (the documented
contract for components tiny next to |z|, e.g. a real part that may be exactly zero).
"""
from fractions import Fraction

import numpy as np
import pytest

from certiroot import distinct_roots

flint = pytest.importorskip("flint")

TINY_REL = 2.0 ** -119


def _frac(x):
    m, e = x.man_exp()
    m, e = int(m), int(e)
    return Fraction(m * 2 ** e) if e >= 0 else Fraction(m, 2 ** -e)


def _verdict(x, v, tiny):
    """'ok', 'bad' or 'undecided' for double v against the arb ball x."""
    M, R = _frac(x.mid()), _frac(x.rad())
    lo, hi = M - R, M + R
    if lo > 0 or hi < 0:
        a, b = float(lo), float(hi)
        if a == b:
            return "ok" if v == a or abs(Fraction(v) - M) + R <= tiny else "bad"
        return "undecided"
    bound = abs(M) + R  # ball contains 0
    if abs(v) + bound <= tiny:
        return "ok"
    return "bad" if abs(v) > bound + tiny else "undecided"


def check_certified(c):
    z, m = distinct_roots(c)
    fr = [Fraction(float(x)) for x in c]
    den = max(f.denominator for f in fr)
    P = flint.fmpz_poly([int(f * den) for f in fr[::-1]])
    old = flint.ctx.prec
    try:
        for prec in (256, 1024, 4096):
            flint.ctx.prec = prec
            ref = P.complex_roots()
            assert len(ref) == len(z), "number of distinct roots"
            used, verdicts = set(), []
            for w, mult in ref:
                wm = complex(float(w.real.mid()), float(w.imag.mid()))
                j = next(int(k) for k in np.argsort(np.abs(z - wm), kind="stable") if int(k) not in used)
                assert int(m[j]) == int(mult), f"root {wm}: multiplicity {m[j]} != {mult}"
                used.add(j)
                tiny = TINY_REL * abs(wm)
                verdicts += [_verdict(w.real, z[j].real, tiny), _verdict(w.imag, z[j].imag, tiny)]
            assert "bad" not in verdicts
            if "undecided" not in verdicts:
                return
        pytest.fail("reference could not decide every component")
    finally:
        flint.ctx.prec = old


def _poly(rs):
    return np.real(np.poly(np.asarray(rs, dtype=complex)))


def _cases():
    rng = np.random.default_rng(2026)
    yield "kac-10", rng.standard_normal(11)
    yield "kac-40", rng.standard_normal(41)
    yield "kac-120", rng.standard_normal(121)
    yield "int9-50", np.r_[9.0, rng.integers(-9, 10, 50).astype(float)]
    yield "unity-17", np.r_[1.0, np.zeros(16), -1.0]
    yield "unity-64", np.r_[1.0, np.zeros(63), -1.0]
    yield "wilkinson-20", _poly(np.arange(1, 21))
    yield "chebyshev-30", np.polynomial.chebyshev.cheb2poly([0] * 30 + [1])[::-1]
    yield "mignotte-a10-20", np.r_[1.0, np.zeros(17), -200.0, 40.0, -2.0]
    yield "mignotte-a100-40", np.r_[1.0, np.zeros(37), -2e4, 400.0, -2.0]
    yield "multiple", np.convolve(np.convolve(_poly([0.5] * 3 + [3]), [1, 1, 1]), [1, 1, 1])
    yield "cluster-1e-6", _poly(1 + 1e-6 * np.exp(2j * np.pi * np.arange(5) / 5))
    yield "imag-axis", _poly([1j * k for k in range(1, 9)] + [-1j * k for k in range(1, 9)])
    yield "spread", _poly([10.0 ** k for k in range(-6, 7, 2)] + [3 + 4j, 3 - 4j])


@pytest.mark.parametrize("name, c", list(_cases()), ids=[n for n, _ in _cases()])
def test_correctly_rounded(name, c):
    check_certified(c)
