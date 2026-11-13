import doctest
import math
import threading
import warnings

import numpy as np
import pytest

import certiroot
from certiroot import distinct_roots, polyroots, roots


def poly_from_int_roots(rs):
    c = [1]
    for r in rs:
        c = np.convolve(c, [1, -r])
    return [float(x) for x in c]


def test_docstrings():
    assert doctest.testmod(certiroot).failed == 0


def test_wilkinson10_is_exact():
    assert roots(poly_from_int_roots(range(1, 11))).tolist() == list(map(float, range(1, 11)))


def test_sqrt2_is_correctly_rounded():
    assert roots([1, 0, -2]).tolist() == [-math.sqrt(2), math.sqrt(2)]
    assert roots([1, 0, 2]).tolist() == [complex(0, -math.sqrt(2)), complex(0, math.sqrt(2))]


def test_exact_multiplicities():
    # x**2 (x - 1)**3 (x + 2)**2 (x**2 + 1)
    c = np.convolve(poly_from_int_roots([0, 0, 1, 1, 1, -2, -2]), [1, 0, 1])
    z, m = distinct_roots(c)
    assert z.tolist() == [-2, -1j, 0, 1j, 1]
    assert m.tolist() == [2, 1, 2, 1, 3]
    r = roots(c)
    assert len(r) == 9 and r.dtype == complex


def test_real_dtype_when_all_roots_real():
    r = roots([1, -6, 11, -6])
    assert r.dtype == np.float64 and r.tolist() == [1.0, 2.0, 3.0]


def test_polyroots_reverses_coefficients():
    c = [3.0, -1.5, 0.25, 2.0, -7.0]
    assert np.array_equal(polyroots(c), roots(c[::-1]))


def test_agrees_with_numpy_on_well_conditioned_input():
    rng = np.random.default_rng(1)
    for n in (5, 20, 60):
        c = rng.standard_normal(n + 1)
        a = np.sort_complex(roots(c).astype(complex))
        b = np.sort_complex(np.roots(c))
        assert len(a) == n
        assert np.max(np.abs(a - b) / np.abs(a)) < 1e-8


def test_leading_and_trailing_zeros():
    assert roots([0, 0, 1, -1]).tolist() == [1.0]
    z, m = distinct_roots([1, -1, 0, 0, 0])
    assert z.tolist() == [0, 1] and m.tolist() == [3, 1]


@pytest.mark.parametrize("p", [[5.0], [0, 3.0], 7.0])
def test_constants_have_no_roots(p):
    assert len(roots(p)) == 0


def test_input_types():
    assert roots((1, -3, 2)).tolist() == [1.0, 2.0]
    assert roots(np.array([1, -3, 2], dtype=np.int64)).tolist() == [1.0, 2.0]
    assert roots(np.array([1, -3, 2], dtype=complex)).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("p, exc", [
    ([0.0, 0.0], ValueError),
    ([], ValueError),
    ([1.0, float("nan"), 2.0], ValueError),
    ([1.0, float("inf")], ValueError),
    ([[1.0, 2.0], [3.0, 4.0]], ValueError),
    ([1.0, 1j], TypeError),
])
def test_invalid_input(p, exc):
    with pytest.raises(exc):
        roots(p)


def test_extreme_scales():
    for s in (2.0 ** -330, 2.0 ** 330):
        assert roots([1.0, -3 * s, 2 * s * s]).tolist() == [s, 2 * s]
    assert roots([1.0, -3e300]).tolist() == [3e300]


def test_no_warning_on_hard_inputs():
    # Mignotte-type close pair, high-degree Kac, roots of unity: certified without fallback warnings
    mig = [1.0] + [0.0] * 39 + [-2 * 100.0 ** 2, 4 * 100.0, -2.0]
    kac = np.random.default_rng(7).standard_normal(301)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for c in (mig, kac, [1.0] + [0.0] * 63 + [-1.0]):
            assert len(roots(c)) == len(c) - 1


def test_thread_safety():
    rng = np.random.default_rng(3)
    polys = [rng.standard_normal(n + 1) for n in (8, 30, 90)] + [poly_from_int_roots([1, 1, 2, 3, 3, 3])]
    expected = [roots(c) for c in polys]
    bad = []

    def work():
        for _ in range(20):
            for c, e in zip(polys, expected):
                if not np.array_equal(roots(c), e):
                    bad.append(1)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not bad
