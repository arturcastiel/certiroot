"""certiroot: certified, correctly rounded polynomial roots with exact multiplicities.

Every real and imaginary part returned is the double nearest to the exact root of the polynomial
whose coefficients are the given doubles, and every multiplicity is exact.
"""
import warnings

import numpy as np

from ._core import COUNT_MISMATCH, USED_EXACT_GCD, USED_FALLBACK, solve_raw

__version__ = "0.1.0"
__all__ = ["roots", "polyroots", "distinct_roots", "solve_raw",
           "COUNT_MISMATCH", "USED_EXACT_GCD", "USED_FALLBACK"]


def _real_coeffs(p):
    c = np.atleast_1d(np.asarray(p))
    if c.ndim != 1:
        raise ValueError("input must be a 1-D array of coefficients")
    if np.iscomplexobj(c):
        if np.any(c.imag != 0):
            raise TypeError("certiroot supports real coefficients only")
        c = c.real
    return np.ascontiguousarray(c, dtype=np.float64)


def _solve(c):
    re, im, mult, status = solve_raw(c)
    if status & COUNT_MISMATCH:
        warnings.warn("certiroot could not certify the root count; the result may be incomplete",
                      RuntimeWarning, stacklevel=3)
    order = np.lexsort((im, re))
    return (re + 1j * im)[order], mult[order].astype(np.intp)


def _expand(z, m):
    r = np.repeat(z, m)
    return r.real.copy() if np.all(r.imag == 0) else r


def distinct_roots(p):
    """Distinct roots of a real polynomial and their exact multiplicities.

    Parameters
    ----------
    p : array_like
        Real coefficients, highest degree first (the ``numpy.roots`` convention).

    Returns
    -------
    z : ndarray of complex
        Distinct roots, correctly rounded, sorted by real part and then imaginary part.
    m : ndarray of int
        Multiplicity of each root; ``m.sum()`` equals the degree.

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, is the zero polynomial, or contains NaN or infinity.
    TypeError
        If ``p`` has a nonzero imaginary part.

    Examples
    --------
    >>> z, m = distinct_roots([1, -1, -3, 5, -2])   # (x - 1)**3 (x + 2)
    >>> z.real, m
    (array([-2.,  1.]), array([1, 3]))
    """
    return _solve(_real_coeffs(p))


def roots(p):
    """Drop-in replacement for ``numpy.roots`` with correctly rounded results.

    Parameters
    ----------
    p : array_like
        Real coefficients, highest degree first.

    Returns
    -------
    ndarray
        All roots, each repeated by its multiplicity, sorted by real part and then imaginary part.
        Real dtype when every root is real, complex otherwise (as in ``numpy.roots``).

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, is the zero polynomial, or contains NaN or infinity. (``numpy.roots``
        returns an empty array for the zero polynomial.)
    TypeError
        If ``p`` has a nonzero imaginary part.

    Examples
    --------
    >>> roots([1, -6, 11, -6])
    array([1., 2., 3.])
    """
    return _expand(*_solve(_real_coeffs(p)))


def polyroots(c):
    """Like :func:`roots`, with coefficients lowest degree first (``numpy.polynomial`` convention).

    Examples
    --------
    >>> polyroots([-2, 0, 1])   # x**2 - 2
    array([-1.41421356,  1.41421356])
    """
    return _expand(*_solve(_real_coeffs(c)[::-1].copy()))
