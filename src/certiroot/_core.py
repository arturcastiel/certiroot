"""ctypes binding to the compiled MAS solver (csrc/mas.cpp)."""
import ctypes
import importlib.machinery
import os

import numpy as np

COUNT_MISMATCH, USED_EXACT_GCD, USED_FALLBACK = 1, 2, 4


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(here, "_libmas" + suffix)
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise ImportError(
        "certiroot's compiled solver (_libmas) was not found next to " + here
        + "; build it with `pip install .` or `python setup.py build_ext --inplace`"
    )


_lib = _load()
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
for _f in (_lib.mas_solve, _lib.mas_solve2):
    _f.argtypes = [_dp, ctypes.c_int, _dp, _dp, _ip, ctypes.POINTER(ctypes.c_int)]
    _f.restype = ctypes.c_int


def solve_raw(coeffs, version=2):
    """Distinct roots of a real polynomial (highest degree first), unsorted.

    Returns ``(re, im, mult, status)``: float64 arrays of the real and imaginary parts, an int32
    array of multiplicities and a bit mask of ``COUNT_MISMATCH | USED_EXACT_GCD | USED_FALLBACK``.
    ``version=2`` is the certified Aberth solver; ``version=1`` is the original Sturm +
    winding-number design, kept for comparison. The GIL is released during the call.
    """
    c = np.ascontiguousarray(coeffs, dtype=np.float64)
    n = len(c)
    re, im, mult = np.empty(n), np.empty(n), np.empty(n, dtype=np.int32)
    status = ctypes.c_int(0)
    k = (_lib.mas_solve2 if version == 2 else _lib.mas_solve)(c, n, re, im, mult, ctypes.byref(status))
    if k == -2:
        raise ValueError("coefficients must be finite (no NaN or inf)")
    if k < 0:
        raise ValueError("the zero polynomial has infinitely many roots")
    return re[:k], im[:k], mult[:k], status.value
