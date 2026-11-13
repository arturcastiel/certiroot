"""Minimal ctypes binding to MPSolve 3.2.1 (libmps) treating double coefficients as exact rationals.

Uses the system libmps.so.3 (Debian/Ubuntu: apt install libmps3), or the directory in $MPSOLVE_LIBDIR
(e.g. an unpacked .deb). Some builds also need LD_PRELOAD=libtcmalloc_minimal.so.4 (see run.sh).
"""
import ctypes
import ctypes.util
import os

_DIR = os.environ.get("MPSOLVE_LIBDIR", "")
_gmp = ctypes.CDLL(ctypes.util.find_library("gmp") or "libgmp.so.10", mode=ctypes.RTLD_GLOBAL)
if _DIR and os.path.exists(os.path.join(_DIR, "libgmpxx.so.4")):
    ctypes.CDLL(os.path.join(_DIR, "libgmpxx.so.4"), mode=ctypes.RTLD_GLOBAL)
_mps = ctypes.CDLL(os.path.join(_DIR, "libmps.so.3") if _DIR else "libmps.so.3")

_vp, _int, _long = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
_mps.mps_context_new.restype = _vp
_mps.mps_context_free.argtypes = [_vp]
_mps.mps_monomial_poly_new.restype = _vp
_mps.mps_monomial_poly_new.argtypes = [_vp, _long]
_mps.mps_monomial_poly_set_coefficient_q.argtypes = [_vp, _vp, _long, _vp, _vp]
_mps.mps_context_set_input_poly.argtypes = [_vp, _vp]
_mps.mps_context_select_algorithm.argtypes = [_vp, _int]
_mps.mps_context_set_output_prec.argtypes = [_vp, _long]
_mps.mps_context_set_output_goal.argtypes = [_vp, _int]
_mps.mps_mpsolve.argtypes = [_vp]
_mps.mps_context_get_degree.argtypes = [_vp]
_mps.mps_context_get_roots_d.argtypes = [_vp, _vp, _vp]
_mps.mps_polynomial_free.argtypes = [_vp, _vp]
_gmp.__gmpq_init.argtypes = [_vp]
_gmp.__gmpq_clear.argtypes = [_vp]
_gmp.__gmpq_set_d.argtypes = [_vp, ctypes.c_double]

GOAL_ISOLATE, GOAL_APPROXIMATE = 0, 1
STANDARD_MPSOLVE, SECULAR_GA = 0, 1


class _Cplx(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


def roots(coeffs, output_bits=53, algorithm=SECULAR_GA):
    """coeffs: highest degree first (numpy convention). Returns list of complex roots
    (approximated to `output_bits` relative bits; coefficients taken as exact)."""
    c = [float(x) for x in coeffs]
    while c and c[0] == 0.0:
        c.pop(0)
    nzero = 0
    while c and c[-1] == 0.0:
        c.pop(); nzero += 1
    n = len(c) - 1
    if n < 1:
        return [0j] * nzero
    ctx = _mps.mps_context_new()
    poly = _mps.mps_monomial_poly_new(ctx, n)
    re, im = ctypes.create_string_buffer(64), ctypes.create_string_buffer(64)
    _gmp.__gmpq_init(re); _gmp.__gmpq_init(im)
    for i, a in enumerate(reversed(c)):  # MPSolve indexes by power
        _gmp.__gmpq_set_d(re, a)
        _mps.mps_monomial_poly_set_coefficient_q(ctx, poly, i, re, im)
    _gmp.__gmpq_clear(re); _gmp.__gmpq_clear(im)
    _mps.mps_context_set_input_poly(ctx, poly)
    _mps.mps_context_select_algorithm(ctx, algorithm)
    _mps.mps_context_set_output_prec(ctx, output_bits)
    _mps.mps_context_set_output_goal(ctx, GOAL_APPROXIMATE)
    _mps.mps_mpsolve(ctx)
    out = (_Cplx * n)()
    p = ctypes.pointer(out)
    _mps.mps_context_get_roots_d(ctx, ctypes.byref(p), None)
    res = [complex(z.re, z.im) for z in out]
    _mps.mps_polynomial_free(ctx, poly)
    _mps.mps_context_free(ctx)
    return res + [0j] * nzero
