"""Adapter exposing the compiled solver with the prototype's API, for the harnesses:
    python stress.py via_certiroot      python sweep.py via_certiroot
"""
import certiroot


def solve_mas_2d(real_coeffs):
    z, m = certiroot.distinct_roots(real_coeffs)
    return [(complex(a), int(k)) for a, k in zip(z, m)]
