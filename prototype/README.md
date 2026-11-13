# Prototype

Where certiroot started: Artur Castiel's MAS idea in pure Python.

| file | what it is |
|---|---|
| `mas_original.py` | The original MAS 2D design: square-free decomposition, Sturm chains with Halley polishing for real roots, and an argument-principle (winding-number) quadtree for complex roots. |
| `mas_patched.py` | The same architecture with the numerical fixes from the first review: exact rational GCD/Sturm, scale-invariant tolerances, certified winding numbers, bracketed Newton. Correct but slow. |
| `stress.py` | 24 named edge cases + 60 random polynomials against an mpmath reference. |
| `sweep.py` | 150 more cases: high degree, repeated integer roots, clusters. |
| `via_certiroot.py` | Adapter so the harnesses can run the compiled solver. |

```bash
pip install mpmath
python stress.py mas_original     # original idea
python stress.py mas_patched      # patched Python version
python stress.py via_certiroot    # compiled certiroot (pip install . first)
python sweep.py  via_certiroot
```

The compiled solver in `../csrc/mas.cpp` keeps the original design as its `v1` path
(`certiroot.solve_raw(c, version=1)`). The default `v2` path replaces the Sturm/quadtree search with
certified Aberth iteration, which is where the speed comes from.
