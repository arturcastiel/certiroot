# Is MAS more efficient than `numpy.roots`? A reproducible study

*Machine: Intel i7-1355U (WSL2, Linux 5.15), one pinned logical CPU, single-threaded BLAS.
numpy 1.26.4 / OpenBLAS, MPSolve 3.2.1, python-flint 0.9 (Arb) for references.
All numbers below are produced by `analyze.py` from `data/timing.jsonl`; the complete tables are in
[`RESULTS.md`](RESULTS.md), and the plots are in [`plots/`](plots).*

## Verdict

| claim | evidence | status |
|---|---|---|
| MAS is faster than `np.roots` | faster on **548/558** cases; geo-mean **2.86×** [95% CI 2.27–3.64]; sign test p = 8e-148 | **supported** |
| …and the gap grows with degree | `t_np/t_MAS ∝ n^g` with g = **0.64** [0.62, 0.67] (Kac, int9, unity, lognormal); **38–105×** at n = 1280 | **supported** |
| MAS is more accurate | meets its output contract (correctly rounded, tiny components ≤ 2^-119·\|z\|) on **558/558** cases with exact multiplicities, bit-exact on 536; worst error 3.2e-49. numpy: **0/558**, 125 cases with relative error > 1e-6 | **supported** |
| numpy is never better on both counts | MAS Pareto-dominates numpy (time, accuracy) on **548/558** cases; numpy dominates MAS on **0** | **supported** |
| MAS beats LAPACK itself at tiny degree | vs bare `eigvals` (no numpy Python overhead), n ≤ 20: 1.06× [0.97, 1.16] | **not shown (tie)** |
| MAS is the efficient way to get certified accuracy | **30.7×** faster than MPSolve [26.8–35.8], faster on 558/558 | **supported** |

**In one sentence:** on this benchmark, MAS returns correctly rounded roots with certified
multiplicities and is still faster than numpy's unverified eigenvalue answer on 98% of inputs. Its
measured cost grows like n^1.9 against numpy's n^2.5, so the advantage widens with degree. numpy is
faster only on 10 extremely ill-conditioned inputs, and on those its answers have between 1 and
8 correct digits.

## 1. What is compared

| name | what it is |
|---|---|
| `numpy` | `np.roots(c)`: companion matrix → LAPACK `dgeev` (balance + Hessenberg QR) |
| `numpy_eig` | `np.linalg.eigvals(A)` on a prebuilt companion matrix, i.e. numpy's algorithm without its Python-level work (n ≤ 320) |
| `numpy_newton` | `np.roots` followed by 2 vectorized Newton steps, the cheap "polish" fix |
| `mpsolve` | MPSolve 3.2.1 (certified multiprecision Aberth), asked for 53 output bits |
| `mas` | MAS v2, i.e. `certiroot` ([`csrc/mas.cpp`](../csrc/mas.cpp)): exact square-free part → Aberth → compensated evaluation → certified isolation → correctly rounded output with multiplicities |

**Test set:** 558 polynomials in 16 families, generated deterministically (`families.py`).
- *Scaling group*, n = 10 … 1280: Kac (Gaussian coefficients), int9 (integers in [-9, 9]), lognormal coefficients, roots of unity, and random roots in the unit disc (n ≤ 320). 20 seeds for n ≤ 80, then 10/6/4/2 seeds.
- *Structured / hard group*: Wilkinson (8–25), Chebyshev in the monomial basis (10–100), Mignotte-type close pairs and triples (a = 10, 100; n ≤ 100), clusters at 1e-6/1e-8/1e-12, multiple roots, purely imaginary roots, and roots spread over magnitudes.

**Ground truth (`reference.py`):** each double-coefficient polynomial is converted *exactly* to an
integer polynomial, and Arb's `complex_roots` returns certified balls with multiplicities, starting at
256 bits and raising precision if needed. A component counts as correctly rounded (CR) when both ends of
its ball round to the same double and the solver returns exactly that double. Any component within
2^-119·|z| absolute error is also accepted. This is MAS's documented output contract (`TINY_REL` in
`csrc/mas.cpp`): a component tiny next to |z| cannot always be rounded correctly, for example a real part
that may be exactly zero. The same rule scores every solver. All 558 references were certified.

**Timing protocol (`run_study.py`):**
- The process is pinned with `taskset` to one logical CPU whose hyperthread sibling is idle, with `OPENBLAS_NUM_THREADS=1`.
- Each solver gets two warm-up calls. Then each round runs every solver once, in a random order, as a batch of k calls lasting at least 4 ms; this is repeated for 7 rounds (3 for solvers slower than 0.3 s).
- GC is disabled during measurement, and the median per call is reported.
- Interleaving the solvers means machine drift affects all of them alike.

**Statistics (`analyze.py`):**
- Paired per-case log ratios, summarized as a geo-mean over (family, n) cells, with a *two-level* bootstrap CI (resample cells, then seeds).
- One-sided Wilcoxon signed-rank and sign tests.
- "Decisive" per case = all 7 rounds of one solver faster than all 7 of the other (exact p ≤ 1/3432).
- Log–log slope fits on n ≥ 80, with bootstrap CIs.

## 2. Theory: why the gap must grow

**numpy.** `np.roots` computes the eigenvalues of the n×n companion matrix with LAPACK `dgeev`:
balancing, Hessenberg reduction, then Francis multishift QR (`dhseqr`).
- The cost is Θ(n³), about 10n³ flops, with Θ(n²) memory; at n = 1280 that is 13 MB, which no longer fits in cache.
- At n ≈ 75, LAPACK switches from `dlahqr` to `dlaqr0` (aggressive early deflation, `NMIN = 75` in `iparmq`). This explains the visible ~10× jump in numpy's time between n = 40 and n = 80 (see `plots/time_vs_n.png`).
- The QR iteration is backward stable *for the matrix*, not for the polynomial's coefficients. The forward error is therefore ≈ κ(root)·u relative to ‖balanced companion‖, and a root of multiplicity m is perturbed by ~u^(1/m).
- No multiplicity information is returned.

**MAS.** Every stage is Θ(n²) or less:

| stage | cost |
|---|---|
| exact square-free test / modular gcd | Θ(n²) word operations |
| Aberth–Ehrlich, k sweeps | Θ(k·n²): n Horner evaluations of O(n) each + O(n²) pair sums; cubic local convergence, k observed ≈ constant |
| compensated (double-double) Newton polish | Θ(n²) |
| certification (Gershgorin-type isolation + per-root bound + conjugate symmetry) | Θ(n²) |
| correct rounding (double-double Newton correction with rigorous error bound) | Θ(n) per root |
| escalation to K-fold compensated evaluation (K = 3, 4) or anchored / multiprecision | only for roots whose separation is below the noise level |

So the predicted ratio is t_np/t_MAS ∝ n^(3−2) = n.

**Measured exponents (n ≥ 80, 95% CI):**

| family | np.roots | MAS | MPSolve | speed-up growth g |
|---|---|---|---|---|
| kac | 2.52 [2.50, 2.55] | **1.88** [1.86, 1.90] | 1.88 | **0.65** [0.62, 0.67] |
| int9 | 2.53 [2.51, 2.55] | **1.89** [1.87, 1.91] | 1.91 | **0.64** [0.62, 0.66] |
| unity | 2.47 | **1.83** | 1.74 | **0.64** |
| lognormal | 2.56 [2.48, 2.66] | **1.95** [1.87, 2.04] | 1.93 | **0.62** [0.51, 0.72] |
| randroots | 2.37 [2.30, 2.43] | 2.56 [2.12, 3.31] | 2.04 | −0.19 [−0.92, 0.24] |

MAS scales as predicted (≈ n²). numpy's measured 2.5 lies below its asymptotic 3 because, at
n ≤ 1280, its n² parts and memory traffic still weigh in. As n grows, its exponent can only rise
toward 3, so the extrapolations (≈ 100× at n = 5000 for Kac/int9/lognormal) are conservative.
"randroots" is the exception and is explained in §4.

**What can and cannot be proved.**
- The complexity statement is a theorem for every stage *given* a bounded number of Aberth sweeps. Aberth's global convergence is not proven in general (this holds for every Aberth/Ehrlich solver, MPSolve included); the bounded sweep count is observed here, not proven.
- The speed and accuracy claims above are statistical statements about this benchmark, with the p-values and CIs shown. They are not universal theorems.
- The accuracy claim is stronger: MAS's output is *certified* by construction (inclusion discs plus a rigorous rounding test), and Arb confirmed it independently on all 558 cases.

## 3. Results

### 3.1 Speed (ratio t(other)/t(MAS); > 1 means MAS is faster)

| baseline | subset | cases | geo-mean [95% CI] | MAS wins | decisive MAS / other | sign-test p |
|---|---|---|---|---|---|---|
| np.roots | all | 558 | **2.86×** [2.27, 3.64] | 548 | 479 / 5 | 8e-148 |
| np.roots | n ≤ 20 | 231 | **1.61×** [1.49, 1.73] | 231 | 180 / 0 | 3e-70 |
| np.roots | n ≥ 80 | 173 | **10.1×** [6.6, 15.4] | 171 | 170 / 2 | 1e-48 |
| bare eigvals | all (n ≤ 320) | 540 | **1.87×** [1.53, 2.32] | 444 | 342 / 20 | 9e-55 |
| bare eigvals | n ≤ 20 | 231 | 1.06× [0.97, 1.16] | 147 | 68 / 13 | 2e-5 |
| np.roots + 2 Newton | all | 558 | **5.5×** [4.6, 6.7] | 555 | 550 / 2 | 3e-161 |
| MPSolve | all | 558 | **30.7×** [26.8, 35.8] | 558 | 558 / 0 | 1e-168 |

Representative cells (median time per case):

| case | np.roots | MAS | ratio |
|---|---|---|---|
| Kac n = 10 | 27 µs | 17 µs | 1.6× |
| Kac n = 80 | 1.83 ms | 255 µs | 7.2× |
| Kac n = 320 | 49.6 ms | 3.21 ms | 15× |
| Kac n = 1280 | 1.81 s | 47 ms | **38×** |
| int9 n = 1280 | 1.81 s | 45 ms | **40×** |
| unity n = 1280 | 3.05 s | 29 ms | **105×** |
| multiple roots n = 60 | 333 µs | 70 µs | 4.7× |

Giving numpy 2 OpenBLAS threads (`timing_mt.jsonl`, n = 320–1280) speeds it up by at most 1.48× (≤ 1.42× at n = 1280), and
single-threaded MAS remains 13–30× faster on those cases. Hessenberg QR parallelizes poorly.

### 3.2 Accuracy (against the certified Arb reference)

| solver | all roots correctly rounded | median correct digits | worst rel. error | cases with error > 1e-6 |
|---|---|---|---|---|
| np.roots | 0 / 558 | 13.3 | 4.3 | 125 |
| np.roots + 2 Newton | 7 / 558 | 15.7 | 7.4 | 97 |
| MPSolve (53 bit) | 1 / 558 | 15.7 | 2.2e-16 | 0 |
| **MAS** | **558 / 558** | exact | 3.2e-49 | 0 |

MAS multiplicities: 558/558 correct (numpy and MPSolve return unlabelled repeated values).
MAS is bit-exact on 536 cases. The other 22 use the 2^-119·|z| tiny-component rule:
- 46 components of size ~1e-155, from roots of unity and multiple roots. In each, the reference ball straddles 0.
- 1 conjugate pair in `mignotte3-a10/100/0`, with imaginary part ±3.19e-35 at |z| = 0.1. The error there is a few ulps (absolute error 3.4e-50, relative to |z| 3.2e-49).

Median correct digits of numpy per family:

| family | median digits |
|---|---|
| Kac / int9 | 14.5 |
| imagaxis | 12.0 |
| randroots | 9.4 |
| Mignotte | 5–7.6 |
| Wilkinson | 4.8 |
| clusters | 4.7–7.3 |
| Chebyshev | 2.6 |
| multiple roots | **1.8** |

Newton polishing barely helps, because Newton stalls or converges linearly at clusters and multiple
roots.

### 3.3 Pareto view (`plots/accuracy_vs_time.png`)

MAS is at least as fast *and* strictly more accurate on 548/558 cases. On the other 10, numpy is
faster but less accurate, with relative errors of 3.5e-8 … 2.3 (median 4.5e-2). There is **no** case
where numpy is both faster and as accurate.

## 4. Where numpy is faster, and why

**Cells (geo-mean over seeds) where np.roots wins: 8 of 94.** All are tiny-but-pathological
inputs, where MAS's work buys correct digits that numpy does not deliver:

| cell | np.roots | MAS | numpy worst rel. error |
|---|---|---|---|
| Wilkinson n = 22, 25 | 63–78 µs | 81–106 µs | 0.08–0.19 |
| Chebyshev n = 50 (tie) | 357 µs | 359 µs | 4.5e-2 |
| Mignotte a = 10, n = 60 | 411 µs | 529 µs | 3.5e-8 |
| Mignotte a = 100, n = 40, 60 | 195–446 µs | 224–687 µs | 5.8e-8 |
| Mignotte-3 a = 10, n = 40, 60 | 190–411 µs | 284–873 µs | 1.1e-5 |

**Single cases where numpy is decisively faster: 5/558.** Two of them are `randroots` (n = 160, seed 3: 8.9 vs 17.4 ms;
n = 320, seed 5: 52 vs 150 ms). That family expands random roots in the unit disc into double
coefficients. The resulting polynomials are ill-conditioned by construction: numpy has **no correct
digit** on any randroots case with n ≥ 160 (relative error 0.87–4.3). The MAS debug trace for seed 5
shows 309 of 320 roots whose double-double evaluation noise exceeds their neighbour distance. Those
roots *need* about 4× double-precision evaluation (level 4) to be separated. That is 107 ms of the
150 ms, and it is the cost of a correct answer, not overhead. (MPSolve needs 189 ms on the same input.)

**Against bare LAPACK `eigvals` at n ≤ 20**, MAS and LAPACK are statistically tied: geo-mean 1.06×,
CI includes 1. Most of MAS's lead at tiny n over `np.roots` comes from numpy's Python-level overhead
(~10 µs). At n ≥ 80, MAS is 6.4× faster even than bare LAPACK.

## 5. A bug the study found and fixed in MAS

The first timing pass showed MAS at n = 1280 taking 13× longer than at n = 640 instead of ~4×. The
cause was `needs_scaling`, which decides when the evaluation must use the overflow-safe scaled path.
It used the binary exponent of 1.5·max(|x|, |y|) with a limit of 2^860. That was wrong in two ways:

1. **Too eager.** For n ≥ 860, every root with |z| ≥ 2/3 (all of a Kac polynomial) went to the
   slower scaled path, which also bypasses the SIMD batch.
2. **Too lax.** It allowed |z|^(n+1) up to 2^860, while later code forms squared moduli
   (sqrt(p_r² + p_i²), |q|² in complex division), which overflow above 2^512. Such roots silently fell
   into escalation and multiprecision. The answers stayed correct, but were slow.

The fix uses the true modulus with limit 2^440, which keeps squares below 2^1000:

| case | before | after |
|---|---|---|
| Kac n = 1280 | 178 ms | 47 ms |
| unity n = 1280 | 164 ms | 29 ms |
| Kac(99)·(z − 100), n = 100 | 21.7 ms | 0.37 ms |
| Kac(59)·(z − 10⁴), n = 60 | 7.1 ms | 0.28 ms |

The fix was re-validated against the full suite before the final run:

| check | result |
|---|---|
| stress | 24/24 + 60/60 |
| sweep | 150/150 |
| broad3 | 5 × 120 cases, 0 not CR |
| sqf | 400/0 bad |
| clusters | clean |
| imaginary-axis | 150/0 |
| Mignotte | 198/0 |
| new scaling-band test (n = 500–1300, |z| ≈ 0.9–1.7) | 60/0 vs Arb |

All study timings use the fixed library.

## 6. Threats to validity and known limitations

**Benchmark scope**
- **One machine**, WSL2, single thread. Absolute times will differ; ratios between Θ(n²) and Θ(n³) methods are robust to that. MKL instead of OpenBLAS might speed numpy's QR by ~1.5–2×, which would not change any conclusion at n ≥ 80.
- **Fast-companion QR (AMVW)** — Aurentz, Mach, Vandebril & Watkins (2015) — is Θ(n²) and backward stable. It is not what numpy uses, and it is not part of this study.
- **Family choice is ours.** The structured families stop at n = 100, so the hard-input claims are small-n claims. Off-axis clusters remain MAS's weak spot (real-only anchors → multiprecision). They are under-represented here: `cluster*` families have real or mixed centres.

**Measurement details**
- **Call overhead is included** for both: ctypes for MAS, the Python wrapper for numpy. `numpy_eig` isolates LAPACK.
- **MPSolve** is called through ctypes with the default (secular) algorithm and 53 output bits. It is faithful (error ≤ 2.2e-16), but not correctly rounded.
- **Reference tolerance:** components within 2^-119·|z| are accepted (MAS's contract; see §3.2). This matters in 22 cases, 21 of which are exact-zero candidates. Under a strict "every component bit-exact" rule, MAS would score 536/558, and every other solver would still score 0–7.

**Known MAS limitations (not in the benchmark)**
- Polynomials whose coefficients span more than ~10^190 at n ≥ 900 still go to the multiprecision fallback. The answer is correct but slow: 1–2 s at n = 900–1300. At n = 2000 (c_k = N(0,1)·1.7^-k) it did not finish within 300 s, while numpy needs a few seconds. This predates the fix in §5.
- The benchmark build uses `-march=native` (AVX2 + FMA). A baseline x86-64-v2 build is 1.5–3.4× slower, or 1.1–1.85× with a Dekker fallback ([`patches/dekker_prod_err.patch`](../patches/dekker_prod_err.patch)). See [`docs/UPSTREAM.md`](../docs/UPSTREAM.md).

**Robustness fixes after the study:** finite-input check, an overflow in exact→double conversion, exact `round_q` rounding, and thread safety. Details are in [`docs/UPSTREAM.md`](../docs/UPSTREAM.md) §3. Outputs are bit-identical on all 558 cases, and speed is unchanged (±2%), so no study result changes.

## 7. Reproduce

```bash
pip install .[study]                                  # from the repository root
cd study
./run.sh refs --jobs 6 --cpus 6,7,8,9,10,11        # certified Arb references -> data/refs.pkl (~3 min)
CPU=4 ./run.sh time --out timing.jsonl              # pinned interleaved timing + scoring (~35 min)
OPENBLAS_NUM_THREADS=2 CPU=4,5 ./run.sh time --out timing_mt.jsonl --solvers numpy \
    --filter '^(kac|int9|lognormal)/(320|640|1280)/[01]$'   # optional threads supplement
python3 analyze.py                                  # RESULTS.md, data/summary.json, plots/*.png
```

Pick a CPU whose hyperthread sibling is idle; on this machine the siblings are (0,1), (2,3), and so on.
Both `refs` and `time` resume from existing output. `data/refs.pkl` (2.5 MB) is not committed; `refs` rebuilds it.
MPSolve is optional: install `libmps3` or set `MPSOLVE_LIBDIR`, or drop it with `--solvers numpy,numpy_eig,numpy_newton,mas`.
