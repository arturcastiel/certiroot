// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Artur Castiel
//
// certiroot: the MAS polynomial root solver.
//
// v2 (mas_solve2, the default):
//   modular square-free test (exact GMP decomposition only when repeated roots exist)
//   -> Aberth-Ehrlich iteration -> compensated (double-double) Newton polish
//   -> Gershgorin-type inclusion discs + conjugate-symmetry certificate
//   -> correct rounding of every component with a rigorous error bound.
//   Roots the double-double noise cannot separate escalate to K-fold compensated evaluation,
//   anchored cluster expansion or multiprecision; factors still uncertified fall back to v1.
// v1 (mas_solve, the original MAS design, see prototype/): exact Sturm real-root isolation ->
//   bracketed Newton -> deflation -> certified winding-number quadtree -> Newton polish.
//
// Must be compiled with -ffp-contract=off: contraction breaks the error-free transformations.
// The C ABI at the end of this file is consumed from Python via ctypes (src/certiroot/_core.py).
#include <gmpxx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {

using cd = std::complex<double>;
using Q = mpq_class;
using PolyQ = std::vector<Q>;   // coefficients, highest degree first
using PolyC = std::vector<cd>;
using PolyR = std::vector<double>;

constexpr double EPS = 2.220446049250313e-16;
// Output guarantee: each component of a root is correctly rounded (round-to-nearest-even of the exact
// value), proven by its enclosure not containing a rounding boundary. A component tiny next to |root|
// (e.g. the real part of a purely imaginary root) can't always be decided; it is accepted once its
// absolute error is <= TINY_REL |root|.
constexpr double TINY_REL = 0x1p-119;

#ifdef MAS_PROFILE  // per-stage wall time, read via mas_profile(): build with -DMAS_PROFILE
double g_prof[16];
struct ProfTimer {
    int k;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ProfTimer(int k) : k(k) {}
    ~ProfTimer() { g_prof[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
#define PROF_CAT2(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT2(a, b)
#define PROF(k) ProfTimer PROF_CAT(prof_, __LINE__)(k)
#else
#define PROF(k)
#endif

// ------------------------------------------------------------------ modular square-free test
using u64 = uint64_t;
using u128 = unsigned __int128;

constexpr u64 M61 = (1ULL << 61) - 1;
inline u64 mod61(u128 x) {
    u64 r = (u64)(x & M61) + (u64)(x >> 61);
    r = (r & M61) + (r >> 61);
    return r >= M61 ? r - M61 : r;
}
inline u64 mul61(u64 a, u64 b) { return mod61((u128)a * b); }
inline u64 sub61(u64 a, u64 b) { return a >= b ? a - b : a + M61 - b; }

// Exact certificate: if gcd(P mod q, P' mod q) = 1 and q does not divide lead(P), then P (the input
// scaled to integers) is square-free over Q. q = 2^61 - 1: reduction is shift-add, 2^e is a rotation.
// false means "not proven" (repeated roots, or an unlucky prime): the caller runs the exact algorithm.
bool squarefree_mod(const std::vector<double>& c) {
    const size_t n = c.size() - 1;
    if (n == 0) return true;
    int emin = INT_MAX;
    for (double v : c)
        if (v != 0.0) {
            int e;
            std::frexp(v, &e);
            emin = std::min(emin, e - 53);
        }
    std::vector<u64> a(n + 1), b(n);  // lowest degree first: the leading coefficient is back()
    for (size_t i = 0; i <= n; ++i) {
        const double v = c[n - i];
        u64 r = 0;
        if (v != 0.0) {
            int e;
            const u64 m = (u64)std::fabs(std::ldexp(std::frexp(v, &e), 53));  // |v| = m 2^(e-53), m < 2^53
            const unsigned sh = (unsigned)((e - 53 - emin) % 61);              // 2^61 = 1 (mod q)
            r = sh ? (((m << sh) & M61) | (m >> (61 - sh))) : m;
            if (r == M61) r = 0;
            if (v < 0 && r) r = M61 - r;
        }
        a[i] = r;
    }
    if (a[n] == 0) return false;
    for (size_t i = 0; i < n; ++i) b[i] = mul61(a[i + 1], (u64)(i + 1));
    auto trim = [](std::vector<u64>& v) {
        while (v.size() > 1 && v.back() == 0) v.pop_back();
    };
    trim(b);
    if (b.size() == 1) return b[0] != 0;
    while (b.size() > 1) {  // Euclid with pseudo-remainders (scaling by units mod q keeps the gcd)
        while (a.size() >= b.size()) {
            const u64 la = a.back(), lb = b.back();
            const size_t off = a.size() - b.size();
            for (size_t j = 0; j < off; ++j) a[j] = mul61(a[j], lb);
            for (size_t j = 0; j + 1 < b.size(); ++j) a[off + j] = sub61(mul61(a[off + j], lb), mul61(la, b[j]));
            a.pop_back();
            trim(a);
            if (a.size() == 1 && a[0] == 0) return false;  // b | a: gcd = b is non-constant
        }
        a.swap(b);
    }
    return b[0] != 0;
}

// ------------------------------------------------------------------ exact polynomial algebra
void strip(PolyQ& p) {
    size_t i = 0;
    while (i + 1 < p.size() && sgn(p[i]) == 0) ++i;
    p.erase(p.begin(), p.begin() + i);
}
bool is_zero(const PolyQ& p) { return p.size() == 1 && sgn(p[0]) == 0; }

PolyQ deriv_q(const PolyQ& p) {
    size_t n = p.size() - 1;
    if (n == 0) return {Q(0)};
    PolyQ d(n);
    for (size_t i = 0; i < n; ++i) d[i] = p[i] * (unsigned long)(n - i);
    return d;
}

void divmod_q(const PolyQ& a, const PolyQ& b, PolyQ& quot, PolyQ& rem) {
    if (a.size() < b.size()) {
        quot = {Q(0)};
        rem = a;
        return;
    }
    PolyQ r = a;
    size_t dd = a.size() - b.size();
    quot.assign(dd + 1, Q(0));
    Q f;
    for (size_t i = 0; i <= dd; ++i) {
        f = r[i] / b[0];
        quot[i] = f;
        if (sgn(f) == 0) continue;
        for (size_t j = 1; j < b.size(); ++j) r[i + j] -= f * b[j];
    }
    rem.assign(r.begin() + dd + 1, r.end());
    if (rem.empty()) rem = {Q(0)};
    strip(rem);
}

PolyQ gcd_q(PolyQ a, PolyQ b) {
    PolyQ q, r;
    while (!is_zero(b)) {
        divmod_q(a, b, q, r);
        a.swap(b);
        b.swap(r);
    }
    Q lead = a[0];
    for (auto& c : a) c /= lead;
    return a;
}

PolyQ exact_div(const PolyQ& a, const PolyQ& b) {
    PolyQ q, r;
    divmod_q(a, b, q, r);
    return q;
}

// Musser square-free decomposition over Q.
std::vector<std::pair<PolyQ, int>> squarefree_exact(const PolyQ& P) {
    std::vector<std::pair<PolyQ, int>> out;
    PolyQ g = gcd_q(P, deriv_q(P));
    PolyQ w = exact_div(P, g);
    int mult = 1;
    while (w.size() > 1) {
        PolyQ y = gcd_q(w, g);
        PolyQ z = exact_div(w, y);
        if (z.size() > 1) out.emplace_back(z, mult);
        w = y;
        g = exact_div(g, y);
        ++mult;
    }
    return out;
}

// ------------------------------------------------------------------ multimodular square-free decomposition
// Primes q = 2^61 - c, c < 2^11: x mod q by two shift-add folds (2^61 = c mod q).
constexpr int NPRIMES = 48;
constexpr u64 PRIME_C[NPRIMES] = {1,    31,   45,   229,  259,  283,  339,  391,  403,  465,  531,  579,
                                  675,  759,  799,  819,  829,  843,  859,  939,  985,  1015, 1153, 1195,
                                  1215, 1281, 1299, 1351, 1371, 1425, 1489, 1525, 1533, 1543, 1609, 1621,
                                  1669, 1741, 1753, 1813, 1845, 1849, 1855, 1863, 1869, 1909, 1921, 1923};

struct ModQ {
    u64 q, c;
    explicit ModQ(u64 c_) : q((1ULL << 61) - c_), c(c_) {}
    u64 red(u128 x) const {  // x < 2^122
        const u128 t = (x & M61) + (x >> 61) * c;                // < 2^73
        const u64 r = (u64)(t & M61) + (u64)(t >> 61) * c;       // < 2^61 + 2^23 < 2q
        return r >= q ? r - q : r;
    }
    u64 mul(u64 a, u64 b) const { return red((u128)a * b); }
    u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + q - b; }
    u64 add(u64 a, u64 b) const { return sub(a, q - b); }
    u64 inv(u64 a) const {  // a != 0; extended Euclid (all intermediates < 2^62 in magnitude)
        int64_t t = 0, nt = 1, r = (int64_t)q, nr = (int64_t)a;
        while (nr) {
            const int64_t k = r / nr;
            int64_t x = t - k * nt;
            t = nt, nt = x;
            x = r - k * nr;
            r = nr, nr = x;
        }
        return (u64)(t < 0 ? t + (int64_t)q : t);
    }
    u64 pow2(u64 e) const {
        if (e < 61) return 1ULL << e;
        u64 r = 1, b = 2;
        for (; e; e >>= 1, b = mul(b, b))
            if (e & 1) r = mul(r, b);
        return r;
    }
    u64 of(const mpz_class& z) const { return mpz_fdiv_ui(z.get_mpz_t(), q); }  // in [0, q)
};
using PolyM = std::vector<u64>;  // lowest degree first; trimmed except where noted

void trim_m(PolyM& a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}
bool zero_m(const PolyM& a) { return a.size() == 1 && a[0] == 0; }
PolyM deriv_m(const ModQ& M, const PolyM& a) {
    if (a.size() == 1) return {0};
    PolyM d(a.size() - 1);
    for (size_t i = 0; i < d.size(); ++i) d[i] = M.mul(a[i + 1], (u64)(i + 1));
    trim_m(d);
    return d;
}
void monic_m(const ModQ& M, PolyM& a) {
    const u64 l = M.inv(a.back());
    for (auto& x : a) x = M.mul(x, l);
}
// Divides a by monic b: quotient returned, a becomes the remainder.
PolyM divrem_m(const ModQ& M, PolyM& a, const PolyM& b) {
    const size_t db = b.size() - 1;
    if (a.size() <= db) return {0};
    PolyM qt(a.size() - db);
    for (size_t i = a.size(); i-- > db;) {
        const u64 f = a[i];
        qt[i - db] = f;
        if (f)
            for (size_t j = 0; j < db; ++j) a[i - db + j] = M.sub(a[i - db + j], M.mul(f, b[j]));
    }
    a.resize(std::max<size_t>(db, 1));
    if (db == 0) a[0] = 0;
    trim_m(a);
    return qt;
}
PolyM gcd_m(const ModQ& M, PolyM a, PolyM b) {  // monic; a != 0
    while (!zero_m(b)) {
        monic_m(M, b);
        divrem_m(M, a, b);
        a.swap(b);
    }
    monic_m(M, a);
    return a;
}
PolyM div_m(const ModQ& M, PolyM a, const PolyM& b) { return divrem_m(M, a, b); }  // b monic, exact
PolyM sub_m(const ModQ& M, PolyM a, const PolyM& b) {
    if (a.size() < b.size()) a.resize(b.size(), 0);
    for (size_t i = 0; i < b.size(); ++i) a[i] = M.sub(a[i], b[i]);
    trim_m(a);
    return a;
}
PolyM mul_m(const ModQ& M, const PolyM& a, const PolyM& b) {  // untrimmed: degree deg a + deg b
    PolyM r(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (size_t j = 0; j < b.size(); ++j) r[i + j] = M.add(r[i + j], M.mul(a[i], b[j]));
    return r;
}

// Yun mod q (q > deg f, lc f != 0): f = lc(f) prod_i A[i]^(i+1), A[i] monic. Returns deg gcd(f, f').
size_t yun_m(const ModQ& M, const PolyM& f, std::vector<PolyM>& A) {
    A.clear();
    const PolyM fp = deriv_m(M, f);
    const PolyM g = gcd_m(M, f, fp);
    if (g.size() == 1) return 0;
    PolyM b = div_m(M, f, g), c = div_m(M, fp, g);
    PolyM d = sub_m(M, c, deriv_m(M, b));
    while (b.size() > 1) {
        PolyM a = gcd_m(M, b, d);
        b = div_m(M, b, a);
        c = div_m(M, d, a);
        d = sub_m(M, c, deriv_m(M, b));
        A.push_back(std::move(a));
    }
    return g.size() - 1;
}

// The input doubles as an integer polynomial A: c_k = +-m_k 2^(sh_k + emin), m_k odd.
struct IntPoly {
    std::vector<u64> m;
    std::vector<u64> sh;
    std::vector<char> neg;
    size_t bits = 0;  // |A_k| < 2^bits
    explicit IntPoly(const std::vector<double>& c) {
        const size_t n = c.size() - 1;
        m.assign(n + 1, 0), sh.assign(n + 1, 0), neg.assign(n + 1, 0);
        std::vector<long> lo(n + 1, 0);
        long emin = LONG_MAX;
        for (size_t i = 0; i <= n; ++i) {
            const double v = c[n - i];
            if (v == 0.0) continue;
            int e;
            u64 mm = (u64)std::ldexp(std::frexp(std::fabs(v), &e), 53);
            const int tz = __builtin_ctzll(mm);
            m[i] = mm >> tz, neg[i] = v < 0, lo[i] = e - 53 + tz;
            emin = std::min(emin, lo[i]);
        }
        for (size_t i = 0; i <= n; ++i)
            if (m[i]) {
                sh[i] = (u64)(lo[i] - emin);
                bits = std::max(bits, (size_t)(64 - __builtin_clzll(m[i])) + sh[i]);
            }
    }
    PolyM image(const ModQ& M) const {  // untrimmed
        PolyM a(m.size());
        for (size_t i = 0; i < m.size(); ++i) {
            const u64 r = m[i] ? M.mul(m[i], M.pow2(sh[i])) : 0;
            a[i] = neg[i] ? M.sub(0, r) : r;
        }
        return a;
    }
};

// Deterministic certificate that A = const * prod F_i^mult_i with prod F_i square-free:
// (1) lc(T) A - lc(A) T = 0 for T = prod F_i^mult_i, checked modulo primes whose product exceeds the
//     coefficient bound of that difference; (2) gcd(S, S') = 1 mod a prime not dividing lc(S), S = prod F_i.
bool certify_sqf(const IntPoly& A, const std::vector<std::vector<mpz_class>>& F, const std::vector<int>& mult) {
    const size_t n = A.m.size() - 1;
    size_t deg = 0, E = A.bits;
    mpz_class one_norm;
    for (size_t i = 0; i < F.size(); ++i) {
        deg += (F[i].size() - 1) * (size_t)mult[i];
        one_norm = 0;
        for (const auto& x : F[i]) one_norm += abs(x);
        E += mpz_sizeinbase(one_norm.get_mpz_t(), 2) * (size_t)mult[i];
    }
    if (deg != n) return false;
    const size_t k = (E + 1) / 60 + 1;  // prod q > 2^(60 k) > 2^(E+1) > |coefficients of the difference|
    if (k > (size_t)NPRIMES) return false;
    for (size_t j = 0; j < k; ++j) {
        const ModQ M(PRIME_C[NPRIMES - 1 - j]);
        const PolyM a = A.image(M);
        PolyM T = {1};
        for (size_t i = 0; i < F.size(); ++i) {
            PolyM f(F[i].size());
            for (size_t t = 0; t < f.size(); ++t) f[t] = M.of(F[i][t]);
            for (int r = 0; r < mult[i]; ++r) T = mul_m(M, T, f);
        }
        const u64 lt = T.back(), la = a.back();
        for (size_t t = 0; t <= n; ++t)
            if (M.mul(lt, a[t]) != M.mul(la, T[t])) return false;
    }
    for (int j = 0; j < 4; ++j) {
        const ModQ M(PRIME_C[j]);
        PolyM S = {1};
        for (const auto& Fi : F) {
            PolyM f(Fi.size());
            for (size_t t = 0; t < f.size(); ++t) f[t] = M.of(Fi[t]);
            S = mul_m(M, S, f);
        }
        if (S.back() == 0) continue;
        if (gcd_m(M, S, deriv_m(M, S)).size() == 1) return true;
    }
    return false;
}

// Square-free decomposition of the integer polynomial behind the doubles c (highest first), multimodular:
// Yun modulo primes, CRT of lc(A) A_i (an integer polynomial since lc(F_i) | lc(A)), primitive parts,
// then the exact certificate. Returns 1 with factors (primitive, lc > 0, highest first), 0 if A is
// proven square-free, -1 if undecided (the caller falls back to rational arithmetic).
int squarefree_modular(const std::vector<double>& c, std::vector<std::pair<PolyQ, int>>& out) {
    out.clear();
    const IntPoly A(c);
    const size_t n = c.size() - 1;
    if (n == 0) return 0;
    size_t best = SIZE_MAX;
    std::vector<size_t> sig;  // degrees of the A_i at the current (presumed lucky) primes
    std::vector<mpz_class> X;
    mpz_class Mprod = 1, half;
    std::vector<PolyM> Ai;
    std::vector<std::vector<mpz_class>> F;
    std::vector<int> mult;
    for (int j = 0; j < NPRIMES; ++j) {
        const ModQ M(PRIME_C[j]);
        PolyM a = A.image(M);
        const u64 la = a.back();
        if (la == 0) continue;
        const size_t gd = yun_m(M, a, Ai);
        if (gd == 0) return 0;  // gcd(A, A') = 1 mod q with q not dividing lc(A): square-free over Q
        std::vector<size_t> s(Ai.size());
        size_t total = 0;
        for (size_t i = 0; i < Ai.size(); ++i) s[i] = Ai[i].size() - 1, total += s[i] + 1;
        if (gd > best) continue;  // unlucky prime
        if (gd < best || s != sig) {  // first prime, or every previous one was unlucky
            best = gd, sig = s, Mprod = 1;
            X.assign(total, mpz_class(0));
        }
        const u64 minv = M.inv(M.of(Mprod));
        bool stable = true;
        size_t p = 0;
        for (const auto& ai : Ai)
            for (u64 v : ai) {
                const u64 h = M.mul(M.sub(M.mul(la, v), M.of(X[p])), minv);
                if (h) {
                    stable = false;
                    mpz_addmul_ui(X[p].get_mpz_t(), Mprod.get_mpz_t(), h);
                }
                ++p;
            }
        const bool first = Mprod == 1;
        mpz_mul_ui(Mprod.get_mpz_t(), Mprod.get_mpz_t(), M.q);
        mpz_fdiv_q_2exp(half.get_mpz_t(), Mprod.get_mpz_t(), 1);
        for (auto& x : X)
            if (x > half) x -= Mprod;
        if (!(first || stable)) continue;
        F.clear(), mult.clear(), p = 0;
        for (size_t i = 0; i < Ai.size(); ++i) {
            const size_t len = Ai[i].size();
            if (len > 1) {
                std::vector<mpz_class> f(X.begin() + (long)p, X.begin() + (long)(p + len));
                mpz_class g = 0;
                for (const auto& x : f) mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), x.get_mpz_t());
                if (f.back() < 0) g = -g;
                for (auto& x : f) mpz_divexact(x.get_mpz_t(), x.get_mpz_t(), g.get_mpz_t());
                F.push_back(std::move(f));
                mult.push_back((int)i + 1);
            }
            p += len;
        }
        if (certify_sqf(A, F, mult)) {
            for (size_t i = 0; i < F.size(); ++i) {
                PolyQ G(F[i].size());
                for (size_t t = 0; t < G.size(); ++t) G[t] = Q(F[i][G.size() - 1 - t]);
                out.emplace_back(std::move(G), mult[i]);
            }
            return 1;
        }
    }
    return -1;
}

// ------------------------------------------------------------------ filtered exact Sturm signs
struct FilteredPoly {
    PolyQ exact;
    PolyR approx;  // exact * 2^-shift, rounded toward zero (positive scaling keeps signs)
};

std::atomic<long> g_exact_evals{0};

int sign_at(const FilteredPoly& f, double x) {
    double v = 0, a = 0, ax = std::fabs(x);
    for (double c : f.approx) {
        v = v * x + c;
        a = a * ax + std::fabs(c);
    }
    double bound = 4.0 * (double)(f.approx.size() + 2) * EPS * a + 1e-290;
    if (std::fabs(v) > bound) return v > 0 ? 1 : -1;  // false for NaN/inf -> exact path
    g_exact_evals.fetch_add(1, std::memory_order_relaxed);
    Q xq(x), r(0);
    for (const auto& c : f.exact) r = r * xq + c;
    return sgn(r);
}

using PolyZ = std::vector<mpz_class>;

// Make integer polynomial primitive with positive content removed (positive scaling keeps signs).
void make_primitive(PolyZ& p) {
    mpz_class g = 0;
    for (const auto& c : p) {
        mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), c.get_mpz_t());
        if (g == 1) return;
    }
    if (g > 1)
        for (auto& c : p) mpz_divexact(c.get_mpz_t(), c.get_mpz_t(), g.get_mpz_t());
}

// Pseudo-remainder: lc(b)^(deg a - deg b + 1) * a  mod  b, computed in Z[x].
PolyZ prem(const PolyZ& a, const PolyZ& b) {
    PolyZ r = a;
    const mpz_class& lb = b[0];
    size_t dd = a.size() - b.size();
    mpz_class f;
    for (size_t i = 0; i <= dd; ++i) {
        f = r[i];
        for (size_t j = i; j < r.size(); ++j) r[j] *= lb;
        for (size_t j = 1; j < b.size(); ++j) r[i + j] -= f * b[j];
    }
    PolyZ rem(r.begin() + dd + 1, r.end());
    size_t k = 0;
    while (k + 1 < rem.size() && rem[k] == 0) ++k;
    rem.erase(rem.begin(), rem.begin() + k);
    if (rem.empty()) rem = {mpz_class(0)};
    return rem;
}

FilteredPoly make_filtered_z(const PolyZ& p) {
    FilteredPoly f{PolyQ(p.size()), PolyR(p.size(), 0.0)};
    long s = LONG_MIN;
    std::vector<long> ex(p.size());
    std::vector<double> man(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        f.exact[i] = p[i];
        if (p[i] == 0) continue;
        man[i] = mpz_get_d_2exp(&ex[i], p[i].get_mpz_t());  // truncated, |man| in [0.5, 1)
        s = std::max(s, ex[i]);
    }
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] != 0) f.approx[i] = std::ldexp(man[i], (int)std::max(ex[i] - s, -2000L));
    return f;
}

// Sturm chain via primitive pseudo-remainder sequence over Z (no rational GCDs per operation).
std::vector<FilteredPoly> sturm_chain(const PolyQ& P) {
    mpz_class den = 1;
    for (const auto& c : P) mpz_lcm(den.get_mpz_t(), den.get_mpz_t(), c.get_den_mpz_t());
    PolyZ A(P.size());
    for (size_t i = 0; i < P.size(); ++i) A[i] = mpz_class(P[i] * den);
    if (A[0] < 0) for (auto& c : A) c = -c;
    make_primitive(A);
    size_t n = A.size() - 1;
    PolyZ B(n > 0 ? n : 1, mpz_class(0));
    for (size_t i = 0; i < n; ++i) B[i] = A[i] * (unsigned long)(n - i);
    make_primitive(B);
    std::vector<PolyZ> chain{A, B};
    while (chain.back().size() > 1) {
        const PolyZ& a = chain[chain.size() - 2];
        const PolyZ& b = chain.back();
        PolyZ r = prem(a, b);
        if (r.size() == 1 && r[0] == 0) break;
        // prem = lc(b)^(delta+1) * rem; Sturm needs -rem up to a POSITIVE factor
        size_t delta = a.size() - b.size();
        bool flip = !(b[0] < 0 && (delta + 1) % 2 == 1);
        if (flip) for (auto& c : r) c = -c;
        make_primitive(r);
        chain.push_back(std::move(r));
    }
    std::vector<FilteredPoly> out;
    out.reserve(chain.size());
    for (const auto& p : chain) out.push_back(make_filtered_z(p));
    return out;
}

int sign_changes(const std::vector<FilteredPoly>& chain, double x) {
    int prev = 0, changes = 0;
    for (const auto& f : chain) {
        int s = sign_at(f, x);
        if (s == 0) continue;
        if (prev && s != prev) ++changes;
        prev = s;
    }
    return changes;
}

struct Chamber { double a, b; int k; };

std::vector<Chamber> isolate_real(const std::vector<FilteredPoly>& chain, double a, double b) {
    std::vector<Chamber> out;
    struct Item { double a, b; int va, vb; };
    std::vector<Item> stack{{a, b, sign_changes(chain, a), sign_changes(chain, b)}};
    while (!stack.empty()) {
        Item it = stack.back();
        stack.pop_back();
        int k = it.va - it.vb;
        if (k <= 0) continue;
        double scale = std::max({1.0, std::fabs(it.a), std::fabs(it.b)});
        if (k == 1 || (it.b - it.a) <= 4 * EPS * scale) {
            out.push_back({it.a, it.b, k});
            continue;
        }
        double mid = 0.5 * (it.a + it.b);
        if (sign_at(chain[0], mid) == 0) mid += 1e-3 * (it.b - it.a);
        int vm = sign_changes(chain, mid);
        stack.push_back({it.a, mid, it.va, vm});
        stack.push_back({mid, it.b, vm, it.vb});
    }
    std::sort(out.begin(), out.end(), [](const Chamber& x, const Chamber& y) { return x.a < y.a; });
    return out;
}

// ------------------------------------------------------------------ floating-point helpers
inline cd horner(const PolyC& p, cd z) {
    cd r = 0;
    for (const auto& c : p) r = r * z + c;
    return r;
}
inline double horner_abs(const PolyC& p, double az) {
    double r = 0;
    for (const auto& c : p) r = r * az + std::abs(c);
    return r;
}
inline double horner_r(const PolyR& p, double x) {
    double r = 0;
    for (double c : p) r = r * x + c;
    return r;
}

PolyC deriv_c(const PolyC& p) {
    size_t n = p.size() - 1;
    if (n == 0) return {0.0};
    PolyC d(n);
    for (size_t i = 0; i < n; ++i) d[i] = p[i] * (double)(n - i);
    return d;
}

double fujiwara(const PolyC& p) {
    size_t n = p.size() - 1;
    if (n == 0) return 0.0;
    double l = std::abs(p[0]), m = 0.0;
    for (size_t k = 1; k < n; ++k) m = std::max(m, std::pow(std::abs(p[k]) / l, 1.0 / (double)k));
    m = std::max(m, std::pow(std::abs(p[n]) / l / 2.0, 1.0 / (double)n));
    return 2.0 * m * 1.001 + 1e-12;
}

double real_root_in_bracket(const PolyR& p, double a, double b) {
    PolyR dp(p.size() - 1);
    for (size_t i = 0; i + 1 < p.size(); ++i) dp[i] = p[i] * (double)(p.size() - 1 - i);
    double fa = horner_r(p, a), fb = horner_r(p, b);
    if (fb == 0.0) return b;
    if (fa == 0.0) return a;
    if ((fa > 0) == (fb > 0)) return 0.5 * (a + b);
    double x = 0.5 * (a + b);
    for (int it = 0; it < 200; ++it) {
        double fx = horner_r(p, x);
        if (fx == 0.0) return x;
        if ((fx > 0) == (fa > 0)) { a = x; fa = fx; } else { b = x; }
        double d = horner_r(dp, x);
        double xn = d != 0.0 ? x - fx / d : 0.5 * (a + b);
        if (!(a < xn && xn < b)) xn = 0.5 * (a + b);
        double sc = std::max(1.0, std::fabs(x));
        if (std::fabs(xn - x) <= 2 * EPS * sc || (b - a) <= 4 * EPS * sc) return xn;
        x = xn;
    }
    return x;
}

bool newton_polish(const PolyC& p, const PolyC& dp, cd& z, int iters) {
    for (int it = 0; it < iters; ++it) {
        cd f = horner(p, z), d = horner(dp, z);
        if (std::abs(f) <= 8.0 * (double)p.size() * EPS * horner_abs(p, std::abs(z))) {
            if (d != 0.0) z -= f / d;  // one extra step: the bound is pessimistic
            return std::isfinite(z.real()) && std::isfinite(z.imag());
        }
        if (d == 0.0) return false;
        cd step = f / d;
        z -= step;
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) return false;
        if (std::abs(step) <= 1e-14 * std::max(1.0, std::abs(z))) return true;
    }
    return false;
}

// ------------------------------------------------------------------ certified winding numbers
struct Workspace {
    PolyC work, taylor;
    std::vector<std::pair<cd, cd>> segs;
};

// Arg change of p along [za, zb]. A segment is accepted when the Taylor bound around its midpoint
// proves p(segment) lies in a disk avoiding 0, so the principal phase difference is exact.
bool segment_arg(const PolyC& p, cd za, cd zb, double min_len, Workspace& ws, double& total) {
    ws.segs.clear();
    ws.segs.emplace_back(za, zb);
    while (!ws.segs.empty()) {
        auto [a, b] = ws.segs.back();
        ws.segs.pop_back();
        cd m = 0.5 * (a + b);
        double r = 0.5 * std::abs(b - a);
        ws.work = p;
        ws.taylor.clear();
        for (size_t len = ws.work.size(); len; --len) {
            for (size_t i = 1; i < len; ++i) ws.work[i] += ws.work[i - 1] * m;
            ws.taylor.push_back(ws.work[len - 1]);
        }
        double bound = 0.0;
        for (size_t k = ws.taylor.size() - 1; k >= 1; --k) bound = bound * r + std::abs(ws.taylor[k]);
        bound *= r;
        if (bound < 0.9 * std::abs(ws.taylor[0])) {
            // arg difference without dividing p(b)/p(a), which can overflow at high degree
            total += std::remainder(std::arg(horner(p, b)) - std::arg(horner(p, a)), 2.0 * M_PI);
        } else if (r < min_len) {
            return false;
        } else {
            ws.segs.emplace_back(m, b);
            ws.segs.emplace_back(a, m);
        }
    }
    return true;
}

bool winding_number(const PolyC& p, double x0, double x1, double y0, double y1, Workspace& ws, int& out) {
    double min_len = 1e-13 * std::max({1.0, std::fabs(x0), std::fabs(x1), std::fabs(y0), std::fabs(y1)});
    cd c[4] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    double total = 0.0;
    for (int i = 0; i < 4; ++i)
        if (!segment_arg(p, c[i], c[(i + 1) % 4], min_len, ws, total)) return false;
    out = (int)std::lround(total / (2.0 * M_PI));
    return true;
}

struct Box { double x0, x1, y0, y1; int k, depth; };

bool split_box(const PolyC& p, const Box& bx, Workspace& ws, Box kids[4]) {
    static const double fr[4][2] = {{0.5, 0.5}, {0.4871, 0.5131}, {0.5237, 0.4763}, {0.4419, 0.5581}};
    for (const auto& f : fr) {
        double xm = bx.x0 + f[0] * (bx.x1 - bx.x0), ym = bx.y0 + f[1] * (bx.y1 - bx.y0);
        kids[0] = {bx.x0, xm, bx.y0, ym, 0, bx.depth + 1};
        kids[1] = {xm, bx.x1, bx.y0, ym, 0, bx.depth + 1};
        kids[2] = {bx.x0, xm, ym, bx.y1, 0, bx.depth + 1};
        kids[3] = {xm, bx.x1, ym, bx.y1, 0, bx.depth + 1};
        bool ok = true;
        for (int i = 0; i < 4 && ok; ++i) ok = winding_number(p, kids[i].x0, kids[i].x1, kids[i].y0, kids[i].y1, ws, kids[i].k);
        if (ok) return true;
    }
    return false;
}

std::vector<cd> isolate_complex(const PolyC& p, double x0, double x1, double y0, double y1, int count, Workspace& ws) {
    const int max_depth = 60;
    PolyC dp = deriv_c(p);
    std::vector<cd> roots;
    std::vector<Box> stack{{x0, x1, y0, y1, count, 0}};
    Box kids[4];
    while (!stack.empty()) {
        Box b = stack.back();
        stack.pop_back();
        if (b.k <= 0) continue;
        cd center(0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1));
        if (b.k == 1) {
            cd z = center;
            bool ok = newton_polish(p, dp, z, 60);
            double sx = 1e-9 * (b.x1 - b.x0), sy = 1e-9 * (b.y1 - b.y0);
            if (ok && b.x0 - sx <= z.real() && z.real() <= b.x1 + sx && b.y0 - sy <= z.imag() && z.imag() <= b.y1 + sy) {
                roots.push_back(z);
                continue;
            }
        }
        double tiny = 1e-12 * std::max({1.0, std::fabs(b.x0), std::fabs(b.x1), std::fabs(b.y0), std::fabs(b.y1)});
        if (b.depth >= max_depth || std::max(b.x1 - b.x0, b.y1 - b.y0) < tiny || !split_box(p, b, ws, kids)) {
            roots.insert(roots.end(), (size_t)b.k, center);  // unresolvable cluster
            continue;
        }
        for (const auto& kid : kids) stack.push_back(kid);
    }
    return roots;
}

// ------------------------------------------------------------------ v2 fast path
// Aberth-Ehrlich approximations (double) -> compensated Newton refinement -> Gerschgorin inclusion
// certificate (isolates every root at once) + per-root error bound -> real roots certified by
// conjugate symmetry. Escalates to compensated Aberth, then GMP multiprecision, then the v1 path.

inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// The exact value lies within err of s + e (s a double, |e| small): true iff it provably rounds to s.
inline bool rounds_to(double s, double e, double err, double tiny) {
    if (err <= tiny) return true;
    const double up = std::nextafter(s, INFINITY) - s, dn = s - std::nextafter(s, -INFINITY);
    const double m = err * (1.0 + 4.0 * EPS) + std::fabs(e) * 4.0 * EPS;
    return std::isfinite(up) && std::isfinite(dn) && e + m < 0.5 * up && e - m > -0.5 * dn;
}

// Real coefficients, highest degree first: a_k = hi_k + lo_k, except |a_k - hi_k - lo_k| <= res_k.
struct CPoly {
    std::vector<double> hi, lo, res;
    bool has_res = false;
    size_t deg() const { return hi.size() - 1; }
};

// Compensated Horner for p(z) and p'(z) at z = x + iy. The rounding errors of the double Horner pass
// are captured exactly by TwoProd (fma) / TwoSum and evaluated by a second Horner pass, so the result
// is as accurate as Horner in twice the precision: error ~ u|p| + O(n u) * T with T ~ u * ptilde.
// With Bound, T (and the analogue for p') is accumulated at run time from the actual rounding errors,
// giving rigorous a-posteriori bounds that are much tighter than the a-priori n u^2 ptilde bound.
//
// Scaled (only if |z|^n could overflow; see needs_scaling): the state is kept multiplied by sc = 2^-S,
// rescaled by 2^-500 whenever it grows past 2^500, and the results are returned that way (*sexp = S).
// p/p' and every relative test are unaffected; values lost to underflow (<= 2^-1074 per operation) are
// carried by ut, uw.
constexpr double SCALE_DOWN = 0x1p-500, UNDERFLOW_STEP = 0x1p-1060;
// Unscaled is safe while |z|^(n+1) <= 2^440: |p|, |p'| and the magnitude Horners stay below
// (n+1)^2 2^440 <= 2^500 (n < 2^30), so the squares formed downstream (moduli, complex division) stay
// finite. A looser limit lets them overflow and sends the root to escalation / multiprecision.
inline bool needs_scaling(size_t n, double x, double y) {  // coefficients |a| < 1
    const double az = std::sqrt(x * x + y * y) * (1.0 + 4.0 * EPS);  // >= |z|; inf if x^2 overflows
    if (!(az > 1.0)) return false;
    int e;
    std::frexp(az, &e);
    if ((double)e * (double)(n + 1) <= 440.0) return false;  // cheap test: log2(az) <= e
    return std::log2(az) * (double)(n + 1) > 440.0;
}

template <bool Bound, bool Scaled>
inline void comp_eval_impl(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi,
                           double* errp, double* errq, int* sexp) {
    const size_t n = P.deg();
    const double* hi = P.hi.data();
    const double* lo = P.lo.data();
    double sr = hi[0], si = 0.0, cr = lo[0], ci = 0.0;  // p: Horner value s, error polynomial c
    double dr = 0.0, di = 0.0, er = 0.0, ei = 0.0;      // p': Horner value d, error polynomial e
    double az = 0.0, t = 0.0, w = 0.0;                  // |z| and magnitude Horners for the bounds
    double sc = 1.0, ut = 0.0, uw = 0.0, rp = 0.0, rq = 0.0;
    int S = 0;
    if (Bound || Scaled) az = std::sqrt(x * x + y * y) * (1.0 + 2.0 * EPS);
    if (Bound) t = std::fabs(lo[0]);
    if (Scaled && P.has_res) rp = P.res[0];
    for (size_t k = 1; k <= n; ++k) {
        const double hk = Scaled ? hi[k] * sc : hi[k], lk = Scaled ? lo[k] * sc : lo[k];
        // d_k = d_{k-1} z + s_{k-1}; e also absorbs c_{k-1}, the error of s_{k-1}
        const double a1 = dr * x, a2 = di * y, a3 = dr * y, a4 = di * x;
        const double b1 = std::fma(dr, x, -a1), b2 = std::fma(di, y, -a2);
        const double b3 = std::fma(dr, y, -a3), b4 = std::fma(di, x, -a4);
        double h1, g1, h2, g2, ndr, g3, ndi, g4;
        two_sum(a1, -a2, h1, g1);
        two_sum(h1, sr, ndr, g3);
        two_sum(a3, a4, h2, g2);
        two_sum(h2, si, ndi, g4);
        const double ner = (er * x - ei * y) + (((b1 - b2) + (g1 + g3)) + cr);
        const double nei = (er * y + ei * x) + (((b3 + b4) + (g2 + g4)) + ci);
        // s_k = s_{k-1} z + a_k
        const double m1 = sr * x, m2 = si * y, m3 = sr * y, m4 = si * x;
        const double e1 = std::fma(sr, x, -m1), e2 = std::fma(si, y, -m2);
        const double e3 = std::fma(sr, y, -m3), e4 = std::fma(si, x, -m4);
        double h3, f1, nsr, f3, nsi, f2;
        two_sum(m1, -m2, h3, f1);
        two_sum(h3, hk, nsr, f3);
        two_sum(m3, m4, nsi, f2);
        const double ncr = (cr * x - ci * y) + (((e1 - e2) + (f1 + f3)) + lk);
        const double nci = (cr * y + ci * x) + ((e3 + e4) + f2);
        if (Bound) {
            const double tq = std::fabs(b1) + std::fabs(b2) + std::fabs(g1) + std::fabs(g3) + std::fabs(b3) +
                              std::fabs(b4) + std::fabs(g2) + std::fabs(g4);
            w = w * az + (tq + 2.0 * t);  // t = t_{k-1} bounds |c_{k-1}| and its error
            const double tp = std::fabs(e1) + std::fabs(e2) + std::fabs(f1) + std::fabs(f3) + std::fabs(lk) +
                              std::fabs(e3) + std::fabs(e4) + std::fabs(f2);
            t = t * az + tp;
        }
        dr = ndr; di = ndi; er = ner; ei = nei;
        sr = nsr; si = nsi; cr = ncr; ci = nci;
        if constexpr (Scaled) {
            if (P.has_res) { rq = rq * az + rp; rp = rp * az + P.res[k] * sc; }
            uw = uw * az + (UNDERFLOW_STEP + 2.0 * ut);
            ut = ut * az + UNDERFLOW_STEP;
            if (std::fabs(sr) + std::fabs(si) + std::fabs(dr) + std::fabs(di) + t + w > 0x1p500) {
                for (double* v : {&sr, &si, &cr, &ci, &dr, &di, &er, &ei, &t, &w, &ut, &uw, &rp, &rq}) *v *= SCALE_DOWN;
                ut += UNDERFLOW_STEP, uw += UNDERFLOW_STEP;
                sc *= SCALE_DOWN;
                S += 500;
            }
        }
    }
    pr = sr + cr; pi = si + ci;
    qr = dr + er; qi = di + ei;
    if (Bound) {
        // error <= u|result| + (g_n + gamma_4) * T (+ coefficient residuals); g = 2x that, in EPS = 2u units
        const double g = (4.0 * (double)n + 8.0) * EPS;
        if (!Scaled && P.has_res)
            for (size_t k = 0; k <= n; ++k) { rq = rq * az + rp; rp = rp * az + P.res[k]; }
        *errp = 1.01 * (EPS * std::sqrt(pr * pr + pi * pi) + g * t + rp + ut);
        *errq = 1.01 * (EPS * std::sqrt(qr * qr + qi * qi) + g * w + rq + uw);
    }
    if (sexp) *sexp = S;
}
// comp_eval_impl<Bound, false> at W points at once: the W Horner chains are independent, so they fill
// the SIMD lanes and hide the latency of the (serial) error-free transformations. Same operations per
// lane, hence the same results. Real (all y == 0): the imaginary parts of the complex recurrence stay
// exactly zero, so dropping them changes nothing but the sign of a zero result.
constexpr int EW = 4;
template <bool Bound, bool Real>
inline void comp_eval_w(const CPoly& P, const double* x, const double* y, double* pr, double* pi, double* qr,
                        double* qi, double* errp, double* errq) {
    const size_t n = P.deg();
    const double* hi = P.hi.data();
    const double* lo = P.lo.data();
    alignas(32) double sr[EW], si[EW], cr[EW], ci[EW], dr[EW], di[EW], er[EW], ei[EW], az[EW], t[EW], w[EW];
    alignas(32) double X[EW], Y[EW];
    for (int l = 0; l < EW; ++l) {
        X[l] = x[l], Y[l] = Real ? 0.0 : y[l];
        sr[l] = hi[0], si[l] = 0.0, cr[l] = lo[0], ci[l] = 0.0;
        dr[l] = di[l] = er[l] = ei[l] = 0.0;
        az[l] = Bound ? std::sqrt(X[l] * X[l] + Y[l] * Y[l]) * (1.0 + 2.0 * EPS) : 0.0;
        t[l] = Bound ? std::fabs(lo[0]) : 0.0, w[l] = 0.0;
    }
    for (size_t k = 1; k <= n; ++k) {
        const double hk = hi[k], lk = lo[k];
#pragma omp simd
        for (int l = 0; l < EW; ++l) {
            const double x_ = X[l], y_ = Y[l];
            if constexpr (Real) {
                const double a1 = dr[l] * x_, b1 = std::fma(dr[l], x_, -a1);
                double ndr, g3;
                two_sum(a1, sr[l], ndr, g3);
                const double ner = er[l] * x_ + ((b1 + g3) + cr[l]);
                const double m1 = sr[l] * x_, e1 = std::fma(sr[l], x_, -m1);
                double nsr, f3;
                two_sum(m1, hk, nsr, f3);
                const double ncr = cr[l] * x_ + ((e1 + f3) + lk);
                if (Bound) {
                    w[l] = w[l] * az[l] + ((std::fabs(b1) + std::fabs(g3)) + 2.0 * t[l]);
                    t[l] = t[l] * az[l] + ((std::fabs(e1) + std::fabs(f3)) + std::fabs(lk));
                }
                dr[l] = ndr, er[l] = ner, sr[l] = nsr, cr[l] = ncr;
            } else {
                const double a1 = dr[l] * x_, a2 = di[l] * y_, a3 = dr[l] * y_, a4 = di[l] * x_;
                const double b1 = std::fma(dr[l], x_, -a1), b2 = std::fma(di[l], y_, -a2);
                const double b3 = std::fma(dr[l], y_, -a3), b4 = std::fma(di[l], x_, -a4);
                double h1, g1, h2, g2, ndr, g3, ndi, g4;
                two_sum(a1, -a2, h1, g1);
                two_sum(h1, sr[l], ndr, g3);
                two_sum(a3, a4, h2, g2);
                two_sum(h2, si[l], ndi, g4);
                const double ner = (er[l] * x_ - ei[l] * y_) + (((b1 - b2) + (g1 + g3)) + cr[l]);
                const double nei = (er[l] * y_ + ei[l] * x_) + (((b3 + b4) + (g2 + g4)) + ci[l]);
                const double m1 = sr[l] * x_, m2 = si[l] * y_, m3 = sr[l] * y_, m4 = si[l] * x_;
                const double e1 = std::fma(sr[l], x_, -m1), e2 = std::fma(si[l], y_, -m2);
                const double e3 = std::fma(sr[l], y_, -m3), e4 = std::fma(si[l], x_, -m4);
                double h3, f1, nsr, f3, nsi, f2;
                two_sum(m1, -m2, h3, f1);
                two_sum(h3, hk, nsr, f3);
                two_sum(m3, m4, nsi, f2);
                const double ncr = (cr[l] * x_ - ci[l] * y_) + (((e1 - e2) + (f1 + f3)) + lk);
                const double nci = (cr[l] * y_ + ci[l] * x_) + ((e3 + e4) + f2);
                if (Bound) {
                    const double tq = std::fabs(b1) + std::fabs(b2) + std::fabs(g1) + std::fabs(g3) + std::fabs(b3) +
                                      std::fabs(b4) + std::fabs(g2) + std::fabs(g4);
                    w[l] = w[l] * az[l] + (tq + 2.0 * t[l]);
                    const double tp = std::fabs(e1) + std::fabs(e2) + std::fabs(f1) + std::fabs(f3) + std::fabs(lk) +
                                      std::fabs(e3) + std::fabs(e4) + std::fabs(f2);
                    t[l] = t[l] * az[l] + tp;
                }
                dr[l] = ndr, di[l] = ndi, er[l] = ner, ei[l] = nei;
                sr[l] = nsr, si[l] = nsi, cr[l] = ncr, ci[l] = nci;
            }
        }
    }
    for (int l = 0; l < EW; ++l) {
        pr[l] = sr[l] + cr[l], pi[l] = Real ? 0.0 : si[l] + ci[l];
        qr[l] = dr[l] + er[l], qi[l] = Real ? 0.0 : di[l] + ei[l];
    }
    if (Bound) {
        const double g = (4.0 * (double)n + 8.0) * EPS;
        double rp[EW] = {}, rq[EW] = {};
        if (P.has_res)
            for (size_t k = 0; k <= n; ++k)
                for (int l = 0; l < EW; ++l) { rq[l] = rq[l] * az[l] + rp[l]; rp[l] = rp[l] * az[l] + P.res[k]; }
        for (int l = 0; l < EW; ++l) {
            errp[l] = 1.01 * (EPS * std::sqrt(pr[l] * pr[l] + pi[l] * pi[l]) + g * t[l] + rp[l] + 0.0);
            errq[l] = 1.01 * (EPS * std::sqrt(qr[l] * qr[l] + qi[l] * qi[l]) + g * w[l] + rq[l] + 0.0);
        }
    }
}

template <bool Bound>
inline void comp_eval(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi,
                      double* errp = nullptr, double* errq = nullptr, int* sexp = nullptr) {
    if (needs_scaling(P.deg(), x, y)) comp_eval_impl<Bound, true>(P, x, y, pr, pi, qr, qi, errp, errq, sexp);
    else comp_eval_impl<Bound, false>(P, x, y, pr, pi, qr, qi, errp, errq, sexp);
}

inline double horner_abs_r(const PolyR& a, double az) {
    double r = 0;
    for (double c : a) r = r * az + std::fabs(c);
    return r;
}
// Newton correction p/p'; uses the reversed polynomial for |z| > 1 (no overflow, better scaling).
inline cd newton_corr(const PolyR& a, cd z, bool& at_noise) {
    const size_t n = a.size() - 1;
    const double az = std::abs(z);
    if (az <= 1.0) {
        cd p = a[0], d = 0;
        double t = std::fabs(a[0]);
        for (size_t k = 1; k <= n; ++k) { d = d * z + p; p = p * z + a[k]; t = t * az + std::fabs(a[k]); }
        at_noise = std::abs(p) <= 4.0 * (double)(n + 1) * EPS * t;
        return p / d;
    }
    cd w = 1.0 / z, q = a[n], dq = 0;
    double aw = 1.0 / az, t = std::fabs(a[n]);
    for (size_t k = n; k-- > 0;) { dq = dq * w + q; q = q * w + a[k]; t = t * aw + std::fabs(a[k]); }
    at_noise = std::abs(q) <= 4.0 * (double)(n + 1) * EPS * t;
    return z * q / ((double)n * q - w * dq);
}

// newton_corr at EW points at once: per lane exactly the scalar recurrence (|z| <= 1: Horner in z,
// else in 1/z on the reversed coefficients), the lanes in SIMD registers.
inline void newton_corr_w(const PolyR& a, const double* zr, const double* zi, cd* N, bool* at_noise) {
    const size_t n = a.size() - 1;
    alignas(32) double wr[EW], wi[EW], aw[EW], sel[EW], pr[EW], pi[EW], dr[EW], di[EW], t[EW];
    for (int l = 0; l < EW; ++l) {
        const cd z(zr[l], zi[l]);
        const double az = std::abs(z);
        sel[l] = az <= 1.0 ? 1.0 : 0.0;
        if (sel[l] != 0.0) wr[l] = zr[l], wi[l] = zi[l], aw[l] = az, pr[l] = a[0], t[l] = std::fabs(a[0]);
        else {
            const cd w = 1.0 / z;
            wr[l] = w.real(), wi[l] = w.imag(), aw[l] = 1.0 / az, pr[l] = a[n], t[l] = std::fabs(a[n]);
        }
        pi[l] = dr[l] = di[l] = 0.0;
    }
    for (size_t k = 1; k <= n; ++k) {
        const double ci = a[k], co = a[n - k];
#pragma omp simd
        for (int l = 0; l < EW; ++l) {
            const double c = sel[l] != 0.0 ? ci : co;
            const double ndr = (dr[l] * wr[l] - di[l] * wi[l]) + pr[l], ndi = (dr[l] * wi[l] + di[l] * wr[l]) + pi[l];
            const double npr = (pr[l] * wr[l] - pi[l] * wi[l]) + c, npi = pr[l] * wi[l] + pi[l] * wr[l];
            t[l] = t[l] * aw[l] + std::fabs(c);
            dr[l] = ndr, di[l] = ndi, pr[l] = npr, pi[l] = npi;
        }
    }
    for (int l = 0; l < EW; ++l) {
        const cd p(pr[l], pi[l]), d(dr[l], di[l]);
        at_noise[l] = std::abs(p) <= 4.0 * (double)(n + 1) * EPS * t[l];
        if (sel[l] != 0.0) N[l] = p / d;
        else N[l] = cd(zr[l], zi[l]) * p / ((double)n * p - cd(wr[l], wi[l]) * d);
    }
}

// Bini's starting points: circles whose radii come from the Newton polygon of |coefficients|.
void bini_start(const PolyR& a, std::vector<cd>& z) {
    const int n = (int)a.size() - 1;
    std::vector<double> lg(n + 1);
    for (int k = 0; k <= n; ++k) {  // ascending powers: coeff of x^k is a[n-k]
        double c = std::fabs(a[n - k]);
        lg[k] = c > 0 ? std::log(c) : -INFINITY;
    }
    std::vector<int> h;
    for (int k = 0; k <= n; ++k) {
        if (!std::isfinite(lg[k])) continue;
        while (h.size() >= 2) {
            int k0 = h[h.size() - 2], k1 = h.back();
            if ((double)(k1 - k0) * (lg[k] - lg[k0]) - (lg[k1] - lg[k0]) * (double)(k - k0) >= 0) h.pop_back();
            else break;
        }
        h.push_back(k);
    }
    z.clear();
    for (size_t s = 0; s + 1 < h.size(); ++s) {
        int m = h[s + 1] - h[s];
        double r = std::exp((lg[h[s]] - lg[h[s + 1]]) / m);
        double offset = 2.0 * M_PI * (double)z.size() / n + 0.7;  // per circle, avoids aligned starts
        for (int l = 0; l < m; ++l) {
            double ang = 2.0 * M_PI * l / m + offset;
            z.emplace_back(r * std::cos(ang), r * std::sin(ang));
        }
    }
}

// K-fold compensated Horner (K >= 3): p and p' at a double point z, ~u^K-accurate (plus u|p|), with a
// rigorous bound. Level 1 is Horner with error-free transformations; the exact rounding errors of
// level j are the coefficients of the level j+1 polynomial (evaluated the same way); level K is plain
// Horner with a running magnitude bound. Real == true: z on the real axis (4x cheaper).
// Scaled: as in comp_eval. lo != nullptr: also the low parts (p = pr + lo[0], ...; p' = qr + lo[2], ...) and
// bounds for these double-doubles (no final-rounding term u|p|).
template <int K, bool Real, bool Scaled>
void eval_k_impl(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi, double& errp,
                 double& errq, int* sexp, double* plo) {
    constexpr int M = 3 * K + 8;  // max inputs per chain and level (grows by 3 per level)
    const size_t n = P.deg();
    const double* hi = P.hi.data();
    const double* lo = P.lo.data();
    double cr[K + 1] = {}, ci[K + 1] = {}, dr[K + 1] = {}, di[K + 1] = {};
    cr[1] = hi[0];
    cr[2] = lo[0];
    const double az = (Real ? std::fabs(x) : std::sqrt(x * x + y * y)) * (1.0 + 2.0 * EPS);
    double t = K == 2 ? std::fabs(lo[0]) : 0.0, w = 0.0;
    double sc = 1.0, ut = 0.0, uw = 0.0, rp = 0.0, rq = 0.0;
    int S = 0;
    if (Scaled && P.has_res) rp = P.res[0];
    int mmax = 0;
    double inR[M], inI[M], dR[M], dI[M], oR[M], oI[M], odR[M], odI[M];
    for (size_t k = 1; k <= n; ++k) {
        int nR = 1, nI = 0, ndR = 0, ndI = 0;
        inR[0] = Scaled ? hi[k] * sc : hi[k];
        for (int j = 1; j < K; ++j) {
            int mR = 0, mI = 0, mdR = 0, mdI = 0;
            double h, e;
            // p' chain: D_j <- D_j z + (errors of D_{j-1}) + C_j(old)
            {
                const double a1 = dr[j] * x;
                odR[mdR++] = std::fma(dr[j], x, -a1);
                h = a1;
                if (!Real) {
                    const double a2 = di[j] * y;
                    odR[mdR++] = -std::fma(di[j], y, -a2);
                    two_sum(h, -a2, h, e); odR[mdR++] = e;
                }
                two_sum(h, cr[j], h, e); odR[mdR++] = e;
                for (int q = 0; q < ndR; ++q) { two_sum(h, dR[q], h, e); odR[mdR++] = e; }
                const double ndr = h;
                double ndi = 0.0;
                if (!Real) {
                    const double a3 = dr[j] * y, a4 = di[j] * x;
                    odI[mdI++] = std::fma(dr[j], y, -a3);
                    odI[mdI++] = std::fma(di[j], x, -a4);
                    two_sum(a3, a4, h, e); odI[mdI++] = e;
                    two_sum(h, ci[j], h, e); odI[mdI++] = e;
                    for (int q = 0; q < ndI; ++q) { two_sum(h, dI[q], h, e); odI[mdI++] = e; }
                    ndi = h;
                }
                dr[j] = ndr; di[j] = ndi;
            }
            // p chain: C_j <- C_j z + (errors of C_{j-1}); level 1 adds the coefficient hi[k]
            {
                const double m1 = cr[j] * x;
                oR[mR++] = std::fma(cr[j], x, -m1);
                h = m1;
                if (!Real) {
                    const double m2 = ci[j] * y;
                    oR[mR++] = -std::fma(ci[j], y, -m2);
                    two_sum(h, -m2, h, e); oR[mR++] = e;
                }
                for (int q = 0; q < nR; ++q) { two_sum(h, inR[q], h, e); oR[mR++] = e; }
                const double ncr = h;
                double nci = 0.0;
                if (!Real) {
                    const double m3 = cr[j] * y, m4 = ci[j] * x;
                    oI[mI++] = std::fma(cr[j], y, -m3);
                    oI[mI++] = std::fma(ci[j], x, -m4);
                    two_sum(m3, m4, h, e); oI[mI++] = e;
                    for (int q = 0; q < nI; ++q) { two_sum(h, inI[q], h, e); oI[mI++] = e; }
                    nci = h;
                }
                cr[j] = ncr; ci[j] = nci;
            }
            if (j == 1) oR[mR++] = Scaled ? lo[k] * sc : lo[k];
            std::copy(oR, oR + mR, inR); nR = mR;
            std::copy(oI, oI + mI, inI); nI = mI;
            std::copy(odR, odR + mdR, dR); ndR = mdR;
            std::copy(odI, odI + mdI, dI); ndI = mdI;
        }
        // level K: plain Horner + magnitude bound
        double sR = 0, sI = 0, sdR = 0, sdI = 0, aR = 0, adR = 0;
        for (int q = 0; q < nR; ++q) sR += inR[q], aR += std::fabs(inR[q]);
        for (int q = 0; q < nI; ++q) sI += inI[q], aR += std::fabs(inI[q]);
        for (int q = 0; q < ndR; ++q) sdR += dR[q], adR += std::fabs(dR[q]);
        for (int q = 0; q < ndI; ++q) sdI += dI[q], adR += std::fabs(dI[q]);
        mmax = std::max(mmax, std::max(std::max(nR, nI), std::max(ndR, ndI)) + 1);
        const double ndr = Real ? dr[K] * x + (sdR + cr[K]) : (dr[K] * x - di[K] * y) + (sdR + cr[K]);
        const double ndi = Real ? 0.0 : (dr[K] * y + di[K] * x) + (sdI + ci[K]);
        const double ncr = Real ? cr[K] * x + sR : (cr[K] * x - ci[K] * y) + sR;
        const double nci = Real ? 0.0 : (cr[K] * y + ci[K] * x) + sI;
        w = w * az + (adR + 2.0 * t);
        t = t * az + aR;
        dr[K] = ndr; di[K] = ndi; cr[K] = ncr; ci[K] = nci;
        if constexpr (Scaled) {
            if (P.has_res) { rq = rq * az + rp; rp = rp * az + P.res[k] * sc; }
            uw = uw * az + (UNDERFLOW_STEP + 2.0 * ut);
            ut = ut * az + UNDERFLOW_STEP;
            if (std::fabs(cr[1]) + std::fabs(ci[1]) + std::fabs(dr[1]) + std::fabs(di[1]) + t + w > 0x1p500) {
                for (int l = 1; l <= K; ++l) cr[l] *= SCALE_DOWN, ci[l] *= SCALE_DOWN, dr[l] *= SCALE_DOWN, di[l] *= SCALE_DOWN;
                for (double* v : {&t, &w, &ut, &uw, &rp, &rq}) *v *= SCALE_DOWN;
                ut += UNDERFLOW_STEP, uw += UNDERFLOW_STEP;
                sc *= SCALE_DOWN;
                S += 500;
            }
        }
    }
    // accurate sum of the K level values (K-1 VecSum passes, smallest level first)
    // also the exact remainder lo of the final addition and sum |v[2..K]| (bounds the rounding of r)
    auto sumk = [](double* v, double& abs_sum, double& lo_, double& rab) {
        abs_sum = 0;
        for (int l = 1; l <= K; ++l) abs_sum += std::fabs(v[l]);
        for (int pass = 0; pass + 1 < K; ++pass)
            for (int l = K; l > 1; --l) two_sum(v[l - 1], v[l], v[l - 1], v[l]);
        double r = 0;
        rab = 0;
        for (int l = K; l > 1; --l) r += v[l], rab += std::fabs(v[l]);
        double h;
        two_sum(v[1], r, h, lo_);
        return h;
    };
    double s1, s2, s3, s4, l1, l2 = 0.0, l3, l4 = 0.0, r1, r2 = 0.0, r3, r4 = 0.0;
    pr = sumk(cr, s1, l1, r1); pi = Real ? 0.0 : sumk(ci, s2, l2, r2);
    qr = sumk(dr, s3, l3, r3); qi = Real ? 0.0 : sumk(di, s4, l4, r4);
    if (Real) s2 = s4 = 0;
    if (!Scaled && P.has_res)
        for (size_t k = 0; k <= n; ++k) { rq = rq * az + rp; rp = rp * az + P.res[k]; }
    const double g = (4.0 * (double)n + 2.0 * mmax + 8.0) * EPS;
    const double sk = std::pow(K * EPS, K);
    // underflow losses: the level-K Horner adds up to M inputs per step
    if (plo) {  // p = pr + plo[0] etc.: no final rounding, only that of r ((K - 2) additions)
        plo[0] = l1, plo[1] = l2, plo[2] = l3, plo[3] = l4;
        const double rr = (double)(K - 2) * EPS;
        errp = 1.01 * (g * t + sk * (s1 + s2) + rr * (r1 + r2) + rp + (double)M * ut);
        errq = 1.01 * (g * w + sk * (s3 + s4) + rr * (r3 + r4) + rq + (double)M * uw);
    } else {
        errp = 1.01 * (EPS * std::sqrt(pr * pr + pi * pi) + g * t + sk * (s1 + s2) + rp + (double)M * ut);
        errq = 1.01 * (EPS * std::sqrt(qr * qr + qi * qi) + g * w + sk * (s3 + s4) + rq + (double)M * uw);
    }
    if (sexp) *sexp = S;
}
// eval_k_impl<3> with the per-level error-term counts fixed at compile time (registers instead of
// arrays): the same operations in the same order, hence bitwise the same results and bounds.
template <int N>
struct Terms {
    double v[N];
};
template <bool Real, bool Scaled>
void eval3_impl(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi, double& errp,
                double& errq, int* sexp, double* plo) {
    constexpr int K = 3;
    constexpr int M = 3 * K + 8;
    // level-1 outputs (inputs of level 2) and level-2 outputs (inputs of level 3)
    constexpr int N1R = Real ? 3 : 5, N1I = Real ? 0 : 3, N1DR = Real ? 2 : 4, N1DI = Real ? 0 : 4;
    constexpr int N2R = (Real ? 1 : 3) + N1R, N2I = Real ? 0 : 3 + N1I;
    constexpr int N2DR = (Real ? 2 : 4) + N1DR, N2DI = Real ? 0 : 4 + N1DI;
    constexpr int MM = std::max(std::max(N2R, N2I), std::max(N2DR, N2DI)) + 1;
    const size_t n = P.deg();
    const double* hi = P.hi.data();
    const double* lo = P.lo.data();
    double c1r = hi[0], c1i = 0.0, c2r = lo[0], c2i = 0.0, c3r = 0.0, c3i = 0.0;
    double d1r = 0.0, d1i = 0.0, d2r = 0.0, d2i = 0.0, d3r = 0.0, d3i = 0.0;
    const double az = (Real ? std::fabs(x) : std::sqrt(x * x + y * y)) * (1.0 + 2.0 * EPS);
    double t = 0.0, w = 0.0;
    double sc = 1.0, ut = 0.0, uw = 0.0, rp = 0.0, rq = 0.0;
    int S = 0;
    if (Scaled && P.has_res) rp = P.res[0];
    double i1R[N1R + 1], i1I[N1I + 1], i1D[N1DR + 1], i1E[N1DI + 1];
    double i2R[N2R + 1], i2I[N2I + 1], i2D[N2DR + 1], i2E[N2DI + 1];
    for (size_t k = 1; k <= n; ++k) {
        const double hk = Scaled ? hi[k] * sc : hi[k];
        double h, e;
        // ---- level 1: p' chain (uses the old c1), then p chain
        {
            int m = 0, mi = 0;
            const double a1 = d1r * x;
            i1D[m++] = std::fma(d1r, x, -a1);
            h = a1;
            if (!Real) {
                const double a2 = d1i * y;
                i1D[m++] = -std::fma(d1i, y, -a2);
                two_sum(h, -a2, h, e); i1D[m++] = e;
            }
            two_sum(h, c1r, h, e); i1D[m++] = e;
            const double ndr = h;
            double ndi = 0.0;
            if (!Real) {
                const double a3 = d1r * y, a4 = d1i * x;
                i1E[mi++] = std::fma(d1r, y, -a3);
                i1E[mi++] = std::fma(d1i, x, -a4);
                two_sum(a3, a4, h, e); i1E[mi++] = e;
                two_sum(h, c1i, h, e); i1E[mi++] = e;
                ndi = h;
            }
            d1r = ndr, d1i = ndi;
        }
        {
            int m = 0, mi = 0;
            const double m1 = c1r * x;
            i1R[m++] = std::fma(c1r, x, -m1);
            h = m1;
            if (!Real) {
                const double m2 = c1i * y;
                i1R[m++] = -std::fma(c1i, y, -m2);
                two_sum(h, -m2, h, e); i1R[m++] = e;
            }
            two_sum(h, hk, h, e); i1R[m++] = e;
            const double ncr = h;
            double nci = 0.0;
            if (!Real) {
                const double m3 = c1r * y, m4 = c1i * x;
                i1I[mi++] = std::fma(c1r, y, -m3);
                i1I[mi++] = std::fma(c1i, x, -m4);
                two_sum(m3, m4, h, e); i1I[mi++] = e;
                nci = h;
            }
            c1r = ncr, c1i = nci;
            i1R[m++] = Scaled ? lo[k] * sc : lo[k];
        }
        // ---- level 2
        {
            int m = 0, mi = 0;
            const double a1 = d2r * x;
            i2D[m++] = std::fma(d2r, x, -a1);
            h = a1;
            if (!Real) {
                const double a2 = d2i * y;
                i2D[m++] = -std::fma(d2i, y, -a2);
                two_sum(h, -a2, h, e); i2D[m++] = e;
            }
            two_sum(h, c2r, h, e); i2D[m++] = e;
#pragma GCC unroll 8
            for (int q = 0; q < N1DR; ++q) { two_sum(h, i1D[q], h, e); i2D[m++] = e; }
            const double ndr = h;
            double ndi = 0.0;
            if (!Real) {
                const double a3 = d2r * y, a4 = d2i * x;
                i2E[mi++] = std::fma(d2r, y, -a3);
                i2E[mi++] = std::fma(d2i, x, -a4);
                two_sum(a3, a4, h, e); i2E[mi++] = e;
                two_sum(h, c2i, h, e); i2E[mi++] = e;
#pragma GCC unroll 8
                for (int q = 0; q < N1DI; ++q) { two_sum(h, i1E[q], h, e); i2E[mi++] = e; }
                ndi = h;
            }
            d2r = ndr, d2i = ndi;
        }
        {
            int m = 0, mi = 0;
            const double m1 = c2r * x;
            i2R[m++] = std::fma(c2r, x, -m1);
            h = m1;
            if (!Real) {
                const double m2 = c2i * y;
                i2R[m++] = -std::fma(c2i, y, -m2);
                two_sum(h, -m2, h, e); i2R[m++] = e;
            }
#pragma GCC unroll 8
            for (int q = 0; q < N1R; ++q) { two_sum(h, i1R[q], h, e); i2R[m++] = e; }
            const double ncr = h;
            double nci = 0.0;
            if (!Real) {
                const double m3 = c2r * y, m4 = c2i * x;
                i2I[mi++] = std::fma(c2r, y, -m3);
                i2I[mi++] = std::fma(c2i, x, -m4);
                two_sum(m3, m4, h, e); i2I[mi++] = e;
#pragma GCC unroll 8
                for (int q = 0; q < N1I; ++q) { two_sum(h, i1I[q], h, e); i2I[mi++] = e; }
                nci = h;
            }
            c2r = ncr, c2i = nci;
        }
        // ---- level 3: plain Horner + magnitude bound
        double sR = 0, sI = 0, sdR = 0, sdI = 0, aR = 0, adR = 0;
#pragma GCC unroll 8
        for (int q = 0; q < N2R; ++q) sR += i2R[q], aR += std::fabs(i2R[q]);
#pragma GCC unroll 8
        for (int q = 0; q < N2I; ++q) sI += i2I[q], aR += std::fabs(i2I[q]);
#pragma GCC unroll 8
        for (int q = 0; q < N2DR; ++q) sdR += i2D[q], adR += std::fabs(i2D[q]);
#pragma GCC unroll 8
        for (int q = 0; q < N2DI; ++q) sdI += i2E[q], adR += std::fabs(i2E[q]);
        const double ndr = Real ? d3r * x + (sdR + c3r) : (d3r * x - d3i * y) + (sdR + c3r);
        const double ndi = Real ? 0.0 : (d3r * y + d3i * x) + (sdI + c3i);
        const double ncr = Real ? c3r * x + sR : (c3r * x - c3i * y) + sR;
        const double nci = Real ? 0.0 : (c3r * y + c3i * x) + sI;
        w = w * az + (adR + 2.0 * t);
        t = t * az + aR;
        d3r = ndr, d3i = ndi, c3r = ncr, c3i = nci;
        if constexpr (Scaled) {
            if (P.has_res) { rq = rq * az + rp; rp = rp * az + P.res[k] * sc; }
            uw = uw * az + (UNDERFLOW_STEP + 2.0 * ut);
            ut = ut * az + UNDERFLOW_STEP;
            if (std::fabs(c1r) + std::fabs(c1i) + std::fabs(d1r) + std::fabs(d1i) + t + w > 0x1p500) {
                for (double* v : {&c1r, &c1i, &c2r, &c2i, &c3r, &c3i, &d1r, &d1i, &d2r, &d2i, &d3r, &d3i}) *v *= SCALE_DOWN;
                for (double* v : {&t, &w, &ut, &uw, &rp, &rq}) *v *= SCALE_DOWN;
                ut += UNDERFLOW_STEP, uw += UNDERFLOW_STEP;
                sc *= SCALE_DOWN;
                S += 500;
            }
        }
    }
    const int mmax = n ? MM : 0;
    // also the exact remainder lo of the final addition and sum |v[2..K]| (bounds the rounding of r)
    auto sumk = [](double* v, double& abs_sum, double& lo_, double& rab) {
        abs_sum = 0;
        for (int l = 1; l <= K; ++l) abs_sum += std::fabs(v[l]);
        for (int pass = 0; pass + 1 < K; ++pass)
            for (int l = K; l > 1; --l) two_sum(v[l - 1], v[l], v[l - 1], v[l]);
        double r = 0;
        rab = 0;
        for (int l = K; l > 1; --l) r += v[l], rab += std::fabs(v[l]);
        double h;
        two_sum(v[1], r, h, lo_);
        return h;
    };
    double cr[K + 1] = {0, c1r, c2r, c3r}, ci[K + 1] = {0, c1i, c2i, c3i};
    double dr[K + 1] = {0, d1r, d2r, d3r}, di[K + 1] = {0, d1i, d2i, d3i};
    double s1, s2, s3, s4, l1, l2 = 0.0, l3, l4 = 0.0, r1, r2 = 0.0, r3, r4 = 0.0;
    pr = sumk(cr, s1, l1, r1); pi = Real ? 0.0 : sumk(ci, s2, l2, r2);
    qr = sumk(dr, s3, l3, r3); qi = Real ? 0.0 : sumk(di, s4, l4, r4);
    if (Real) s2 = s4 = 0;
    if (!Scaled && P.has_res)
        for (size_t k = 0; k <= n; ++k) { rq = rq * az + rp; rp = rp * az + P.res[k]; }
    const double g = (4.0 * (double)n + 2.0 * mmax + 8.0) * EPS;
    const double sk = std::pow(K * EPS, K);
    if (plo) {  // p = pr + plo[0] etc.: no final rounding, only that of r ((K - 2) additions)
        plo[0] = l1, plo[1] = l2, plo[2] = l3, plo[3] = l4;
        const double rr = (double)(K - 2) * EPS;
        errp = 1.01 * (g * t + sk * (s1 + s2) + rr * (r1 + r2) + rp + (double)M * ut);
        errq = 1.01 * (g * w + sk * (s3 + s4) + rr * (r3 + r4) + rq + (double)M * uw);
    } else {
        errp = 1.01 * (EPS * std::sqrt(pr * pr + pi * pi) + g * t + sk * (s1 + s2) + rp + (double)M * ut);
        errq = 1.01 * (EPS * std::sqrt(qr * qr + qi * qi) + g * w + sk * (s3 + s4) + rq + (double)M * uw);
    }
    if (sexp) *sexp = S;
}
template <int K, bool Real>
inline void eval_k(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi, double& errp,
                   double& errq, int* sexp = nullptr, double* lo = nullptr) {
    if constexpr (K == 3) {
        if (needs_scaling(P.deg(), x, y)) eval3_impl<Real, true>(P, x, y, pr, pi, qr, qi, errp, errq, sexp, lo);
        else eval3_impl<Real, false>(P, x, y, pr, pi, qr, qi, errp, errq, sexp, lo);
    } else {
        if (needs_scaling(P.deg(), x, y)) eval_k_impl<K, Real, true>(P, x, y, pr, pi, qr, qi, errp, errq, sexp, lo);
        else eval_k_impl<K, Real, false>(P, x, y, pr, pi, qr, qi, errp, errq, sexp, lo);
    }
}

// p, p' and their error bounds with evaluation level L: 2 = compensated (u^2), 3, 4 = K-fold.
// Values may come back scaled by 2^-*sexp (huge |z|; see comp_eval).
template <int L>
inline void eval_lvl(const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi, double& ep,
                     double& eq, int* sexp = nullptr) {
    if constexpr (L == 2) comp_eval<true>(P, x, y, pr, pi, qr, qi, &ep, &eq, sexp);
    else if (y == 0.0) eval_k<L, true>(P, x, 0.0, pr, pi, qr, qi, ep, eq, sexp);
    else eval_k<L, false>(P, x, y, pr, pi, qr, qi, ep, eq, sexp);
}

// sum_{j != i} 1 / (z_i - z_j), vectorized over j (1/d = conj(d) / |d|^2, no library complex division).
inline cd aberth_sum(const double* xr, const double* xi, size_t n, size_t i) {
    const double x = xr[i], y = xi[i];
    double a = 0.0, b = 0.0;
#pragma omp simd reduction(+ : a, b)
    for (size_t j = 0; j < n; ++j) {
        const double dx = x - xr[j], dy = y - xi[j];
        const double inv = j == i ? 0.0 : 1.0 / (dx * dx + dy * dy);
        a += dx * inv;
        b -= dy * inv;
    }
    return cd(a, b);
}

// Aberth-Ehrlich (Gauss-Seidel) on points (xr, xi) with evaluation level L (1 = double). Each root
// stops at the noise floor of that level; higher levels separate roots that are indistinguishable at
// lower ones.
template <int L>
bool aberth(const CPoly& P, std::vector<double>& xr, std::vector<double>& xi, int maxit) {
    const size_t n = P.deg();
    std::vector<char> done(n, 0);
    size_t left = n;
    // Gauss-Seidel in groups of EW: p(z_i) depends on z_i only, so evaluating the next EW active points
    // together before updating them in order is the same iteration
    size_t g[EW];
    cd Ng[EW];
    bool nz[EW];
    alignas(32) double gx[EW], gy[EW];
    for (int it = 0; it < maxit && left; ++it) {
        for (size_t j = 0; j < n;) {
            int m = 0;
            for (; j < n && m < EW; ++j)
                if (!done[j]) g[m++] = j;
            if (!m) break;
            if constexpr (L == 1) {
                for (int l = 0; l < EW; ++l) gx[l] = xr[g[l < m ? l : 0]], gy[l] = xi[g[l < m ? l : 0]];
                newton_corr_w(P.hi, gx, gy, Ng, nz);
            } else {
                for (int l = 0; l < m; ++l) {
                    double pr, pi, qr, qi, ep, eq;
                    eval_lvl<L>(P, xr[g[l]], xi[g[l]], pr, pi, qr, qi, ep, eq);
                    nz[l] = std::hypot(pr, pi) <= ep;
                    Ng[l] = cd(pr, pi) / cd(qr, qi);
                }
            }
            for (int l = 0; l < m; ++l) {
                const size_t i = g[l];
                const cd z(xr[i], xi[i]);
                const bool noise = nz[l];
                const cd N = Ng[l];
                if (noise) { done[i] = 1; --left; continue; }
                const cd corr = N / (1.0 - N * aberth_sum(xr.data(), xi.data(), n, i));
                if (!std::isfinite(corr.real()) || !std::isfinite(corr.imag())) {
#ifdef MAS_DEBUG
                    fprintf(stderr, "aberth<%d> nonfinite at %zu (%.17g,%.6e)\n", L, i, xr[i], xi[i]);
#endif
                    return false;
                }
                xr[i] -= corr.real();
                xi[i] -= corr.imag();
                // converged: correction below an ulp (z may not even move); polishing does the rest
                if (std::abs(corr) <= EPS * std::abs(z)) { done[i] = 1; --left; }
            }
        }
    }
#ifdef MAS_DEBUG
    if (left) {
        fprintf(stderr, "aberth<%d> left=%zu:", L, left);
        for (size_t i = 0; i < n; ++i) if (!done[i]) fprintf(stderr, " (%.17g,%.6e)", xr[i], xi[i]);
        fprintf(stderr, "\n");
    }
#endif
    return left == 0;
}

// distance from point i to the nearest other point
inline double nn_dist(const std::vector<double>& xr, const std::vector<double>& xi, size_t i) {
    const size_t n = xr.size();
    const double x = xr[i], y = xi[i];
    double m = INFINITY;
#pragma omp simd reduction(min : m)
    for (size_t j = 0; j < n; ++j) {
        const double dx = x - xr[j], dy = y - xi[j], d = dx * dx + dy * dy;
        m = std::min(m, j == i ? INFINITY : d);
    }
    return std::sqrt(m);
}
// Real polynomial: a point whose imaginary part is negligible against the distance to its nearest
// neighbour (a conjugate pair has the two points 2|y| apart) is moved onto the real axis, where the
// K-fold evaluation is 4x cheaper. A heuristic only: the certificate decides what is real.
inline bool snap_real(std::vector<double>& xr, std::vector<double>& xi, size_t i) {
    if (xi[i] == 0.0) return true;
    if (std::fabs(xi[i]) <= 0x1p-10 * nn_dist(xr, xi, i)) { xi[i] = 0.0; return true; }
    return false;
}

// aberth<L> on the points listed in idx only (the others stay fixed). Level >= 3: points snapped to the
// real axis stay there (real part of the correction only).
template <int L>
bool aberth_subset(const CPoly& P, std::vector<double>& xr, std::vector<double>& xi, const std::vector<size_t>& idx,
                   int maxit) {
    const size_t n = P.deg();
    std::vector<char> done(idx.size(), 0);
    size_t left = idx.size();
    size_t g[EW];
    alignas(32) double gx[EW], gy[EW], o[6][EW];
    for (int it = 0; it < maxit && left; ++it) {
        for (size_t u = 0; u < idx.size();) {
            // level 2: the next EW active points evaluated together (Gauss-Seidel as in aberth)
            int m = 0;
            if constexpr (L == 2) {
                bool all_real = true;
                for (; u < idx.size() && m < EW; ++u) {
                    if (done[u]) continue;
                    g[m++] = u;
                    all_real = all_real && xi[idx[u]] == 0.0;
                }
                if (!m) break;
                for (int l = 0; l < EW; ++l) gx[l] = xr[idx[g[l < m ? l : 0]]], gy[l] = xi[idx[g[l < m ? l : 0]]];
                if (all_real) comp_eval_w<true, true>(P, gx, gy, o[0], o[1], o[2], o[3], o[4], o[5]);
                else comp_eval_w<true, false>(P, gx, gy, o[0], o[1], o[2], o[3], o[4], o[5]);
                for (int l = 0; l < m; ++l)
                    if (needs_scaling(n, gx[l], gy[l]))
                        eval_lvl<2>(P, gx[l], gy[l], o[0][l], o[1][l], o[2][l], o[3][l], o[4][l], o[5][l]);
            } else {
                for (; u < idx.size() && !m; ++u)
                    if (!done[u]) g[m++] = u;
                if (!m) break;
            }
            for (int l = 0; l < m; ++l) {
                const size_t t = g[l], i = idx[t];
                const bool real = L >= 3 && snap_real(xr, xi, i);
                const cd z(xr[i], xi[i]);
                double pr, pi, qr, qi, ep, eq;
                if constexpr (L == 2) pr = o[0][l], pi = o[1][l], qr = o[2][l], qi = o[3][l], ep = o[4][l], eq = o[5][l];
                else eval_lvl<L>(P, z.real(), z.imag(), pr, pi, qr, qi, ep, eq);
                (void)eq;
                if (std::hypot(pr, pi) <= ep) { done[t] = 1; --left; continue; }
                const cd N = cd(pr, pi) / cd(qr, qi);
                const cd corr = N / (1.0 - N * aberth_sum(xr.data(), xi.data(), n, i));
                if (!std::isfinite(corr.real()) || !std::isfinite(corr.imag())) return false;
                xr[i] -= corr.real();
                if (!real) xi[i] -= corr.imag();
                if (std::abs(real ? cd(corr.real(), 0.0) : corr) <= EPS * std::abs(z)) { done[t] = 1; --left; }
            }
        }
#ifdef MAS_DEBUG
        if (!left || it + 1 == maxit) fprintf(stderr, "  aberth_subset<%d> |idx|=%zu its=%d left=%zu\n", L, idx.size(), it + 1, left);
#endif
    }
    return left == 0;
}

// Newton at evaluation level L >= 2: limiting accuracy ~ u + u^L * cond instead of u * cond.
template <int L>
inline void polish(const CPoly& P, double& x, double& y) {
    double prev = INFINITY;
    for (int it = 0; it < 6; ++it) {
        double pr, pi, qr, qi, ep, eq;
        if constexpr (L == 2) comp_eval<false>(P, x, y, pr, pi, qr, qi);
        else eval_lvl<L>(P, x, y, pr, pi, qr, qi, ep, eq);
        const cd corr = cd(pr, pi) / cd(qr, qi);
        const double c = std::abs(corr);
        if (!(c < prev)) break;  // stagnation (or NaN): keep current iterate
        x -= corr.real();
        y -= corr.imag();
        prev = c;
        if (c <= EPS / 16 * std::hypot(x, y)) break;
    }
}

// polish<2> on the points idx[0..cnt), EW at a time (each lane runs exactly the scalar iteration; the
// lanes of a group are all real or all complex; points needing scaling are polished one by one)
template <bool Real>
inline void polish2_group(const CPoly& P, std::vector<double>& xr, std::vector<double>& xi, const size_t* g, int m) {
    alignas(32) double X[EW], Y[EW], pr[EW], pi[EW], qr[EW], qi[EW];
    double prev[EW];
    bool act[EW];
    for (int l = 0; l < EW; ++l) {
        const size_t i = g[l < m ? l : 0];
        X[l] = xr[i], Y[l] = Real ? 0.0 : xi[i], prev[l] = INFINITY, act[l] = l < m;
    }
    for (int it = 0; it < 6; ++it) {
        comp_eval_w<false, Real>(P, X, Y, pr, pi, qr, qi, nullptr, nullptr);
        bool any = false;
        for (int l = 0; l < m; ++l) {
            if (!act[l]) continue;
            if constexpr (Real) {
                const double corr = pr[l] / qr[l], c = std::fabs(corr);
                if (!(c < prev[l])) { act[l] = false; continue; }
                X[l] -= corr;
                prev[l] = c;
                if (c <= EPS / 16 * std::fabs(X[l])) act[l] = false;
            } else {
                const cd corr = cd(pr[l], pi[l]) / cd(qr[l], qi[l]);
                const double c = std::abs(corr);
                if (!(c < prev[l])) { act[l] = false; continue; }
                X[l] -= corr.real();
                Y[l] -= corr.imag();
                prev[l] = c;
                if (c <= EPS / 16 * std::hypot(X[l], Y[l])) act[l] = false;
            }
            any = any || act[l];
        }
        if (!any) break;
    }
    for (int l = 0; l < m; ++l) xr[g[l]] = X[l], xi[g[l]] = Real ? xi[g[l]] : Y[l];
}
template <class Idx>
void polish2_all(const CPoly& P, std::vector<double>& xr, std::vector<double>& xi, const Idx& idx) {
    size_t gr[EW], gc[EW];
    int mr = 0, mc = 0;
    for (size_t i : idx) {
        if (needs_scaling(P.deg(), xr[i], xi[i])) { polish<2>(P, xr[i], xi[i]); continue; }
        if (xi[i] == 0.0) {
            gr[mr++] = i;
            if (mr == EW) polish2_group<true>(P, xr, xi, gr, mr), mr = 0;
        } else {
            gc[mc++] = i;
            if (mc == EW) polish2_group<false>(P, xr, xi, gc, mc), mc = 0;
        }
    }
    if (mr) polish2_group<true>(P, xr, xi, gr, mr);
    if (mc) polish2_group<false>(P, xr, xi, gc, mc);
}
struct IotaRange {  // 0, 1, ..., n-1 without a vector
    size_t n;
    struct It { size_t i; size_t operator*() const { return i; } It& operator++() { ++i; return *this; } bool operator!=(const It& o) const { return i != o.i; } };
    It begin() const { return {0}; }
    It end() const { return {n}; }
};

enum CertResult { CERT_OK, CERT_NOT_ISOLATED, CERT_UNRESOLVED, CERT_HOPELESS };

// min over pairs of |z_i - z_j| / max(|z_i|, |z_j|)
inline double min_rel_dist(const std::vector<double>& xr, const std::vector<double>& xi) {
    const size_t n = xr.size();
    double m = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const double x = xr[i], y = xi[i], a2 = x * x + y * y;
#pragma omp simd reduction(min : m)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = x - xr[j], dy = y - xi[j];
            m = std::min(m, (dx * dx + dy * dy) / std::max(a2, xr[j] * xr[j] + xi[j] * xi[j]));
        }
    }
    return std::sqrt(m);
}

// Certificate for all n approximations at once.
// 1. Gerschgorin: roots of p are the eigenvalues of diag(z) - W 1^T with W_i = p(z_i) / (a_0 prod_{j!=i}
//    (z_i - z_j)); discs D(z_i, n|W_i|) that are pairwise disjoint hold exactly one root each.
// 2. Per-root bound: p'/p(z) = sum_j 1/(z - zeta_j) and |z_i - zeta_j| >= |z_i - z_j| - rad_j give
//    |z_i - zeta_i| <= |p| / (|p'| - |p| (n-1) / min_j(|z_i - z_j| - rad_j))  (~|p/p'|, no factor n).
// 3. Conjugate symmetry: a disc meeting the real axis whose mirror meets no other disc holds a real
//    root; a disc off the axis must mirror onto exactly one partner disc (the conjugate root).
// All bounds use |p|, |p'| enclosures from comp_eval<true> and are rounded outward with margin.
// Final correctly rounded estimate from an enclosure of p, p' at z (|p - p^| <= ep, |p' - q^| <= eq) for
// an isolated root zeta (all other roots at distance >= gap, |z - zeta| <= rad). With e = z - zeta and
// S = sum_{j != i} 1/(z - zeta_j), |S| <= (n-1)/gap: p/p' = e/(1 + eS), so the Newton step N = p^/q^
// gives |zeta - (z - N)| <= |N - p/p'| + delta^2 |S| / (1 - delta |S|), delta a rigorous bound on |e|.
// Each component of zeta is then enclosed in an interval around z - N, checked by rounds_to (a real root
// only needs its real part).
// (x, y) local coordinates around the real anchor oh + ol (+- oe): ox is then the absolute real part.
// Optional second-order (Aberth) correction: e = N / (1 - N S) exactly, so with an estimate S^ of S
// (|S^ - S| <= ds, from the other certified discs) z - N^/(1 - N^ S^) is off by ~ delta^2 ds instead of
// delta^2 |S|: past the double-centre limit (delta ~ ulp|z|) for a root with a tiny component.
struct AbCorr {
    double sr = 0.0, si = 0.0, ds = INFINITY;
};
// lo != nullptr: (pr, pi, qr, qi) + lo are double-double values of p, p' (ep, eq bound their errors) and N is
// formed in double-double (N0 = p_h / q_h, then (p - N0 q) / q_h): no u|N| rounding, so the step resolves
// components far below ulp(|N|) (e.g. the real part of a root near the imaginary axis).
inline bool resolve_point(double x, double y, double pr, double pi, double qr, double qi, double ep, double eq,
                          size_t n, double gap, double rad, bool real, double& ox, double& oy, double oh = 0.0,
                          double ol = 0.0, double oe = 0.0, const AbCorr* ac = nullptr, const double* lo = nullptr) {
    const double pa = std::sqrt(pr * pr + pi * pi), qa = std::sqrt(qr * qr + qi * qi);
    const double pu = (pa + ep) * (1.0 + 4.0 * EPS), dl = qa * (1.0 - 4.0 * EPS) - eq;
    const double s = (double)(n - 1) / gap * (1.0 + 4.0 * EPS);
    const double den = dl - pu * s;
    if (!(gap > 0.0 && den > 0.0)) return false;
    const double delta = std::min(rad, pu / den * (1.0 + 8.0 * EPS));
    const double t = delta * s;
    if (!(t < 0.5)) return false;
    const double den2 = qr * qr + qi * qi;  // N = p q^* / |q|^2
    double nr = (pr * qr + pi * qi) / den2, ni = (pi * qr - pr * qi) / den2;
    double lr = 0.0, li = 0.0;  // the step is (nr + lr) + i (ni + li)
    double na = std::sqrt(nr * nr + ni * ni), eN;
    if (lo) {
        // R = p - N0 q with the products split exactly (fma); tail terms are O(u |p|)
        const double h1 = nr * qr, e1 = std::fma(nr, qr, -h1), h2 = ni * qi, e2 = std::fma(ni, qi, -h2);
        const double h3 = nr * qi, e3 = std::fma(nr, qi, -h3), h4 = ni * qr, e4 = std::fma(ni, qr, -h4);
        double a, ea, b, eb;
        two_sum(pr, -h1, a, ea);
        two_sum(a, h2, b, eb);
        const double Rr = b + (((ea + eb) - e1 + e2) + ((lo[0] - nr * lo[2]) + ni * lo[3]));
        two_sum(pi, -h3, a, ea);
        two_sum(a, -h4, b, eb);
        const double Ri = b + (((ea + eb) - e3 - e4) + ((lo[1] - nr * lo[3]) - ni * lo[2]));
        const double bsum = std::fabs(pr) + std::fabs(pi) + std::fabs(h1) + std::fabs(h2) + std::fabs(h3) + std::fabs(h4);
        const double Ra = std::fabs(Rr) + std::fabs(Ri), qla = std::fabs(lo[2]) + std::fabs(lo[3]);
        const double eR = 1.01 * EPS * Ra + 64.0 * EPS * EPS * bsum;
        lr = (Rr * qr + Ri * qi) / den2, li = (Ri * qr - Rr * qi) / den2;
        na += std::fabs(lr) + std::fabs(li);
        // |N1 - (p^/q^ - N0)| <= (|R~ - R^| + |R~| |q_lo| / |q_h|) / |q^| + division rounding
        eN = (ep + na * eq) / dl + ((eR + Ra * qla / qa) / (qa * (1.0 - 4.0 * EPS)) + 8.0 * EPS * Ra / qa) * 1.01;
    } else {
        eN = (ep + na * eq) / dl + 8.0 * EPS * na;  // |N - N^|
    }
    double err = (eN + delta * t / (1.0 - t)) * 1.01;
    if (ac && std::isfinite(ac->ds)) {
        // e - N^/g = ((N - N^) - N N^ (S^ - S)) / ((1 - N S) g), g = 1 - N^ S^; N^/g = N^ + N^ w / g, w = N^ S^
        // (the correction term is computed from the doubles; its rounding and the dropped low part are
        // O(u |N|^2 |S^| / |g|))
        const double nu = na + eN, sa = std::hypot(ac->sr, ac->si), ws = nu * std::max(s, sa);
        const double wr = nr * ac->sr - ni * ac->si, wi = nr * ac->si + ni * ac->sr;
        const double gr = 1.0 - wr, gi = -wi;
        const double g2 = gr * gr + gi * gi, ga = std::sqrt(g2) - 8.0 * EPS * (1.0 + na * sa);
        if (ws < 0.5 && ga > 0.0) {
            const double Qr = (wr * gr + wi * gi) / g2, Qi = (wi * gr - wr * gi) / g2;
            const double c2r = nr * Qr - ni * Qi, c2i = nr * Qi + ni * Qr;
            const double err2 = ((eN + nu * na * ac->ds) / ((1.0 - ws) * ga) + 32.0 * EPS * na * na * sa / ga) * 1.01;
            if (err2 < err) err = err2, lr += c2r, li += c2i, na += std::fabs(c2r) + std::fabs(c2i);
        }
    }
    double ex, ey, tiny;
    if (oh == 0.0) {
        tiny = TINY_REL * (std::sqrt(x * x + y * y) - delta);
        two_sum(x, -nr, ox, ex);
        double erx = err;
        if (lr != 0.0) {
            const double e2 = ex - lr;
            erx += 1.01 * EPS * (std::fabs(ex) + std::fabs(lr));
            two_sum(ox, e2, ox, ex);
        }
        if (!rounds_to(ox, ex, erx, tiny)) return false;
    } else {  // oh + ol + x - nr - lr = ox + ex up to the rounding of e1 + e2 (- lr) + ol
        double s1, e1, s2, e2;
        two_sum(oh, x, s1, e1);
        two_sum(s1, -nr, s2, e2);
        double re;
        if (lr != 0.0) {
            two_sum(s2, ((e1 + e2) - lr) + ol, ox, ex);
            re = 3.0 * EPS * (std::fabs(e1) + std::fabs(e2) + std::fabs(lr) + std::fabs(ol)) + oe;
        } else {
            two_sum(s2, (e1 + e2) + ol, ox, ex);
            re = 2.0 * EPS * (std::fabs(e1) + std::fabs(e2) + std::fabs(ol)) + oe;
        }
        tiny = TINY_REL * (std::hypot(s1, y) * (1.0 - 4.0 * EPS) - std::fabs(e1) - std::fabs(ol) - oe - delta);
        if (!rounds_to(ox, ex, err + re, tiny)) return false;
    }
    if (real) { oy = 0.0; return true; }
    two_sum(y, -ni, oy, ey);
    double ery = err;
    if (li != 0.0) {
        const double e2 = ey - li;
        ery += 1.01 * EPS * (std::fabs(ey) + std::fabs(li));
        two_sum(oy, e2, oy, ey);
    }
    return rounds_to(oy, ey, ery, tiny);
}

inline void eval_at(int lv, const CPoly& P, double x, double y, double& pr, double& pi, double& qr, double& qi,
                    double& ep, double& eq, int* sexp = nullptr) {
    if (lv <= 2) eval_lvl<2>(P, x, y, pr, pi, qr, qi, ep, eq, sexp);
    else if (lv == 3) eval_lvl<3>(P, x, y, pr, pi, qr, qi, ep, eq, sexp);
    else eval_lvl<4>(P, x, y, pr, pi, qr, qi, ep, eq, sexp);
}

// Root i is evaluated at level lev[i] (any level gives a valid enclosure). bad[i] = its disc is not
// isolated (meets another disc, or the conjugate-symmetry test fails): the roots to work on further.
CertResult certify(const CPoly& P, const std::vector<double>& xr, const std::vector<double>& xi,
                   const std::vector<int>& lev, std::vector<int>& partner, std::vector<double>& rad,
                   std::vector<char>& res_ok, std::vector<double>& ox, std::vector<double>& oy,
                   std::vector<char>& bad, std::vector<double>* noise = nullptr) {
    const size_t n = xr.size();
    bad.assign(n, 1);
    if (noise) noise->assign(n, INFINITY);
    const double a0 = std::fabs(P.hi[0]) - std::fabs(P.lo[0]) - (P.has_res ? P.res[0] : 0.0);
    if (!(a0 > 0.0)) return CERT_NOT_ISOLATED;
    std::vector<double> pu(n), d2(n);
    std::vector<std::array<double, 6>> ev(n);
    std::vector<int> sx(n, 0);  // values scaled by 2^-sx: only rad needs it
    {  // level-2 points EW at a time (real and complex groups), the others one by one
        size_t gr[EW], gc[EW];
        int mr = 0, mc = 0;
        auto flush = [&](auto real, const size_t* g, int m) {
            alignas(32) double X[EW], Y[EW], o[6][EW];
            for (int l = 0; l < EW; ++l) X[l] = xr[g[l < m ? l : 0]], Y[l] = xi[g[l < m ? l : 0]];
            comp_eval_w<true, decltype(real)::value>(P, X, Y, o[0], o[1], o[2], o[3], o[4], o[5]);
            for (int l = 0; l < m; ++l) ev[g[l]] = {o[0][l], o[1][l], o[2][l], o[3][l], o[4][l], o[5][l]};
        };
        for (size_t i = 0; i < n; ++i) {
            if (lev[i] <= 2 && !needs_scaling(n, xr[i], xi[i])) {
                if (xi[i] == 0.0) {
                    gr[mr++] = i;
                    if (mr == EW) flush(std::true_type(), gr, mr), mr = 0;
                } else {
                    gc[mc++] = i;
                    if (mc == EW) flush(std::false_type(), gc, mc), mc = 0;
                }
                continue;
            }
            auto& e = ev[i];
            eval_at(lev[i], P, xr[i], xi[i], e[0], e[1], e[2], e[3], e[4], e[5], &sx[i]);
        }
        if (mr) flush(std::true_type(), gr, mr);
        if (mc) flush(std::false_type(), gc, mc);
    }
    rad.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double pr = ev[i][0], pi = ev[i][1], qr = ev[i][2], qi = ev[i][3], ep = ev[i][4], eq = ev[i][5];
        const int S = sx[i];
        pu[i] = (std::sqrt(pr * pr + pi * pi) + ep) * (1.0 + 4.0 * EPS);
        if (noise) {  // root uncertainty from the evaluation error alone (heuristic, for level choice)
            const double ql = std::sqrt(qr * qr + qi * qi) - eq;
            if (ql > 0.0) (*noise)[i] = (double)n * ep / ql;
        }
        const double x = xr[i], y = xi[i];
#pragma omp simd
        for (size_t j = 0; j < n; ++j) {
            const double dx = x - xr[j], dy = y - xi[j];
            d2[j] = dx * dx + dy * dy;
        }
        double prod = 1.0;  // prod_{j != i} |z_i - z_j|^2 = prod * 2^ex, renormalized to avoid over/underflow
        long ex = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            if (!(d2[j] > 0x1p-500 && d2[j] < 0x1p500)) return CERT_NOT_ISOLATED;
            prod *= d2[j];
            if (!(prod > 0x1p-500 && prod < 0x1p500)) {
                int e;
                prod = std::frexp(prod, &e);
                ex += e;
            }
        }
        if (ex & 1) { prod *= 2.0; --ex; }
        // (1 + 16 n EPS) covers the rounding of the n-1 squared distances, their product and sqrt
        rad[i] = std::ldexp((double)n * pu[i] / (a0 * std::sqrt(prod)), (int)(-ex / 2) + S) *
                 (1.0 + 16.0 * (double)n * EPS) * 1.01;
        if (!std::isfinite(rad[i]) || !std::isfinite(pu[i])) return CERT_NOT_ISOLATED;
    }
    partner.assign(n, -1);
    res_ok.assign(n, 0);
    ox.resize(n), oy.resize(n);
    bool resolved = true, isolated = true;
    const double grow = 1.0 + 16.0 * EPS;
    for (size_t i = 0; i < n; ++i) {
        const double x = xr[i], y = xi[i], ri = rad[i];
        int overlap = 0, mirror = 0, own = 0;
        double gap = INFINITY;
        const double r16 = 16.0 * ri * ri;
#pragma omp simd reduction(+ : overlap, mirror, own) reduction(min : gap)
        for (size_t j = 0; j < n; ++j) {
            const double dx = x - xr[j], dy = y - xi[j], cy = y + xi[j];
            const double s = ri + rad[j], s2 = s * s * grow;
            const double dd = dx * dx + dy * dy, dc = dx * dx + cy * cy;
            const bool other = j != i;
            overlap += (other && dd <= s2) ? 1 : 0;
            mirror += (other && dc <= s2) ? 1 : 0;  // conj(D_i) meets D_j (over-approximated)
            // conflicts this disc is responsible for (radius >= 1/4 of the distance; of two overlapping
            // discs at least one is): the other disc may be small and well placed
            own += (other && (dd <= s2 || dc <= s2) && r16 >= std::min(dd, dc)) ? 1 : 0;
            gap = std::min(gap, other ? std::sqrt(dd) * (1.0 - 4.0 * EPS) - rad[j] : INFINITY);
        }
        const bool axis = std::fabs(y) <= ri * grow;
        if (overlap || (axis ? mirror != 0 : mirror != 1)) {
            isolated = false;
            bad[i] = own > 0 || (!axis && mirror != 1);
            continue;
        }
        bad[i] = 0;
        if (!axis)
            for (size_t j = 0; j < n; ++j) {
                const double dx = x - xr[j], cy = y + xi[j], s = ri + rad[j];
                if (j != i && dx * dx + cy * cy <= s * s * grow) { partner[i] = (int)j; break; }
            }
        if (!isolated) continue;  // no certificate this round: skip the (costly) rounding
        const auto& e = ev[i];
        res_ok[i] = (!axis && y < 0.0) ||
                    resolve_point(x, y, e[0], e[1], e[2], e[3], e[4], e[5], n, gap, ri, axis, ox[i], oy[i]);
        resolved = resolved && res_ok[i];  // lower pair members are not emitted, nor checked
    }
    for (size_t i = 0; i < n; ++i)
        if (partner[i] != -1 && partner[partner[i]] != (int)i) {
            isolated = false;
            bad[i] = 1, bad[(size_t)partner[i]] = 1;
        }
    if (!isolated) return CERT_NOT_ISOLATED;
    return resolved ? CERT_OK : CERT_UNRESOLVED;
}


// Refine root i of an isolated set (discs (xr, xi, rad), each holding exactly one root) by Newton at
// double points with K-fold compensated evaluation; real roots stay on the axis. For j != i the root
// zeta_j lies in D(z_j, rad_j), which gives the gap for resolve_point.
template <int K>
bool refine_root(const CPoly& P, const std::vector<double>& xr, const std::vector<double>& xi,
                 const std::vector<double>& rad, size_t i, bool real, double& ox, double& oy) {
    const size_t n = xr.size();
    double x = xr[i], y = real ? 0.0 : xi[i];
    double best = INFINITY, bx = x, by = y;
    std::array<double, 6> be{};
    double bl[4] = {};
    for (int it = 0; it < 6; ++it) {
        double pr, pi, qr, qi, ep, eq, l[4];
        if (real) eval_k<K, true>(P, x, 0.0, pr, pi, qr, qi, ep, eq, nullptr, l);
        else eval_k<K, false>(P, x, y, pr, pi, qr, qi, ep, eq, nullptr, l);
        const double c = std::sqrt((pr * pr + pi * pi) / (qr * qr + qi * qi));
        if (!(c < best)) break;
        best = c, bx = x, by = y, be = {pr, pi, qr, qi, ep, eq};
        std::copy(l, l + 4, bl);
        const cd corr = cd(pr, pi) / cd(qr, qi);
        const double nx = x - corr.real(), ny = real ? 0.0 : y - corr.imag();
        if (nx == x && ny == y) break;
        x = nx, y = ny;
    }
    double gap = INFINITY;
    AbCorr ac;  // S^ = sum_{j != i} 1/(b - z_j); |1/(b - zeta_j) - 1/(b - z_j)| <= rad_j / (d_j (d_j - rad_j))
    double ds = 0.0;
    for (size_t j = 0; j < n; ++j) {
        if (j == i) continue;
        const double dx = bx - xr[j], dy = by - xi[j], d2 = dx * dx + dy * dy, d = std::sqrt(d2) * (1.0 - 4.0 * EPS);
        gap = std::min(gap, d - rad[j]);
        ac.sr += dx / d2, ac.si -= dy / d2;
        ds += rad[j] / (d * (d - rad[j])) + (double)(n + 8) * EPS / d;
    }
    if (gap > 0.0 && std::isfinite(ds) && std::isfinite(ac.sr) && std::isfinite(ac.si)) ac.ds = ds * 1.01;
    // rad_i still bounds |b - zeta_i| via |b - z_i| + rad_i
    const double r = std::hypot(bx - xr[i], by - xi[i]) * (1.0 + 4.0 * EPS) + rad[i];
    return resolve_point(bx, by, be[0], be[1], be[2], be[3], be[4], be[5], n, gap, r, real, ox, oy, 0.0, 0.0, 0.0, &ac,
                         bl);
}

// Resolve the unresolved roots of an isolated set one by one (K = 3, 4 above the level used). Writes (ur, ui).
bool refine_unresolved(const CPoly& P, const std::vector<double>& xr, const std::vector<double>& xi,
                       const std::vector<double>& rad, const std::vector<int>& partner, const std::vector<char>& res_ok,
                       const std::vector<int>& lev, std::vector<double>& ox, std::vector<double>& oy) {
    const size_t n = xr.size();
    for (size_t i = 0; i < n; ++i) {
        const int pj = partner[i];
        if (pj != -1 && xi[i] < 0.0) continue;  // a pair is emitted from its upper member
        if (res_ok[i]) continue;
        double x, y;
        if (!(lev[i] < 3 && refine_root<3>(P, xr, xi, rad, i, pj == -1, x, y)) &&
            !(lev[i] < 4 && refine_root<4>(P, xr, xi, rad, i, pj == -1, x, y)))
            return false;
        ox[i] = x, oy[i] = y;
    }
    return true;
}

// ------------------------------------------------------------------ anchored clusters
// Roots closer than double resolution (relative) cannot be separated by double centres. A cluster near
// the real axis is re-centred at a real double anchor c: Q(y) = p(c + y) is computed exactly (integer
// Taylor shift of the exact coefficients) and rounded to hi + lo with residual bounds. In the local
// coordinate y the cluster is wide relative to |y|, so the double machinery separates it. Each root keeps
// the polynomial and coordinates it is best represented in (Mixed); cluster mates subtract local
// coordinates, so their differences carry no anchor rounding.

// Exact integer times 2^sh -> hi + lo, residual bound (|T 2^sh - hi - lo| <= res; underflow included).
inline bool mpz_to_dd(const mpz_class& T, long sh, double& hi, double& lo, double& res) {
    hi = lo = res = 0.0;
    if (sgn(T) == 0) return true;
    long e1;
    const double d1 = mpz_get_d_2exp(&e1, T.get_mpz_t());  // truncated, 0.5 <= |d1| < 1
    mpz_class H, R, L;
    mpz_set_d(H.get_mpz_t(), std::ldexp(d1, 53));
    if (e1 >= 53) mpz_mul_2exp(H.get_mpz_t(), H.get_mpz_t(), (unsigned long)(e1 - 53));
    else mpz_tdiv_q_2exp(H.get_mpz_t(), H.get_mpz_t(), (unsigned long)(53 - e1));
    R = T - H;
    hi = std::ldexp(d1, (int)(e1 + sh));
    if (!std::isfinite(hi)) return false;
    if (std::ldexp(hi, (int)-(e1 + sh)) != d1) res += 0x1p-1073;  // underflow
    if (sgn(R) == 0) return true;
    long e2;
    const double d2 = mpz_get_d_2exp(&e2, R.get_mpz_t());
    mpz_set_d(L.get_mpz_t(), std::ldexp(d2, 53));
    if (e2 >= 53) mpz_mul_2exp(L.get_mpz_t(), L.get_mpz_t(), (unsigned long)(e2 - 53));
    else mpz_tdiv_q_2exp(L.get_mpz_t(), L.get_mpz_t(), (unsigned long)(53 - e2));
    R -= L;
    lo = std::ldexp(d2, (int)(e2 + sh));
    if (std::ldexp(lo, (int)-(e2 + sh)) != d2) res += 0x1p-1073;
    if (sgn(R) != 0) {
        long e3;
        const double d3 = mpz_get_d_2exp(&e3, R.get_mpz_t());
        res += std::ldexp(std::fabs(d3) + 0x1p-52, (int)(e3 + sh)) * (1.0 + 2.0 * EPS) + 0x1p-1074;
    }
    return true;
}

// Exact P(anchor + y) for an anchor c_1 + c_2 + ... (doubles), kept as sum_k B_k (y 2^-F)^k up to a
// positive power of two (B integers). P = hi + lo exactly (no residuals).
struct Shift {
    std::vector<mpz_class> B;  // ascending
    long F = 0;
    static void split(double v, u64& m, long& e) {  // |v| = m 2^e, m odd
        int ee;
        m = (u64)std::ldexp(std::frexp(std::fabs(v), &ee), 53);
        const int tz = __builtin_ctzll(m);
        m >>= tz;
        e = (long)ee - 53 + tz;
    }
    bool init(const CPoly& P) {
        if (P.has_res) return false;
        const size_t n = P.deg();
        long E = LONG_MAX;
        for (size_t k = 0; k <= n; ++k)
            for (double v : {P.hi[k], P.lo[k]})
                if (v != 0.0) {
                    u64 m;
                    long e;
                    split(v, m, e);
                    E = std::min(E, e);
                }
        if (E == LONG_MAX) return false;
        B.assign(n + 1, mpz_class(0));
        mpz_class t;
        for (size_t j = 0; j <= n; ++j)
            for (double v : {P.hi[n - j], P.lo[n - j]})
                if (v != 0.0) {
                    u64 m;
                    long e;
                    split(v, m, e);
                    mpz_set_ui(t.get_mpz_t(), (unsigned long)m);
                    mpz_mul_2exp(t.get_mpz_t(), t.get_mpz_t(), (unsigned long)(e - E));
                    if (v < 0) B[j] -= t;
                    else B[j] += t;
                }
        F = 0;
        return true;
    }
    // y = c + y': with c = +-m 2^f and t = y 2^-F = 2^g (+-m + s), g = f - F, y' = 2^f s
    bool shift_by(double c) {
        if (c == 0.0 || !std::isfinite(c)) return false;
        const size_t n = B.size() - 1;
        u64 m;
        long f;
        split(c, m, f);
        const long g = f - F;
        if (std::labs(g) * (long)n > 400000) return false;
        const long off = g < 0 ? -g * (long)n : 0;  // keeps the coefficients integers
        for (size_t j = 0; j <= n; ++j)
            mpz_mul_2exp(B[j].get_mpz_t(), B[j].get_mpz_t(), (unsigned long)(g * (long)j + off));
        for (size_t i = 0; i < n; ++i)  // Taylor shift by +-m: B_j <- B_j +- m B_{j+1}, j = n-1 .. i
            for (size_t j = n; j-- > i;) {
                if (c > 0) mpz_addmul_ui(B[j].get_mpz_t(), B[j + 1].get_mpz_t(), (unsigned long)m);
                else mpz_submul_ui(B[j].get_mpz_t(), B[j + 1].get_mpz_t(), (unsigned long)m);
            }
        F = f;
        return true;
    }
    // coefficient of y^k: B_k 2^(-F k), scaled so that all are < 1
    bool to_cpoly(CPoly& Q) const {
        const size_t n = B.size() - 1;
        long G = LONG_MIN;
        for (size_t k = 0; k <= n; ++k)
            if (sgn(B[k]) != 0) G = std::max(G, (long)mpz_sizeinbase(B[k].get_mpz_t(), 2) - F * (long)k);
        if (G == LONG_MIN || sgn(B[n]) == 0) return false;
        Q.hi.assign(n + 1, 0.0), Q.lo.assign(n + 1, 0.0), Q.res.assign(n + 1, 0.0);
        Q.has_res = false;
        for (size_t k = 0; k <= n; ++k) {
            const size_t i = n - k;
            if (!mpz_to_dd(B[k], -F * (long)k - G, Q.hi[i], Q.lo[i], Q.res[i])) return false;
            Q.has_res = Q.has_res || Q.res[i] != 0.0;
        }
        return Q.hi[0] != 0.0;
    }
};

struct Mixed {
    const CPoly* P = nullptr;
    std::vector<CPoly> Q;             // shifted polynomials
    std::vector<double> oh, ol, oe;   // their anchors: |anchor - oh - ol| <= oe
    std::vector<int> a;               // per root: -1 (P, absolute coordinates) or index into Q
    std::vector<double> wr, wi;
    const CPoly& poly(size_t i) const { return a[i] < 0 ? *P : Q[(size_t)a[i]]; }
    double ohi(size_t i) const { return a[i] < 0 ? 0.0 : oh[(size_t)a[i]]; }
    double olo(size_t i) const { return a[i] < 0 ? 0.0 : ol[(size_t)a[i]]; }
    double oerr(size_t i) const { return a[i] < 0 ? 0.0 : oe[(size_t)a[i]]; }
    // z - z_j for z = (x, y) in root i's coordinates (conj: conj(z) - z_j); e bounds the absolute error
    // beyond a relative EPS per component
    cd diff(size_t i, double x, double y, size_t j, bool conj, double& e) const {
        const double dy = conj ? y + wi[j] : y - wi[j];
        e = 0.0;
        if (a[i] == a[j]) return cd(x - wr[j], dy);
        double s, t;
        two_sum(ohi(i), -ohi(j), s, t);
        const double lo = olo(i) - olo(j), t2 = t + lo, w = x - wr[j], u = w + t2;
        e = 2.0 * EPS * (std::fabs(w) + 2.0 * std::fabs(t2) + std::fabs(t) + 2.0 * std::fabs(lo)) + oerr(i) + oerr(j);
        return cd(s + u, dy);
    }
    double dist_lo(size_t i, double x, double y, size_t j, bool conj = false) const {
        double e;
        const cd d = diff(i, x, y, j, conj, e);
        return std::hypot(d.real(), d.imag()) * (1.0 - 4.0 * EPS) - e;
    }
    // Vectorized forms over all j (same formulas as diff): anchor data per root, refresh() after the
    // anchors or a change
    std::vector<double> jh, jl, je, ja;
    void refresh() {
        const size_t n = wr.size();
        jh.resize(n), jl.resize(n), je.resize(n), ja.resize(n);
        for (size_t j = 0; j < n; ++j) jh[j] = ohi(j), jl[j] = olo(j), je[j] = oerr(j), ja[j] = (double)a[j];
    }
    // d[j] = dist_lo(i, x, y, j, conj), d[i] = INFINITY. sqrt(dx^2 + dy^2) is within 2u of hypot,
    // inside the 4 EPS margin, while the components stay in [2^-450, 2^450] (else hypot).
    void dist_lo_all(size_t i, double x, double y, bool conj, double* d) const {
        const size_t n = wr.size();
        const double hi_ = jh[i], li = jl[i], ei = je[i], ai = ja[i], sg = conj ? -1.0 : 1.0;
        int out = 0;
#pragma omp simd reduction(| : out)
        for (size_t j = 0; j < n; ++j) {
            const double dy = y - sg * wi[j], w = x - wr[j];
            double s, t;
            two_sum(hi_, -jh[j], s, t);
            const double lo = li - jl[j], t2 = t + lo, u = w + t2;
            const bool same = ja[j] == ai;
            const double dx = same ? w : s + u;
            const double e = same ? 0.0
                                  : 2.0 * EPS * (std::fabs(w) + 2.0 * std::fabs(t2) + std::fabs(t) + 2.0 * std::fabs(lo)) +
                                        ei + je[j];
            const double m = std::max(std::fabs(dx), std::fabs(dy));
            out |= (j != i && !(m >= 0x1p-450 && m <= 0x1p450)) ? 1 : 0;
            d[j] = std::sqrt(dx * dx + dy * dy) * (1.0 - 4.0 * EPS) - e;
        }
        if (out)
            for (size_t j = 0; j < n; ++j)
                if (j != i) d[j] = dist_lo(i, x, y, j, conj);
        d[i] = INFINITY;
    }
    // sum_{j != i} 1 / (z - z_j) in root i's coordinates (Aberth; no error bound needed)
    cd aberth_sum_mixed(size_t i, double x, double y) const {
        const size_t n = wr.size();
        const double hi_ = jh[i], li = jl[i], ai = ja[i];
        double a_ = 0.0, b_ = 0.0;
        int out = 0;
#pragma omp simd reduction(+ : a_, b_) reduction(| : out)
        for (size_t j = 0; j < n; ++j) {
            const double dy = y - wi[j], w = x - wr[j];
            double s, t;
            two_sum(hi_, -jh[j], s, t);
            const double dx = ja[j] == ai ? w : s + (w + (t + (li - jl[j])));
            const double m = std::max(std::fabs(dx), std::fabs(dy));
            const bool skip = j == i;
            out |= (!skip && !(m >= 0x1p-450 && m <= 0x1p450)) ? 1 : 0;
            const double inv = skip ? 0.0 : 1.0 / (dx * dx + dy * dy);
            a_ += dx * inv;
            b_ -= dy * inv;
        }
        if (!out) return cd(a_, b_);
        cd S = 0.0;
        for (size_t j = 0; j < n; ++j) {
            double e;
            if (j != i) S += 1.0 / diff(i, x, y, j, false, e);
        }
        return S;
    }
};

// Aberth on the members of one cluster in its local coordinates (the other roots stay fixed). A member
// whose correction is not finite (it met another point) is frozen and the result is false.
template <int L>
bool aberth_local(Mixed& M, const std::vector<size_t>& mem, int maxit) {
    bool ok = true;
    M.refresh();
    std::vector<char> done(mem.size(), 0);
    size_t left = mem.size();
    for (int it = 0; it < maxit && left; ++it)
        for (size_t t = 0; t < mem.size(); ++t) {
            if (done[t]) continue;
            const size_t i = mem[t];
            const double x = M.wr[i], y = M.wi[i];
            double pr, pi, qr, qi, ep, eq;
            eval_lvl<L>(M.poly(i), x, y, pr, pi, qr, qi, ep, eq);
            if (std::hypot(pr, pi) <= ep) { done[t] = 1; --left; continue; }
            const cd N = cd(pr, pi) / cd(qr, qi);
            const cd S = M.aberth_sum_mixed(i, x, y);
            const cd corr = N / (1.0 - N * S);
            if (!std::isfinite(corr.real()) || !std::isfinite(corr.imag())) {
                ok = false;
                done[t] = 1, --left;
                continue;
            }
            M.wr[i] -= corr.real();
            M.wi[i] -= corr.imag();
            if (std::abs(corr) <= EPS * std::abs(cd(x, y))) { done[t] = 1; --left; }
        }
    return ok && left == 0;
}

// certify<L> for mixed representations (same three steps; distances from Mixed::dist_lo).
template <int L>
CertResult certify_mixed(const Mixed& M, std::vector<int>& partner, std::vector<double>& rad,
                         std::vector<char>& res_ok, std::vector<double>& ox, std::vector<double>& oy) {
    const size_t n = M.wr.size();
    std::vector<std::array<double, 6>> ev(n);
    std::vector<int> sx(n, 0);
    const_cast<Mixed&>(M).refresh();
    {  // roots in absolute coordinates: level 2 EW at a time (as in certify); members one by one
        size_t gr[EW], gc[EW];
        int mr = 0, mc = 0;
        const CPoly& P = *M.P;
        auto flush = [&](auto real, const size_t* g, int m) {
            alignas(32) double X[EW], Y[EW], o[6][EW];
            for (int l = 0; l < EW; ++l) X[l] = M.wr[g[l < m ? l : 0]], Y[l] = M.wi[g[l < m ? l : 0]];
            comp_eval_w<true, decltype(real)::value>(P, X, Y, o[0], o[1], o[2], o[3], o[4], o[5]);
            for (int l = 0; l < m; ++l) ev[g[l]] = {o[0][l], o[1][l], o[2][l], o[3][l], o[4][l], o[5][l]};
        };
        for (size_t i = 0; i < n; ++i) {
            if (L == 2 && M.a[i] < 0 && !needs_scaling(P.deg(), M.wr[i], M.wi[i])) {
                if (M.wi[i] == 0.0) {
                    gr[mr++] = i;
                    if (mr == EW) flush(std::true_type(), gr, mr), mr = 0;
                } else {
                    gc[mc++] = i;
                    if (mc == EW) flush(std::false_type(), gc, mc), mc = 0;
                }
                continue;
            }
            auto& e = ev[i];
            eval_lvl<L>(M.poly(i), M.wr[i], M.wi[i], e[0], e[1], e[2], e[3], e[4], e[5], &sx[i]);
        }
        if (mr) flush(std::true_type(), gr, mr);
        if (mc) flush(std::false_type(), gc, mc);
    }
    rad.assign(n, 0.0);
    std::vector<double> d(n), dcv(n);
    for (size_t i = 0; i < n; ++i) {
        const CPoly& Pi = M.poly(i);
        const double a0 = std::fabs(Pi.hi[0]) - std::fabs(Pi.lo[0]) - (Pi.has_res ? Pi.res[0] : 0.0);
        if (!(a0 > 0.0)) return CERT_NOT_ISOLATED;
        const double pr = ev[i][0], pi = ev[i][1], ep = ev[i][4];
        const int S = sx[i];
        const double pu = (std::sqrt(pr * pr + pi * pi) + ep) * (1.0 + 4.0 * EPS);
        M.dist_lo_all(i, M.wr[i], M.wi[i], false, d.data());
        double prod = 1.0;  // renormalized only near over/underflow (scaling by 2^e is exact)
        long ex = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            if (!(d[j] > 0.0)) return CERT_NOT_ISOLATED;
            prod *= d[j];
            if (!(prod > 0x1p-500 && prod < 0x1p500)) {
                int e;
                prod = std::frexp(prod, &e);
                ex += e;
            }
        }
        rad[i] = std::ldexp((double)n * pu / (a0 * prod), (int)(S - ex)) * (1.0 + 16.0 * (double)n * EPS) * 1.01;
        if (!std::isfinite(rad[i]) || !std::isfinite(pu)) return CERT_NOT_ISOLATED;
    }
    partner.assign(n, -1);
    res_ok.assign(n, 0);
    ox.resize(n), oy.resize(n);
    bool resolved = true;
    for (size_t i = 0; i < n; ++i) {
        const double x = M.wr[i], y = M.wi[i], ri = rad[i];
        int mirror = 0;
        double gap = INFINITY;
        M.dist_lo_all(i, x, y, false, d.data());
        M.dist_lo_all(i, x, y, true, dcv.data());
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dj = d[j], dc = dcv[j];
            if (dj <= ri + rad[j]) return CERT_NOT_ISOLATED;
            if (dc <= ri + rad[j]) {
                if (mirror++ == 0) partner[i] = (int)j;
            }
            gap = std::min(gap, dj - rad[j]);
        }
        const bool axis = std::fabs(y) <= ri * (1.0 + 16.0 * EPS);
        if (axis ? mirror != 0 : mirror != 1) return CERT_NOT_ISOLATED;
        if (axis) partner[i] = -1;
        const auto& e = ev[i];
        res_ok[i] = (!axis && y < 0.0) || resolve_point(x, y, e[0], e[1], e[2], e[3], e[4], e[5], n, gap, ri, axis,
                                                        ox[i], oy[i], M.ohi(i), M.olo(i), M.oerr(i));
        resolved = resolved && res_ok[i];
    }
    for (size_t i = 0; i < n; ++i)
        if (partner[i] != -1 && partner[partner[i]] != (int)i) return CERT_NOT_ISOLATED;
    return resolved ? CERT_OK : CERT_UNRESOLVED;
}

template <int K>
bool refine_mixed(const Mixed& M, const std::vector<double>& rad, size_t i, bool real, double& ox, double& oy) {
    const size_t n = M.wr.size();
    const CPoly& Pi = M.poly(i);
    double x = M.wr[i], y = real ? 0.0 : M.wi[i];
    double best = INFINITY, bx = x, by = y;
    std::array<double, 6> be{};
    double bl[4] = {};
    for (int it = 0; it < 6; ++it) {
        double pr, pi, qr, qi, ep, eq, l[4];
        if (real) eval_k<K, true>(Pi, x, 0.0, pr, pi, qr, qi, ep, eq, nullptr, l);
        else eval_k<K, false>(Pi, x, y, pr, pi, qr, qi, ep, eq, nullptr, l);
        const double c = std::sqrt((pr * pr + pi * pi) / (qr * qr + qi * qi));
        if (!(c < best)) break;
        best = c, bx = x, by = y, be = {pr, pi, qr, qi, ep, eq};
        std::copy(l, l + 4, bl);
        const cd corr = cd(pr, pi) / cd(qr, qi);
        const double nx = x - corr.real(), ny = real ? 0.0 : y - corr.imag();
        if (nx == x && ny == y) break;
        x = nx, y = ny;
    }
    double gap = INFINITY, ds = 0.0;
    AbCorr ac;  // as in refine_root; the computed difference d^ is off by <= e + 2 EPS |d^|
    for (size_t j = 0; j < n; ++j) {
        if (j == i) continue;
        double e;
        const cd dj = M.diff(i, bx, by, j, false, e);
        const double d2 = std::norm(dj), d = std::sqrt(d2) * (1.0 - 4.0 * EPS) - e;
        gap = std::min(gap, d - rad[j]);
        ac.sr += dj.real() / d2, ac.si -= dj.imag() / d2;
        ds += (rad[j] + e) / (d * (d - rad[j])) + (double)(n + 12) * EPS / d;
    }
    if (gap > 0.0 && std::isfinite(ds) && std::isfinite(ac.sr) && std::isfinite(ac.si)) ac.ds = ds * 1.01;
    const double r = std::hypot(bx - M.wr[i], by - M.wi[i]) * (1.0 + 4.0 * EPS) + rad[i];
    return resolve_point(bx, by, be[0], be[1], be[2], be[3], be[4], be[5], n, gap, r, real, ox, oy, M.ohi(i), M.olo(i),
                         M.oerr(i), &ac, bl);
}

// After CERT_HOPELESS (or roots left unresolved): anchor every cluster of the Aberth points (xr, xi) and
// certify the mixed set (non-members keep their polished points ur, ui). Clusters: points within 2^-36
// (relative), and unresolved roots with their neighbours within 2^-16 (a double centre limits the
// absolute accuracy to ~ulp(|z|)^2 n / gap, e.g. too coarse for the tiny imaginary parts of a split
// double root). False if some cluster is not centred near the real axis.
bool solve_anchored(const CPoly& P, const std::vector<double>& xr, const std::vector<double>& xi,
                    const std::vector<double>& ur, const std::vector<double>& ui, std::vector<cd>& roots,
                    const std::vector<char>* unresolved = nullptr) {
    const size_t n = xr.size();
    std::vector<size_t> comp(n);
    for (size_t i = 0; i < n; ++i) comp[i] = i;
    auto find = [&](size_t i) {
        while (comp[i] != i) i = comp[i] = comp[comp[i]];
        return i;
    };
    const double thr = 0x1p-36, thr_u = 0x1p-16;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            const double d = std::hypot(xr[i] - xr[j], xi[i] - xi[j]);
            const double t = unresolved && ((*unresolved)[i] || (*unresolved)[j]) ? thr_u : thr;
            if (d <= t * std::max(std::hypot(xr[i], xi[i]), std::hypot(xr[j], xi[j]))) comp[find(i)] = find(j);
        }
    Mixed M;
    M.P = &P;
    M.a.assign(n, -1);
    M.wr = ur, M.wi = ui;
    std::vector<std::vector<size_t>> clusters;
    std::vector<int> cid(n, -1);
    for (size_t i = 0; i < n; ++i) {
        const size_t r = find(i);
        if (r == i) continue;
        if (cid[r] < 0) {
            cid[r] = (int)clusters.size();
            clusters.push_back({r});
        }
        clusters[(size_t)cid[r]].push_back(i);
    }
#ifdef MAS_DEBUG
    fprintf(stderr, "solve_anchored n=%zu clusters=%zu\n", n, clusters.size());
#endif
    if (clusters.empty()) return false;
    Shift S0;
    if (!S0.init(P)) return false;
    const size_t nc = clusters.size();
    M.oh.assign(nc, 0.0), M.ol.assign(nc, 0.0), M.oe.assign(nc, 0.0);
    M.Q.resize(nc);
    constexpr int MAXA = 8;
    std::vector<Shift> SK(nc);
    std::vector<std::array<double, MAXA>> anc(nc);
    std::vector<int> na(nc, 0);
    // Anchor cluster k (force: one more level at the current local centre of its members, e.g. when a
    // double-double Q cannot resolve them).
    auto develop = [&](size_t k, bool force) {
        const auto& mem = clusters[k];
        const size_t m = mem.size();
        Shift& S = SK[k];
        if (!force) {
            double mx = 0.0, my = 0.0;
            for (size_t i : mem) mx += xr[i], my += xi[i];
            mx /= (double)m, my /= (double)m;
            if (!(std::fabs(my) <= 0x1p-30 * std::fabs(mx))) return false;  // complex cluster: multiprecision
            S = S0;
            if (!S.shift_by(mx)) return false;
            anc[k][0] = mx, na[k] = 1;
            for (size_t i : mem) M.a[i] = (int)k;
            M.oh[k] = mx;  // local differences of the members only need oh (all of them share it)
        } else {
            if (na[k] == MAXA) return false;
            double cx = 0.0, cy = 0.0;
            for (size_t i : mem) cx += M.wr[i], cy += M.wi[i];
            cx /= (double)m, cy /= (double)m;
            if (!(std::fabs(cy) <= 0x1p-30 * std::fabs(cx)) || !S.shift_by(cx)) return false;
            anc[k][(size_t)na[k]++] = cx;
        }
        // Re-centre while the cluster stays below double resolution in local coordinates: the converged
        // local mean is the next anchor term (each level adds ~53 bits of anchor precision). The local
        // members of an unresolved m-cluster scatter by ~(u^2 kappa)^(1/m) around their (well-conditioned)
        // centroid, hence the m-dependent threshold; a needless re-centring costs a shift, not accuracy.
        const double thr_m = std::max(thr, std::exp2(-90.0 / (double)m));
        for (;;) {
            CPoly& Q = M.Q[k];
            if (!S.to_cpoly(Q)) return false;
            // members start on a circle of the local root radius max_j (|q_j| / |q_m|)^(1/(m-j))
            const size_t nq = Q.deg();
            double r0 = 0.0;
            const double qm = std::fabs(Q.hi[nq - m]);
            for (size_t j = 0; j < m && qm > 0.0; ++j)
                r0 = std::max(r0, std::pow(std::fabs(Q.hi[nq - j]) / qm, 1.0 / (double)(m - j)));
            if (!(r0 > 0.0 && std::isfinite(r0))) r0 = thr * std::fabs(M.oh[k]);
            for (size_t t = 0; t < m; ++t) {
                const double th = 2.0 * M_PI * (double)t / (double)m + 0.7;
                M.wr[mem[t]] = r0 * std::cos(th), M.wi[mem[t]] = r0 * std::sin(th);
            }
            if (na[k] == MAXA) break;
            aberth_local<2>(M, mem, 60 + 4 * (int)m);
            double cx = 0.0, cy = 0.0, md = INFINITY;
            for (size_t t = 0; t < m; ++t) {
                const size_t i = mem[t];
                cx += M.wr[i], cy += M.wi[i];
                for (size_t u = t + 1; u < m; ++u) {
                    const size_t j = mem[u];
                    md = std::min(md, std::hypot(M.wr[i] - M.wr[j], M.wi[i] - M.wi[j]) /
                                          std::max(std::hypot(M.wr[i], M.wi[i]), std::hypot(M.wr[j], M.wi[j])));
                }
            }
            if (!(md <= thr_m)) break;  // no longer a cluster at this anchor (or Aberth failed: certify decides)
            cx /= (double)m, cy /= (double)m;
            if (!(std::fabs(cy) <= 0x1p-30 * std::fabs(cx)) || !S.shift_by(cx)) break;
            anc[k][(size_t)na[k]++] = cx;
        }
        // anchor = anc[0] + (anc[1] + anc[2] + ...), |anc[j+1]| ~< ulp(anc[j]): oh + ol with error oe
        double lo = 0.0, le = 0.0;
        for (int j = na[k] - 1; j >= 1; --j) {
            lo += anc[k][(size_t)j];
            le += EPS * std::fabs(lo);
        }
        M.ol[k] = lo, M.oe[k] = le * 1.01;
        return true;
    };
    for (size_t k = 0; k < nc; ++k)
        if (!develop(k, false)) return false;
    std::vector<int> partner;
    std::vector<double> rad, ox, oy;
    std::vector<char> res_ok;
    CertResult cr = CERT_NOT_ISOLATED;
    int level = 2;
    auto stage = [&](auto lvl) {
        constexpr int L = decltype(lvl)::value;
        level = L;
        for (const auto& mem : clusters)
            if (!aberth_local<L>(M, mem, 60 + 4 * (int)mem.size())) return;
        const std::vector<double> ar = M.wr, ai = M.wi;
        for (const auto& mem : clusters)
            for (size_t i : mem) polish<L>(M.poly(i), M.wr[i], M.wi[i]);
        cr = certify_mixed<L>(M, partner, rad, res_ok, ox, oy);
#ifdef MAS_DEBUG
        fprintf(stderr, "anchored L=%d cr=%d clusters=%zu\n", L, (int)cr, clusters.size());
        for (const auto& mem : clusters) for (size_t i : mem) fprintf(stderr, "  i=%zu org=%.17g+%.6e w=(%.6e,%.6e) rad=%.3e\n", i, M.ohi(i), M.olo(i), M.wr[i], M.wi[i], i < rad.size() ? rad[i] : -1.0);
#endif
        if (cr == CERT_NOT_ISOLATED) M.wr = ar, M.wi = ai;  // polishing may have merged two points
    };
    // A cluster that the double-double Q(y) cannot resolve (e.g. tiny imaginary parts at a local centre
    // far from the anchor) gets one more anchor level, then everything is certified again.
    for (int attempt = 0;; ++attempt) {
        stage(std::integral_constant<int, 2>());
        if (cr == CERT_NOT_ISOLATED) stage(std::integral_constant<int, 3>());
        if (cr == CERT_NOT_ISOLATED) stage(std::integral_constant<int, 4>());
        std::vector<char> fail(nc, 0);
        if (cr == CERT_UNRESOLVED) {
            bool ok = true;
            for (size_t i = 0; i < n; ++i) {
                if (res_ok[i] || (partner[i] != -1 && M.wi[i] < 0.0)) continue;
                const bool real = partner[i] == -1;
                if ((level < 3 && refine_mixed<3>(M, rad, i, real, ox[i], oy[i])) ||
                    (level < 4 && refine_mixed<4>(M, rad, i, real, ox[i], oy[i])))
                    continue;
                ok = false;
                if (M.a[i] < 0) return false;  // not a cluster member: re-anchoring cannot help
                fail[(size_t)M.a[i]] = 1;
            }
            if (ok) cr = CERT_OK;
        } else if (cr == CERT_NOT_ISOLATED)
            fail.assign(nc, 1);
        if (cr == CERT_OK || cr == CERT_HOPELESS || attempt == 2) break;
        for (size_t k = 0; k < nc; ++k)
            if (fail[k] && !develop(k, true)) return false;
    }
    if (cr != CERT_OK) return false;
    for (size_t i = 0; i < n; ++i) {
        if (partner[i] == -1) roots.emplace_back(ox[i], 0.0);
        else if (M.wi[i] > 0.0) {
            roots.emplace_back(ox[i], std::fabs(oy[i]));
            roots.emplace_back(ox[i], -std::fabs(oy[i]));
        }
    }
    return true;
}

// Solve a square-free polynomial of degree >= 2 with nonzero constant term. On success appends the
// certified roots; otherwise returns false and leaves the best approximations in approx.
bool solve_cpoly(const CPoly& P, std::vector<cd>& roots, std::vector<cd>& approx) {
    const size_t n = P.deg();
    std::vector<cd> z0;
    bini_start(P.hi, z0);
    if (z0.size() != n) return false;
    std::vector<double> xr(n), xi(n);
    for (size_t i = 0; i < n; ++i) xr[i] = z0[i].real(), xi[i] = z0[i].imag();
    {
        PROF(2);
        aberth<1>(P, xr, xi, 60 + 4 * (int)n);
    }
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(xr[i]) || !std::isfinite(xi[i])) {
            for (size_t k = 0; k < n; ++k) xr[k] = z0[k].real(), xi[k] = z0[k].imag();
            break;
        }
    std::vector<int> partner;
    std::vector<double> rad;
    std::vector<char> res_ok;
    std::vector<double> ox, oy;  // resolved (correctly rounded) roots
    std::vector<double> ur, ui;
    auto emit = [&]() {  // a conjugate pair is emitted from its upper member
        for (size_t i = 0; i < n; ++i) {
            if (partner[i] == -1) roots.emplace_back(ox[i], 0.0);
            else if (ui[i] > 0.0) {
                roots.emplace_back(ox[i], std::fabs(oy[i]));
                roots.emplace_back(ox[i], -std::fabs(oy[i]));
            }
        }
    };
    // Polish copies of the Aberth points (Newton may merge two of them; later rounds restart from the
    // Aberth points). Escalation is local: only the roots whose discs are not isolated are iterated again
    // (compensated Aberth, then evaluation levels 3, 4) with the others fixed at their polished points;
    // isolated but unresolved roots are refined one by one at a higher level.
    CertResult cr = CERT_NOT_ISOLATED;
    std::vector<int> lev(n, 2);
    std::vector<char> bad;
    ur = xr, ui = xi;
    {
        PROF(3);
        polish2_all(P, ur, ui, IotaRange{n});
    }
    std::vector<double> noise;
    {
        PROF(4);
        cr = certify(P, ur, ui, lev, partner, rad, res_ok, ox, oy, bad, &noise);
    }
#ifdef MAS_DEBUG
    fprintf(stderr, "initial certify n=%zu cr=%d\n", n, (int)cr);
    for (size_t i = 0; i < n; ++i)
        fprintf(stderr, "   %zu z=(%.17g,%.6g) rad=%.3g res_ok=%d bad=%d\n", i, ur[i], ui[i], i < rad.size() ? rad[i] : -1.0,
                i < res_ok.size() ? (int)res_ok[i] : -1, i < bad.size() ? (int)bad[i] : -1);
#endif
    std::vector<size_t> B;
    std::vector<double> wr, wi;
    for (int round = 0; round < 3 && cr == CERT_NOT_ISOLATED; ++round) {
        // round 0: compensated Aberth, or directly level 3 / 4 for roots whose level-2 noise radius
        // already reaches their neighbours (each level gains a factor ~u); later rounds: one level up
        B.clear();
        for (size_t i = 0; i < n; ++i) {
            if (!bad[i]) continue;
            if (round == 0) {
                const double d = 0.125 * nn_dist(ur, ui, i);
                lev[i] = noise[i] < d ? 2 : (noise[i] * EPS < d ? 3 : 4);
            } else lev[i] = std::min(lev[i] + 1, 4);
            B.push_back(i);
        }
        wr = ur, wi = ui;  // fixed roots at their polished points, the others restart from Aberth
        for (size_t i : B) wr[i] = xr[i], wi[i] = xi[i];
        bool ok = true;
        {
            PROF(5);
            const int it = 30 + (int)n;
            std::vector<size_t> sub;
            for (int L = 2; L <= 4 && ok; ++L) {  // the lower-level groups first
                sub.clear();
                for (size_t i : B)
                    if (lev[i] == L) sub.push_back(i);
                if (sub.empty()) continue;
                ok = L == 2 ? aberth_subset<2>(P, wr, wi, sub, it)
                            : (L == 3 ? aberth_subset<3>(P, wr, wi, sub, it) : aberth_subset<4>(P, wr, wi, sub, it));
            }
            for (size_t i : B) xr[i] = wr[i], xi[i] = wi[i];
        }
        if (!ok) continue;
        {
            PROF(3);
            std::vector<size_t> b2;
            for (size_t i : B) {
                ur[i] = xr[i], ui[i] = xi[i];
                if (lev[i] == 2) b2.push_back(i);
                else if (lev[i] == 3) polish<3>(P, ur[i], ui[i]);
                else polish<4>(P, ur[i], ui[i]);
            }
            polish2_all(P, ur, ui, b2);
        }
        {
            PROF(4);
            cr = certify(P, ur, ui, lev, partner, rad, res_ok, ox, oy, bad);
        }
#ifdef MAS_DEBUG
        {
            int h[5] = {};
            for (size_t i : B) h[lev[i]]++;
            fprintf(stderr, "round %d |B|=%zu (L2 %d, L3 %d, L4 %d) cr=%d\n", round, B.size(), h[2], h[3], h[4], (int)cr);
            if (round == 0) for (size_t i : B) fprintf(stderr, "   %zu z=(%.4g,%.3g) noise=%.3g nn=%.3g bad=%d rad=%.3g\n", i, ur[i], ui[i], noise[i], nn_dist(ur, ui, i), (int)bad[i], rad[i]);
        }
#endif
        // converged points a few ulps apart: roots closer than double resolution, which no
        // double-centered certificate can separate -> anchored / multiprecision directly
        if (round == 0 && cr == CERT_NOT_ISOLATED && min_rel_dist(xr, xi) <= 16.0 * EPS) cr = CERT_HOPELESS;
    }
    if (cr == CERT_UNRESOLVED) {  // isolated, but some roots need a higher evaluation level
        PROF(8);
        if (refine_unresolved(P, ur, ui, rad, partner, res_ok, lev, ox, oy)) cr = CERT_OK;
    }
    if (cr == CERT_OK) { emit(); return true; }
    {  // anchored local coordinates before multiprecision: clusters below double resolution, and the
       // unresolved / still overlapping roots with their close neighbours (a cluster whose spread is
       // noise-limited at every level: its centroid is well conditioned, its members are not)
        PROF(9);
        std::vector<char> loose;
        if (cr == CERT_UNRESOLVED) {
            loose.resize(n);
            for (size_t i = 0; i < n; ++i) loose[i] = !res_ok[i];
        } else if (cr == CERT_NOT_ISOLATED) loose = bad;
        const size_t before = roots.size();
        if (solve_anchored(P, xr, xi, ur, ui, roots, loose.empty() ? nullptr : &loose)) return true;
        roots.resize(before);
    }
    approx.resize(n);
    for (size_t i = 0; i < n; ++i) approx[i] = cd(ur[i], ui[i]);
    return false;
}

// Real roots by conjugate symmetry (multiprecision path): a certified disc that meets the real axis and
// whose mirror image meets no other disc holds a real root. dconj(i, j) = |conj(z_i) - z_j|.
template <class DistConj>
bool conj_partners(size_t n, const std::vector<double>& rad, const std::vector<double>& absim, DistConj dconj,
                   std::vector<int>& partner) {
    partner.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
        bool meets_axis = absim[i] <= rad[i];
        for (size_t j = 0; j < n; ++j) {
            if (j == i || dconj(i, j) > rad[i] + rad[j]) continue;
            if (meets_axis || partner[i] != -1) return false;  // ambiguous
            partner[i] = (int)j;
        }
        if (!meets_axis && partner[i] == -1) return false;
    }
    for (size_t i = 0; i < n; ++i)
        if (partner[i] != -1 && partner[partner[i]] != (int)i) return false;
    return true;
}

struct PrecGuard {
    mp_bitcnt_t old;
    explicit PrecGuard(mp_bitcnt_t p) : old(mpf_get_default_prec()) { mpf_set_default_prec(p); }
    ~PrecGuard() { mpf_set_default_prec(old); }
};
struct cmpf { mpf_class re, im; };
inline double round_d(const mpf_class& x) {  // round to nearest even (mpf_get_d truncates)
    const double t = x.get_d();
    const double a = std::nextafter(t, sgn(x) < 0 ? -INFINITY : INFINITY);
    if (!std::isfinite(a)) return t;
    mpf_class mid(t, 64);
    mid += a;
    mid /= 2;  // exact: t, a have 53 bits
    const int c = cmp(abs(x), abs(mid));
    if (c != 0) return c < 0 ? t : a;
    int e;
    const double m = std::frexp(t, &e) * 0x1p53;  // tie: the one with an even last mantissa bit
    return std::fmod(m, 2.0) == 0.0 ? t : a;
}

// Escalation for conditioning beyond double-double: the same Aberth + Gerschgorin + conjugate-symmetry
// scheme in GMP multiprecision, doubling the precision until every root is certified (like MPSolve).
bool solve_factor_mp(const PolyQ& G, const std::vector<cd>& start, std::vector<cd>& roots) {
    // PrecGuard sets GMP's process-wide default mpf precision: serialize concurrent callers.
    static std::mutex mp_mutex;
    const std::lock_guard<std::mutex> lock(mp_mutex);
    const size_t n = G.size() - 1;
    std::vector<mpf_class> ab(G.size());  // |coefficients| at 64 bits (no overflow, unlike double)
    for (size_t i = 0; i < G.size(); ++i) {
        ab[i].set_prec(64);
        mpf_set_q(ab[i].get_mpf_t(), G[i].get_mpq_t());
        ab[i] = abs(ab[i]);
    }
    std::vector<cd> z0 = start;
    const size_t given = z0.size();
    z0.resize(n);
    for (size_t i = 0; i < n; ++i)  // missing or diverged approximations must not reach mpf_set_d
        if (i >= given || !std::isfinite(z0[i].real()) || !std::isfinite(z0[i].imag()))
            z0[i] = std::polar(1.0, 0.4 + 6.283185307179586 * (double)i / (double)n);
    for (size_t i = 0; i < n; ++i)  // separate coincident starts (Aberth needs distinct points)
        for (size_t j = 0; j < i; ++j)
            if (std::abs(z0[i] - z0[j]) <= 1e-12 * (1.0 + std::abs(z0[i])))
                z0[i] += 1e-8 * (1.0 + std::abs(z0[i])) * std::polar(1.0, (double)i);
    std::vector<cmpf> z;
    for (mp_bitcnt_t prec : {128u, 256u, 512u, 1024u, 2048u, 4096u}) {
        PrecGuard guard(prec);
        std::vector<mpf_class> A(G.size());
        for (size_t i = 0; i < G.size(); ++i) mpf_set_q(A[i].get_mpf_t(), G[i].get_mpq_t());
        std::vector<cmpf> zn(n);
        for (size_t i = 0; i < n; ++i) {
            if (z.empty()) { zn[i].re = z0[i].real(); zn[i].im = z0[i].imag(); }
            else { zn[i].re = z[i].re; zn[i].im = z[i].im; }
        }
        z.swap(zn);
        const unsigned long ubits = prec - 3;  // unit = 2^-ubits > relative error of one mpf operation
        mpf_class pr, pi, dr, di, tr, ti, cr, ci;
        auto eval = [&](const cmpf& w) {  // p and p' at w
            pr = A[0]; pi = 0; dr = 0; di = 0;
            for (size_t k = 1; k <= n; ++k) {
                tr = dr * w.re - di * w.im; ti = dr * w.im + di * w.re; dr = tr + pr; di = ti + pi;
                tr = pr * w.re - pi * w.im; ti = pr * w.im + pi * w.re; pr = tr + A[k]; pi = ti;
            }
        };
        auto absd = [](const mpf_class& x, const mpf_class& y) { return std::hypot(x.get_d(), y.get_d()); };
        auto diff = [&](size_t i, size_t j) {  // z_i - z_j, rounded to double after exact-ish subtraction
            tr = z[i].re - z[j].re; ti = z[i].im - z[j].im;
            return cd(tr.get_d(), ti.get_d());
        };
        // (2n+4) unit sum |a_k| |z|^(n-k) (rounding of the Horner pass), at 64 bits; 1.02 covers the
        // truncations of this low-precision pass
        mpf_class nb(0, 64), az(0, 64), t64(0, 64), p2, c2;
        auto noise_bound = [&](size_t i) {
            az = z[i].re * z[i].re;
            t64 = z[i].im * z[i].im;
            az = sqrt(az + t64);
            nb = ab[0];
            for (size_t k = 1; k <= n; ++k) nb = nb * az + ab[k];
            nb *= (2.0 * (double)n + 4.0) * 1.02;
            mpf_div_2exp(nb.get_mpf_t(), nb.get_mpf_t(), ubits);
        };
        auto at_noise = [&](size_t i) {  // |p| <= noise bound
            noise_bound(i);
            p2 = pr * pr + pi * pi;
            return cmp(p2, nb * nb) <= 0;
        };
        std::vector<char> done(n, 0);
        size_t left = n;
        for (int it = 0; it < 100 + 2 * (int)n && left; ++it) {
            for (size_t i = 0; i < n; ++i) {
                if (done[i]) continue;
                eval(z[i]);
                if (at_noise(i)) { done[i] = 1; --left; continue; }
                tr = dr * dr + di * di;
                if (sgn(tr) == 0) break;
                cr = (pr * dr + pi * di) / tr;  // Newton correction N = p / p'
                ci = (pi * dr - pr * di) / tr;
                cd S = 0;
                for (size_t j = 0; j < n; ++j)
                    if (j != i) S += 1.0 / diff(i, j);
                cd q = 1.0 / (1.0 - cd(cr.get_d(), ci.get_d()) * S);  // Aberth factor, double is enough
                if (!std::isfinite(q.real()) || !std::isfinite(q.imag())) break;
                tr = cr * q.real() - ci * q.imag();
                ti = cr * q.imag() + ci * q.real();
                z[i].re -= tr;
                z[i].im -= ti;
                c2 = tr * tr + ti * ti;  // |correction| <= unit |z|
                p2 = z[i].re * z[i].re + z[i].im * z[i].im;
                mpf_div_2exp(p2.get_mpf_t(), p2.get_mpf_t(), 2 * ubits);
                if (cmp(c2, p2) <= 0) { done[i] = 1; --left; }
            }
        }
        if (left) continue;
        std::vector<double> rad(n), absim(n);
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) {
            eval(z[i]);
            noise_bound(i);
            p2 = sqrt(pr * pr + pi * pi) + nb;  // |p| + noise, may exceed the double range
            double prod = 1.0;
            long ex = 0;
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                double d = std::abs(diff(i, j));
                if (d == 0.0) { ok = false; break; }
                int e;
                prod = std::frexp(prod * d, &e);
                ex += e;
            }
            if (!ok) break;
            p2 /= prod;
            if (ex > 0) mpf_div_2exp(p2.get_mpf_t(), p2.get_mpf_t(), (unsigned long)ex);
            else mpf_mul_2exp(p2.get_mpf_t(), p2.get_mpf_t(), (unsigned long)-ex);
            // get_d truncates; 1 + 8 EPS covers it and the mpf roundings of |p|
            rad[i] = (double)n * p2.get_d() * (1.0 + 8.0 * EPS) * (1.0 + 16.0 * (double)n * EPS) * 1.01;
            absim[i] = std::fabs(z[i].im.get_d());
            ok = ok && std::isfinite(rad[i]);
        }
        for (size_t i = 0; i < n && ok; ++i)
            for (size_t j = i + 1; j < n && ok; ++j) ok = std::abs(diff(i, j)) > rad[i] + rad[j];
        std::vector<int> partner;
        ok = ok && conj_partners(n, rad, absim, [&](size_t i, size_t j) {
            tr = z[i].re - z[j].re; ti = z[i].im + z[j].im;  // conj(z_i) - z_j = (re_i-re_j) - i(im_i+im_j)
            return std::hypot(tr.get_d(), ti.get_d());
        }, partner);
        if (!ok) continue;
        // root i lies in D(z_i, rad_i): a component is emitted once its interval provably rounds to one
        // double (see TINY_REL); otherwise retry at a higher precision
        std::vector<cd> out;
        mpf_class rm;
        auto comp = [&](const mpf_class& c, double r, double za, double& o) {
            o = round_d(c);
            if (r <= 0.5 * TINY_REL * za) return true;  // the mpf margin below is < 2^-123 za
            rm = za;  // + rounding of c -+ r in mpf
            mpf_div_2exp(rm.get_mpf_t(), rm.get_mpf_t(), ubits - 2);
            rm += r;
            mpf_class lo = c - rm, hi = c + rm;
            return round_d(lo) == o && round_d(hi) == o;
        };
        for (size_t i = 0; i < n && ok; ++i) {
            if (partner[i] != -1 && sgn(z[i].im) <= 0) continue;
            const double za = absd(z[i].re, z[i].im);
            const double r = rad[i] * 1.01;
            double x, y = 0.0;
            ok = comp(z[i].re, r, za - r, x) && (partner[i] == -1 || comp(z[i].im, r, za - r, y));
            out.emplace_back(x, y);
            if (partner[i] != -1) out.emplace_back(x, -y);
        }
        if (!ok) continue;
        roots.insert(roots.end(), out.begin(), out.end());
        return true;
    }
    return false;
}

// Round to nearest even (mpq get_d truncates toward zero).
double round_q(const Q& q) {
    const int s = sgn(q);
    const double t = s < 0 ? -std::fabs(q.get_d()) : q.get_d();  // keep -0.0 for tiny negatives
    if (!std::isfinite(t)) return t;
    const double a = std::nextafter(t, s < 0 ? -INFINITY : INFINITY);
    // midpoint between t and its successor away from zero; past DBL_MAX the successor is the
    // overflow threshold DBL_MAX + ulp/2 (a tie there rounds to infinity)
    const Q mid = std::isfinite(a) ? Q((Q(t) + Q(a)) / 2) : Q(Q(t) + Q(std::copysign(0x1p970, t)));
    const int c = cmp(abs(q), abs(mid));
    if (c != 0) return c < 0 ? t : a;
    if (!std::isfinite(a)) return a;
    int e;
    const double m = std::frexp(t, &e) * 0x1p53;  // tie: the one with an even last mantissa bit
    return std::fmod(m, 2.0) == 0.0 ? t : a;
}

// Exact doubles scaled by a power of two (exact unless a coefficient underflows; then res covers it).
CPoly cpoly_from_double(const std::vector<double>& c) {
    double mx = 0.0;
    for (double v : c) mx = std::max(mx, std::fabs(v));
    int e;
    std::frexp(mx, &e);
    CPoly P;
    P.hi.resize(c.size());
    P.lo.assign(c.size(), 0.0);
    P.res.assign(c.size(), 0.0);
    for (size_t k = 0; k < c.size(); ++k) {
        P.hi[k] = std::ldexp(c[k], -e);
        if (std::ldexp(P.hi[k], e) != c[k]) { P.res[k] = 0x1p-1074; P.has_res = true; }
    }
    return P;
}

// Rational factor -> primitive integer polynomial -> hi + lo (exact up to 106 bits, else residual).
CPoly cpoly_from_q(const PolyQ& G) {
    mpz_class L = 1, g = 0;
    for (const auto& q : G) mpz_lcm(L.get_mpz_t(), L.get_mpz_t(), q.get_den_mpz_t());
    std::vector<mpz_class> Z(G.size());
    for (size_t k = 0; k < G.size(); ++k) {
        Z[k] = G[k].get_num() * (L / G[k].get_den());
        mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), Z[k].get_mpz_t());
    }
    size_t bits = 0;
    for (auto& v : Z) {
        v /= g;
        bits = std::max(bits, mpz_sizeinbase(v.get_mpz_t(), 2));
    }
    const int s = -(int)bits;
    CPoly P;
    P.hi.resize(G.size());
    P.lo.resize(G.size());
    P.res.assign(G.size(), 0.0);
    for (size_t k = 0; k < G.size(); ++k) {  // exponent split: |Z_k| may exceed the double range
        double rr;
        mpz_to_dd(Z[k], s, P.hi[k], P.lo[k], rr);  // |Z_k| 2^s <= 1: always finite
        if (rr != 0.0) { P.res[k] = rr; P.has_res = true; }
    }
    return P;
}

// Solve one square-free factor (monic, exact rational coefficients). Returns false if not certified.
bool solve_factor_fast(const PolyQ& F, std::vector<cd>& roots) {
    roots.clear();
    PolyQ G = F;
    if (G.size() > 1 && sgn(G.back()) == 0) {  // square-free: x divides at most once
        roots.emplace_back(0.0, 0.0);
        G.pop_back();
    }
    const size_t n = G.size() - 1;
    if (n == 0) return true;
    if (n == 1) {
        roots.emplace_back(round_q(Q(-G[1] / G[0])), 0.0);
        return true;
    }
    std::vector<cd> approx;
    if (solve_cpoly(cpoly_from_q(G), roots, approx)) return true;
    PROF(6);
    return solve_factor_mp(G, approx, roots);
}

// Same for the (common) square-free input given as exact doubles: no rational arithmetic unless the
// fast path fails and multiprecision is needed.
bool solve_double_fast(const std::vector<double>& c, std::vector<cd>& roots) {
    roots.clear();
    std::vector<double> cc = c;
    if (cc.size() > 1 && cc.back() == 0.0) {
        roots.emplace_back(0.0, 0.0);
        cc.pop_back();
    }
    const size_t n = cc.size() - 1;
    if (n == 0) return true;
    if (n == 1) {
        roots.emplace_back(-cc[1] / cc[0], 0.0);
        return true;
    }
    std::vector<cd> approx;
    if (solve_cpoly(cpoly_from_double(cc), roots, approx)) return true;
    PROF(6);
    PolyQ G(cc.size());
    for (size_t k = 0; k < cc.size(); ++k) G[k] = Q(cc[k]);
    const Q lead = G[0];
    for (auto& v : G) v /= lead;
    return solve_factor_mp(G, approx, roots);
}

}  // namespace


// ------------------------------------------------------------------ C ABI
extern "C" {

// Status bits written to *status.
enum { MAS_COUNT_MISMATCH = 1, MAS_USED_EXACT_GCD = 2, MAS_USED_FALLBACK = 4 };

long mas_exact_evals() { return g_exact_evals.load(std::memory_order_relaxed); }

#ifdef MAS_PROFILE
double* mas_profile() { return g_prof; }
#endif

// coeffs: highest degree first, len entries. Output arrays need capacity len.
// Returns number of root entries written (each entry has its multiplicity), -1 for the zero polynomial,
// or -2 if a coefficient is NaN or infinite.
static int solve_impl(const double* coeffs, int len, double* out_re, double* out_im, int* out_mult, int* status,
                      bool fast) {
    const int cap = len;
    *status = 0;
    int start = 0;
    while (start < len && coeffs[start] == 0.0) ++start;
    if (start == len) return -1;
    for (int i = start; i < len; ++i)
        if (!std::isfinite(coeffs[i])) return -2;  // GMP traps on NaN/inf (SIGFPE)
    std::vector<double> c(coeffs + start, coeffs + len);
    if (c.size() == 1) return 0;

    int nout = 0;
    Workspace ws;
    auto emit = [&](cd z, int m) {
        if (nout >= cap) { *status |= MAS_COUNT_MISMATCH; return; }
        out_re[nout] = z.real();
        out_im[nout] = z.imag();
        out_mult[nout] = m;
        ++nout;
    };
    bool squarefree;
    {
        PROF(1);
        squarefree = squarefree_mod(c);
    }
    std::vector<std::pair<PolyQ, int>> factors;
    int modular = -1;
    if (!squarefree) {
        PROF(7);
        modular = squarefree_modular(c, factors);
        squarefree = modular == 0;
    }
    std::vector<cd> fast_roots;
    if (fast && squarefree) {  // common case: no rational arithmetic at all
        if (solve_double_fast(c, fast_roots)) {
            for (cd z : fast_roots) emit(z, 1);
            return nout;
        }
    }

    PolyQ P(c.size());
    for (size_t i = 0; i < c.size(); ++i) P[i] = Q(c[i]);  // exact conversion
    Q lead = P[0];
    for (auto& x : P) x /= lead;

    if (squarefree) {
        factors.assign(1, {P, 1});
    } else {
        *status |= MAS_USED_EXACT_GCD;
        if (modular != 1) factors = squarefree_exact(P);
    }
    for (const auto& [F, m] : factors) {
        if (fast) {
            if (!(squarefree && m == 1) && solve_factor_fast(F, fast_roots)) {
                for (cd z : fast_roots) emit(z, m);
                continue;
            }
            *status |= MAS_USED_FALLBACK;
        }
        PolyC factor(F.size());
        PolyR factor_r(F.size());
        for (size_t i = 0; i < F.size(); ++i) factor_r[i] = F[i].get_d(), factor[i] = factor_r[i];
        double R = fujiwara(factor);
        if (!std::isfinite(R)) {  // root bound overflows double: v1 cannot represent this factor
            *status |= MAS_COUNT_MISMATCH;
            continue;
        }
        auto chain = sturm_chain(F);
        std::vector<double> reals;
        for (const auto& ch : isolate_real(chain, -R, R)) {
            double r = real_root_in_bracket(factor_r, ch.a, ch.b);
            for (int i = 0; i < ch.k; ++i) reals.push_back(r);
        }
        PolyC deflated = factor;
        std::vector<double> order = reals;
        std::sort(order.begin(), order.end(), [](double x, double y) { return std::fabs(x) < std::fabs(y); });
        for (double r : order) {  // forward deflation, small roots first
            PolyC qd(deflated.size() - 1);
            cd acc = 0;
            for (size_t i = 0; i + 1 < deflated.size(); ++i) qd[i] = acc = acc * r + deflated[i];
            deflated.swap(qd);
        }
        for (double r : reals) emit(cd(r, 0.0), m);

        int n_cplx = (int)deflated.size() - 1;
        if (n_cplx <= 0) continue;
        double Rc = fujiwara(deflated), y0 = 0.0;
        if (!std::isfinite(Rc)) {
            *status |= MAS_COUNT_MISMATCH;
            continue;
        }
        int k_top = 0;
        if (!winding_number(deflated, -Rc, Rc, y0, Rc, ws, k_top)) {
            y0 = -1e-9 * Rc;  // residual root on the real axis: drop the bottom edge slightly
            if (!winding_number(deflated, -Rc, Rc, y0, Rc, ws, k_top)) k_top = n_cplx / 2;
        }
        std::vector<cd> found;
        if (k_top > 0) found = isolate_complex(deflated, -Rc, Rc, y0, Rc, k_top, ws);
        PolyC dfac = deriv_c(factor);
        for (size_t i = 0; i < found.size(); ++i) {
            cd z0 = found[i], z = z0;
            double sep = INFINITY;
            for (double r : reals) sep = std::min(sep, std::abs(z0 - r));
            for (size_t j = 0; j < found.size(); ++j) {
                if (j != i) sep = std::min(sep, std::abs(z0 - found[j]));
                sep = std::min(sep, std::abs(z0 - std::conj(found[j])));
            }
            bool ok = newton_polish(factor, dfac, z, 20);
            if (!(ok && std::abs(z - z0) < 0.5 * sep)) z = z0;
            emit(cd(z.real(), std::fabs(z.imag())), m);
            emit(cd(z.real(), -std::fabs(z.imag())), m);
        }
        if (2 * (int)found.size() != n_cplx) *status |= MAS_COUNT_MISMATCH;
    }
    return nout;
}

// v1: Sturm chain + certified winding-number quadtree (the original MAS design).
int mas_solve(const double* coeffs, int len, double* out_re, double* out_im, int* out_mult, int* status) {
    return solve_impl(coeffs, len, out_re, out_im, out_mult, status, false);
}

// v2: Aberth + compensated Newton + Gerschgorin/conjugate-symmetry certificate; v1 path as fallback.
int mas_solve2(const double* coeffs, int len, double* out_re, double* out_im, int* out_mult, int* status) {
    return solve_impl(coeffs, len, out_re, out_im, out_mult, status, true);
}

}  // extern "C"
