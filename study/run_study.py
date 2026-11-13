"""MAS vs numpy study driver.

  refs : certified Arb references for every case (parallel)      -> data/refs.pkl
  time : interleaved, pinned timing of each solver + accuracy     -> data/<out>.jsonl
Use run.sh (sets the single-core pinning and BLAS threads). Needs certiroot installed (pip install .).
"""
import argparse, gc, json, os, pickle, platform, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")

import numpy as np
import families as F


# ---------------------------------------------------------------- references
def _ref_worker(args):
    cid, c, cpus = args
    if cpus:
        os.sched_setaffinity(0, cpus)
    import reference as R
    t = time.time()
    r = R.reference(c)
    r["time"] = time.time() - t
    return cid, r


def cmd_refs(a):
    from multiprocessing import Pool
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "refs.pkl")
    refs = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    cpus = set(int(x) for x in a.cpus.split(",")) if a.cpus else None
    todo = [(cid, c, cpus) for cid, *_, c in F.cases() if cid not in refs]
    todo.sort(key=lambda t: -len(t[1]))  # big ones first
    print(f"{len(todo)} references to compute", flush=True)
    with Pool(a.jobs) as pool:
        for i, (cid, r) in enumerate(pool.imap_unordered(_ref_worker, todo)):
            refs[cid] = r
            if i % 25 == 0 or r["time"] > 2:
                print(f"  {i + 1}/{len(todo)} {cid} {r['time']:.2f}s prec={r['prec']} complete={r['complete']}", flush=True)
                pickle.dump(refs, open(path, "wb"))
    pickle.dump(refs, open(path, "wb"))
    print("incomplete:", [k for k, v in refs.items() if not v["complete"]])


# ---------------------------------------------------------------- solvers
def solvers(names):
    import certiroot._core as M
    out = {}

    def np_roots(c):
        return np.roots(c)

    def np_eig_factory(c):  # numpy's own algorithm without the Python-level work of np.roots
        p = np.trim_zeros(c, "f")
        A = np.diag(np.ones(len(p) - 2), -1)
        A[0, :] = -p[1:] / p[0]
        return lambda c_: np.linalg.eigvals(A)

    def np_newton(c, steps=2):
        r = np.roots(c).astype(complex)
        dc = np.polyder(c)
        for _ in range(steps):
            d = np.polyval(dc, r)
            ok = d != 0
            r = np.where(ok, r - np.polyval(c, r) / np.where(ok, d, 1), r)
        return r

    def mas(c):
        return M.solve_raw(c, 2)

    table = {"numpy": lambda c: np_roots, "numpy_eig": np_eig_factory, "numpy_newton": lambda c: np_newton,
             "mas": lambda c: mas}
    if "mpsolve" in names:
        import mps_ctypes
        table["mpsolve"] = lambda c: (lambda c_: mps_ctypes.roots(list(map(float, c_)), 53))
    return {k: table[k] for k in names}


def roots_of(name, r):
    if name == "mas":
        re_, im_, mu, st = r
        return np.repeat(re_ + 1j * im_, mu)
    return np.asarray(r, dtype=complex)


def measure(fns, c, batch=0.004, reps=7, slow=0.3, rng=None):
    """Interleaved timing: each round runs every solver once (random order) as a batch of k calls."""
    first, res = {}, {}
    for name, f in fns.items():
        t0 = time.perf_counter(); res[name] = f(c); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); f(c); t2 = time.perf_counter() - t0
        first[name] = min(t1, t2)
    k = {nm: max(1, int(batch / max(t, 1e-7))) for nm, t in first.items()}
    nrep = {nm: (reps if t < slow else 3) for nm, t in first.items()}
    times = {nm: [] for nm in fns}
    gc.disable()
    try:
        for rnd in range(reps):
            order = [nm for nm in fns if rnd < nrep[nm]]
            rng.shuffle(order)
            for nm in order:
                f, kk = fns[nm], k[nm]
                t0 = time.perf_counter()
                for _ in range(kk):
                    f(c)
                times[nm].append((time.perf_counter() - t0) / kk)
    finally:
        gc.enable()
    out = {}
    for nm, ts in times.items():
        ts = np.array(ts)
        out[nm] = dict(t_med=float(np.median(ts)), t_min=float(ts.min()), t_max=float(ts.max()), reps=len(ts), k=k[nm])
    return out, res


def meta():
    cpu = ""
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    return dict(kind="meta", time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                numpy=np.__version__, cpu=cpu, kernel=platform.release(), affinity=sorted(os.sched_getaffinity(0)),
                openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"), argv=sys.argv)


def cmd_time(a):
    import reference as R
    refs = pickle.load(open(os.path.join(DATA, "refs.pkl"), "rb"))
    names = a.solvers.split(",")
    path = os.path.join(DATA, a.out)
    done = set()
    if os.path.exists(path):
        for line in open(path):
            row = json.loads(line)
            if row.get("kind") == "case":
                done.add(row["cid"])
    rng = random.Random(1)
    pat = re.compile(a.filter) if a.filter else None
    with open(path, "a") as fo:
        fo.write(json.dumps(meta()) + "\n")
        for cid, fam, group, n, seed, c in F.cases():
            if cid in done or (pat and not pat.search(cid)):
                continue
            use = [nm for nm in names if not (nm == "numpy_eig" and n > a.eig_max)]
            fac = solvers(use)
            fns = {nm: fac[nm](c) for nm in use}
            t, res = measure(fns, c, reps=a.reps, rng=rng)
            ref = refs[cid]
            row = dict(kind="case", cid=cid, family=fam, group=group, n=n, seed=seed,
                       ndistinct=int(len(ref["mult"])), maxmult=int(ref["mult"].max()) if len(ref["mult"]) else 0)
            for nm in use:
                sc = R.score(roots_of(nm, res[nm]), ref)
                row[nm] = {**t[nm], **sc}
            if "mas" in use:
                row["mas"]["status"] = int(res["mas"][3])
                row["mas"]["mult_ok"] = bool(R.mult_ok(res["mas"][2], ref))
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            msg = " ".join(f"{nm}={row[nm]['t_med'] * 1e3:.3f}ms" for nm in use)
            print(f"{cid:24s} {msg}  masCR={row.get('mas', {}).get('cr_frac', '-')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--cpus", default="")
    p = sub.add_parser("time")
    p.add_argument("--out", default="timing.jsonl")
    p.add_argument("--solvers", default="numpy,numpy_eig,numpy_newton,mpsolve,mas")
    p.add_argument("--filter", default="")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--eig-max", type=int, default=320)
    a = ap.parse_args()
    {"refs": cmd_refs, "time": cmd_time}[a.cmd](a)
