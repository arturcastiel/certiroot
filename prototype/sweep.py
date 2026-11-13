import random, sys, time, warnings, importlib, numpy as np, io, contextlib
warnings.simplefilter("ignore")
sys.argv = [sys.argv[0], sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    import stress
random.seed(12345)
ok = n = 0; t0 = time.time(); fails = []
for trial in range(150):
    kind = trial % 3
    roots = []
    if kind == 0:   # higher-degree random real+complex
        roots = [complex(random.uniform(-10,10)) for _ in range(random.randint(0,10))]
        for _ in range(random.randint(1,8)):
            z = complex(random.uniform(-10,10), random.uniform(0.01,10)); roots += [z, z.conjugate()]
    elif kind == 1: # integer roots / gaussian-integer pairs with repeats (exact coefficients)
        for _ in range(random.randint(1,4)):
            r = random.randint(-4,4); roots += [complex(r)] * random.randint(1,3)
        for _ in range(random.randint(0,2)):
            z = complex(random.randint(-3,3), random.randint(1,3)); m = random.randint(1,2); roots += [z, z.conjugate()] * m
    else:           # unit-circle-ish clusters
        for _ in range(random.randint(2,10)):
            th = random.uniform(0.05, 3.1); z = complex(np.cos(th), np.sin(th)) * random.uniform(0.9,1.1); roots += [z, z.conjugate()]
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        good = stress.run(f"sweep #{trial}", roots)
    n += 1; ok += good
    if not good: fails.append(buf.getvalue().strip())
print(f"{sys.argv[1]}: sweep passed {ok}/{n} in {time.time()-t0:.1f}s")
for f in fails[:8]: print("  ", f[:150])
