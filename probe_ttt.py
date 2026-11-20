# Groundwork: does the learning rate make a visible difference on tic-tac-toe, at which training length,
# how much do identical runs wobble, and how long is one run?
import sys, time
import numpy as np
sys.path.insert(0, ".")
from tttlib import material, Student

tr, te = material()
print("boards: train %d, held-out %d" % (len(tr[0]), len(te[0])))
s = Student(0); print("untrained: loss %.3f top-is-perfect %.3f value err %.3f" % s.test(*te))


def run(sched, steps, seed, batch=64):
    r = np.random.default_rng(1000 + seed)
    s = Student(seed)
    for t in range(steps):
        b = r.integers(0, len(tr[0]), batch)
        l = s.step(tr[0][b], tr[1][b], tr[2][b], tr[3][b], sched(t / max(1, steps - 1)))
        if not np.isfinite(l):
            return 9.0, 0.0, 9.0
    return s.test(*te)


SET = [("const 0.01", lambda f: 0.01), ("const 0.03", lambda f: 0.03), ("const 0.1", lambda f: 0.1), ("const 0.3", lambda f: 0.3),
       ("const 1", lambda f: 1.0), ("const 3", lambda f: 3.0), ("1 -> 0.01", lambda f: 10 ** (-2 * f)), ("0.3 -> 0.003", lambda f: 0.3 * 10 ** (-2 * f))]
for steps in (100, 300, 1000, 5000):
    print("=== %d fixes (batch 64)" % steps)
    for name, sc in SET:
        t0 = time.time()
        rs = np.array([run(sc, steps, s) for s in range(5)])
        print("  %-13s loss %.3f (spread %.3f) | top-is-perfect %.3f (spread %.3f) | %.3fs/run" % (
            name, rs[:, 0].mean(), rs[:, 0].max() - rs[:, 0].min(), rs[:, 1].mean(), rs[:, 1].max() - rs[:, 1].min(), (time.time() - t0) / 5), flush=True)
