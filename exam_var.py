# Final exam for the varied students.
# Part 1: 300 fresh students of the 6 kinds the teacher was raised on.
# Part 2: kinds the teacher never met while being raised (other net sizes, other amounts of material).
#  A) one hand-picked schedule for everybody (best of a grid on part 1; the same one is reused in part 2)
#  B) a hand-picked schedule per kind, i.e. someone who is TOLD the kind (best of the grid for that kind)
#  C) the teacher net (not told the kind)
#  D) the teacher's average curve from part 1, replayed for everybody (no looking at the student)
import numpy as np
import teach_var as T
import varlib

se = lambda x: x.std() / np.sqrt(len(x))
GRID = {}
for c in (0.7, 1.0, 1.5, 2.0, 3.0, 4.5):
    GRID["const %.1f" % c] = lambda f, c=c: c
    GRID["%.1f, last 40%% down to 1/8" % c] = lambda f, c=c: c if f < 0.6 else c * 8 ** (-(f - 0.6) / 0.4)
    GRID["%.1f -> 1/3" % c] = lambda f, c=c: c * 3 ** (-f)
th = np.load("teacher_var.npy")


def measure(seeds, kind):
    g = {k: np.array([T.course(None, s, f, kind=kind)[0] for s in seeds]) for k, f in GRID.items()}
    tr = {}
    C = np.array([T.course(th, s, trace=tr.setdefault(s, []), kind=kind)[0] for s in seeds])
    return g, C, np.array([[lr for _, lr, _ in tr[s]] for s in seeds])


NK = len(varlib.KINDS)
res = [measure(range(900000 + k, 900000 + k + 50 * NK, NK), k) for k in range(NK)]
bestA = max(GRID, key=lambda k: np.mean([r[0][k].mean() for r in res]))
mean = np.exp(np.log(np.concatenate([r[2] for r in res])).mean(0))
fD = lambda f: float(mean[min(len(mean) - 1, int(f * (T.FIXES - 1)) // T.CHUNK)])


def show(title, kinds, res):
    print(title)
    print("  %-18s %-7s %-7s %-7s %-7s  C-A (+-)        C-D (+-)        B's schedule" % ("kind", "A", "B", "C", "D"))
    tot = {x: [] for x in "ABCD"}
    for (k, (hid, n)), (g, C, cur) in zip(kinds, res):
        seeds = range(900000 + k, 900000 + k + 50 * NK, NK) if k < NK else range(950000, 950050)
        D = np.array([T.course(None, s, fD, kind=k)[0] for s in seeds])
        nb = max(g, key=lambda x: g[x].mean())
        A, B = g[bestA], g[nb]
        for x, v in zip("ABCD", (A, B, C, D)):
            tot[x].append(v)
        print("  hid %3d mat %4d:  %.3f   %.3f   %.3f   %.3f   %+.3f (%.3f)  %+.3f (%.3f)  %s" % (
            hid, n, A.mean(), B.mean(), C.mean(), D.mean(), (C - A).mean(), se(C - A), (C - D).mean(), se(C - D), nb))
        print("      teacher's lr: " + " ".join("%.2f" % x for x in np.exp(np.log(cur).mean(0))[::2]))
    t = {x: np.concatenate(v) for x, v in tot.items()}
    print("  ALL               %.3f   %.3f   %.3f   %.3f   %+.4f (%.4f)  %+.4f (%.4f)   C-B %+.4f (%.4f)" % (
        t["A"].mean(), t["B"].mean(), t["C"].mean(), t["D"].mean(), (t["C"] - t["A"]).mean(), se(t["C"] - t["A"]),
        (t["C"] - t["D"]).mean(), se(t["C"] - t["D"]), (t["C"] - t["B"]).mean(), se(t["C"] - t["B"])), flush=True)


print("A = [%s] for everybody" % bestA)
print("D = teacher's average curve: " + " ".join("%.2f" % x for x in mean))
show("PART 1: the 6 kinds the teacher was raised on (50 fresh students each)", list(enumerate(varlib.KINDS)), res)
NEW = [(16, 3616), (64, 3616), (256, 3616), (32, 1000), (64, 1000), (16, 300)]
varlib.KINDS.extend(NEW)
kinds2 = [(NK + i, kd) for i, kd in enumerate(NEW)]
res2 = [measure(range(950000, 950050), k) for k, _ in kinds2]
show("PART 2: kinds the teacher never met (50 fresh students each)", kinds2, res2)
