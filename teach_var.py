# A net that teaches VARIED students (net size and amount of material differ; the teacher is not told which).
# Same method as teach.py. The teacher sees: progress, loss now / first loss, recent change, loss now (absolute), 1.
# usage: python -X utf8 teach_var.py <minutes>
import json
import sys
import time

import numpy as np

from varlib import TR, TE, KINDS, make

FIXES, CHUNK, HID, NIN = 300, 15, 8, 5
LO, HI = -2.5, 0.8
POP, SIGMA, STEP = 16, 0.15, 0.08
NT = NIN * HID + HID + HID + 1


def teacher_lr(th, obs):
    W1 = th[:NIN * HID].reshape(NIN, HID); b1 = th[NIN * HID:(NIN + 1) * HID]
    W2 = th[(NIN + 1) * HID:(NIN + 2) * HID]; b2 = th[(NIN + 2) * HID]
    out = np.tanh(obs @ W1 + b1) @ W2 + b2
    return float(10.0 ** (LO + (HI - LO) / (1 + np.exp(-out))))


def course(th, seed, fixed=None, trace=None, kind=None):
    """Returns (share of test boards whose top move is perfect, test loss)."""
    s, rows, _ = make(seed, kind)
    r = np.random.default_rng(100000 + seed)
    first = prev = None
    lr, recent = 0.0, []
    for t in range(FIXES):
        if th is None:
            lr = fixed(t / max(1, FIXES - 1))
        elif t % CHUNK == 0:
            now = float(np.mean(recent)) if recent else 2.3
            if recent and first is None:
                first = now
            f0 = first if first is not None else now
            prev = now if prev is None else prev
            lr = teacher_lr(th, np.array([t / FIXES, now / f0, (now - prev) / f0 * 5, now / 2.3, 1.0], np.float32))
            if trace is not None:
                trace.append((t, lr, now))
            prev, recent = now, []
        b = rows[r.integers(0, len(rows), 64)]
        l = s.step(TR[0][b], TR[1][b], TR[2][b], TR[3][b], lr)
        if not np.isfinite(l):
            return 0.0, 9.0
        recent.append(l)
    loss, ok, _ = s.test(*TE)
    return ok, min(loss, 9.0)


def grade(th, seeds):
    r = [course(th, s) for s in seeds]
    return float(np.mean([ok - 0.05 * loss for ok, loss in r]))


if __name__ == "__main__":
    MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    t0 = time.time()
    rng = np.random.default_rng(0)
    th = rng.normal(0, 0.1, NT).astype(np.float32)
    th[(NIN + 2) * HID] = 1.0          # start near learning rate 0.8
    log, rnd, nseed = [], 0, 0
    NK = len(KINDS)
    print("teacher has %d numbers, %d kinds of students" % (NT, NK), flush=True)
    while time.time() - t0 < MINUTES * 60:
        rnd += 1
        eps = rng.normal(0, 1, (POP, NT)).astype(np.float32)
        seeds = list(range(nseed, nseed + NK)); nseed += NK      # one fresh student of every kind
        d = np.array([grade(th + SIGMA * e, seeds) - grade(th - SIGMA * e, seeds) for e in eps])
        if d.std() > 0:
            th = (th + STEP * (eps.T @ (d / d.std())) / POP).astype(np.float32)
        if rnd % 10 == 0 or rnd == 1:
            chk = [course(th, s)[0] for s in range(500000, 500030)]
            log.append((rnd, round(time.time() - t0), float(np.mean(chk))))
            print("round %4d  %5.0fs  check students %.3f" % (rnd, time.time() - t0, np.mean(chk)), flush=True)
            np.save("teacher_var.npy", th)
            json.dump(log, open("teach_var_log.json", "w"))
    np.save("teacher_var.npy", th)
    json.dump(log, open("teach_var_log.json", "w"))
    print("saved teacher_var.npy after %d rounds" % rnd)
