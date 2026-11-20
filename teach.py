# A net that teaches a net (tic-tac-toe).
# Teacher: a tiny net. During a student's course it looks at how the course is going and picks the learning rate.
# Student: the 938-number tic-tac-toe net, a fresh random one for every course.
# The teacher is graded by the test score of the students it raised. It is improved by
# "make slightly changed copies of the teacher, see whose students did better, move toward those"
# (the grade only arrives at the end of a course, so there is no per-step error to push back through).
# usage: python -X utf8 teach.py <minutes> [course_fixes]
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from tttlib import material, Student

FIXES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CHUNK = 15                 # the teacher decides again every 15 fixes
HID = 8                    # teacher: 4 inputs -> 8 -> 1 output  (49 numbers)
LO, HI = -2.5, 0.8         # learning rate = 10 ** (something between LO and HI), i.e. about 0.003 .. 6.3
POP, SIGMA, STEP = 16, 0.15, 0.08   # copies per round (each also mirrored), size of the change, teacher's own step
NSTU = 3                   # students raised per copy
NT = 4 * HID + HID + HID + 1
TR, TE = material()


def teacher_lr(th, obs):
    W1 = th[:4 * HID].reshape(4, HID); b1 = th[4 * HID:5 * HID]; W2 = th[5 * HID:6 * HID]; b2 = th[6 * HID]
    out = np.tanh(obs @ W1 + b1) @ W2 + b2
    return float(10.0 ** (LO + (HI - LO) / (1 + np.exp(-out))))


def course(th, seed, fixed=None, trace=None):
    """Raise one fresh student (numbered by seed). th = teacher numbers, or None with fixed = a function of progress.
    Returns (share of test boards whose top move is a perfect move, test loss)."""
    r = np.random.default_rng(100000 + seed)
    s = Student(seed)
    first = prev = None
    lr = 0.0
    recent = []
    for t in range(FIXES):
        if th is None:
            lr = fixed(t / max(1, FIXES - 1))
        elif t % CHUNK == 0:
            now = float(np.mean(recent)) if recent else 2.3
            first = now if first is None else first
            prev = now if prev is None else prev
            # what the teacher sees: progress, how wrong the student is now (relative to the start), recent change
            lr = teacher_lr(th, np.array([t / FIXES, now / first, (now - prev) / first * 5, 1.0], np.float32))
            if trace is not None:
                trace.append((t, lr, now))
            prev, recent = now, []
        b = r.integers(0, len(TR[0]), 64)
        l = s.step(TR[0][b], TR[1][b], TR[2][b], TR[3][b], lr)
        if not np.isfinite(l):
            return 0.0, 9.0
        recent.append(l)
    loss, ok, _ = s.test(*TE)
    return ok, loss


def grade(th, seeds):
    """Mostly the share of perfect top moves, with a little of the loss so that near-ties are still ordered."""
    r = [course(th, s) for s in seeds]
    return float(np.mean([ok - 0.05 * loss for ok, loss in r]))


if __name__ == "__main__":
    MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    t0 = time.time()
    rng = np.random.default_rng(0)
    th = rng.normal(0, 0.1, NT).astype(np.float32)
    th[6 * HID] = -1.5          # start the teacher at a deliberately low learning rate (about 0.012)
    log, rnd, nseed = [], 0, 0
    print("course %d fixes, teacher has %d numbers" % (FIXES, NT), flush=True)
    while time.time() - t0 < MINUTES * 60:
        rnd += 1
        eps = rng.normal(0, 1, (POP, NT)).astype(np.float32)
        seeds = list(range(nseed, nseed + NSTU)); nseed += NSTU       # the same fresh students for every copy this round
        d = np.array([grade(th + SIGMA * e, seeds) - grade(th - SIGMA * e, seeds) for e in eps])
        if d.std() > 0:
            th = (th + STEP * (eps.T @ (d / d.std())) / POP).astype(np.float32)
        if rnd % 10 == 0 or rnd == 1:
            chk = [course(th, s)[0] for s in range(500000, 500010)]   # check students, never used for improving
            log.append((rnd, round(time.time() - t0), float(np.mean(chk))))
            print("round %4d  %5.0fs  courses %6d  check students %.3f" % (rnd, time.time() - t0, rnd * POP * 2 * NSTU, np.mean(chk)), flush=True)
            np.save("teacher_f%d.npy" % FIXES, th)
            json.dump(log, open("teach_log_f%d.json" % FIXES, "w"))
    np.save("teacher_f%d.npy" % FIXES, th)
    json.dump(log, open("teach_log_f%d.json" % FIXES, "w"))
    print("saved teacher_f%d.npy after %d rounds, %d courses" % (FIXES, rnd, rnd * POP * 2 * NSTU))
