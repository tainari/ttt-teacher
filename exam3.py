# Fairness check for experiment 1: give the hand-picked side the "hold, then drop late" shape too.
import numpy as np
import teach
from teach import course
S = range(900000, 900200)
T = np.array([course(np.load("teacher_f300.npy"), s)[0] for s in S])
print("teacher  %.3f" % T.mean())
for c in (2.0, 3.0, 4.5):
    for late, div in ((0.6, 8), (0.5, 8), (0.6, 16)):
        r = np.array([course(None, s, lambda f, c=c: c if f < late else c * div ** (-(f - late) / (1 - late)))[0] for s in S])
        print("%.1f, hold until %.0f%%, then down to 1/%d : %.3f   teacher - this %+.4f (+-%.4f)" % (c, late * 100, div, r.mean(), (T - r).mean(), (T - r).std() / np.sqrt(len(S))))
