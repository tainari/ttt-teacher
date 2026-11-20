# Groundwork: does the best learning rate really differ between kinds of students?
import numpy as np
from varlib import KINDS, vcourse
LRS = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5)
SH = [("const", lambda c: (lambda f: c)), ("x -> x/8 late", lambda c: (lambda f: c if f < 0.6 else c * 8 ** (-(f - 0.6) / 0.4)))]
for name, mk in SH:
    print(name, "   lr:", "  ".join("%5.1f" % c for c in LRS))
    for k, (hid, n) in enumerate(KINDS):
        row = [np.mean([vcourse(800000 + i, mk(c), kind=k) for i in range(20)]) for c in LRS]
        print("  hid %3d mat %4d: " % (hid, n) + "  ".join("%.3f" % x for x in row) + "   best lr %.1f" % LRS[int(np.argmax(row))], flush=True)
