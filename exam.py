# Final exam: the teacher net against hand-picked learning rates, on students never used while the teacher was raised.
# usage: python -X utf8 exam.py [n_students] [course_fixes]
import sys

import numpy as np

sys.path.insert(0, ".")
import teach
from teach import course

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEEDS = range(900000, 900000 + N)


def report(name, th=None, fixed=None):
    r = np.array([course(th, s, fixed)[0] for s in SEEDS])
    print("%-22s %.3f  (+-%.3f)   worst student %.3f" % (name, r.mean(), r.std() / np.sqrt(N), r.min()), flush=True)
    return r.mean()


print("course %d fixes, %d fresh students each" % (teach.FIXES, N))
for c in (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
    report("const %.1f" % c, fixed=lambda f, c=c: c)
for a, b in ((2.0, 0.5), (2.0, 0.2), (3.0, 0.3), (1.5, 0.5), (3.0, 1.0)):
    report("%.1f -> %.1f" % (a, b), fixed=lambda f, a=a, b=b: a * (b / a) ** f)
th = np.load("teacher_f%d.npy" % teach.FIXES)
report("TEACHER NET", th=th)
print("what the teacher chose (fix number: learning rate, student's recent loss):")
for s in (900000, 900001, 900002):
    tr = []
    ok = course(th, s, trace=tr)[0]
    print("  student %d -> %.3f | " % (s, ok) + "  ".join("%d:%.2f" % (t, lr) for t, lr, _ in tr))
