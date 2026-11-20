# Does the teacher react to each student, or is it just one fixed curve?
# Replay the teacher's AVERAGE curve as a fixed schedule on the same fresh students and compare.
import sys
import numpy as np
sys.path.insert(0, ".")
import teach
from teach import course
N = 200
SEEDS = range(900000, 900000 + N)
th = np.load("teacher_f%d.npy" % teach.FIXES)
T, R = [], []
for s in SEEDS:
    tr = []
    R.append(course(th, s, trace=tr)[0])
    T.append([lr for _, lr, _ in tr])
T = np.array(T); R = np.array(R)
mean = np.exp(np.log(T).mean(0))
print("teacher            %.3f (+-%.3f)" % (R.mean(), R.std() / np.sqrt(N)))
print("average curve :", " ".join("%.2f" % x for x in mean))
print("min over students:", " ".join("%.2f" % x for x in T.min(0)))
print("max over students:", " ".join("%.2f" % x for x in T.max(0)))
nch = len(mean)
f = lambda fr: float(mean[min(nch - 1, int(fr * (teach.FIXES - 1)) // teach.CHUNK)])
A = np.array([course(None, s, f)[0] for s in SEEDS])
print("average curve replayed as fixed  %.3f (+-%.3f)" % (A.mean(), A.std() / np.sqrt(N)))
d = R - A
print("teacher minus replay, same students: %.4f (+-%.4f)" % (d.mean(), d.std() / np.sqrt(N)))
