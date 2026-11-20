# Varied students: the size of the net and the amount of teaching material differ from student to student.
# kind = seed % 6 :  hidden size (8, 32, 128)  x  material (all 3616 boards, only 300 boards)
import numpy as np
from tttlib import material, Student

TR, TE = material()
HIDS = (8, 32, 128)
NMAT = (len(TR[0]), 300)
KINDS = [(h, n) for h in HIDS for n in NMAT]


class VStudent(Student):
    def __init__(self, seed, hid):
        r = np.random.default_rng(seed)
        self.W1 = (r.random((18, hid), np.float32) - 0.5); self.b1 = np.zeros(hid, np.float32)
        self.Wp = (r.random((hid, 9), np.float32) - 0.5); self.bp = np.zeros(9, np.float32)
        self.Wv = (r.random(hid, np.float32) - 0.5); self.bv = np.float32(0)


def make(seed, kind=None):
    """Returns (student, row numbers of the boards this student may learn from, kind)."""
    kind = seed % len(KINDS) if kind is None else kind
    hid, n = KINDS[kind]
    rows = np.random.default_rng(700000 + seed).permutation(len(TR[0]))[:n]
    return VStudent(seed, hid), rows, kind


def vcourse(seed, lr_fn, fixes=300, kind=None):
    """lr_fn(progress 0..1) -> learning rate. Returns share of test boards whose top move is perfect."""
    s, rows, _ = make(seed, kind)
    r = np.random.default_rng(100000 + seed)
    for t in range(fixes):
        b = rows[r.integers(0, len(rows), 64)]
        l = s.step(TR[0][b], TR[1][b], TR[2][b], TR[3][b], lr_fn(t / max(1, fixes - 1)))
        if not np.isfinite(l):
            return 0.0
    return s.test(*TE)[1]
