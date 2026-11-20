# Tic-tac-toe pieces for the "a net that teaches a net" experiment.
# Teaching material: every reachable unfinished board with its perfect answer (worked out by full search).
# Student: a small net (18 -> 32 tanh -> 9 moves + 1 value, 938 numbers).
import functools
import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]:
            return b[i]
    return 0


@functools.lru_cache(maxsize=None)
def solve(b):
    """b: tuple of 9 from the mover's view (1 = mover, -1 = other). Returns (value for mover, tuple of best moves)."""
    best, moves = -2, []
    for a in range(9):
        if b[a] == 0:
            nb = list(b); nb[a] = 1
            if winner(nb) == 1:
                v = 1
            elif all(nb):
                v = 0
            else:
                v = -solve(tuple(-x for x in nb))[0]
            if v > best:
                best, moves = v, [a]
            elif v == best:
                moves.append(a)
    return best, tuple(moves)


def all_boards():
    seen, todo = set(), [tuple([0] * 9)]
    while todo:
        b = todo.pop()
        if b in seen:
            continue
        seen.add(b)
        for a in range(9):
            if b[a] == 0:
                nb = list(b); nb[a] = 1
                if winner(nb) == 0 and not all(nb):
                    todo.append(tuple(-x for x in nb))
    return sorted(seen)


def material(seed=0, hold=0.2):
    """Returns train and held-out arrays: X (n,18), PI (n,9) even over the perfect moves, Z (n,), LEGAL (n,9)."""
    bs = all_boards()
    X = np.zeros((len(bs), 18), np.float32); PI = np.zeros((len(bs), 9), np.float32)
    Z = np.zeros(len(bs), np.float32); L = np.zeros((len(bs), 9), bool)
    for i, b in enumerate(bs):
        v, mv = solve(b)
        a = np.array(b)
        X[i, :9] = a == 1; X[i, 9:] = a == -1; L[i] = a == 0
        PI[i, list(mv)] = 1.0 / len(mv); Z[i] = v
    k = np.random.default_rng(seed).permutation(len(bs))
    n = int(len(bs) * (1 - hold))
    tr, te = k[:n], k[n:]
    return (X[tr], PI[tr], Z[tr], L[tr]), (X[te], PI[te], Z[te], L[te])


class Student:
    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.W1 = (r.random((18, 32), np.float32) - 0.5); self.b1 = np.zeros(32, np.float32)
        self.Wp = (r.random((32, 9), np.float32) - 0.5); self.bp = np.zeros(9, np.float32)
        self.Wv = (r.random(32, np.float32) - 0.5); self.bv = np.float32(0)

    def forward(self, X, L):
        h = np.tanh(X @ self.W1 + self.b1)
        lg = np.where(L, h @ self.Wp + self.bp, -1e9)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        v = np.tanh(h @ self.Wv + self.bv)
        return h, p, v

    def step(self, X, PI, Z, L, lr):
        """One fix (plain gradient descent, written out by hand). Returns the loss before the fix."""
        h, p, v = self.forward(X, L)
        n = len(X)
        loss = -(PI * np.log(p + 1e-12)).sum(1).mean() + ((v - Z) ** 2).mean()
        dl = (p - PI) / n
        dv = 2 * (v - Z) * (1 - v * v) / n
        ds = (dl @ self.Wp.T + np.outer(dv, self.Wv)) * (1 - h * h)
        self.Wp -= lr * (h.T @ dl); self.bp -= lr * dl.sum(0)
        self.Wv -= lr * (h.T @ dv); self.bv -= lr * dv.sum()
        self.W1 -= lr * (X.T @ ds); self.b1 -= lr * ds.sum(0)
        return float(loss)

    def test(self, X, PI, Z, L):
        """loss (lower is better), share of boards where the top move is a perfect move, value error."""
        h, p, v = self.forward(X, L)
        pol = -(PI * np.log(p + 1e-12)).sum(1).mean()
        val = ((v - Z) ** 2).mean()
        ok = (PI[np.arange(len(X)), p.argmax(1)] > 0).mean()
        return float(pol + val), float(ok), float(val)
