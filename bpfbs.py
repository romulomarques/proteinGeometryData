import numpy as np
from numpy.linalg import norm, solve
import unittest
import itertools


def solveEQ3(a, b, c, da, db, dc, stol=1e-4):
    u = b - a
    A11 = norm(u)
    v = c - a
    A22 = norm(v)
    u = u / A11
    v = v / A22
    w = np.cross(u, v)  # w perp u, v
    w = w / norm(w)
    uv = np.inner(u, v)
    A12 = A11 * uv
    A21 = A22 * uv
    # Let y = x - a, then x = y + a and y = y0*u + y1*v + y2*w
    # Using the constraints, we get
    # ||x - a|| = ||y|| = da
    # ||x - b|| = ||y - (b - a)|| = db
    # ||x - c|| = ||y - (c - a)|| = dc
    # Subtrating the square of the first from the one of two last equations, we have
    A = [[A11, A12], [A21, A22]]
    B = [(da ** 2 - db ** 2 + A11 ** 2) / 2.0, (da ** 2 - dc ** 2 + A22 ** 2) / 2.0]
    y0, y1 = solve(A, B)
    s = da ** 2 - y0 ** 2 - y1 ** 2 - 2.0 * y0 * y1 * uv
    if s < 0 and np.abs(s) < stol:
        # print('Warning: base is almost plane (s=%g)' % s)
        s = 0
    if s < 0:  # there is no solution
        # print('solveEQ3:: there is no solution (s=%g)' % s)
        return False, None, None

    proj_x = a + y0 * u + y1 * v  # proj on the plane(a,b,c)
    y2 = np.sqrt(s)

    w = y2 * w
    return proj_x, w


class TreeSearch:
    def __init__(self, D):
        self.D = D
        self.T = self.init_T(D)
        self.x = D.init_x()

    def next(self, x, i, is_feasible):
        pass

    def backtracking(self, i):
        pass

    def init_T(self):
        pass

    def calc_x(self, x, i, child, debug=False):
        ifirst = 4 * i
        ilast = 4 * (i + 1)
        ia, ib, ic = self.D.parents[ifirst:ilast]
        da, db, dc = self.D.d[ifirst:ilast]
        A, B, C = x[ia], x[ib], x[ic]
        p, w = solveEQ3(A, da, B, db, C, dc)
        x[i] = p + w if child else p - w

        if debug:
            dax = norm(A - x[i])
            dbx = norm(B - x[i])
            dcx = norm(C - x[i])
            assert np.abs(dax - da) < 1e-4, "dax=%g, da=%g" % (dax, da)
            assert np.abs(dbx - db) < 1e-4, "dbx=%g, db=%g" % (dbx, db)
            assert np.abs(dcx - dc) < 1e-4, "dcx=%g, dc=%g" % (dcx, dc)


class DFS(TreeSearch):
    def init_T(self):
        T = [False] * self.D.n
        T[:4] = True
        return T

    def next(self, x, i, is_feasible):
        if not is_feasible:
            i = self.backtracking(i)
            if self.T[i] == False:
                self.T[i] = True
                x, i = self.calc_x(x, i, self.T[i])
            else:
                print("No solution")
                exit(0)
        else:
            i += 1
            self.T[i] = False
            x, i = self.calc_x(x, i, False)

    def backtracking(self, i):
        while self.T[i]:
            self.T[i] = False
            i -= 1
        return i


def bp(D, TreeSearch):
    x = D.init_x()  # set the first four points
    T = TreeSearch(D)
    i = 3  # index of the last fixed point
    is_feasible = True
    while True:
        x, i = T.next(x, i, is_feasible)
        is_feasible = D.is_feasible(x, i)
        if not is_feasible:
            continue
        if i == D.n:
            print("Found a feasible solution:", x)
            return x


class TestSolveEQ3(unittest.TestCase):
    def test_solveEQ3(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 1, 0])
        xref = np.array([0.2, 0.3, 0.4])
        da = norm(xref - a)
        db = norm(xref - b)
        dc = norm(xref - c)
        p, w = solveEQ3(a, b, c, da, db, dc)

        xpos = p + w
        dax = norm(a - xpos)
        dbx = norm(b - xpos)
        dcx = norm(c - xpos)
        self.assertAlmostEqual(dax, da, places=4)
        self.assertAlmostEqual(dbx, db, places=4)
        self.assertAlmostEqual(dcx, dc, places=4)

        xneg = p - w
        dax = norm(a - xneg)
        dbx = norm(b - xneg)
        dcx = norm(c - xneg)
        self.assertAlmostEqual(dax, da, places=4)
        self.assertAlmostEqual(dbx, db, places=4)
        self.assertAlmostEqual(dcx, dc, places=4)


class DDGP:
    def __init__(self, d: dict) -> None:
        self.d = d
        self.n = np.max(list(d.keys())) + 1

    def init_x(self):
        x = np.zeros((self.n, 3))

        # set x1
        d01 = self.d[0][1]
        x[1][0] = d01

        # set x2
        d02 = self.d[0][2]
        d12 = self.d[1][2]
        cos_theta = (d02 * d02 + d01 * d01 - d12 * d12) / (2 * d01 * d02)
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)
        x[2][0] = d02 * cos_theta
        x[2][1] = d02 * sin_theta

        # set x3
        d03 = self.d[0][3]
        d13 = self.d[1][3]
        d23 = self.d[2][3]
        p, w = solveEQ3(x[0], x[1], x[2], d03, d13, d23)
        x[3] = p - w
        return x

    def is_feasible(self, x, i, dtol=1e-4):
        if i < 4:
            return True
        di = self.d[i]
        for j in di.keys():
            dij = di[j]
            dij_calc = norm(x[i] - x[j])
            if np.abs(dij - dij_calc) > dtol:
                return False
        return True


def fake_DDGP(n, num_prune_edges=1, seed=0):
    # def random seed
    np.random.seed(seed)
    xsol = np.random.rand(n, 3)
    parents = [[] for i in range(n)]
    d = {}
    for i in range(4):
        d[i] = {}
        for j in range(i, 4):
            d[i][j] = norm(xsol[i] - xsol[j])

    for i in range(4, n):
        parents[i] = sorted(np.random.choice(i, 3, replace=False))    
        d[i] = {}
        for j in parents[i]:
            d[i][j] = norm(xsol[i] - xsol[j])
        if i % 2 == 0:
            continue
        # A: set of antecessor points that are not parents
        A = list(set(range(i)) - set(parents[i]))
        A = np.random.choice(A, np.min([num_prune_edges, len(A)]))
        for j in A:
            d[i][j] = norm(xsol[i] - xsol[j])

    # symmetrize
    for i in range(n):
        for j in d[i].keys():
            d[j][i] = d[i][j]
    return DDGP(d), xsol


class TestDDGP(unittest.TestCase):
    def test_init_x(self):
        D, _ = fake_DDGP(7)
        x = D.init_x()
        self.assertAlmostEqual(norm(x[1] - x[0]), D.d[0][1], places=4)
        self.assertAlmostEqual(norm(x[2] - x[0]), D.d[0][2], places=4)
        self.assertAlmostEqual(norm(x[3] - x[0]), D.d[0][3], places=4)
        self.assertAlmostEqual(norm(x[2] - x[1]), D.d[1][2], places=4)
        self.assertAlmostEqual(norm(x[3] - x[1]), D.d[1][3], places=4)
        self.assertAlmostEqual(norm(x[3] - x[2]), D.d[2][3], places=4)


    def test_is_feasible(self):
        D, xsol = fake_DDGP(7)
        self.assertTrue(D.is_feasible(xsol, 6))

        x = xsol.copy()
        x[6] = np.zeros(3)
        self.assertFalse(D.is_feasible(x, 6))


if __name__ == "__main__":
    unittest.main()
