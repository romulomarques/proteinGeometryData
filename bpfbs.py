# Sugestão:

# 1. Teste com distâncias de poda exatas;
# 2. Teste com distâncias de poda inexatas, onde podemos postergar a poda tomando intervalos maiores;

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
    if s < 0 and np.abs(s) > stol:  # there is no solution
        # print('solveEQ3:: there is no solution (s=%g)' % s)
        # unfeasible solution
        w = np.array([np.inf, 0, 0])
        proj_x = np.zeros(3)
        return proj_x, w

    if s < 0:
        # print('Warning: base is almost plane (s=%g)' % s)
        s = 0

    proj_x = a + y0 * u + y1 * v  # proj on the plane(a,b,c)
    y2 = np.sqrt(s)

    w = y2 * w
    return proj_x, w


def calc_x(i, b, x, D, debug=False):
    ia, ib, ic = D.parents[i]
    da, db, dc = [D.d[i][ia], D.d[i][ib], D.d[i][ic]]
    A, B, C = x[ia], x[ib], x[ic]
    p, w = solveEQ3(A, B, C, da, db, dc)
    x[i] = p + w if (b == 1) else p - w

    if debug:
        dax = norm(A - x[i])
        dbx = norm(B - x[i])
        dcx = norm(C - x[i])
        assert np.abs(dax - da) < 1e-4, "dax=%g, da=%g" % (dax, da)
        assert np.abs(dbx - db) < 1e-4, "dbx=%g, db=%g" % (dbx, db)
        assert np.abs(dcx - dc) < 1e-4, "dcx=%g, dc=%g" % (dcx, dc)


def init_x(D):
    n = D.n
    x = np.zeros((n, 3), dtype=float)

    # set x1
    d01 = D.d[0][1]
    x[1][0] = d01

    # set x2
    d02 = D.d[0][2]
    d12 = D.d[1][2]
    cos_theta = (d02 * d02 + d01 * d01 - d12 * d12) / (2 * d01 * d02)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    x[2][0] = d02 * cos_theta
    x[2][1] = d02 * sin_theta

    # set x3
    d03 = D.d[0][3]
    d13 = D.d[1][3]
    d23 = D.d[2][3]
    p, w = solveEQ3(x[0], x[1], x[2], d03, d13, d23)
    x[3] = p + w
    return x


class DDGP:
    def __init__(self, d: dict, parents: list) -> None:
        self.d = d
        self.parents = parents
        self.n = np.max(list(d.keys())) + 1
        self.triang_bounds = []

        # triangular inequalities
        neighs = [set(d[i].keys()) for i in range(self.n)]
        for i in range(self.n):
            ui = {}
            for j in range(0, i):
                neighs_ij = neighs[i].intersection(neighs[j])
                for k in neighs_ij:
                    if k < i:
                        continue
                    sij = d[i][k] + d[j][k]
                    uij = ui.get(j, np.inf)
                    if sij < uij:
                        ui[j] = sij
            self.triang_bounds.append(list(ui.items()))

    def is_feasible(self, x, i, dtol=1e-4):
        if i < 4:
            return True
        di = self.d[i]
        xi = x[i]
        L = list(di.keys())
        # feasibility with respect to antecessors
        for j in L:
            if j < i:
                dij = di[j]
                dij_calc = norm(xi - x[j])
                if np.abs(dij - dij_calc) > dtol:
                    return False
            else:
                break
        # feasibility with respect to triangular inequalities
        for k, uik in self.triang_bounds[i]:
            dik = norm(xi - x[k])
            if dik >= uik + dtol:
                return False
        return True


class DFS:
    def __init__(self, D, last_node=1):
        self.D = D  # este atributo deve ser do tipo DDGP
        self.n = self.D.n
        self.x = init_x(D)  # set the first four points
        self.T = np.zeros(self.n, dtype=int)
        self.T[:4] = last_node
        self.last_node = last_node

    def backtracking(self, i):
        while self.T[i] == self.last_node:
            self.T[i] = 0
            i -= 1
        return i

    def next(self, i, is_feasible):
        if not is_feasible:
            i = self.backtracking(i)
            if i < 0:
                raise Exception("No solution")
            self.T[i] += 1  # next node
        else:
            i += 1
            self.T[i] = 0  # first node
        calc_x(i, self.T[i], self.x, self.D)
        return i


class FBS:
    def __init__(self, D, states, p):
        self.x = init_x(D)
        self.states = states
        self.p = p
        self.filter_states()
        self.last_node = len(self.states) - 1
        self.D = D
        self.n = self.D.n
        self.T = [0 for i in range(self.n)]  # vector of states id
        self.F = [0 for i in range(self.n)]  # vector of flips
        self.F[0] = 1
        self.TLvl = 0  # level of the current state
        self.TVal = self.states[0]  # state of the current node
        self.iX = np.zeros(self.n, dtype=int)  # index of the last fixed point
        self.iX[0] = 4

    def filter_states(self):
        # ToDo Idealmente este método deve levar em consideração o
        # comprimento do estado 's' e a probabilidade 'p'
        return True

    def calc_x(self, i):
        k = i - self.iX[self.TLvl]
        b = int(self.TVal[k])
        calc_x(i, b, self.x, self.D)

    def backtracking(self):
        while self.T[self.TLvl] == self.last_node:
            self.T[self.TLvl] = 0
            self.F[self.TLvl] = 0
            self.TLvl -= 1
        if self.TLvl < 0:
            raise Exception("No solution")
        self.T[self.TLvl] += 1
        s = self.states[self.T[self.TLvl]]
        s = self.config_state(s)
        self.TVal = s
        if self.TLvl > 0:
            k = self.T[self.TLvl - 1]
            self.iX[self.TLvl] = self.iX[self.TLvl - 1] + len(self.states[k])
        return self.iX[self.TLvl]

    def config_state(self, s):
        if self.TLvl == 0:
            self.F[self.TLvl] = 1
            s = [1 - x for x in s]
        else:
            k = self.T[self.TLvl - 1]
            f = self.F[self.TLvl - 1]
            b = self.states[k][-1]
            # (f == 1 and b == 0) or (f == 0 and b == 1)
            if f + b == 1:
                # flip s
                s = [1 - x for x in s]
                self.F[self.TLvl] = 1
        return s

    def next(self, i, is_feasible):
        if not is_feasible:
            i = self.backtracking()
        else:
            i += 1
            if i == self.iX[self.TLvl] + len(self.TVal):
                self.TLvl += 1
                self.T[self.TLvl] = 0
                s = self.states[0]
                s = self.config_state(s)
                self.TVal = s
                self.iX[self.TLvl] = i
        if (self.TLvl == 0) and (i == 4):
            s = self.states[self.T[self.TLvl]]
            s = self.config_state(s)
            self.TVal = s
        self.calc_x(i)
        return i


def bp(D: DDGP, TS):
    i = 3  # index of the last fixed vertice
    is_feasible = True
    while True:
        i = TS.next(i, is_feasible)
        is_feasible = D.is_feasible(TS.x, i)
        if not is_feasible:
            continue
        if i == (D.n - 1):
            print("Found a feasible solution.")
            return TS.x


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


def fake_DDGP(n, num_prune_edges=1, seed=0):
    # def random seed
    np.random.seed(seed)
    xsol = np.random.rand(n, 3)
    parents = [[] for i in range(n)]
    d = {}

    d01 = norm(xsol[1] - xsol[0])
    d02 = norm(xsol[2] - xsol[0])
    d12 = norm(xsol[2] - xsol[1])
    d03 = norm(xsol[3] - xsol[0])
    d13 = norm(xsol[3] - xsol[1])
    d23 = norm(xsol[3] - xsol[2])

    xsol[0], xsol[1], xsol[2], xsol[3] = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # set x1
    xsol[1][0] = d01

    # set x2
    cos_theta = (d02 * d02 + d01 * d01 - d12 * d12) / (2 * d01 * d02)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    xsol[2][0] = d02 * cos_theta
    xsol[2][1] = d02 * sin_theta

    # set x3
    p, w = solveEQ3(xsol[0], xsol[1], xsol[2], d03, d13, d23)
    xsol[3] = p + w

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

    return DDGP(d, parents), xsol


class TestInitX(unittest.TestCase):
    def test_init_x(self):
        D, _ = fake_DDGP(7)
        x = init_x(D)
        self.assertAlmostEqual(norm(x[1] - x[0]), D.d[0][1], places=4)
        self.assertAlmostEqual(norm(x[2] - x[0]), D.d[0][2], places=4)
        self.assertAlmostEqual(norm(x[3] - x[0]), D.d[0][3], places=4)
        self.assertAlmostEqual(norm(x[2] - x[1]), D.d[1][2], places=4)
        self.assertAlmostEqual(norm(x[3] - x[1]), D.d[1][3], places=4)
        self.assertAlmostEqual(norm(x[3] - x[2]), D.d[2][3], places=4)


class TestDDGP(unittest.TestCase):
    def test_is_feasible(self):
        D, xsol = fake_DDGP(7)
        self.assertTrue(D.is_feasible(xsol, 6))

        x = xsol.copy()
        x[6] = np.zeros(3)
        self.assertFalse(D.is_feasible(x, 6))


def determineT(D, x, dtol=1e-4):
    T = np.array([0] * len(x))
    for i in range(4):
        T[i] = 1
    for i in range(4, len(x)):
        ia, ib, ic = D.parents[i]
        da, db, dc = [D.d[i][ia], D.d[i][ib], D.d[i][ic]]
        A, B, C = x[ia], x[ib], x[ic]
        p, w = solveEQ3(A, B, C, da, db, dc)
        if norm(x[i] - (p - w)) < dtol:
            T[i] = 0
        else:
            if norm(x[i] - (p + w)) < dtol:
                T[i] = 1
            else:
                print("Error creating T from x!")
                exit(0)
    return T


class TesteDFS(unittest.TestCase):
    def test_backtracking_dfs_1(self):
        D, _ = fake_DDGP(100)
        dfs = DFS(D)
        Tsol = dfs.T.copy()

        end_bt, start_bt = sorted(
            np.random.choice(np.asarray(range(4, D.n)), 2, replace=False)
        )
        for i in range(start_bt, end_bt, -1):
            dfs.T[i] = 1

        i_bt = dfs.backtracking(start_bt)
        self.assertEqual(
            i_bt,
            end_bt,
            "Backtracking failed: It does not stop in the first FALSE found before the current TRUE!",
        )
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Backtracking failed: The vertex between the current vertex and the backtracked vertex are not all set to the FALSE node.",
        )

    def test_backtracking_dfs_2(self):
        D, _ = fake_DDGP(100)
        dfs = DFS(D)

        end_bt, start_bt = sorted(
            np.random.choice(np.asarray(range(4, D.n)), 2, replace=False)
        )

        dfs.T[end_bt + 1 : start_bt] = 1

        Tsol = dfs.T.copy()

        i_bt = dfs.backtracking(start_bt)
        self.assertEqual(
            i_bt,
            start_bt,
            "Backtracking failed: It does not stop in the same current vertex!",
        )
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Backtracking failed: The current vertex has not been set to the TRUE node OR another vertex has been set to the wrong node.",
        )

    def test_backtracking_dfs_3(self):
        D, _ = fake_DDGP(7)
        dfs = DFS(D)

        dfs.T[4] = True
        dfs.T[5] = True
        dfs.T[6] = True

        Tsol = np.array([0] * D.n)

        i_bt = dfs.backtracking(D.n - 1)
        self.assertEqual(
            i_bt,
            -1,
            "Backtracking failed: It do not identify the absence of solutions!",
        )
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Backtracking failed: Some vertex has been set to True.",
        )

    # Current vertex is on the node FALSE and it is infeasible.
    def test_next_dfs_1(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        # Testing if 'next' goes to the correct level of the BP tree.
        dfs = DFS(D)
        dfs.x = xsol
        dfs.T = T
        Tsol = [dfs.T[i] for i in range(dfs.D.n)]
        is_feasible = False
        for i in range(4, dfs.D.n):
            if dfs.T[i] == 0:
                break
        Tsol[i] = 1
        j = dfs.next(i, is_feasible)
        self.assertEqual(
            i,
            j,
            "Next failed: It do not stay on the current vertex when the current vertex is on the node FALSE and it is infeasible!",
        )

        # Testing if 'next' goes to the correct node of the BP tree.
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Next failed: The current vertex has not been set to the TRUE node OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node FALSE and it is feasible.
    def test_next_dfs_2(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        # Testing if 'next' goes to the correct level of the BP tree.
        dfs = DFS(D)
        dfs.x = xsol
        dfs.T = T
        Tsol = [dfs.T[i] for i in range(dfs.D.n)]
        is_feasible = True
        for i in range(4, dfs.D.n):
            if dfs.T[i] == False:
                break
        Tsol[i + 1] = False
        j = dfs.next(i, is_feasible)
        self.assertEqual(
            i + 1,
            j,
            "Next failed: It do not go to the successor vertex when the current vertex is on the node False and it is feasible!",
        )

        # Testing if 'next' goes to the correct node of the BP tree.
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Next failed: The successor vertex of the current vertex is not set to the FALSE child node OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node TRUE and it is infeasible.
    def test_next_dfs_3(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        dfs = DFS(D)
        dfs.x = xsol.copy()
        dfs.T = T
        for i in range(4, dfs.D.n):
            if dfs.T[i] == 0 and dfs.T[i + 1] == 1:
                break
        for j in range(i + 2, dfs.D.n):
            if dfs.T[j] == 0:
                break

        Tsol = dfs.T.copy()
        Tsol[i + 1 : j] = 0
        Tsol[i] = 1

        is_feasible = False
        i_bt = dfs.next(j - 1, is_feasible)

        # Testing if 'next' updates correctly the coordinates of the realized vertices.
        ia, ib, ic = dfs.D.parents[i]
        A, B, C = xsol[[ia, ib, ic]]
        da, db, dc = norm(A - dfs.x[i]), norm(B - dfs.x[i]), norm(C - dfs.x[i])
        self.assertAlmostEqual(da, dfs.D.d[i][ia], places=4)
        self.assertAlmostEqual(db, dfs.D.d[i][ib], places=4)
        self.assertAlmostEqual(dc, dfs.D.d[i][ic], places=4)
        self.assertTrue(
            norm(dfs.x[i] - xsol[i]) > 1e-4,
            "The current vertex is not on the TRUE node.",
        )

        # Testing if 'next' goes to the correct level of the BP tree.
        self.assertEqual(
            i,
            i_bt,
            "Next failed: It do not go to the first previous vertex set to FALSE when the current vertex is on the node TRUE and it is infeasible!",
        )

        # Testing if 'next' goes to the correct node of the BP tree.
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Next failed: A vertex between the current vertex and the first previous FALSE has not been set to FALSE, OR the first previous FALSE has not been set to TRUE, OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node TRUE and it is feasible.
    def test_next_dfs_4(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        dfs = DFS(D)
        dfs.x = xsol
        dfs.T = T
        is_feasible = True
        for i in range(4, dfs.D.n):
            if dfs.T[i] == 1:
                break
        Tsol = dfs.T.copy()
        j = dfs.next(i, is_feasible)
        Tsol[j] = 0

        if j == dfs.D.n - 1:
            raise ValueError("Reformulate the test!")

        self.assertEqual(
            i + 1,
            j,
            "Next failed: It do not go to the successor vertex when the current vertex is on the node True and it is feasible!",
        )

        # Testing if 'next' goes to the correct node of the BP tree.
        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Next failed: The successor vertex of the current vertex is not set to the FALSE child node OR another vertex has been set to the wrong node.",
        )


class TestBP(unittest.TestCase):
    def test_bp_dfs(self):
        D, xsol = fake_DDGP(10)
        dfs = DFS(D)
        x = bp(D, dfs)

    def test_bp_fbs(self):
        D, xsol = fake_DDGP(10)
        states = [(1, 1, 1), (0, 0, 0), (1, 1, 0), (1, 1), (1, 0), (0, 0), (0, 1)]
        p = [0.7, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1]
        fbs = FBS(D, states, p)
        x = bp(D, fbs)


if __name__ == "__main__":
    unittest.main()
