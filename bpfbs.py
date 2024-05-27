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


def calc_x(self, i, b, x, D, debug=False):
    ia, ib, ic = D.parents[i]
    da, db, dc = [D.d[i][ia], D.d[i][ib], D.d[i][ic]]
    A, B, C = x[ia], x[ib], x[ic]
    p, w = solveEQ3(A, B, C, da, db, dc)
    x[i] = p + w if (b == 0) else p - w

    if debug:
        dax = norm(A - x[i])
        dbx = norm(B - x[i])
        dcx = norm(C - x[i])
        assert np.abs(dax - da) < 1e-4, "dax=%g, da=%g" % (dax, da)
        assert np.abs(dbx - db) < 1e-4, "dbx=%g, db=%g" % (dbx, db)
        assert np.abs(dcx - dc) < 1e-4, "dcx=%g, dc=%g" % (dcx, dc)


def init_x(self, D):
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
    x[3] = p - w
    return x


class DDGP:
    def __init__(self, d: dict, parents: list) -> None:
        self.d = d
        self.parents = parents
        self.n = np.max(list(d.keys())) + 1

    def is_feasible(self, x, i, dtol=1e-4):
        if i <= -1:
            return False
        if i < 4:
            return True
        di = self.d[i]
        for j in di.keys():
            dij = di[j]
            dij_calc = norm(x[i] - x[j])
            if np.abs(dij - dij_calc) > dtol:
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
        return self.T

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
    def ___init__(self, fn, D):
        self.states = []
        self.p = []  # probabilidade de cada estado
        with open(fn, "r") as f:
            for line in f:
                data = [x for x in line.split()]
                p = float(data[0])
                s = [int(x) for x in data[1:]]
                if self.check_acceptance(p, s):
                    self.p.append(p)
                    self.states.append(s)
                # ToDo - verificar se o estado deve ser aceito
        self.last_node = len(self.states) - 1
        self.x = init_x(D)
        self.D = D
        self.iX = []  # index of the last fixed point
        self.n = self.D.n
        self.T = [0 for i in range(self.n)]
        self.TLvl = 0  # level of the current state
        self.TVal = ""  # state of the current node

    def check_acceptance(self, p, s):
        # ToDo Idealmente este método deve levar em consideração o
        # comprimento do estado 's' e a probabilidade 'p'
        return True

    def calc_x(self, i):
        k = self.iX[self.TLvl] - i
        b = self.TVal[k]
        calc_x(i, b, self.x, self.D)

    def backtracking(self):
        while self.T[self.TLvl] == self.last_node:
            self.T[self.TLvl] = 0
            self.Tvl -= 1
        if self.TLvl < 0:
            raise Exception("No solution")
        self.T[self.TLvl] += 1
        self.TVal = self.states[self.T[self.TLvl]]
        self.iX[self.TLvl] = self.iX[self.TLvl - 1] + len(self.TVal)
        return self.iX[self.TLvl]

    def next(self, i, is_feasible):
        if not is_feasible:
            i = self.backtracking()
        else:
            self.TLvl += 1
            self.T[self.TLvl] = 0
            self.TVal = self.states[self.T[self.Tvl]]
            self.iX[self.TLvl] = self.iX[self.Tvl - 1] + len(self.TVal)
        self.calc_x(i)
        return i


def bp(D, TreeSearch):
    # Acho que nao precisa da linha abaixo, visto que o x jah eh inicializado na TreeSearch
    # x = D.init_x()  # set the first four points
    T = TreeSearch(D)
    i = 3  # index of the last fixed point
    is_feasible = True
    while True:
        i = T.next(i, is_feasible)
        is_feasible = D.is_feasible(T.x, i)
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
    return DDGP(d, parents), xsol


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


def determineT(D, x, dtol=1e-4):
    T = np.array([False] * len(x))
    for i in range(4):
        T[i] = True
    for i in range(4, len(x)):
        ia, ib, ic = D.parents[i]
        da, db, dc = [D.d[i][ia], D.d[i][ib], D.d[i][ic]]
        A, B, C = x[ia], x[ib], x[ic]
        p, w = solveEQ3(A, B, C, da, db, dc)
        if norm(x[i] - (p - w)) < dtol:
            T[i] = False
        else:
            if norm(x[i] - (p + w)) < dtol:
                T[i] = True
            else:
                print("Error creating T from x!")
                exit(0)
    return T


class TesteDFS(unittest.TestCase):
    def test_backtracking_1(self):
        D, _ = fake_DDGP(100)
        dfs = DFS(D)
        Tsol = dfs.T.copy()

        end_bt, start_bt = sorted(
            np.random.choice(np.asarray(range(4, D.n)), 2, replace=False)
        )
        for i in range(start_bt, end_bt, -1):
            dfs.T[i] = True

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

    def test_backtracking_2(self):
        D, _ = fake_DDGP(100)
        dfs = DFS(D)

        end_bt, start_bt = sorted(
            np.random.choice(np.asarray(range(4, D.n)), 2, replace=False)
        )

        dfs.T[end_bt + 1 : start_bt] = True

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

    def test_backtracking_3(self):
        D, _ = fake_DDGP(7)
        dfs = DFS(D)

        dfs.T[4] = True
        dfs.T[5] = True
        dfs.T[6] = True

        Tsol = np.array([False] * D.n)

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
    def test_next_1(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        dfs = DFS(D)
        dfs.x = xsol
        dfs.T = T
        Tsol = [dfs.T[i] for i in range(dfs.D.n)]
        is_feasible = False
        for i in range(4, dfs.D.n):
            if dfs.T[i] == False:
                break
        Tsol[i] = True
        j = dfs.next(i, is_feasible)
        self.assertEqual(
            i,
            j,
            "Next failed: It do not go to the brother node TRUE when the current vertex is on the node FALSE and it is infeasible!",
        )

        diff = np.array(dfs.T) == np.array(Tsol)
        is_equal = True
        for i in range(dfs.D.n):
            is_equal = is_equal and diff[i]
        self.assertTrue(
            is_equal,
            "Next failed: The current vertex has not been set to the TRUE node OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node FALSE and it is feasible.
    def test_next_2(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

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
            "Next failed: It do not go to the child node FALSE when the current vertex is on the node False and it is feasible!",
        )

        diff = np.array(dfs.T) == np.array(Tsol)
        is_equal = True
        for i in range(dfs.D.n):
            is_equal = is_equal and diff[i]
        self.assertTrue(
            is_equal,
            "Next failed: The successor vertex of the current vertex is not set to the FALSE child node OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node TRUE and it is infeasible.
    def test_next_3(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        dfs = DFS(D)
        dfs.x = xsol.copy()
        dfs.T = T
        for i in range(4, dfs.D.n):
            if dfs.T[i] == False and dfs.T[i + 1] == True:
                break
        for j in range(i + 2, dfs.D.n):
            if dfs.T[j] == False:
                break

        Tsol = dfs.T.copy()
        Tsol[i + 1 : j] = False
        Tsol[i] = True

        is_feasible = False
        i_bt = dfs.next(j - 1, is_feasible)

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

        self.assertEqual(
            i,
            i_bt,
            "Next failed: It do not go to the first previous node FALSE when the current vertex is on the node TRUE and it is infeasible!",
        )

        self.assertTrue(
            np.all(np.array(dfs.T) == np.array(Tsol)),
            "Next failed: The current vertex has not been set to the TRUE node OR another vertex has been set to the wrong node.",
        )

    # Current vertex is on the node TRUE and it is feasible.
    def test_next_4(self):
        D, xsol = fake_DDGP(20)
        T = determineT(D, xsol)

        dfs = DFS(D)
        dfs.x = xsol
        dfs.T = T
        is_feasible = True
        for i in range(4, dfs.D.n):
            if dfs.T[i] == True:
                break
        j = dfs.next(i, is_feasible)

        if j == dfs.D.n - 1:
            raise ValueError("Reformulate the test!")

        self.assertEqual(
            i + 1,
            j,
            "Next failed: It do not go to the child node FALSE when the current vertex is on the node True and it is feasible!",
        )

        self.assertFalse(dfs.T[j])

        # fazer os 4 casos: (False, infeasible); (False, feasible); (True, infeasible); (True, feasible)


if __name__ == "__main__":
    unittest.main()
