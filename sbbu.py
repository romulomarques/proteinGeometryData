from bp import *
from typing import Tuple,List,Dict
import numpy as np

def create_reflection_operator(a:np.ndarray, b:np.ndarray, c:np.ndarray)->np.ndarray:
    """
    Create a reflection operator for a plane defined by three points. It returns a point in the plane,

    Args:
    a, b, c (np.ndarray): 3D coordinates of three points

    Returns:
    np.ndarray: Reflection operator
    """
    u = b - a
    A11 = np.linalg.norm(u)
    v = c - a
    A22 = np.linalg.norm(v)

    u = u / A11
    v = v / A22

    w = np.cross(u, v)
    w = w / np.linalg.norm(w)

    uv = np.inner(u, v)

    A12 = A11 * uv
    A21 = A22 * uv
    A = [[A11, A12], [A21, A22]]

    return np.linalg.inv(A)


def sbbu_solve_constraint(i:int, j:int, dij:float, xi:np.ndarray, xj:np.ndarray, r:list)->Tuple[float, float]:
    b = np.zeros(len(r), dtype=int)




def sbbu(edges:List[DGEgde], x:np.ndarray, b:List[int], i:int, D:Dict[int, Dict[int, float]])->Tuple[np.ndarray, List[int]]:
    if i == len(x):
        return x, b

    p, w = calc_pw(i, x, D)

    # Try positive direction
    x[i] = p - w
    b[i] = 0
    if is_feasible(D, i, x) and i > 3:
        x, b = sbbu(edges, x, b, i+1, D)
        if x is not None:
            return x, b

    # Try negative direction
    x[i] = p + w
    b[i] = 1
    if is_feasible(D, i, x):
        x, b = sbbu(edges, x, b, i+1, D)
        if x is not None:
            return x, b

    return None, None