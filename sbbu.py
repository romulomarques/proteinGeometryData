from bp import *
from typing import Tuple,List,Dict
import numpy as np

class DGEdge:
    def __init__(self, i:int, j:int, w:float, s:str):
        i, j = sorted([i, j])
        self.i = i
        self.j = j
        self.w = w
        self.s = s

    def __lt__(self, other):
        if self.i == other.i:
            return self.j < other.j
        return self.i < other.i
    
    def __eq__(self, other):
        return self.i == other.i and self.j == other.j
    
    def is_independent(self, other:'DGEdge')->bool:
        if self.i > other.i:
            self, other = other, self

        return self.j < other.i + 3


def sort_edges(edges:List[DGEdge])->List[DGEdge]:
    # sort by (j - i, i, j)
    edges = sorted(edges, key=lambda x: (x.j - x.i, x.i, x.j))

    num_sorted = 0
