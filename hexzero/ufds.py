from __future__ import annotations
import numpy as np


class UFDS: 
    def __init__(self, N: int):
        self.num_sets = N
        self.parent   = np.full(N, -1)


    def find(self, cell: int) -> int:
        if self.parent[cell] < 0:
            return cell
        
        self.parent[cell] = self.find(self.parent[cell])
        return self.parent[cell]


    def related(self, first_cell: int, second_cell: int) -> bool:
        return self.find(first_cell) == self.find(second_cell)


    def join(self, first_cell: int, second_cell: int):
        root_first_cell  = self.find(first_cell)
        root_second_cell = self.find(second_cell)

        if root_first_cell == root_second_cell:
            return
        if self.parent[root_first_cell] < self.parent[root_second_cell]:
            root_first_cell, root_second_cell = root_second_cell, root_first_cell # Swap

        self.parent[root_second_cell] = self.parent[root_second_cell] + self.parent[root_first_cell]
        self.parent[root_first_cell]  = root_second_cell

        self.num_sets = self.num_sets - 1


    def copy(self) -> UFDS:
        new_ufds = UFDS(len(self.parent))
        new_ufds.parent = self.parent.copy()
        new_ufds.num_sets = self.num_sets
        return new_ufds

