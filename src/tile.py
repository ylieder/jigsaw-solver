import numpy as np


class Vertex:
    def __init__(self) -> None:
        self.vertices = np.arange(123)

class Tile:
    def __init__(self, rows, cols) -> None:
        self.rows = rows
        self.cols = cols

        self.tile_ids = np.arange(self.rows * self.cols)
        self.vertices = None
    
    @property
    def vertices(self):
        if self.vertices is None:
            raise
        
