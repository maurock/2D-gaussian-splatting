"""
This defines a Scene, which is a collection of 2D Gaussians.
"""
from gaussian_model import Gaussian2D
import numpy as np

class Scene:

    def __init__(self, N: int):
        self.N = N
        self.gaussians = [Gaussian2D() for _ in range(N)]
        self.height = 0
        self.width = 0
    
    def initialise_size(self, image: np.ndarray):
        self.height, self.width = image.shape[:2]
    
    def initialise_gaussians(self):
        for gaussian in self.gaussians:
            gaussian.initialise_parameters(self.width, self.height)
