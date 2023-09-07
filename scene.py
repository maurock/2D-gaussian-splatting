"""
This defines a Scene, which is a collection of 2D Gaussians.
"""
from gaussian_model import Gaussian2D
import numpy as np
import torch

class Scene:

    def __init__(self, image, N: int):
        self.N = N
        self.height, self.width = image.shape[:2]
        self.gaussians = [Gaussian2D(self.width, self.height) for _ in range(N)]

def render(scene: Scene):
    """
    Process:
    1. Create a blank image.
        For each pixel:
        2. Compute the footprint (value of each gaussian at that point)
        3. Multiply each footprint value by its RGB
        4. Average the resulting values
    """ 
    image = torch.zeros(size=(scene.width, scene.height, 3))

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            densities = torch.tensor([gaussian.get_density(torch.tensor([i, j], dtype=torch.float32)) for gaussian in scene.gaussians])
            colours = torch.cat([gaussian.colour for gaussian in scene.gaussians], axis=0)
            opacities = torch.tensor([gaussian.opacity for gaussian in scene.gaussians])

            image[i, j, :] = torch.sum(densities[:, None] * colours * opacities[:, None], axis=0)
    
    # image = image/image.max()
    return image


