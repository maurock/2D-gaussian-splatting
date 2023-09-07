"""
This defines the 2D Gaussian model.
"""
import numpy as np
import torch

class Gaussian2D:
    """Definition of a 2D Gaussian."""

    def __init__(self, width=100, height=100):
        self.mean = torch.tensor([np.random.uniform(0, width), np.random.uniform(0, height)], requires_grad=True, dtype=torch.float32)
        self.scaling = torch.tensor(np.random.uniform(0, min(width, height) / 5, (2,1)), requires_grad=True, dtype=torch.float32)
        #self.rotation = torch.tensor(np.random.uniform(0, 2 * np.pi), requires_grad=True, dtype=torch.float32)
        self.rotation = torch.tensor(np.pi/4, requires_grad=True, dtype=torch.float32)
        self.opacity = torch.tensor(np.random.uniform(0, 1), requires_grad=True, dtype=torch.float32)
        self.colour = torch.tensor(np.random.uniform(0, 1, (1, 3)), requires_grad=True, dtype=torch.float32)

    def get_covariance(self):
        """Calculate the covariance matrix. """
        scaling_matrix = torch.diag(self.scaling.squeeze())
        rotation_matrix = torch.tensor([[torch.cos(self.rotation), -torch.sin(self.rotation)],
                                    [torch.sin(self.rotation), torch.cos(self.rotation)]])
        return rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T  # TODO: turn scaling into a matrix?
    
    def get_density(self, x):
        # TODO: add normalisation term if needed
        return torch.exp(-0.5 *  (x - self.mean)[:, None].T  @ torch.linalg.inv(self.get_covariance()) @ (x - self.mean))




def is_positive_semi_definite(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Compute eigenvalues
    eigenvalues, _ = torch.eig(matrix, eigenvectors=False)
    
    # Check if all eigenvalues are non-negative
    if (eigenvalues[:, 0] >= 0).all():
        return True
    else:
        return False