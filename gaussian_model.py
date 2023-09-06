"""
This defines the 2D Gaussian model.
"""
import numpy as np

class Gaussian2D:
    """Definition of a 2D Gaussian."""

    def __init__(self):
        self.mean = np.zeros(2)
        self.scaling = np.array([1, 1])
        self.rotation = np.array([0.5])  # radians
        self.opacity = np.array([0.5])
        self.colour = np.array([0.5, 0.5, 0.5])

    def initialise_parameters(self, width, height):
        """Initialise the Gaussian with a uniform distribution."""
        self.mean = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
        self.scaling = np.random.uniform(0, min(width, height) / 10, 2)
        self.rotation = np.random.uniform(0, 2 * np.pi)
        self.opacity = np.random.uniform(0, 1)
        self.colour = np.random.uniform(0, 1, 3)

    def get_covariance(self):
        """Calculate the covariance matrix."""
        scaling_matrix = np.diag(self.scaling)
        rotation_matrix = np.array([[np.cos(self.rotation), -np.sin(self.rotation)],
                                    [np.sin(self.rotation), np.cos(self.rotation)]])
        return scaling_matrix @ rotation_matrix
    