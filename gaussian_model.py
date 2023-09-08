"""
This defines the 2D Gaussian model.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple

from utils import get_new_keys






class Gaussian2D(NamedTuple):
    """Definition of a 2D Gaussian."""
    mean: jnp.ndarray
    scaling: jnp.ndarray
    rotation: jnp.ndarray
    opacity: jnp.ndarray
    colour: jnp.ndarray



def init_gaussian(key, width=256., height=256.) -> Gaussian2D:
    """Returns the initial model params."""
    keys = get_new_keys(key, 5)

    ## Uniformly initialise parameters of a 2D gaussian
    mean = jax.random.uniform(keys[0], (2,1), minval=0, maxval=min(width, height))
    scaling = jax.random.uniform(keys[1], (2,1), minval=0, maxval=min(width, height)/5)
    rotation = jax.random.uniform(keys[2], (1,), minval=0, maxval=2*jnp.pi)
    opacity = jax.random.uniform(keys[3], (1,), minval=0, maxval=1)
    colour = jax.random.uniform(keys[4], (1,3), minval=0, maxval=1)

    return Gaussian2D(mean, scaling, rotation, opacity, colour)


def get_covariance(gaussian: Gaussian2D):
    """Calculate the covariance matrix. """
    scaling_matrix = jnp.diag(gaussian.scaling.squeeze())

    cos, sin = jnp.cos(gaussian.rotation), jnp.sin(gaussian.rotation)
    rotation_matrix = jnp.array([[cos, -sin], [sin, cos]]).squeeze()

    # print("Covariance mat shape: ", (rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T).shape)

    return rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T 


def get_density(gaussian, x):
    """Calculate the density of the gaussian at a given point."""
    ret = jnp.exp(-0.5 * (x - gaussian.mean).T  @ jnp.linalg.inv(get_covariance(gaussian)) @ (x - gaussian.mean))

    # print(" shapes: ", x.shape, gaussian.mean.shape,)
    # print("Density shape: ", ((x - gaussian.mean).T  @ jnp.linalg.inv(get_covariance(gaussian)) @ (x - gaussian.mean)).shape)
    # print("Density shape: ", ret.shape)

    return ret.squeeze()
    # return ret


def get_colour(gaussian):
    """Return the colour of the gaussian."""
    return gaussian.colour

def get_opacity(gaussian):
    """Return the opacity of the gaussian."""
    return gaussian.opacity


def is_positive_semi_definite(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Compute eigenvalues
    eigenvalues, _ = jnp.linalg.eig(matrix, eigenvectors=False)
    
    # Check if all eigenvalues are non-negative
    if (eigenvalues[:, 0] >= 0).all():
        return True
    else:
        return False
