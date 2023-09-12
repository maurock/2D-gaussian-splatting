#%%

"""
This defines a Scene, which is a collection of 2D Gaussians.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm
import time

from typing import List, NamedTuple
from gaussian_model import *
from functools import partial

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# jax.config.update('jax_platform_name', 'cpu')           ## CPU is faster here !
# jax.config.update("jax_enable_x64", True)

Scene2D = jnp.ndarray

#%%

# x = jnp.linspace(0,10,101)
# y = jnp.sin(x)

# plt.plot(x,y)



#%%


def init_scene(key, image, N: int) -> Scene2D:
    """Returns the initial model params."""
    keys = get_new_keys(key, N)
    gaussians = [init_gaussian(keys[i], image.shape[0], image.shape[1]) for i in range(N)]
    return jnp.stack(gaussians, axis=0)

    ## Concat all atributes of a
    # gaussians = [init_gaussian(key, image.shape[0], image.shape[1]) for _ in range(N)]
    # return Scene2D(gaussians)


# @partial(jax.jit, static_argnums=(0,))
def render_pixel(scene: Scene2D, x: jnp.ndarray):
    """Render a single pixel."""
    # densities = jnp.concatenate([get_density(gaussian, x) for gaussian in scene.gaussians], axis=0)

    # densities = jnp.array([get_density(gaussian, x) for gaussian in scene.gaussians])[:, None]
    # colours = jnp.concatenate([gaussian.colour for gaussian in scene.gaussians], axis=0)
    # opacities = jnp.array([gaussian.opacity for gaussian in scene.gaussians])

    # ## Ugly trick to add extra dimensions to colours and opacities (For VMAPPING)
    # col_len = len(colours.shape)
    # extra_len = len(densities.shape) - col_len
    # extra_dims = tuple([col_len+i for i in range(extra_len)])
    # colours = jnp.expand_dims(colours, axis=tuple(extra_dims))
    # opacities = jnp.expand_dims(opacities, axis=tuple(extra_dims))


    means = scene[:, :2]
    scalings = scene[:, 2:4]
    rotations = scene[:, 4:5]
    opacities = scene[:, 5:6]
    colours = scene[:, 6:]

    densities = jax.vmap(get_density, in_axes=(0, 0, 0, None))(means, scalings, rotations, x)[:, None]
    # colours = jax.vmap(get_colour, in_axes=0)(scene.gaussians)
    # opacities = jax.vmap(get_opacity, in_axis=0)(scene.gaussians)

    return jnp.sum(densities * colours * opacities, axis=0)
    # return jax.nn.sigmoid(jnp.sum(densities * colours * opacities, axis=0))



render_pixels_1D = jax.vmap(render_pixel, in_axes=(None, 0), out_axes=0)
render_pixels_2D = jax.vmap(render_pixels_1D, in_axes=(None, 1), out_axes=1)



def render(scene: Scene2D, ref_image: jnp.ndarray):
    """
    Process:
    1. Create a blank image.
        For each pixel:
        2. Compute the footprint (value of each gaussian at that point)
        3. Multiply each footprint value by its RGB
        4. Average the resulting values
    """

    ## X is the meshgrids
    meshgrid = jnp.meshgrid(jnp.arange(0, ref_image.shape[0]), jnp.arange(0, ref_image.shape[1]))
    pixels = jnp.stack(meshgrid, axis=0).T

    image = render_pixels_2D(scene, pixels)

    # image = image/image.max()
    # image = jnp.abs(jax.nn.tanh(image))
    # image = jax.nn.sigmoid(image)

    return image.squeeze()

def penalty_loss(image):
    return jnp.mean(jnp.where(image > 1., image, 0.))

def mse_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    ## Add penalty for values greater than 1
    return jnp.mean((image - ref_image) ** 2)

def mae_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    ## Add penalty for values greater than 1
    # return jnp.mean(jnp.abs(image - ref_image)) + 0.*penalty_loss(image)
    return jnp.mean(jnp.abs(image - ref_image))


def dice_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    ## Add penalty for values greater than 1
    return (jnp.sum(image * ref_image) * 2 / jnp.sum(image) + jnp.sum(ref_image)) + 1.*penalty_loss(image)


@partial(jax.jit, static_argnums=(3,))
def train_step(scene: Scene2D, ref_image: jnp.ndarray, opt_state, optimiser):
    """Perform a single training step."""
    loss, grad = jax.value_and_grad(mae_loss)(scene, ref_image)
    updates, new_opt_state = optimiser.update(grad, opt_state)
    new_scene = optax.apply_updates(scene, updates)
    return new_scene, new_opt_state, loss


#%%
## Print the Scene2D as a pytree


# if __name__=='__main__':
    # %timeit


# key = jax.random.PRNGKey(42)
key = jax.random.PRNGKey(time.time_ns())
# key = None

# scene = init_scene(key, jnp.zeros((256, 256)), 1000)
scene = init_scene(key, jnp.zeros((100, 100)), 2000)

# load image called luna.jpeg and save it as a numpy array
# ref_image = plt.imread('luna.jpeg')/255.

ref_image = plt.imread('earth.jpeg')[...,:3]/255.

plt.imshow(ref_image)
plt.show()




image = render(scene, ref_image)

plt.imshow(image)
plt.show()


nb_iter = 2000
## Init optimiser
## Set exponential smoothing parameter to 0.9
# scheduler = optax.constant_decay(1e-3)
scheduler = optax.exponential_decay(1e-3, nb_iter, 0.9)
optimiser = optax.adam(scheduler)
opt_state = optimiser.init(scene)

losses = []
# start_time = time.time()
## Training loop
for i in tqdm(range(1, nb_iter+1), disable=True):
    scene, opt_state, loss = train_step(scene, ref_image, opt_state, optimiser)
    # print("Loss: ", loss)
    ## Print loss and iteration number
    losses.append(loss)
    if i % 100 == 0 or i <= 3:
        print(f'Iteration: {i}  Loss: {loss:.3f}')
# wall_time = time.time() - start_time

## Print time and number of params in scene
print(f'Number of params: {jnp.size(scene)}')
print("Number of pixels: ", jnp.size(ref_image))

## Evaluate the final scene
image = render(scene, ref_image)

fig, (ax) = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title("Final render")

ax[1].imshow(ref_image)
ax[1].set_title("Reference")
plt.show()


## Plot loss in log scale
# plt.plot(losses)

print('Done')

#%%
# plt.savefig("final_render")
# #%%
# plt.imread("final_render.png")
# #%%
