#%%

"""
This defines a Scene, which is a collection of 2D Gaussians.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from typing import List, NamedTuple
from gaussian_model import *
from functools import partial


class Scene2D(NamedTuple):
    """Definition of a 2D Scene."""
    gaussians: List[Gaussian2D]
    # width: int
    # height: int

    # gaussians: jnp.ndarray



def init_scene(key, image, N: int) -> Scene2D:
    """Returns the initial model params."""
    keys = get_new_keys(key, N)
    gaussians = [init_gaussian(keys[i], image.shape[0], image.shape[1]) for i in range(N)]
    return Scene2D(gaussians)

    ## Concat all atributes of a
    # gaussians = [init_gaussian(key, image.shape[0], image.shape[1]) for _ in range(N)]
    # return Scene2D(gaussians)


# @partial(jax.jit, static_argnums=(0,))
def render_pixel(scene: Scene2D, x: jnp.ndarray):
    """Render a single pixel."""
    # densities = jnp.concatenate([get_density(gaussian, x) for gaussian in scene.gaussians], axis=0)

    densities = jnp.array([get_density(gaussian, x) for gaussian in scene.gaussians])[:, None]
    colours = jnp.concatenate([gaussian.colour for gaussian in scene.gaussians], axis=0)
    opacities = jnp.array([gaussian.opacity for gaussian in scene.gaussians])

    # ## Ugly trick to add extra dimensions to colours and opacities (For VMAPPING)
    # col_len = len(colours.shape)
    # extra_len = len(densities.shape) - col_len
    # extra_dims = tuple([col_len+i for i in range(extra_len)])
    # colours = jnp.expand_dims(colours, axis=tuple(extra_dims))
    # opacities = jnp.expand_dims(opacities, axis=tuple(extra_dims))

    # densities = jax.vmap(get_density, in_axes=0)(scene.gaussian, x)[:, None]
    # colours = jax.vmap(get_colour, in_axes=0)(scene.gaussians)
    # opacities = jax.vmap(get_opacity, in_axis=0)(scene.gaussians)

    return jnp.sum(densities * colours * opacities, axis=0)



    # densities = [get_density(gaussian, x)[None, :] for gaussian in scene.gaussians]
    # colours = [gaussian.colour for gaussian in scene.gaussians]
    # opacities = [gaussian.opacity for gaussian in scene.gaussians]

    # pytree = jax.tree_map(lambda d, c, o: d*c*o, densities, colours, opacities)
    # return jnp.sum(pytree, axis=0)


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
    image = jnp.zeros_like(ref_image)

    # for (int i = 0; i < N-1; i++){
    #     for (int j = 0; j < M-1; j++){
    #         l = i + j*(N-1);

    # for i in range(0, image.shape[0]):
    #     for j in range(0, image.shape[1]):

    #         x = jnp.array([i, j], dtype=jnp.float32)[:, None]

    #         # densities = jnp.concatenate([get_density(gaussian, x) for gaussian in scene.gaussians], axis=0)
    #         # colours = jnp.concatenate([gaussian.colour for gaussian in scene.gaussians], axis=0)
    #         # opacities = jnp.array([gaussian.opacity for gaussian in scene.gaussians])

    #         # image = image.at[i, j, :].set(jnp.sum(densities * colours * opacities, axis=0))

    #         image = image.at[i, j, :].set(render_pixel(scene, x))



    ## X is the meshgrids
    meshgrid = jnp.meshgrid(jnp.arange(0, ref_image.shape[0]), jnp.arange(0, ref_image.shape[1]))
    xs = jnp.stack(meshgrid, axis=0).T[..., None]

    # print("xs shape: ", xs.shape)
    image = render_pixels_2D(scene, xs)


    # image = image/image.max()
    return image.squeeze()


# @jax.value_and_grad
def mse_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    ## Add penalty for values greater than 1
    return jnp.mean((image - ref_image) ** 2) + 1.*jnp.mean(jnp.where(image > 1., image, 0.))


@partial(jax.jit, static_argnums=(3,))
def train_step(scene: Scene2D, ref_image: jnp.ndarray, opt_state, optimiser):
    """Perform a single training step."""
    loss, grad = jax.value_and_grad(mse_loss)(scene, ref_image)
    updates, new_opt_state = optimiser.update(grad, opt_state)
    new_scene = optax.apply_updates(scene, updates)
    return new_scene, new_opt_state, loss


## Print the Scene2D as a pytree


if __name__=='__main__':

    key = jax.random.PRNGKey(2)
    # key = None

    scene = init_scene(key, jnp.zeros((256, 256)), 50)

    # load image called luna.jpeg and save it as a numpy array
    ref_image = plt.imread('luna.jpeg')/255.
    plt.imshow(ref_image)
    plt.show()

    image = render(scene, ref_image)

    plt.imshow(image)
    plt.show()


    nb_iter = 10000
    ## Init optimiser
    ## Set exponential smoothing parameter to 0.9
    # scheduler = optax.constant_decay(1e-3)
    scheduler = optax.exponential_decay(1e-2, nb_iter, 0.90)
    optimiser = optax.adam(scheduler)
    opt_state = optimiser.init(scene)

    ## Training loop
    for i in range(nb_iter):
        scene, opt_state, loss = train_step(scene, ref_image, opt_state, optimiser)
        # print("Loss: ", loss)
        ## Print loss and iteration number
        if i % 100 == 0 or i < 3:
            print(f'Iteration: {i}            Loss: {loss:.3f}')


    ## Evaluate the final scene
    image = render(scene, ref_image)
    plt.imshow(image)
    plt.show()


    print('Done')
