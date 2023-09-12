# from gaussian_model import Gaussian2D
# from scene import Scene, render
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
if __name__=='__main__':
    # our_scene = Scene(N=10, image=np.zeros((100, 100)))
    # image = render(our_scene)
    #
    # # load image called luna.jpeg and save it as a numpy array
    # # image = plt.imread('luna.jpeg')/255
    # # plt.imshow(image)
    # # plt.show()
    #
    # plt.imshow(image.detach().numpy())
    # plt.show()


    x = jnp.linspace(0,10,101)
    y = jnp.sin(x)

    plt.plot(x,y)
    print('Done')


#%%
