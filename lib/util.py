import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from PIL import Image

def cutout_image(image, offsetx, offsety, size, output_size=224):
    """Computes (square) cutouts of an image given x and y offsets and size."""
    (c, h, w) = image.shape
    # offsetx * scale[1] + translation[1] = 0
    # (offsetx + size) * scale[1] + translation[1] = output_size
    # subtract...
    # size * scale[1] = output_size
    # scale[1] = output_size / size
    # translation[1] = -offsetx * output_size / size

    scale = jnp.stack([output_size / size, output_size / size])
    translation = jnp.stack([-offsety * output_size / size, -offsetx * output_size / size])
    return jax.image.scale_and_translate(image,
                                         shape=(c, output_size, output_size),
                                         spatial_dims=(1,2),
                                         scale=scale,
                                         translation=translation,
                                         method='linear')

# vmapped version of cutout_image. Accepts one image, but a tensor of offsets and sizes.
cutouts_image = jax.vmap(cutout_image, in_axes=(None, 0, 0, 0), out_axes=0)

# Accepts a batch of images and returns (the same) cutouts for each image
cutouts_images = jax.vmap(cutouts_image, in_axes=(0, None, None, None), out_axes=0)

def pil_to_tensor(pil_image):
  img = np.array(pil_image).astype('float32')
  img = jnp.array(img) / 255
  img = img.rearrange('h w c -> c h w')
  return img

def pil_from_tensor(image):
  image = image.rearrange('c h w -> h w c')
  image = (image * 256).clamp(0, 255)
  image = np.array(image).astype('uint8')
  return Image.fromarray(image)

# def pil_from_tensor(image):
#   image = image.transpose(1,2,0)
#   image = jnp.clip(image * 256, 0, 255)
#   image = np.array(image).astype('uint8')
#   return Image.fromarray(image)
