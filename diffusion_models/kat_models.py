# Kat's models from <https://github.com/crowsonkb/v-diffusion-jax>

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

import diffusion as v_diffusion

from diffusion_models.common import *
from diffusion_models.schedules import cosine
from diffusion_models.lazy import LazyParams

@jax.tree_util.register_pytree_node_class
class KatModel(object):
    def __init__(self, model, params, **kwargs):
      self.model = model
      self.params = params
      self.kwargs = kwargs
    @jax.jit
    def __call__(self, x, cosine_t, key):
        n = x.shape[0]
        alpha, sigma = cosine.to_alpha_sigma(cosine_t)
        v = self.model.apply(self.params, key, x, cosine_t.broadcast_to([n]), self.kwargs)
        pred = x * alpha - v * sigma
        eps = x * sigma + v * alpha
        return DiffusionOutput(v, pred, eps)
    def tree_flatten(self):
        return [self.params, self.kwargs], [self.model]
    def tree_unflatten(static, dynamic):
        [params, kwargs] = dynamic
        [model] = static
        return KatModel(model, params, **kwargs)


danbooru_128_model = v_diffusion.get_model('danbooru_128')
danbooru_128_params = LazyParams(lambda: v_diffusion.load_params(LazyParams.fetch('https://the-eye.eu/public/AI/models/v-diffusion/danbooru_128.pkl')))

wikiart_256_model = v_diffusion.get_model('wikiart_256')
wikiart_256_params = LazyParams(lambda: v_diffusion.load_params(LazyParams.fetch('https://the-eye.eu/public/AI/models/v-diffusion/wikiart_256.pkl')))

wikiart_128_model = v_diffusion.get_model('wikiart_128')
wikiart_128_params = LazyParams(lambda: v_diffusion.load_params(LazyParams.fetch('https://the-eye.eu/public/AI/models/v-diffusion/wikiart_128.pkl')))

imagenet_128_model = v_diffusion.get_model('imagenet_128')
imagenet_128_params = LazyParams(lambda: v_diffusion.load_params(LazyParams.fetch('https://the-eye.eu/public/AI/models/v-diffusion/imagenet_128.pkl')))

def danbooru_128():
    return KatModel(danbooru_128_model, danbooru_128_params())
def wikiart_128():
    return KatModel(wikiart_128_model, wikiart_128_params())
def wikiart_256():
    return KatModel(wikiart_256_model, wikiart_256_params())
def imagenet_128():
    return KatModel(imagenet_128_model, imagenet_128_params())
