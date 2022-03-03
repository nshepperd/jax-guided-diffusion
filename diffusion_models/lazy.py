import jax
import jax.numpy as jnp
import requests, io
import numpy as np
import jaxtorch


from diffusion_models.cache import WeakCache

class LazyParams(object):
    """Lazily download parameters and load onto gpu. Parameters are kept in cpu memory and only loaded to gpu as long as needed."""
    fetch = None

    def __init__(self, load=None, params=None):
        self.cache = WeakCache(jnp.array)
        self.load = load
        self.params = jax.tree_util.tree_map(np.asarray, params)
    def __call__(self, *args, **kwargs):
        if self.params is None:
            self.params = jax.tree_util.tree_map(np.asarray, self.load(*args, **kwargs))
        return jax.tree_util.tree_map(lambda x: self.cache(x) if type(x) is np.ndarray else x, self.params)

    @staticmethod
    def pt(url, key=None):
        def load():
            params = jaxtorch.pt.load(LazyParams.fetch(url))
            if key is not None:
                params = params[key]
            return params
        return LazyParams(load)