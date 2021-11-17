import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module
from jaxtorch import init

class Upsample2d(Module):
    """Upsample image or features to 2x. Input: (n, c, h, w)."""
    def __init__(self, method='linear'):
        self.method = method
    def forward(self, cx, x):
        return jaxtorch.image.upsample2x(x, method=self.method)

class Downsample2d(Module):
    """Downsample image or features to 1/2x. Input: (n, c, h, w)."""
    def __init__(self, method='linear'):
        self.method = method
    def forward(self, cx, x):
        return jaxtorch.image.downsample2x(x, method=self.method)
