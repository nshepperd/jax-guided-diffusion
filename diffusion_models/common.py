import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init
from dataclasses import dataclass
from functools import partial
import math

# Common nn modules.
class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return jnp.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = init.normal(out_features // 2, in_features, stddev=std)
        self.std = std

    def forward(self, cx, input):
        f = 2 * math.pi * input @ cx[self.weight].transpose()
        return jnp.concatenate([f.cos(), f.sin()], axis=-1)


class AvgPool2d(nn.Module):
    def forward(self, cx, x):
        [n, c, h, w] = x.shape
        x = x.reshape([n, c, h//2, 2, w//2, 2])
        x = x.mean((3,5))
        return x


def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(list(input.shape) + [shape[2], shape[3]])

Tensor = None

@dataclass
@jax.tree_util.register_pytree_node_class
class DiffusionOutput:
    v: Tensor
    pred: Tensor
    eps: Tensor

    def tree_flatten(self):
        return [self.v, self.pred, self.eps], []

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        return cls(*dynamic)

    def __mul__(self, scalar):
        return DiffusionOutput(self.v * scalar, self.pred * scalar, self.eps * scalar)
    def __add__(self, other):
        return DiffusionOutput(self.v + other.v,
                               self.pred + other.pred,
                               self.eps + other.eps)


@jax.tree_util.register_pytree_node_class
class Partial(object):
  """Wrap a function with arguments as a jittable object."""
  def __init__(self, f, *args, **kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs
    self.p = partial(f, *args, **kwargs)
  def __call__(self, *args, **kwargs):
    return self.p(*args, **kwargs)
  def tree_flatten(self):
    return [self.args, self.kwargs], [self.f]
  def tree_unflatten(static, dynamic):
    [args, kwargs] = dynamic
    [f] = static
    return Partial(f, *args, **kwargs)

def make_partial(f):
  def p(*args, **kwargs):
    return Partial(f, *args, **kwargs)
  return p

def make_cosine_model(model, lazy_params=None):
    @make_partial
    @jax.jit
    def forward(params, x, cosine_t, key, **kwargs):
        n = x.shape[0]
        cx = Context(params, key).eval_mode_()
        return model(cx, x, cosine_t.broadcast_to([n]), **kwargs)
    if lazy_params is not None:
        def wrapper(**kwargs):
            return forward(lazy_params(), **kwargs)
        return wrapper
    else:
        def wrapper(params, **kwargs):
            return forward(params, **kwargs)
        return wrapper

@jax.jit
def blur_fft(image, std):
  std = jnp.asarray(std).clamp(1e-18)
  [*_, h, w] = image.shape
  dy = jnp.arange(-(h-1)//2, (h+1)//2)
  dy = jnp.roll(dy, -(h-1)//2)
  dx = jnp.arange(-(w-1)//2, (w+1)//2)
  dx = jnp.roll(dx, -(w-1)//2)
  distance = dy[:, None]**2 + dx[None, :]**2
  kernel = jnp.exp(-0.5 * distance / std**2)
  kernel /= kernel.sum()
  return jnp.fft.ifft2(jnp.fft.fft2(image, norm='forward') * jnp.fft.fft2(kernel, norm='backward'), norm='forward').real

def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3,1,1)
    std = jnp.array(std).reshape(3,1,1)
    def forward(image):
        return (image - mean) / std
    return forward

def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt().clamp(1e-12)

@jax.tree_util.register_pytree_node_class
class Static(object):
    def __init__(self, value):
        self.value = value
    def __bool__(self):
        return bool(self.value)
    def tree_flatten(self):
        return [], [self.value]
    @classmethod
    def tree_unflatten(cls, static, dynamic):
        return cls(*static)

def expand_batched(array, n):
    return jnp.asarray(array).broadcast_to([n])[:,None,None,None]

@jax.tree_util.register_pytree_node_class
class LerpModels(object):
    """Linear combination of diffusion models."""
    def __init__(self, models):
        self.models = models
    def __call__(self, x, t, key):
        n = x.shape[0]
        outputs = [(m(x,t,key), expand_batched(w, n)) for (m,w) in self.models]
        v = sum(out.v * w for (out, w) in outputs)
        pred = sum(out.pred * w for (out, w) in outputs)
        eps = sum(out.eps * w for (out, w) in outputs)
        return DiffusionOutput(v, pred, eps)
    def tree_flatten(self):
        return [self.models], []
    def tree_unflatten(static, dynamic):
        return LerpModels(*dynamic)