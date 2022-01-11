"""Implement a bunch of convenience methods for jax arrays.

"""

import sys
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import functools
from einops import rearrange

def register(**kwargs):
    for (attr, fun) in kwargs.items():
        if hasattr(jnp.zeros([]), attr):
            print(f'Not monkeypatching DeviceArray and Tracer with `{attr}`, because that method is already implemented.', file=sys.stderr)
            continue
        setattr(jaxlib.xla_extension.DeviceArrayBase, attr, fun)
        setattr(jax.core.Tracer, attr, fun)

def broadcast_to(arr, shape):
  shape = (shape,) if jnp.ndim(shape) == 0 else shape
  shape = jax.core.canonicalize_shape(shape)  # check that shape is concrete
  arr_shape = arr.shape
  if jax.core.symbolic_equal_shape(arr_shape, shape):
    return arr
  else:
    nlead = len(shape) - len(arr_shape)
    shape_tail = shape[nlead:]
    compatible = all(jax.core.symbolic_equal_one_of_dim(arr_d, [1, shape_d])
                      for arr_d, shape_d in zip(arr_shape, shape_tail))
    if nlead < 0 or not compatible:
      msg = "Incompatible shapes for broadcasting: {} and requested shape {}"
      raise ValueError(msg.format(arr_shape, shape))
    diff, = np.where(tuple(not jax.core.symbolic_equal_dim(arr_d, shape_d)
                           for arr_d, shape_d in zip(arr_shape, shape_tail)))
    new_dims = tuple(range(nlead)) + tuple(nlead + diff)
    kept_dims = tuple(np.delete(np.arange(len(shape)), new_dims))
    return jax.lax.broadcast_in_dim(jnp.squeeze(arr, tuple(diff)), shape, kept_dims)

register(
    square = lambda arr: arr**2,
    sqrt = jnp.sqrt,
    abs = jnp.abs,
    add = lambda a, b: a + b,
    sub = lambda a, b: a - b,
    div = lambda a, b: a / b,
    mul = lambda a, b: a * b,
    sin = jnp.sin,
    cos = jnp.cos,
    arcsin = jnp.arcsin,
    arccos = jnp.arccos,
    log = jnp.log,
    exp = jnp.exp,
    clamp = lambda a, minval=None, maxval=None: jnp.clip(a, a_min=minval, a_max=maxval),
    unsqueeze = lambda arr, axis=0: jnp.expand_dims(arr, axis),
    rearrange = rearrange,
    broadcast_to = broadcast_to,
    isfinite = jnp.isfinite,
    isnan = jnp.isnan,
)
