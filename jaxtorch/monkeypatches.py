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
        if (hasattr(jaxlib.xla_extension.DeviceArrayBase, attr) or hasattr(jax.core.Tracer, attr)):
            print(f'Not monkeypatching DeviceArray and Tracer with `{attr}`, because that method is already implemented.', file=sys.stderr)
            continue
        setattr(jaxlib.xla_extension.DeviceArrayBase, attr, fun)
        setattr(jax.core.Tracer, attr, fun)

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
    broadcast_to = jnp.broadcast_to,
    isfinite = jnp.isfinite,
    isnan = jnp.isnan,
)
