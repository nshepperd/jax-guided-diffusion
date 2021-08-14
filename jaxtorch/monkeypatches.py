"""Implement a bunch of convenience methods for jax arrays.

"""

import sys
import jax
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
    sqrt = lambda arr: jax.numpy.sqrt(arr),
    add = lambda a, b: a + b,
    sub = lambda a, b: a - b,
    div = lambda a, b: a / b,
    mul = lambda a, b: a * b,
    arcsin = lambda a: jax.numpy.arcsin(a),
    clamp = lambda a, minval, maxval: jax.numpy.clip(a, a_min=minval, a_max=maxval),
    unsqueeze = lambda arr, axis=0: jax.numpy.expand_dims(arr, axis),
    rearrange = rearrange,
)
