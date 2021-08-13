import jax
import jaxlib
import numpy as np
import functools

def register(**kwargs):
    for (attr, fun) in kwargs.items():
        setattr(jaxlib.xla_extension.DeviceArrayBase, attr, fun)
        setattr(jax.core.Tracer, attr, fun)

register(
    square = lambda arr: arr**2,
    sqrt = lambda arr: jax.numpy.sqrt(arr),
    add = lambda a, b: a + b,
    div = lambda a, b: a / b,
    mul = lambda a, b: a * b,
    arcsin = lambda a: jax.numpy.arcsin(a),
    clamp = lambda a, minval, maxval: jax.numpy.clip(a, a_min=minval, a_max=maxval)
)
