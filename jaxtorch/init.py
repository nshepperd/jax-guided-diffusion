import jax
import numpy as np
from jaxtorch import core

def desc(name, *args, **kwargs):
    return name + '(' + ', '.join([repr(x) for x in args] + [f'{k}={v}' for (k,v) in kwargs.items()]) + ')'

def zeros(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape=shape, initializer=jax.nn.initializers.zeros, desc=desc('zeros', *shape))

def ones(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape=shape, initializer=jax.nn.initializers.ones, desc=desc('ones', *shape))

def normal(*shape, stddev=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape=shape, initializer=jax.nn.initializers.normal(stddev=stddev), desc=desc('normal', *shape, stddev=stddev))

def const(vals):
    shape = jax.core.canonicalize_shape(vals.shape)
    return core.Param(shape=shape, initializer=lambda key, shape: vals)

def glorot_normal(*shape):
    shape = jax.core.canonicalize_shape(shape)
    scale = np.sqrt(2.0 / (shape[0] + shape[1]))
    return core.Param(shape=shape, initializer=jax.nn.initializers.normal(stddev=scale))

def uniform(*shape, min=-1.0, max=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape=shape, initializer=lambda key, shape: jax.random.uniform(key, shape, minval=min, maxval=max))