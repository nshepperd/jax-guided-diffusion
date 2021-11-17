import math
import jax
import jax.numpy as jnp
import numpy as np
from jaxtorch import core

def mkdesc(name, *args, **kwargs):
    return name + '(' + ', '.join([repr(x) for x in args] + [f'{k}={v}' for (k,v) in kwargs.items()]) + ')'

def zeros(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jnp.zeros(shape), desc=mkdesc('zeros', *shape))

def ones(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jnp.ones(shape), desc=mkdesc('ones', *shape))

def normal(*shape, stddev=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: stddev * jax.random.normal(key, shape), desc=mkdesc('normal', *shape, stddev=stddev))

def const(tensor):
    shape = jax.core.canonicalize_shape(tensor.shape)
    return core.Param(lambda key: tensor)

def glorot_normal(*shape):
    shape = jax.core.canonicalize_shape(shape)
    fan_out = shape[0] * np.prod(shape[2:])
    fan_in = shape[1] * np.prod(shape[2:])
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    return core.Param(lambda key: stddev * jax.random.normal(key, shape), desc=mkdesc('glorot_normal', *shape))

def uniform(*shape, min=-1.0, max=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jax.random.uniform(key, shape, minval=min, maxval=max))

def kaiming_uniform(*shape, a=0):
    fan_in = np.prod(shape[1:])
    gain = math.sqrt(2.0 / (1 + a ** 2))
    bound = gain * math.sqrt(3.0 / fan_in)
    return uniform(*shape, min=-bound, max=bound)
