import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, PRNG, Context, ParamState
from jaxtorch import init

def square(x):
    return x**2

class Identity(Module):
    def forward(self, cx, x):
        return x


class ModuleList(Module):
    def __init__(self, *modules):
        self.modules = []
        for item in modules:
            if isinstance(item, Module):
                self.modules.append(item)
            elif isinstance(item, (list, tuple)):
                self.modules.extend(item)
            else:
                raise ValueError("Expected module or sequence to ModuleList()")

    def __iter__(self):
        return iter(self.modules)

    def append(self, mod):
        self.modules.append(mod)

    def forward(self, cx, x):
        raise NotImplementedError

    def gen_named_modules(self):
        for (i, m) in enumerate(self.modules):
            yield (f'{i}', m)
            for (k, p) in m.gen_named_modules():
                yield (f'{i}.{k}', p)

    def gen_named_parameters(self):
        for (i, m) in enumerate(self.modules):
            for (k, p) in m.gen_named_parameters():
                yield (f'{i}.{k}', p)


class Sequential(ModuleList):
    def forward(self, cx, x):
        for module in self.modules:
            x = module(cx, x)
        return x

class Linear(Module):
    def __init__(self, c1, c2, bias=True):
        self.c1 = c1
        self.c2 = c2
        self.weight = init.glorot_normal(c2, c1)
        if bias:
            self.bias = init.zeros(c2)
        else:
            self.bias = None

    def __repr__(self):
        return f'Linear({self.c1}, {self.c2})'

    def forward(self, cx, x):
        y = x @ jnp.transpose(cx[self.weight])
        if self.bias:
            y = y + cx[self.bias]
        return y

class Embedding(Module):
    def __init__(self, n, c):
        self.n = n
        self.c = c
        self.weight = init.normal(n, c)

    def __repr__(self):
        return f'Embedding({self.n}, {self.c})'

    def forward(self, cx, x):
        return cx[self.weight][x]


class Tanh(Module):
    def forward(self, cx, x):
        return jnp.tanh(x)


class Dropout(Module):
  def __init__(self, p=0.5):
    self.rate = p

  def forward(self, cx, x):
    key = cx.rng.split()
    p = jax.random.bernoulli(key, 1.0 - self.rate, shape=x.shape)
    return x * p / (1.0 - self.rate)

class GELU(Module):
    def forward(self, cx, x):
        return jax.nn.gelu(x)

class LayerNorm(Module):
    def __init__(self, *normalized_shape):
        self.normalized_shape = normalized_shape
        self.weight = init.ones(*normalized_shape)
        self.bias = init.zeros(*normalized_shape)
        self.axes = tuple(-i for i in range(1, len(normalized_shape)+1))

    def forward(self, cx, x):
        mu = x.mean(axis=self.axes, keepdims=True)
        sigma = jnp.sqrt(square(x - mu).mean(axis=self.axes, keepdims=True))
        normed = (x - mu) / sigma
        return cx[self.weight] * normed + cx[self.bias]


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, zero_init=False):
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = init.glorot_normal(out_channels, in_channels//groups, kernel_size)
        if zero_init:
            self.weight = init.zeros(out_channels, in_channels//groups, kernel_size)
        self.use_bias = bias
        if self.use_bias:
            self.bias = init.zeros(out_channels)
        else:
            self.bias = None

    def forward(self, cx, x):
        return jaxtorch.nn.functional.conv1d(x, cx[self.weight], cx[self.bias] if self.use_bias else None,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             groups=self.groups)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, zero_init=False):
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = init.glorot_normal(out_channels, in_channels//groups, kernel_size, kernel_size)
        if zero_init:
            self.weight = init.zeros(out_channels, in_channels//groups, kernel_size, kernel_size)
        self.use_bias = bias
        if self.use_bias:
            self.bias = init.zeros(out_channels)
        else:
            self.bias = None

    def forward(self, cx, x):
        return jaxtorch.nn.functional.conv2d(x, cx[self.weight], cx[self.bias] if self.use_bias else None,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation,
                                             groups=self.groups)


class SiLU(Module):
    def forward(self, cx, x):
        return jax.nn.silu(x)

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert self.num_channels % self.num_groups == 0
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = init.ones(num_channels)
            self.bias = init.zeros(num_channels)
        else:
            self.weight = None
            self.bias = None

    def forward(self, cx, x):
        B, C, *rest = x.shape
        assert C == self.num_channels
        x = x.reshape([B, self.num_groups, C//self.num_groups, *rest])
        mu = x.mean(axis=tuple(range(2,len(x.shape))), keepdims=True)
        var = x.var(axis=tuple(range(2,len(x.shape))), keepdims=True)
        y = (x - mu) / jnp.sqrt(var + self.eps)
        y = y.reshape([B, C, *rest])
        if self.affine:
            broadcast_shape = [self.num_channels] + [1] * len(rest)
            weight = cx[self.weight].reshape(broadcast_shape)
            bias = cx[self.bias].reshape(broadcast_shape)
            y = y * weight + bias
        return y
