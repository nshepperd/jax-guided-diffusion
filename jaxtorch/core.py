import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import functools
import jaxtorch.monkeypatches
import sys

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Param(object):
    """Represents a parameter of a Module, and specifies its shape and initialization."""
    def __init__(self, shape, initializer):
        self.shape = shape
        self.initializer = initializer
        self.name = None

    def __repr__(self):
        if self.name is not None:
            return f'<Param at {self.name}>'
        else:
            return super().__repr__()

class PRNG(object):
    """Just a stateful wrapper for a jax.random.PRNGKey."""
    def __init__(self, key):
        self.key = key
    def split(self):
        (self.key, subkey) = jax.random.split(self.key)
        return subkey

class ContextRandom(object):
    """Lives inside a Context and provides convenience functions for
random number generation that use the Context's stateful PRNG.

    """
    def __init__(self, rng):
        self.rng = rng

    def _wrap(f):
        return lambda self, *args, **kwargs: f(self.rng.split(), *args, **kwargs)

    bernoulli = _wrap(jax.random.bernoulli)
    beta = _wrap(jax.random.beta)
    categorical = _wrap(jax.random.categorical)
    cauchy = _wrap(jax.random.cauchy)
    choice = _wrap(jax.random.choice)
    dirichlet = _wrap(jax.random.dirichlet)
    double_sided_maxwell = _wrap(jax.random.double_sided_maxwell)
    exponential = _wrap(jax.random.exponential)
    gamma = _wrap(jax.random.gamma)
    gumbel = _wrap(jax.random.gumbel)
    laplace = _wrap(jax.random.laplace)
    logistic = _wrap(jax.random.logistic)
    maxwell = _wrap(jax.random.maxwell)
    multivariate_normal = _wrap(jax.random.multivariate_normal)
    normal = _wrap(jax.random.normal)
    pareto = _wrap(jax.random.pareto)
    permutation = _wrap(jax.random.permutation)
    poisson = _wrap(jax.random.poisson)
    rademacher = _wrap(jax.random.rademacher)
    randint = _wrap(jax.random.randint)
    shuffle = _wrap(jax.random.shuffle)
    t = _wrap(jax.random.t)
    truncated_normal = _wrap(jax.random.truncated_normal)
    uniform = _wrap(jax.random.uniform)
    weibull_min = _wrap(jax.random.weibull_min)

@jax.tree_util.register_pytree_node_class
class Context(object):
    """Wraps a parameter dictionary and a PRNG."""
    def __init__(self, px, key, mode='train'):
        self.px = px
        self.rng = PRNG(key)
        self.random = ContextRandom(self.rng)
        self.mode = mode

    def train_mode_(self):
        self.mode = 'train'
        return self

    def eval_mode_(self):
        self.mode = 'eval'
        return self

    def __getitem__(self, par):
        if isinstance(par, Param):
            return self.px[par.name]
        elif isinstance(par, str):
            return self.px[par]
        else:
            raise TypeError('Expected a Param for indexing into Context')

    def __setitem__(self, par, value):
        if isinstance(par, Param):
            self.px[par.name] = value
        elif isinstance(par, str):
            self.px[par] = value
        else:
            raise TypeError('Expected a Param for indexing into Context')

    def tree_flatten(self):
        return (self.px, self.rng.split()), (self.mode,)

    @staticmethod
    def tree_unflatten(aux, values):
        (px, key) = values
        (mode,) = aux
        return Context(px, key, mode=mode)


class Module(object):
    def __call__(self, cx: Context, *args, **kwargs):
        return self.forward(cx, *args, **kwargs)

    def forward(self, cx: Context, *args, **kwargs):
        """Implements the forward pass. Must take Context as the first argument."""
        raise NotImplementedError

    def self_named_modules(self):
        """Yields a sequence of (str, Module) for direct children of this
           module. May be overridden.

        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Module):
                yield (name, val)

    def self_named_parameters(self):
        """Yields a sequence of (str, Param) for direct children of this
           module. May be overridden.

        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Param):
                yield (name, val)

    def self_init_weights(self, cx):
        """Initializes weights for this network's parameters. May be overriden
           for custom initialization. Child modules are initialized
           before parents.

        """
        for (name, par) in self.self_named_parameters():
            if par.initializer is not None:
                cx[par] = par.initializer(cx.rng.split())

    def init_weights(self, key):
        """Attaches names to parameters and returns initialized dict of
           parameters by name.

        """
        self.labeled_parameters_()
        cx = Context({}, key)
        for module in self.gen_postorder_modules():
            module.self_init_weights(cx)
        self.self_init_weights(cx)
        return cx.px

    def labeled_parameters_(self):
        for (name, par) in self.named_parameters():
            par.name = name
        return self.parameters()

    def gen_named_modules(self):
        "Yields (str, Module) for all descendants of this module."
        for (name, val) in self.self_named_modules():
            yield (name, val)
            for (k, v) in val.gen_named_modules():
                yield (name+'.'+k, v)

    def gen_postorder_modules(self):
        "Yields Module for all descendants of this module (postorder traversal)."
        for (name, mod) in self.self_named_modules():
            for submod in mod.gen_postorder_modules():
                yield submod
            yield mod

    def gen_named_parameters(self):
        "Yields (str, Param) for this module and all descendants."
        for (name, par) in self.self_named_parameters():
            yield (name, par)

        for (name, mod) in self.self_named_modules():
            for (k, v) in mod.gen_named_parameters():
                yield (name+'.'+k, v)

    def named_parameters(self):
        return list(self.gen_named_parameters())

    def modules(self):
        return [m for (k, m) in self.gen_named_modules()]

    def parameters(self):
        return [p for (k, p) in self.gen_named_parameters()]

    def state_dict(self, px):
        return dict(px)

    def load_state_dict(self, px, state, strict=True):
        """Load a previously saved state_dict into px. Returns a new copy of px."""
        px = dict(px)
        for (k, p) in self.gen_named_parameters():
            if k not in state:
                if strict:
                    raise ValueError(f'Not loading missing parameter: {k}')
                else:
                    print(f'Not loading missing parameter: {k}', file=sys.stderr)
                    continue

            if px[p.name].shape != state[k].shape:
                msg = f'Not loading parameter from incompatible shape: {k} ({px[p.name].shape} vs {state[k].shape})'
                if strict:
                    raise ValueError(msg)
                else:
                    print(msg, file=sys.stderr)
                    continue

            px[p.name] = jax.numpy.asarray(state[k])
        return px

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self.__dict__.items():
            if isinstance(module, Module):
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
