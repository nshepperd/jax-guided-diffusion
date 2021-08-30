import sys
sys.path = ['.'] + sys.path
import torch
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, Context, ParamState, PRNG

torch.backends.cudnn.deterministic = True

def check(old, new):
    old_np = np.array(old)
    new_np = np.array(new)
    assert old_np.shape == new_np.shape
    difference = np.abs(old_np - new_np).max()
    magnitude = np.mean((np.abs(old_np) + np.abs(new_np))/2)
    # Pretty loose bounds due to float32 precision. There seem to be
    # some implementation differences resulting in different error
    # bounds betwen jax and torch.
    assert difference/magnitude < 1e-6, difference/magnitude

def totorch(x):
    return torch.tensor(np.array(x))
def fromtorch(x):
    return jnp.array(np.array(x))

@torch.no_grad()
def test_conv1d():
    rng = PRNG(jax.random.PRNGKey(0))
    x = jax.random.normal(key=rng.split(), shape=[1, 4, 5])
    w = jax.random.normal(key=rng.split(), shape=[3, 4, 3])
    b = jax.random.normal(key=rng.split(), shape=[3])

    x_torch = totorch(x)
    w_torch = totorch(w)
    b_torch = totorch(b)

    new_result = jaxtorch.nn.functional.conv1d(x, w, bias=b, padding='same')
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, bias=b_torch, padding='same')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid')
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', stride=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', stride=2)
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', dilation=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', dilation=2)
    check(old_result, new_result)

    w = jax.random.normal(key=rng.split(), shape=[4, 2, 3])
    w_torch = totorch(w)
    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', groups=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', groups=2)
    check(old_result, new_result)

@torch.no_grad()
def test_conv2d():
    rng = PRNG(jax.random.PRNGKey(0))
    x = jax.random.normal(key=rng.split(), shape=[1, 4, 5, 5])
    w = jax.random.normal(key=rng.split(), shape=[3, 4, 3, 3])
    b = jax.random.normal(key=rng.split(), shape=[3])

    x_torch = totorch(x)
    w_torch = totorch(w)
    b_torch = totorch(b)

    new_result = jaxtorch.nn.functional.conv2d(x, w, bias=b, padding='same')
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch, padding='same')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid')
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', stride=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', stride=2)
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', dilation=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', dilation=2)
    check(old_result, new_result)

    w = jax.random.normal(key=rng.split(), shape=[4, 2, 3, 3])
    w_torch = totorch(w)
    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', groups=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', groups=2)
    check(old_result, new_result)

@torch.no_grad()
def test_groupnorm():
    rng = PRNG(jax.random.PRNGKey(0))
    new = jaxtorch.nn.GroupNorm(8, 32, batched=False)
    old = torch.nn.GroupNorm(8, 32)

    px = ParamState(new.parameters())
    px.initialize(rng.split())

    old.weight.data.copy_(totorch(px[new.weight]))
    old.bias.data.copy_(totorch(px[new.bias]))

    x = jax.random.normal(key=rng.split(), shape=[1, 32, 2])
    x_torch = totorch(x)

    new_result = new(px, x.squeeze(0)).unsqueeze(0)
    old_result = old(x_torch)
    check(old_result, new_result)

if __name__ == '__main__':
    test_conv2d()