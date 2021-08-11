import sys
sys.path = ['.', './guided-diffusion'] + sys.path
import torch
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, Context, ParamState, PRNG

from lib import unet
from guided_diffusion import unet as old_unet

# diffusion_state_dict = torch.load('256x256_diffusion_uncond.pt')

torch.backends.cudnn.deterministic = True

def check(old, new):
    old_np = np.array(old)
    new_np = np.array(new)
    assert old_np.shape == new_np.shape, f'{new_np.shape} vs {old_np.shape}'
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

def torch_state_dict(module, px):
    return {name : torch.tensor(t) for (name, t) in module.state_dict(px).items()}


@torch.no_grad()
def test_QKVAttentionLegacy():
    H = 4
    C = 4
    T = 2
    new_module = unet.QKVAttentionLegacy(n_heads=H)
    old_module = old_unet.QKVAttentionLegacy(n_heads=H)

    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=[2, (H * 3 * C), T])
    x_torch = torch.tensor(np.array(x))
    cx = Context(ParamState([]), jax.random.PRNGKey(1))

    new_result = np.array(new_module(cx, x))
    old_result = np.array(old_module(x_torch))
    difference = np.abs(new_result - old_result).max()
    assert difference < 1e-6, difference

@torch.no_grad()
def test_Upsample2D():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 4
    new_module = unet.Upsample2D(C, use_conv=True)
    old_module = old_unet.Upsample(C, use_conv=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    x_torch = torch.tensor(np.array(x))

    new_result = new_module(px, x)
    old_result = old_module(x_torch)
    check(old_result, new_result)


@torch.no_grad()
def test_Downsample2D():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 4
    new_module = unet.Downsample2D(C, use_conv=True)
    old_module = old_unet.Downsample(C, use_conv=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    x_torch = torch.tensor(np.array(x))

    new_result = new_module(px, x)
    old_result = old_module(x_torch)
    check(old_result, new_result)

    new_module = unet.Downsample2D(C, use_conv=False)
    old_module = old_unet.Downsample(C, use_conv=False)

    new_result = new_module(ParamState([]), x)
    old_result = old_module(x_torch)
    check(old_result, new_result)

@torch.no_grad()
def test_ResBlock():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 64
    D = 6
    new_module = unet.ResBlock(C, D, dropout=0.1, use_conv=True)
    old_module = old_unet.ResBlock(C, D, dropout=0.1, use_conv=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    emb = jax.random.normal(key=rng.split(), shape=[1, D])
    x_torch = totorch(x)
    emb_torch = totorch(emb)

    new_result = new_module(Context(px, rng.split()), x, emb)
    old_result = old_module(x_torch, emb_torch)
    check(old_result, new_result)

    # Upsample
    new_module = unet.ResBlock(C, D, dropout=0.1, use_conv=True, up=True)
    old_module = old_unet.ResBlock(C, D, dropout=0.1, use_conv=True, up=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    emb = jax.random.normal(key=rng.split(), shape=[1, D])
    x_torch = totorch(x)
    emb_torch = totorch(emb)

    new_result = new_module(Context(px, rng.split()), x, emb)
    old_result = old_module(x_torch, emb_torch)
    check(old_result, new_result)

    # Downsample
    new_module = unet.ResBlock(C, D, dropout=0.1, use_conv=True, down=True)
    old_module = old_unet.ResBlock(C, D, dropout=0.1, use_conv=True, down=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    emb = jax.random.normal(key=rng.split(), shape=[1, D])
    x_torch = totorch(x)
    emb_torch = totorch(emb)

    new_result = new_module(Context(px, rng.split()), x, emb)
    old_result = old_module(x_torch, emb_torch)
    check(old_result, new_result)

    # Downsample
    new_module = unet.ResBlock(C, D, dropout=0.1, use_conv=True, use_scale_shift_norm=True)
    old_module = old_unet.ResBlock(C, D, dropout=0.1, use_conv=True, use_scale_shift_norm=True)

    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    emb = jax.random.normal(key=rng.split(), shape=[1, D])
    x_torch = totorch(x)
    emb_torch = totorch(emb)

    new_result = new_module(Context(px, rng.split()), x, emb)
    old_result = old_module(x_torch, emb_torch)
    check(old_result, new_result)
