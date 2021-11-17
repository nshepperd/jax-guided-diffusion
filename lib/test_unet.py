import sys
sys.path = ['.', '../guided-diffusion'] + sys.path
import torch
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, Context, PRNG

from lib import unet
from guided_diffusion import unet as old_unet

torch.backends.cudnn.deterministic = True

diffusion_state_dict = torch.load('models/256x256_diffusion_uncond.pt')

def check(old, new):
    old_np = np.array(old)
    new_np = np.array(new)
    assert old_np.shape == new_np.shape, f'{new_np.shape} vs {old_np.shape}'
    difference = np.abs(old_np - new_np).max()
    magnitude = np.mean((np.abs(old_np) + np.abs(new_np))/2)
    # Pretty loose bounds due to float32 precision. There seem to be
    # some implementation differences resulting in different error
    # bounds betwen jax and torch.
    assert difference/magnitude < 1e-6, (difference, magnitude)

def totorch(x):
    return torch.tensor(np.array(x))
def fromtorch(x):
    return jnp.array(np.array(x))

def torch_state_dict(module, px):
    return {name : totorch(t) for (name, t) in module.state_dict(px).items()}


@torch.no_grad()
def test_QKVAttentionLegacy():
    H = 4
    C = 4
    T = 2
    new_module = unet.QKVAttentionLegacy(n_heads=H)
    old_module = old_unet.QKVAttentionLegacy(n_heads=H)

    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=[2, (H * 3 * C), T])
    x_torch = torch.tensor(np.array(x))
    cx = Context({}, jax.random.PRNGKey(1))

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

    px = new_module.init_weights(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    x_torch = torch.tensor(np.array(x))

    new_result = new_module(Context(px, rng.split()), x)
    old_result = old_module(x_torch)
    check(old_result, new_result)


@torch.no_grad()
def test_Downsample2D():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 4
    new_module = unet.Downsample2D(C, use_conv=True)
    old_module = old_unet.Downsample(C, use_conv=True)

    px = new_module.init_weights(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    x_torch = torch.tensor(np.array(x))

    new_result = new_module(Context(px, rng.split()), x)
    old_result = old_module(x_torch)
    check(old_result, new_result)

    new_module = unet.Downsample2D(C, use_conv=False)
    old_module = old_unet.Downsample(C, use_conv=False)
    px = new_module.init_weights(rng.split())

    new_result = new_module(px, x)
    old_result = old_module(x_torch)
    check(old_result, new_result)

@torch.no_grad()
def test_ResBlock():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 64
    D = 6
    new_module = unet.ResBlock(C, D, dropout=0.1, use_conv=True)
    old_module = old_unet.ResBlock(C, D, dropout=0.1, use_conv=True)

    px = new_module.init_weights(rng.split())

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

    px = new_module.init_weights(rng.split())

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

    px = new_module.init_weights(rng.split())

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
    px = new_module.init_weights(rng.split())

    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    emb = jax.random.normal(key=rng.split(), shape=[1, D])
    x_torch = totorch(x)
    emb_torch = totorch(emb)

    new_result = new_module(Context(px, rng.split()), x, emb)
    old_result = old_module(x_torch, emb_torch)
    check(old_result, new_result)

@torch.no_grad()
def test_AttentionBlock():
    rng = PRNG(jax.random.PRNGKey(0))

    C = 64
    new_module = unet.AttentionBlock(C)
    old_module = old_unet.AttentionBlock(C)
    px = new_module.init_weights(rng.split())
    old_module.load_state_dict(torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, C, 8, 8])
    x_torch = totorch(x)

    new_result = new_module(Context(px, rng.split()), x)
    old_result = old_module(x_torch)
    check(old_result, new_result)

@torch.no_grad()
def test_UNetModel():
    rng = PRNG(jax.random.PRNGKey(0))

    config = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )

    config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_scale_shift_norm': True,
    })

    attention_ds = []
    for res in config['attention_resolutions'].split(","):
        attention_ds.append(config['image_size'] // int(res))

    args = dict(
        image_size=config['image_size'],
        in_channels=3,
        model_channels=256,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=attention_ds,
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),

        num_heads=config['num_heads'],
        num_head_channels=config['num_head_channels'],
        num_heads_upsample=config['num_heads_upsample'],
        use_scale_shift_norm=config['use_scale_shift_norm'],
        resblock_updown=config['resblock_updown'],
        use_new_attention_order=config['use_new_attention_order']
    )

    new_module = unet.UNetModel(**args)
    old_module = old_unet.UNetModel(**args)
    px = new_module.init_weights(rng.split())

    px = new_module.load_state_dict(px, {name : par.cpu().numpy() for (name, par) in diffusion_state_dict.items()})
    old_module.load_state_dict(diffusion_state_dict) #torch_state_dict(new_module, px))

    x = jax.random.normal(key=rng.split(), shape=[1, 3, 32, 32])
    ts = jnp.array([1])
    x_torch = totorch(x)
    ts_torch = totorch(ts)

    new_result = new_module(Context(px, rng.split()), x, ts)
    old_result = old_module(x_torch, ts_torch)
    check(old_result, new_result)
