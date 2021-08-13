import sys
sys.path.append('.')
import torch
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, Context, ParamState, PRNG

from lib import unet
from tqdm import tqdm

with open('256x256_diffusion_uncond.cbor', 'rb') as fp:
    state_dict = jaxtorch.cbor.load(fp)

def main():
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
    px = ParamState(new_module.labeled_parameters_())
    px.initialize(rng.split())

    new_module.load_state_dict(px, state_dict)

    x = jax.random.normal(key=rng.split(), shape=[1, 3, 32, 32])
    ts = jnp.array([1])

    @jax.jit
    def execute(px, x, ts, key):
        cx = Context(px, key)
        return new_module(cx, x, ts)

    with tqdm() as pbar:
        while True:
            val = execute(px, x, ts, rng.split())
            pbar.update()

if __name__ == '__main__':
    main()
