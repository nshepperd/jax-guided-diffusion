import math
import io
import sys
import time
import os
import functools
from functools import partial

from PIL import Image
import requests

import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import PRNG, Context, ParamState, Module
from tqdm import tqdm
from glob import glob
from einops import rearrange
import jax.experimental.optimizers

from lib.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from lib.util import pil_to_tensor, pil_from_tensor

def make_cutout(image, key):
    rng = PRNG(key)
    [c, h, w] = image.shape
    scale = 512/min(h,w)
    new_h = round(scale * h)
    new_w = round(scale * w)
    resized = jax.image.resize(image,
                               (c, new_h, new_w),
                               method='bilinear')
    y0 = jax.random.randint(rng.split(), [], 0, new_h-512+1)
    x0 = jax.random.randint(rng.split(), [], 0, new_w-512+1)
    return resized[:, y0:y0+512, x0:x0+512]


def load_torch(checkpoint):
    import torch
    with torch.no_grad():
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        jax_state_dict = {name : par.cpu().numpy() for (name, par) in state_dict.items()}
        return jax_state_dict


# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_scale_shift_norm': True,
})


# Load models

model, diffusion = create_model_and_diffusion(**model_config)

def exec_model(model_params, x, timesteps, y=None, key=None):
    cx = Context(model_params, key)
    return model(cx, x, timesteps, y=y)
# exec_model_jit = functools.partial(jax.jit(exec_model), model_params)

def exec_loss(model_params, x, t, key):
    rng = PRNG(key)
    run_model = partial(exec_model, model_params, key=rng.split())
    return diffusion.training_losses(run_model, x, t, rng)['loss'].mean()
exec_loss_jit = jax.jit(exec_loss)
exec_grad_jit = jax.jit(jax.value_and_grad(exec_loss))

lr = 1.0e-4

def image_loop():
    rng = PRNG(jax.random.PRNGKey(0))
    for filename in glob('/mnt/data2/data/openimages/train/data/*.jpg'):
        image = Image.open(filename).convert('RGB')
        image = pil_to_tensor(image)
        image = make_cutout(image, rng.split())
        image = image * 2 - 1
        image = image.unsqueeze(0) # [1, c, h, w]
        yield image
image_loop = image_loop()

def get_batch(bs=1):
    batch = []
    for i in range(bs):
        image = next(image_loop)
        batch.append(image)
    return jnp.concatenate(batch, axis=0)

def main():
    model_params = ParamState(model.labeled_parameters_())
    model_params.initialize(jax.random.PRNGKey(0))

    print('Loading state dict...')
    jax_state_dict = load_torch('512x512_diffusion_uncond_finetune_008100.pt')
    # with open('512x512_diffusion_uncond_finetune_008100.cbor', 'rb') as fp:
    #     jax_state_dict = jaxtorch.cbor.load(fp)
    model.load_state_dict(model_params, jax_state_dict)

    rng = PRNG(jax.random.PRNGKey(0))

    # Adam seems to OOM on 3090
    # Plain SGD works though

    # opt_init, opt_update, opt_params = jax.experimental.optimizers.adam(1e-4, b1=0.9, b2=0.999, eps=1e-08)
    # opt = opt_init(model_params)

    counter = 1
    while True:
        batch = get_batch(1)
        t = jax.random.randint(rng.split(), [batch.shape[0]], 0, diffusion.num_timesteps)
        loss, grad = exec_grad_jit(model_params, batch, t, rng.split())
        model_params = jax.tree_util.tree_map(lambda x, g: x - lr*g, model_params, grad)
        # opt = opt_update(counter, grad, opt)
        # model_params = opt_params(opt)
        print(counter, loss)
        counter += 1

        # {'vb': DeviceArray([6.8819056e-05], dtype=float32), 'mse': DeviceArray([0.00934565], dtype=float32), 'loss': DeviceArray([0.00941447], dtype=float32)}

if __name__ == '__main__':
    main()