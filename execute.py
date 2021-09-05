# Based on Katherine Crowson's CLIP guided diffusion notebook
# (https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj)
# including a port of https://github.com/crowsonkb/guided-diffusion to jax.

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

sys.path.append('./CLIP_JAX')
import clip_jax

from lib.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from lib import util
from lib.util import pil_from_tensor, pil_to_tensor

# Define necessary functions

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def load_torch(checkpoint):
    import torch
    with torch.no_grad():
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        jax_state_dict = {name : par.cpu().numpy() for (name, par) in state_dict.items()}
        return jax_state_dict

class MakeCutouts(object):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def key(self):
        return (self.cut_size,self.cutn,self.cut_pow)
    def __hash__(self):
        return hash(self.key())
    def __eq__(self, other):
        if isinstance(other, MakeCutouts):
            return type(self) is type(other) and self.key() == other.key()
        return NotImplemented

    def __call__(self, input, key):
        [b, c, h, w] = input.shape
        rng = PRNG(key)
        max_size = min(h, w)
        min_size = min(h, w, self.cut_size)
        cut_us = jax.random.uniform(rng.split(), shape=[self.cutn])**self.cut_pow
        sizes = (min_size + cut_us * (max_size - min_size + 1)).astype(jnp.int32).clamp(min_size, max_size)
        offsets_x = jax.random.randint(rng.split(), [self.cutn], 0, w - sizes + 1)
        offsets_y = jax.random.randint(rng.split(), [self.cutn], 0, h - sizes + 1)
        cutouts = util.cutouts_images(input, offsets_x, offsets_y, sizes)
        cutouts = cutouts.rearrange('b n c h w -> (n b) c h w')
        return cutouts

class StaticCutouts(MakeCutouts):
    def __init__(self, cut_size, cutn, size):
        self.cut_size = cut_size
        self.cutn = cutn
        self.size = size

    def key(self):
        return (self.cut_size,self.cutn,self.size)

    def __call__(self, input, key):
        [b, c, h, w] = input.shape
        rng = PRNG(key)
        sizes = jnp.array([self.size]*self.cutn).astype(jnp.int32)
        offsets_x = jax.random.randint(rng.split(), [self.cutn], 0, w - sizes + 1)
        offsets_y = jax.random.randint(rng.split(), [self.cutn], 0, h - sizes + 1)
        cutouts = util.cutouts_images(input, offsets_x, offsets_y, sizes)
        cutouts = cutouts.rearrange('b n c h w -> (n b) c h w')
        return cutouts

def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3,1,1)
    std = jnp.array(std).reshape(3,1,1)
    def forward(image):
        return (image - mean) / std
    return forward

def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt()

def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)

def cborfile(path):
    with fetch(path) as fp:
      return jaxtorch.cbor.load(fp)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    # input = jnp.pad(input, ((0,0), (0,0), (0,1), (0,1)), mode='edge')
    # x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    # y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    # return (x_diff**2 + y_diff**2).mean([1, 2, 3])
    x_diff = input[..., :, 1:] - input[..., :, :-1]
    y_diff = input[..., 1:, :] - input[..., :-1, :]
    return x_diff.square().mean([1,2,3]) + y_diff.square().mean([1,2,3])

def downscale2d(image, f):
  [c, n, h, w] = image.shape
  return jax.image.resize(image, [c, n, h//f, w//f], method='linear')

def rms(x):
  return x.square().mean().sqrt()

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
    # 'use_fp16': True,
    'use_scale_shift_norm': True,
})


# Load models

model, diffusion = create_model_and_diffusion(**model_config)
model_params = ParamState(model.labeled_parameters_())
model_params.initialize(jax.random.PRNGKey(0))

print('Loading state dict...')
jax_state_dict = load_torch('512x512_diffusion_uncond_finetune_008100.pt')
# with open('512x512_diffusion_uncond_finetune_008100.cbor', 'rb') as fp:
#     jax_state_dict = jaxtorch.cbor.load(fp)

model.load_state_dict(model_params, jax_state_dict)

def exec_model(model_params, x, timesteps, y=None):
    cx = Context(model_params, jax.random.PRNGKey(0))
    return model(cx, x, timesteps, y=y)
exec_model_jit = functools.partial(jax.jit(exec_model), model_params)
exec_model_par_base = jax.pmap(exec_model, in_axes=(None, 0, 0), out_axes=0, devices=jax.devices()[1:])
def exec_model_par(x, timesteps, y=None):
    return exec_model_par_base(model_params, x.unsqueeze(1), timesteps.unsqueeze(1)).squeeze(1)

def base_cond_fn(x, t, cur_t, params, key, make_cutouts, make_cutouts_style):
    text_embed, style_embed, model_params, clip_params, clip_guidance_scale, style_guidance_scale, tv_scale, sat_scale = params

    rng = PRNG(key)
    n = x.shape[0]

    def denoise(x):
      my_t = jnp.ones([n], dtype=jnp.int32) * cur_t
      out = diffusion.p_mean_variance(functools.partial(exec_model,model_params),
                                      x, my_t,
                                      clip_denoised=False,
                                      model_kwargs={})
      fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
      x_in = out['pred_xstart'] * fac + x * (1 - fac)
      return x_in
    (x_in, backward) = jax.vjp(denoise, x)

    def main_clip_loss(x_in, key):
      clip_in = normalize(make_cutouts(x_in.add(1).div(2), key))
      image_embeds = emb_image(clip_in, clip_params).reshape([make_cutouts.cutn, n, 512])
      # Method 1. Average the clip embeds, then compute great circle distance.
      # losses = spherical_dist_loss(image_embeds.mean(0), text_embed)
      # Method 2. Compute great circle losses for clip embeds, then average.
      losses = spherical_dist_loss(image_embeds, text_embed).mean(0)
      return losses.sum() * clip_guidance_scale

    # Scan method, should reduce jit times...
    num_cuts = 4
    keys = jnp.stack([rng.split() for _ in range(num_cuts)])
    main_clip_grad = jax.lax.scan(lambda total, key: (total + jax.grad(main_clip_loss)(x_in, key), key),
                                  jnp.zeros_like(x_in),
                                  keys)[0] / num_cuts

    if style_embed is not None:
      def style_loss(x_in, key):
        clip_in = normalize(make_cutouts_style(x_in.add(1).div(2), key))
        image_embeds = emb_image(clip_in, clip_params).reshape([make_cutouts_style.cutn, n, 512])
        style_losses = spherical_dist_loss(image_embeds, style_embed).mean(0)
        return style_losses.sum() * style_guidance_scale
      main_clip_grad += jax.grad(style_loss)(x_in, rng.split())

    def sum_tv_loss(x_in, f=None):
      if f is not None:
        x_in = downscale2d(x_in, f)
      return tv_loss(x_in).sum() * tv_scale
    tv_grad_512 = jax.grad(sum_tv_loss)(x_in)
    tv_grad_256 = jax.grad(partial(sum_tv_loss,f=2))(x_in)
    tv_grad_128 = jax.grad(partial(sum_tv_loss,f=4))(x_in)
    main_clip_grad += tv_grad_512 + tv_grad_256 + tv_grad_128

    def saturation_loss(x_in):
      return jnp.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
    sat_grad = sat_scale * jax.grad(saturation_loss)(x_in)
    main_clip_grad += sat_grad

    return -backward(main_clip_grad)[0]
# base_cond_fn = jax.jit(base_cond_fn, static_argnames=['make_cutouts', 'make_cutouts_style'])
base_cond_fn = jax.pmap(base_cond_fn, in_axes = (0, 0, None, None, 0, None, None), out_axes=0, static_broadcasted_argnums=(5,6),
                        devices=jax.devices()[1:])

print('Loading CLIP model...')
image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/32') #, "cpu")
clip_size = 224
normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])


def txt(prompt):
  """Returns normalized embedding."""
  text = clip_jax.tokenize([prompt])
  text_embed = text_fn(clip_params, text)
  return norm1(text_embed.reshape(512))

def emb_image(image, clip_params=None):
    return norm1(image_fn(clip_params, image))

title = ['the portal to hell was discovered on a siberian plateau. trending on ArtStation']
prompt = [txt(t) for t in title]
style_embed = norm1(jnp.array(cborfile('data/openimages_512x_png_embed224.cbor'))) - norm1(jnp.array(cborfile('data/imagenet_512x_jpg_embed224.cbor')))
batch_size = 7
clip_guidance_scale = 2000
style_guidance_scale = 300
tv_scale = 150
sat_scale = 150
cutn = 32 # effective cutn is 4x this because we do 4 iterations in base_cond_fn
cut_pow = 0.5
style_cutn = 32
n_batches = len(title)
init_image = None
skip_timesteps = 0
seed = 1

# Actually do the run
print('Starting run...')

def run():
    rng = PRNG(jax.random.PRNGKey(seed))

    text_embed = prompt

    init = None
    # if init_image is not None:
    #     init = Image.open(fetch(init_image)).convert('RGB')
    #     init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
    #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow=cut_pow)
    make_cutouts_style = StaticCutouts(clip_size, style_cutn, size=224)

    def cond_fn(x, t, y=None):
        # x : [n, c, h, w]
        n = x.shape[0]
        # Triggers recompilation if cutout parameters have changed (cutn or cut_pow).
        grad = base_cond_fn(x.unsqueeze(1), jnp.array(t).unsqueeze(1), jnp.array(cur_t),
                            (text_embed, style_embed, model_params, clip_params, clip_guidance_scale, style_guidance_scale, tv_scale, sat_scale),
                            jnp.stack([rng.split() for _ in range(n)]),
                            make_cutouts,
                            make_cutouts_style)
        # grad : [n, 1, c, h, w]
        grad = grad.squeeze(1)
        # grad : [n, c, h, w]
        magnitude = grad.square().mean(axis=(1,2,3), keepdims=True).sqrt()
        grad = grad / magnitude * magnitude.clamp(max=0.1)
        return grad

    for i in range(n_batches):
        if type(prompt) is list:
          text_embed = prompt[i % len(prompt)]
          this_title = title[i % len(prompt)]
        else:
          text_embed = prompt
          this_title = title

        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = diffusion.p_sample_loop_progressive(
            exec_model_par,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            rng=rng,
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=tqdm,
            skip_timesteps=skip_timesteps,
            init_image=init,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 100 == 0 or cur_t == -1:
                print()
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{i * batch_size + k:05}.png'
                    # print(k, type(image).mro())
                    # For some reason this comes out as a numpy array. Huh?
                    image = pil_from_tensor(jnp.array(image).add(1).div(2))
                    image.save(filename)
                    print(f'Wrote {filename}')

        for k in range(batch_size):
            filename = f'progress_{i * batch_size + k:05}.png'
            timestring = time.strftime('%Y%m%d%H%M%S')
            os.makedirs('samples', exist_ok=True)
            dname = f'samples/{timestring}_{k}_{this_title}.png'
            with open(filename, 'rb') as fp:
              data = fp.read()
            with open(dname, 'wb') as fp:
              fp.write(data)

run()
