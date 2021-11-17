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
import jax.profiler
import jax.numpy as jnp
import jaxtorch
from jaxtorch import PRNG, Context, ParamState, Module
from tqdm import tqdm
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading, json

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

def grey(image):
    [*_, c, h, w] = image.shape
    return jnp.broadcast_to(image.mean(axis=-3, keepdims=True), image.shape)

class MakeCutouts(object):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.p_grey = 0.2

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
        cut_us = jax.random.uniform(rng.split(), shape=[self.cutn//2])**self.cut_pow
        sizes = (min_size + cut_us * (max_size - min_size + 1)).astype(jnp.int32).clamp(min_size, max_size)
        offsets_x = jax.random.uniform(rng.split(), [self.cutn//2], minval=0, maxval=w - sizes)
        offsets_y = jax.random.uniform(rng.split(), [self.cutn//2], minval=0, maxval=h - sizes)
        cutouts = util.cutouts_images(input, offsets_x, offsets_y, sizes)

        lcut_us = jax.random.uniform(rng.split(), shape=[self.cutn//2])
        lsizes = (max_size + 10 + lcut_us * 10).astype(jnp.int32)
        loffsets_x = jax.random.uniform(rng.split(), [self.cutn//2], minval=w - lsizes, maxval=0)
        loffsets_y = jax.random.uniform(rng.split(), [self.cutn//2], minval=h - lsizes, maxval=0)
        lcutouts = util.cutouts_images(input, loffsets_x, loffsets_y, lsizes)

        cutouts = jnp.concatenate([cutouts, lcutouts], axis=1)

        grey_us = jax.random.uniform(rng.split(), shape=[b, self.cutn, 1, 1, 1])
        cutouts = jnp.where(grey_us < self.p_grey, grey(cutouts), cutouts)
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

RANDOM_FLIP=True
RANDOM_SHIFT=True

def exec_model(model_params, x, timesteps, key, y=None):
    [b, c, h, w] = x.shape
    rng = PRNG(key)
    if RANDOM_FLIP:
        flip_us = jax.random.uniform(rng.split(), [b, 1, 1, 1])
        x = jnp.where(flip_us < 0.5, jnp.flip(x, axis=3), x)
    if RANDOM_SHIFT:
        shift = jax.random.randint(rng.split(), [2], minval=-10, maxval=11)
        x = jnp.roll(x, shift, (2, 3))
    cx = Context(model_params, rng.split())
    out = model(cx, x, timesteps, y=y)
    if RANDOM_SHIFT:
        out = jnp.roll(out, -shift, (2, 3))
    if RANDOM_FLIP:
        out = jnp.where(flip_us < 0.5, jnp.flip(out, axis=3), out)
    return out
exec_model_jit = functools.partial(jax.jit(exec_model), model_params)
exec_model_par_base = jax.pmap(exec_model, in_axes=(None, 0, 0, None), out_axes=0, devices=jax.devices()[1:])
def exec_model_par(x, timesteps, key, y=None):
    return exec_model_par_base(model_params, x.unsqueeze(1), timesteps.unsqueeze(1), key).squeeze(1)

def base_cond_fn(x, t, cur_t, text_embed, params, key, make_cutouts, make_cutouts_style, cut_batches):
    style_embed, model_params, clip_params, clip_guidance_scale, style_guidance_scale, tv_scale, sat_scale = params

    rng = PRNG(key)
    n = x.shape[0]

    def denoise(key, x):
      my_t = jnp.ones([n], dtype=jnp.int32) * cur_t
      out = diffusion.p_mean_variance(functools.partial(exec_model,model_params),
                                      x, my_t, key,
                                      clip_denoised=False,
                                      model_kwargs={})
      fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
      x_in = out['pred_xstart'] * fac + x * (1 - fac)
      return x_in
    (x_in, backward) = jax.vjp(partial(denoise, rng.split()), x)

    def main_clip_loss(x_in, key):
      clip_in = normalize(make_cutouts(x_in.add(1).div(2), key))
      image_embeds = emb_image(clip_in, clip_params).reshape([make_cutouts.cutn, n, 512])
      # Method 1. Average the clip embeds, then compute great circle distance.
      # losses = spherical_dist_loss(image_embeds.mean(0), text_embed)
      # Method 2. Compute great circle losses for clip embeds, then average.
      losses = spherical_dist_loss(image_embeds, text_embed).mean(0)
      return losses.sum() * clip_guidance_scale

    # Scan method, should reduce jit times...
    num_cuts = cut_batches
    cut_rng = PRNG(rng.split())
    keys = jnp.stack([cut_rng.split() for _ in range(num_cuts)])
    main_clip_grad = jax.lax.scan(lambda total, key: (total + jax.grad(main_clip_loss)(x_in, key), key),
                                  jnp.zeros_like(x_in),
                                  keys)[0] / num_cuts

    if style_embed is not None:
      def style_loss(x_in, key):
        clip_in = normalize(make_cutouts_style(x_in.add(1).div(2), key))
        image_embeds = emb_image(clip_in, clip_params).reshape([make_cutouts_style.cutn, n, 512])
        style_losses = spherical_dist_loss(image_embeds, style_embed).mean(0)
        return style_losses.sum() * style_guidance_scale
      # main_clip_grad += jax.grad(style_loss)(x_in, rng.split())

      num_cuts = cut_batches
      cut_rng = PRNG(rng.split())
      keys = jnp.stack([cut_rng.split() for _ in range(num_cuts)])
      main_clip_grad += jax.lax.scan(lambda total, key: (total + jax.grad(style_loss)(x_in, key), key),
                                     jnp.zeros_like(x_in),
                                     keys)[0] / num_cuts

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
base_cond_fn = jax.pmap(base_cond_fn, in_axes = (0, # x,
                                                 0, # t,
                                                 None, # cur_t,
                                                 0, # text_embed,
                                                 None, # params,
                                                 0, # key,
                                                 None, # make_cutouts,
                                                 None, # make_cutouts_style,
                                                 None, # cut_batches,
                                                 ), out_axes=0, static_broadcasted_argnums=(6,7,8),
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

def cosim(a, b):
    return (txt(a) * txt(b)).sum(-1)

def average(*xs):
    total = 0
    count = 0
    for x in xs:
        total += x
        count += 1
    return total / count

title = ['curse breaker | trending on ArtStation']
prompt = [txt('curse breaker | trending on ArtStation')]
style_embed = norm1(jnp.array(cborfile('data/openimages_512x_png_embed224.cbor'))) - norm1(jnp.array(cborfile('data/imagenet_512x_jpg_embed224.cbor')))
batch_size = 7

prompt_jitter_scale = 500
clip_guidance_scale = 2000
style_guidance_scale = 300
tv_scale = 150
sat_scale = 600

cutn = 32 # effective cutn is cut_batches * this
cut_pow = 0.5
cut_batches = 16
style_cutn = 32

n_batches = len(title) * 5
init_image = None #'https://zlkj.in/dalle/generated/2bcef7cbfa690a06a77acca7ac209718.png'
skip_timesteps = 0
seed = 30

# Actually do the run

def proc_init_image(init_image):
    if init_image.endswith(':parts512'):
        url = init_image.rsplit(':', 1)[0]
        init = Image.open(fetch(url)).convert('RGB')
        # init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
        init = pil_to_tensor(init).mul(2).sub(1)
        [c, h, w] = init.shape
        indices = [(x, y)
                   for y in range(0, h, 512)
                   for x in range(0, w, 512)]
        indices = (indices * batch_size)[:batch_size]
        parts = [init[:, y:y+512, x:x+512] for (x, y) in indices]
        init = jnp.stack(parts)
        init = jax.image.resize(init, [batch_size, c, model_config['image_size'], model_config['image_size']], method='lanczos3')
        return init

    init = Image.open(fetch(init_image)).convert('RGB')
    init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
    init = pil_to_tensor(init).unsqueeze(0).mul(2).sub(1)
    return init

def run():
    print('Starting run...')
    rng = PRNG(jax.random.PRNGKey(seed))

    init = None
    if init_image is not None:
        if type(init_image) is list:
            init = jnp.concatenate([proc_init_image(url) for url in init_image], axis=0)
        else:
            init = proc_init_image(init_image)

    cur_t = None

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow=cut_pow)
    make_cutouts_style = StaticCutouts(clip_size, style_cutn, size=224)

    def cond_fn(x, t, y=None):
        # x : [n, c, h, w]
        n = x.shape[0]
        # Triggers recompilation if cutout parameters have changed (cutn or cut_pow).
        grad = base_cond_fn(x.unsqueeze(1), jnp.array(t).unsqueeze(1), jnp.array(cur_t), text_embed,
                            (style_embed, model_params, clip_params, clip_guidance_scale, style_guidance_scale, tv_scale, sat_scale),
                            jnp.stack([rng.split() for _ in range(n)]),
                            make_cutouts,
                            make_cutouts_style,
                            cut_batches)
        # grad : [n, 1, c, h, w]
        grad = grad.squeeze(1)
        # grad : [n, c, h, w]
        magnitude = grad.square().mean(axis=(1,2,3), keepdims=True).sqrt()
        grad = grad / magnitude * magnitude.clamp(max=0.2)
        return grad

    for i in range(n_batches):
        timestring = time.strftime('%Y%m%d%H%M%S')

        if type(prompt) is list:
          text_embed = prompt[i % len(prompt)]
          this_title = title[i % len(prompt)]
        else:
          text_embed = prompt
          this_title = title

        # Apply jitter
        text_embed = text_embed.broadcast_to([batch_size, 512])
        text_embed = norm1(text_embed + jax.random.normal(rng.split(), [batch_size, 512]) / math.sqrt(prompt_jitter_scale * 512.0))

        with open('text_embeds.txt', 'w') as fp:
            print(title, file=fp)
            print(text_embed.tolist(), file=fp)

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
                    if j >= 500:
                        os.makedirs('progress', exist_ok=True)
                        filename = f'progress/{timestring}_{this_title}_{k:05}_{j:03}.png'
                        image.save(filename)

        for k in range(batch_size):
            filename = f'progress_{i * batch_size + k:05}.png'
            os.makedirs('samples', exist_ok=True)
            dname = f'samples/{timestring}_{k}_{this_title}.png'
            with open(filename, 'rb') as fp:
              data = fp.read()
            with open(dname, 'wb') as fp:
              fp.write(data)


run()


class FIFOLock(object):
    def __init__(self):
        self.cv = threading.Condition()
        self.waiting = []
        self.active = None
    def __enter__(self):
        self.acquire()
    def __exit__(self, type, value, traceback):
        self.release()

    def can_go(self, tid):
        return self.active is None and self.waiting[:1] == [tid]

    def acquire(self):
        tid = threading.currentThread().ident
        self.waiting.append(tid)
        with self.cv:
            self.cv.wait_for(functools.partial(self.can_go, tid))
            self.active = tid
            del self.waiting[0]

    def release(self):
        with self.cv:
            self.active = None
            self.cv.notify_all()

def handler(ui):
    class Handler(BaseHTTPRequestHandler):
        def sendmsg(self, msg):
            self.wfile.write(json.dumps(msg).encode('utf-8'))
            self.wfile.write(b'\n')

        def do_POST(self):
            path = '/' + self.path.strip('/')
            if path == '/generate':
                content_length = int(self.headers['Content-Length'])
                data = self.rfile.read(content_length)
                params = json.loads(data)
                with ui.lock:
                    globals()['title'] = params['title']
                    globals()['prompt'] = txt(params['title'])
                    globals()['n_batches'] = 1
                    globals()['init_image'] = params.get('init_image', None)
                    if 'init_image' in params:
                        globals()['skip_timesteps'] = int(
                            params.get('skip_timesteps', 500))
                    else:
                        globals()['skip_timesteps'] = 0
                    globals()['seed'] = int(params['seed'])
                    globals()['cut_batches'] = int(params.get('cut_batches', 4))
                    globals()['n_batches'] = 1
                    run()
                self.send_response(200)
                self.end_headers()

    return Handler

class UiServer(object):
    def __init__(self, port=8544):
        self.httpd = ThreadingHTTPServer(('0.0.0.0', port), handler(self))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.lock = FIFOLock()

def run_server():
    try:
        server = UiServer()
        print('Ready to accept requests!', file=sys.stderr)
        server.thread.join()
    except KeyboardInterrupt:
        exit()

run_server()
