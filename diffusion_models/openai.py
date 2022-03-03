import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.lazy import LazyParams
from diffusion_models.schedules import cosine, ddpm
from lib import openai

def make_openai_model(model):
    def forward(params, x, cosine_t, key):
        n = x.shape[0]
        ddpm_t = ddpm.from_cosine(cosine_t)
        openai_t = (ddpm_t * 1000).broadcast_to([n])

        cx = Context(params, key).eval_mode_()
        eps = model(cx, x, openai_t)[:, :3, :, :]

        alpha, sigma = cosine.to_alpha_sigma(cosine_t)
        pred = (x - eps * sigma) / alpha
        v    = (eps - x * sigma) / alpha
        return DiffusionOutput(v, pred, eps)
    return make_partial(jax.jit(forward))

def make_openai_finetune_model(model):
    def forward(params, x, cosine_t, key):
        n = x.shape[0]
        alpha, sigma = cosine.to_alpha_sigma(cosine_t)
        cx = Context(params, key).eval_mode_()
        openai_t = (cosine_t * 1000).broadcast_to([n])
        v = model(cx, x, openai_t)[:, :3, :, :]
        pred = x * alpha - v * sigma
        eps = x * sigma + v * alpha
        return DiffusionOutput(v, pred, eps)
    return make_partial(jax.jit(forward))

use_checkpoint = False # Set to True to save some memory

openai_512_model = openai.create_openai_512_model(use_checkpoint=use_checkpoint)
openai_512_model.labeled_parameters_()
openai_512_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_finetune_008100.pt')
openai_512_wrap = make_openai_model(openai_512_model)
def openai_512():
    return openai_512_wrap(openai_512_params())

openai_256_model = openai.create_openai_256_model(use_checkpoint=use_checkpoint)
openai_256_model.labeled_parameters_()
openai_256_params = LazyParams.pt('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt')
openai_256_wrap = make_openai_model(openai_256_model)
def openai_256():
    return openai_256_wrap(openai_256_params())

openai_512_finetune_wrap = make_openai_finetune_model(openai_512_model)
openai_512_finetune_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_openimages_epoch28_withfilter.pt')
def openai_512_finetune():
    return openai_512_finetune_wrap(openai_512_finetune_params())
