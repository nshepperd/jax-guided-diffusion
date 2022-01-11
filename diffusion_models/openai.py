import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.schedules import cosine, ddpm

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
