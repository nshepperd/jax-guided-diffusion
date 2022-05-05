import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.lazy import LazyParams
from diffusion_models.schedules import cosine, ddpm

aesthetic_model = nn.Linear(512, 10)
aesthetic_model.labeled_parameters_()
aesthetic_model_params = LazyParams.pt('https://the-eye.eu/public/AI/models/v-diffusion/ava_vit_b_16_full.pth')

def exec_aesthetic_model(params, embed):
  return jax.nn.log_softmax(aesthetic_model(Context(params, None), embed), axis=-1)

# Losses and cond fn.

def AestheticLoss_forward(params, target, scale, embed):
  [k, n, d] = embed.shape
  log_probs = exec_aesthetic_model(params, embed)
  return -(scale * log_probs[:, :, target-1].mean(0)).sum()

def AestheticLoss(target, scale):
  return Partial(AestheticLoss_forward, aesthetic_model_params(), target, scale)

def AestheticExpected_forward(params, scale, embed):
  [k, n, d] = embed.shape
  probs = jax.nn.softmax(exec_aesthetic_model(params, embed))
  expected = (probs * (1 + jnp.arange(10))).sum(-1)
  return -(scale * expected.mean(0)).sum()

def AestheticExpected(scale):
  return Partial(AestheticExpected_forward, aesthetic_model_params(), scale)
