import random
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import PRNG
from tqdm import tqdm

from diffusion_models.schedules import cosine
from diffusion_models.common import DiffusionOutput
from dataclasses import dataclass

@dataclass
class ConditionedDiffusionOutput(DiffusionOutput):
    pred0: None # Unconditioned pred.

def transfer(x, eps, t_1, t_2):
    alpha1, sigma1 = cosine.to_alpha_sigma(t_1)
    alpha2, sigma2 = cosine.to_alpha_sigma(t_2)
    pred = (x - eps * sigma1) / alpha1
    x = pred * alpha2 + eps * sigma2
    return x

def prk_step(model_fn, x, t_1, t_2, key):
    rng = PRNG(key)
    t_mid = (t_2 + t_1) / 2
    out_1 = model_fn(x, t_1, rng.split())
    eps_1 = out_1.eps
    x_1 = transfer(x, eps_1, t_1, t_mid)
    eps_2 = model_fn(x_1, t_mid, rng.split()).eps
    x_2 = transfer(x, eps_2, t_1, t_mid)
    eps_3 = model_fn(x_2, t_mid, rng.split()).eps
    x_3 = transfer(x, eps_3, t_1, t_2)
    eps_4 = model_fn(x_3, t_2, rng.split()).eps
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, out_1.pred0

def plms_step(model_fn, x, old_eps, t_1, t_2, key):
    out = model_fn(x, t_1, key)
    eps, pred0 = out.eps, out.pred0
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps, pred0

def ddim_step(model_fn, x, t1, t2, eta, key):
    rng = PRNG(key)

    [n, c, h, w] = x.shape
    alpha1, sigma1 = cosine.to_alpha_sigma(t1)
    alpha2, sigma2 = cosine.to_alpha_sigma(t2)

    out = model_fn(x, t1, rng.split())
    eps, pred0 = out.eps, out.pred0
    pred = (x - eps * sigma1) / alpha1

    # Negative eta allows more extreme levels of noise.
    ddpm_sigma = (sigma2**2 / sigma1**2).sqrt() * (1 - alpha1**2 / alpha2**2).sqrt()
    ddim_sigma = jnp.where(eta >= 0.0,
                           eta * ddpm_sigma, # Normal: eta interpolates between ddim and ddpm
                           -eta * sigma2)    # Extreme: eta interpolates between ddim and q_sample(pred)
    adjusted_sigma = (sigma2**2 - ddim_sigma**2).sqrt()

    x_new = pred * alpha2 + eps * adjusted_sigma + jax.random.normal(rng.split(), x.shape) * ddim_sigma
    return x_new, pred0


def mk_model_fn(model, cond_fn):
    if cond_fn is None:
        def model_fn(x, ts, key):
            out = model(x, ts, key)
            return ConditionedDiffusionOutput(out.v, out.pred, out.eps, out.pred)
        return model_fn

    def model_fn(x, ts, key):
        alpha, sigma = cosine.to_alpha_sigma(ts)

        out = model(x, ts, key)
        score = cond_fn(x, ts, key)

        v = out.v - sigma / alpha * score
        eps = out.eps - sigma * score
        pred = (x - eps * sigma) / alpha
        return ConditionedDiffusionOutput(v, pred, eps, out.pred)
    return model_fn

def prk_sample_loop(model, cond_fn, x, schedule, key, x_fn=None):
    """Draws samples from a model given starting noise using Pseudo Runge-Kutta."""
    rng = PRNG(key)
    model_fn = mk_model_fn(model, cond_fn)
    for i in tqdm(range(schedule.shape[0]-1)):
        if x_fn is not None:
            x = x_fn(x, schedule[i])
        y, _, pred = prk_step(model_fn, x, schedule[i], schedule[i + 1], rng.split())
        yield {'step': i, 'x': x, 'pred': pred}
        x = y
    i = schedule.shape[0]-1
    if schedule[i] > 0:
        pred = model(x, schedule[i], rng.split()).pred
    yield {'step': i, 'x': x, 'pred': pred}

def plms_sample_loop(model, cond_fn, x, schedule, key, x_fn=None):
    """Draws samples from a model given starting noise using Pseudo Linear Multistep."""
    rng = PRNG(key)
    model_fn = mk_model_fn(model, cond_fn)
    old_eps = []
    for i in tqdm(range(schedule.shape[0]-1)):
        if x_fn is not None:
            x = x_fn(x, schedule[i])
        if len(old_eps) < 3:
            y, eps, pred = prk_step(model_fn, x, schedule[i], schedule[i + 1], rng.split())
        else:
            y, eps, pred = plms_step(model_fn, x, old_eps, schedule[i], schedule[i + 1], rng.split())
            old_eps.pop(0)
        old_eps.append(eps)
        yield {'step': i, 'x': x, 'pred': pred}
        x = y
    i = schedule.shape[0]-1
    if schedule[i] > 0:
        pred = model(x, schedule[i], rng.split()).pred
    yield {'step': i, 'x': x, 'pred': pred}

def ddim_sample_loop(model, cond_fn, x, schedule, key, eta=1.0, x_fn=None):
    """Draws samples from a model given starting noise using DDIM (eta=0) / DDPM (eta=1)."""
    rng = PRNG(key)
    model_fn = mk_model_fn(model, cond_fn)
    for i in tqdm(range(schedule.shape[0]-1)):
        if x_fn is not None:
            x = x_fn(x, schedule[i])
        y, pred = ddim_step(model_fn, x, schedule[i], schedule[i + 1], eta, rng.split())
        yield {'step': i, 'x': x, 'pred': pred}
        x = y
    i = schedule.shape[0]-1
    if schedule[i] > 0:
        pred = model(x, schedule[i], rng.split()).pred
    yield {'step': i, 'x': x, 'pred': pred}
