import random
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import PRNG
from tqdm import tqdm

from diffusion_models.schedules import cosine

def transfer(x, eps, t_1, t_2):
    alpha1, sigma1 = cosine.to_alpha_sigma(t_1)
    alpha2, sigma2 = cosine.to_alpha_sigma(t_2)
    pred = (x - eps * sigma1) / alpha1
    x = pred * alpha2 + eps * sigma2
    return x, pred

def prk_step(model, x, t_1, t_2, key):
    rng = PRNG(key)
    t_mid = (t_2 + t_1) / 2
    eps_1 = model(x, t_1, rng.split()).eps
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = model(x_1, t_mid, rng.split()).eps
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = model(x_2, t_mid, rng.split()).eps
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = model(x_3, t_2, rng.split()).eps
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred

def plms_step(model, x, old_eps, t_1, t_2, key):
    out = model(x, t_1, key)
    eps, pred = out.eps, out.pred
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps, pred

def prk_sample_loop(model, x, schedule, key):
    """Draws samples from a model given starting noise using Pseudo Runge-Kutta."""
    rng = PRNG(key)
    for i in tqdm(range(schedule.shape[0]-1)):
        y, _, pred = prk_step(model, x, schedule[i], schedule[i + 1], rng.split())
        yield i, x, pred
        x = y
    i = schedule.shape[0]-1
    pred = model(x, schedule[i], rng.split()).pred
    yield i, x, pred

def plms_sample_loop(model, x, schedule, key):
    """Draws samples from a model given starting noise using Pseudo Linear Multistep."""
    rng = PRNG(key)
    old_eps = []
    for i in tqdm(range(schedule.shape[0]-1)):
        if len(old_eps) < 3:
            y, eps, pred = prk_step(model, x, schedule[i], schedule[i + 1], rng.split())
        else:
            y, eps, pred = plms_step(model, x, old_eps, schedule[i], schedule[i + 1], rng.split())
            old_eps.pop(0)
        old_eps.append(eps)
        yield i, x, pred
        x = y
    i = schedule.shape[0]-1
    pred = model(x, schedule[i], rng.split()).pred
    yield i, x, pred
