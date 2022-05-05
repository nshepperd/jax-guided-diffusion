import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
import math

def alpha_sigma_to_t(alpha, sigma):
    return jnp.arctan2(sigma, alpha) * 2 / math.pi

def get_cosine_alphas_sigmas(t):
    return jnp.cos(t * math.pi/2), jnp.sin(t * math.pi/2)


def get_ddpm_alphas_sigmas(t, initial_snr=10.0):
    log_snrs = -jnp.expm1(1e-4 + initial_snr * t**2).log()
    alphas_squared = jax.nn.sigmoid(log_snrs)
    sigmas_squared = jax.nn.sigmoid(-log_snrs)
    return alphas_squared.sqrt(), sigmas_squared.sqrt()

def ddpm_t_to_cosine(t, initial_snr=10.0):
    alpha, sigma = get_ddpm_alphas_sigmas(t, initial_snr)
    return alpha_sigma_to_t(alpha, sigma)

def cosine_t_to_ddpm(t, initial_snr=10.0):
    alpha, sigma = get_cosine_alphas_sigmas(t)
    log_snr = jnp.log(alpha**2 / sigma**2)
    return ((jnp.log1p(jnp.exp(-log_snr)) - 1e-4) / initial_snr).clamp(0,1).sqrt()


class NoiseSchedule(object):
    def to_cosine(self, t):
        raise NotImplementedError
    def from_cosine(self, t):
        raise NotImplementedError

    def to_alpha_sigma(self, t):
        return get_cosine_alphas_sigmas(self.to_cosine(t))
    def from_alpha_sigma(self, alpha, sigma):
        return self.from_cosine(alpha_sigma_to_t(alpha, sigma))

class Cosine(NoiseSchedule):
    def to_cosine(self, t):
        return t
    def from_cosine(self, t):
        return t

class DDPM(NoiseSchedule):
    def __init__(self, initial_snr=10.0):
        self.initial_snr = initial_snr
    def to_cosine(self, t):
        return ddpm_t_to_cosine(t, self.initial_snr)
    def to_alpha_sigma(self, t):
        return get_ddpm_alphas_sigmas(t, self.initial_snr)
    def from_cosine(self, t):
        return cosine_t_to_ddpm(t, self.initial_snr)

@jax.jit
def reweight_ts(ts, initial_snr=10.0):
    Z = jax.scipy.special.erfinv(1.0 * jax.scipy.special.erf(jnp.sqrt(initial_snr)))
    return jax.scipy.special.erfinv(ts * jax.scipy.special.erf(jnp.sqrt(initial_snr)))/Z

@jax.jit
def inv_reweight_ts(ts, initial_snr=10.0):
    Z = jnp.sqrt(initial_snr)
    return jax.scipy.special.erf(ts * Z) / jax.scipy.special.erf(1.0 * Z)

class ReweightedDDPM(NoiseSchedule):
    def __init__(self, initial_snr=10.0):
        self.ddpm = DDPM(initial_snr)
        self.initial_snr = initial_snr

    def to_cosine(self, t):
        return self.ddpm.to_cosine(reweight_ts(t, self.initial_snr))

    def from_cosine(self, t):
        return inv_reweight_ts(self.ddpm.from_cosine(t), self.initial_snr)

class LinearLogSnr(NoiseSchedule):
    def __init__(self, initial_snr=10.0, final_snr=-10):
        self.initial_snr = initial_snr
        self.final_snr = final_snr
    def to_cosine(self, t):
        alpha, sigma = self.to_alpha_sigma(t)
        return jnp.arctan2(sigma, alpha) * 2 / math.pi
    def to_alpha_sigma(self, t):
        log_snrs = self.initial_snr * (1-t) + self.final_snr * t
        alphas_squared = jax.nn.sigmoid(log_snrs)
        sigmas_squared = jax.nn.sigmoid(-log_snrs)
        return alphas_squared.sqrt(), sigmas_squared.sqrt()
    def from_cosine(self, t):
        alpha, sigma = cosine.to_alpha_sigma(t)
        log_snr = jnp.log(alpha**2 / sigma**2)
        ct = (log_snr - self.initial_snr) / (self.final_snr - self.initial_snr)
        return ct.clamp(0,1)

class Spliced(NoiseSchedule):
    # Fixed to initial_snr=10 for now.
    def to_cosine(self, t):
        crossover_ddpm = 0.48536712
        crossover_cosine = 0.80074257
        big_t = t * (crossover_cosine + 1 - crossover_ddpm)
        return jnp.where(big_t < crossover_cosine,
                         big_t,
                         ddpm_t_to_cosine(big_t - crossover_cosine + crossover_ddpm)
                         )

cosine = Cosine()
ddpm = DDPM()
ddpm2 = DDPM(14.0)
spliced = Spliced()
reweighted = ReweightedDDPM()
