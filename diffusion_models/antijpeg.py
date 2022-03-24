import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.schedules import cosine
from diffusion_models.lazy import LazyParams

# Anti-JPEG model
class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.LeakyReLU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
        ], skip)


CHANNELS=64
class JPEGModel(nn.Module):
    def __init__(self, c=CHANNELS):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16, std=1.0)
        self.class_embed = nn.Embedding(3, 16)

        self.arch = '11(22(22(2)22)22)11'

        self.net = nn.Sequential(
            nn.Conv2d(3 + 16 + 16, c, 1),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.image.Downsample2d(),
                ResConvBlock(c,     c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.image.Downsample2d(),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, 2 * 2, c * 2),
                    SkipBlock([
                        nn.image.Downsample2d(),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.image.Upsample2d(),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.image.Upsample2d(),
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.image.Upsample2d(),
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )

    def forward(self, cx, input, ts, cond):
        [n, c, h, w] = input.shape
        cond = jnp.asarray(cond).broadcast_to([n])
        timestep_embed = expand_to_planes(self.timestep_embed(cx, ts[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
        v = self.net(cx, jnp.concatenate([input, timestep_embed, class_embed], axis=1))
        alphas, sigmas = cosine.to_alpha_sigma(ts)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


jpeg_model = JPEGModel()
jpeg_model.labeled_parameters_()
jpeg_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/jpeg-db-oi-614.pt', key='params_ema')
anti_jpeg = make_cosine_model(jpeg_model, jpeg_params)
def anti_jpeg_cfg():
    return LerpModels([(anti_jpeg(cond=0), 1.0), # clean
                       (anti_jpeg(cond=2), -1.0),# mixed
                       ])

# Secondary Anti-JPEG Classifier

CHANNELS=64
class Classifier(nn.Module):
    def __init__(self, c=CHANNELS):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16, std=1.0)

        self.arch = '11-22-22-22'

        self.net = nn.Sequential(
            nn.Conv2d(3 + 16, c, 1),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            nn.image.Downsample2d(),
            ResConvBlock(c,     c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c * 2),
            nn.image.Downsample2d(),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, 2 * 2, c * 2),
            nn.image.Downsample2d(),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, 1, dropout=False),
        )

    def forward(self, cx, input, ts):
        [n, c, h, w] = input.shape
        timestep_embed = expand_to_planes(self.timestep_embed(cx, ts[:, None]), input.shape)
        return self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))

    def score(self, cx, reals, ts, cond, flood_level, blur_size):
        cond = cond[:, None, None, None]
        logits = self.forward(cx, reals, ts)
        logits = blur_fft(logits, blur_size)
        loss = -jax.nn.log_sigmoid(jnp.where(cond==0, logits, -logits))
        loss = loss.clamp(minval=flood_level, maxval=None)
        return loss.mean()

# @jax.jit
# def classifier_probs(classifier_params, x, ts):
#   n = x.shape[0]
#   cx = Context(classifier_params, jax.random.PRNGKey(0)).eval_mode_()
#   probs = jax.nn.sigmoid(classifier_model(cx, x, ts.broadcast_to([n])))
#   return probs

jpeg_classifier_model = Classifier()
jpeg_classifier_model.labeled_parameters_()
jpeg_classifier_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/jpeg-classifier-72.pt', 'params_ema')

@make_partial
@jax.jit
def jpeg_classifier_wrap(params, x, cosine_t, key, guidance_scale, flood_level=0.7, blur_size=3.0):
    n = x.shape[0]
    cond = jnp.array([0]*n)
    def fwd(x):
        cx = Context(params, key).eval_mode_()
        return guidance_scale * jpeg_classifier_model.score(cx, x, cosine_t.broadcast_to([n]), cond, flood_level, blur_size)
    return -jax.grad(fwd)(x)

def jpeg_classifier(guidance_scale, flood_level=0.7, blur_size=3.0):
    return jpeg_classifier_wrap(jpeg_classifier_params(),
                                guidance_scale=guidance_scale,
                                flood_level=flood_level, blur_size=blur_size)