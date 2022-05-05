import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.schedules import cosine

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, cx, input):
        return self.ln(cx, input.rearrange('n c h w -> n h w c')).rearrange('n h w c -> n c h w')

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)

class ShiftScale(nn.Module):
    def __init__(self, c):
        self.c = c
        self.shift = init.zeros(c)
        self.scale = init.ones(c)

    def forward(self, cx, x):
        shift = cx[self.shift][:, None,None]
        scale = cx[self.scale][:, None,None]
        return x * scale + shift


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.GELU() if not is_last else nn.Identity(),
            ShiftScale(c_out),
        ], skip)

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return jnp.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = init.normal(out_features // 2, in_features, stddev=std)
        self.std = std

    def forward(self, cx, input):
        if len(input.shape) == 2:
            f = 2 * math.pi * input @ cx[self.weight].T
        elif len(input.shape) == 4:
            f = 2 * math.pi * jnp.einsum('nchw,kc->nkhw', input, cx[self.weight])
        return jnp.concatenate([f.cos(), f.sin()], axis=1)

class Rearrange(nn.Module):
    def __init__(self, pattern, **kwargs):
        self.pattern = pattern
        self.kwargs = kwargs
    def forward(self, cx, x):
        return x.rearrange(self.pattern, **self.kwargs)


def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(input.shape[:2] + shape[2:])

def downsample(c1, c2):
    return nn.Sequential(nn.image.Downsample2d(), nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity())

def upsample(c1, c2):
    return nn.Sequential(nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity(), nn.image.Upsample2d())

CHANNELS=192
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        c = CHANNELS  # The base channel count

        self.timestep_embed = FourierFeatures(1, 64, std=100.0)
        self.cond_gate = nn.Linear(64, 1)

        self.arch = '12488/gelu/reone'

        muls = [1, 2, 4, 8, 8]
        cs = [CHANNELS * m for m in muls]

        self.net = nn.Sequential(
            nn.Conv2d(3 + 3 + 1 + 64, cs[0], 1),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            SkipBlock([
                downsample(cs[0], cs[1]), # 2x2
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                SkipBlock([
                    downsample(cs[1], cs[2]),  # 4x4
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        downsample(cs[2], cs[3]),  # 8x8
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            downsample(cs[3], cs[4]),  # 16x16
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            upsample(cs[4],cs[3]),
                        ]),
                        ResConvBlock(cs[3]*2, cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        upsample(cs[3],cs[2]),
                    ]),
                    ResConvBlock(cs[2]*2, cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    upsample(cs[2],cs[1]),
                ]),
                ResConvBlock(cs[1]*2, cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                upsample(cs[1],cs[0]),
            ]),
            ResConvBlock(cs[0]*2, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], 3, is_last=True),
        )

    def self_init_weights(self, cx):
        super().self_init_weights(cx)
        cx[self.cond_gate.weight] = jnp.zeros_like(cx[self.cond_gate.weight])
        cx[self.cond_gate.bias] = jnp.ones_like(cx[self.cond_gate.bias])
        for mod in self.net[5:].modules():
            if isinstance(mod, ShiftScale):
                cx[mod.scale] = jnp.zeros_like(cx[mod.scale])

    def forward(self, cx, input, ts, cond, has_cond):
        [n, c, h, w] = input.shape
        ts = ts.broadcast_to([n])
        has_cond = has_cond.broadcast_to([n]).astype(jnp.float32)

        timestep_embed = self.timestep_embed(cx, ts[:, None])
        has_cond = expand_to_planes(has_cond[:, None], input.shape)
        cond_gate = self.cond_gate(cx, timestep_embed)[:, :, None, None]
        timestep_embed = expand_to_planes(timestep_embed, input.shape)

        v = self.net(cx, jnp.concatenate([input, cond * has_cond * cond_gate, has_cond, timestep_embed], axis=1))
        alphas, sigmas = get_cosine_alphas_sigmas(ts)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

upscaler_2_model = DiffusionModel()
upscaler_2_model.labeled_parameters_()

upscaler_2_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/upscaler/2/upscaler-cfg-mixed-12488-265.pt', key='params_ema')

@make_partial
@jax.jit
def upscaler_2_wrap(params, x, cosine_t, key, cond, has_cond):
    [n, c, h, w] = x.shape
    cx = Context(params, key).eval_mode_()
    upscaler_2_model(cx, x, cosine_t, cond, has_cond)

def upscaler_2(cond, has_cond):
    return upscaler_2_wrap(upscaler_2_params(), cond=cond, has_cond=has_cond)
def upscaler_2_cfg(cond, scale):
    if scale == 0.0:
        return upscaler_2(cond, False)
    elif scale == 1.0:
        return upscaler_2(cond, True)
    else:
        scale = jnp.asarray(scale)
        return LerpModels([(upscaler_2(cond, True), scale),
                           (upscaler_2(cond, False), 1 - scale)])