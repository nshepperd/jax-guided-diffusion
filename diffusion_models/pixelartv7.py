import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init

from diffusion_models.common import *
from diffusion_models.schedules import cosine
from diffusion_models.lazy import LazyParams

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)

CHANNELS=192
class PixelArtV7_IC(nn.Module):
    def __init__(self, timestep_channels=64):
        super().__init__()
        c = CHANNELS  # The base channel count

        self.timestep_embed = FourierFeatures(1, timestep_channels, std=1.0)

        self.arch = '122222'

        muls = [1, 2, 2, 2, 2, 2]
        cs = [CHANNELS * m for m in muls]

        def downsample(c1, c2):
            return nn.Sequential(nn.image.Downsample2d(), nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity())

        def upsample(c1, c2):
            return nn.Sequential(nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity(), nn.image.Upsample2d())

        class ResConvBlock(ResidualBlock):
            def __init__(self, c_in, c_mid, c_out, dropout=True):
                skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
                super().__init__([
                    nn.Conv2d(c_in, c_mid, 3, padding=1),
                    nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
                    nn.ReLU(),
                    nn.Conv2d(c_mid, c_out, 3, padding=1),
                    nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
                    nn.ReLU() if dropout else nn.Identity(),
                ], skip)


        self.net = nn.Sequential(
            ResConvBlock(3 + 3 + timestep_channels, cs[0], cs[0]),
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
                            SkipBlock([
                                downsample(cs[4], cs[5]),  # 32x32
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                upsample(cs[5],cs[4]),
                            ]),
                            ResConvBlock(cs[4]*2, cs[4], cs[4]),
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
            ResConvBlock(cs[0], cs[0], 3, dropout=False),
        )

    def forward(self, cx, input, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, cond, timestep_embed], axis=1))
        alphas, sigmas = cosine.to_alpha_sigma(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

pixelartv7_ic_model = PixelArtV7_IC()
pixelartv7_ic_model.labeled_parameters_()

@make_partial
@jax.jit
def pixelartv7_ic_wrap(params, x, cosine_t, key, cond=None, cfg_guidance_scale=None):
    [n, c, h, w] = x.shape
    cx = Context(params, key).eval_mode_()
    return (pixelartv7_ic_model(cx, x, cosine_t.broadcast_to([n]), cond) * cfg_guidance_scale +
            pixelartv7_ic_model(cx, x, cosine_t.broadcast_to([n]), 0*cond) * (1.0-cfg_guidance_scale))

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, cx, input):
        return self.ln(cx, input.rearrange('n c h w -> n h w c')).rearrange('n h w c -> n c h w')

class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = LayerNorm2d(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, cx, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(cx, self.norm(cx, input))
        qkv = qkv.rearrange('n (k n_head c_head) h w -> n k n_head (h w) c_head', k=3, n_head = self.n_head, c_head = c//self.n_head)
        q, k, v = [qkv[:, 0], qkv[:, 1], qkv[:, 2]]
        scale = k.shape[3]**-0.25
        att = jnp.einsum('nhlc,nhLc->nhlL', q*scale, k*scale)
        att = jax.nn.softmax(att, axis=3)
        y = jnp.einsum('nhlL,nhLc->nhlc', att, v)
        y = y.rearrange('n n_head (h w) c_head -> n (n_head c_head) h w', h=h, w=w)
        return input + self.dropout(cx, self.out_proj(cx, y))

class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU() if dropout else nn.Identity(),
        ], skip)


CHANNELS=192
class PixelArtV7_IC_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        c = CHANNELS  # The base channel count

        self.timestep_embed = FourierFeatures(1, 64, std=100.0)

        self.arch = '122222'

        muls = [1, 2, 2, 2, 2, 2]
        cs = [CHANNELS * m for m in muls]

        def downsample(c1, c2):
            return nn.Sequential(nn.image.Downsample2d(), nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity())

        def upsample(c1, c2):
            return nn.Sequential(nn.Conv2d(c1, c2, 1) if c1!=c2 else nn.Identity(), nn.image.Upsample2d())

        self.net = nn.Sequential(
            ResConvBlock(3 + 3 + 1 + 64, cs[0], cs[0]),
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
                        SelfAttention2d(cs[3], cs[3] // 64),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            downsample(cs[3], cs[4]),  # 16x16
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            SkipBlock([
                                downsample(cs[4], cs[5]),  # 32x32
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                upsample(cs[5],cs[4]),
                            ]),
                            ResConvBlock(cs[4]*2, cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            upsample(cs[4],cs[3]),
                        ]),
                        ResConvBlock(cs[3]*2, cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
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
            ResConvBlock(cs[0], cs[0], 3, dropout=False),
        )

    def forward(self, cx, input, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, cond, timestep_embed], axis=1))
        alphas, sigmas = cosine.to_alpha_sigma(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


pixelartv7_ic_attn_model = PixelArtV7_IC_Attn()
pixelartv7_ic_attn_model.labeled_parameters_()

@make_partial
@jax.jit
def pixelartv7_ic_attn_wrap(params, x, cosine_t, key, cond=None, ic_guidance_scale=None):
    [n, c, h, w] = x.shape
    cx = Context(params, key).eval_mode_()
    if ic_guidance_scale is not None:
        ic_guidance_scale = ic_guidance_scale.broadcast_to([n])[:,None,None,None]
        cond = jnp.concatenate([cond, jnp.ones([n, 1, h, w])], axis=1)
        return (pixelartv7_ic_attn_model(cx, x, cosine_t.broadcast_to([n]), cond) * ic_guidance_scale +
                pixelartv7_ic_attn_model(cx, x, cosine_t.broadcast_to([n]), 0*cond) * (1.0-ic_guidance_scale))
    elif cond is not None:
        cond = jnp.concatenate([cond, jnp.ones([n, 1, h, w])], axis=1)
        return pixelartv7_ic_attn_model(cx, x, cosine_t.broadcast_to([n]), cond)
    else:
        cond = jnp.zeros([n, 4, h, w])
        return pixelartv7_ic_attn_model(cx, x, cosine_t.broadcast_to([n]), cond)


pixelartv7_ic_params = LazyParams.pt(
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-ic-1400.pt'
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v7-large-ic-700.pt'
    , key='params_ema'
)

pixelartv7_ic_attn_params = LazyParams.pt(
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v7-large-ic-attn-600.pt'
    , key='params_ema'
)

def pixelartv7_ic_attn(cond, ic_guidance_scale):
    return pixelartv7_ic_attn_wrap(pixelartv7_ic_attn_params(), cond=cond, ic_guidance_scale=ic_guidance_scale)