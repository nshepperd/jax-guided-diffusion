import os
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
class PixelArtV4(nn.Module):
    def __init__(self, timestep_channels=16):
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
            ResConvBlock(3 + timestep_channels, cs[0], cs[0]),
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

    def forward(self, cx, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))
        alphas, sigmas = cosine.to_alpha_sigma(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

pixelartv4_model = PixelArtV4()
pixelartv4_model.labeled_parameters_()
pixelartv4_wrap = make_cosine_model(pixelartv4_model)

pixelartv6_model = PixelArtV4(timestep_channels=64)
pixelartv6_model.labeled_parameters_()
pixelartv6_wrap = make_cosine_model(pixelartv6_model)
