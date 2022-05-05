# CC12M_1 model, ported from <https://github.com/crowsonkb/v-diffusion-pytorch>

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

class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([
            nn.Linear(f_in, f_mid),
            nn.ReLU(),
            nn.Linear(f_mid, f_out),
            nn.ReLU() if not is_last else nn.Identity(),
        ], skip)

class Modulation2d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, cx, input):
        scales, shifts = jnp.split(self.layer(cx, self.state['cond']), 2, axis=-1)
        return shifts[..., None, None] + input * (scales[..., None, None] + 1)

class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.GroupNorm(1, c_mid, affine=False),
            Modulation2d(state, feats_in, c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(),
            Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(),
            nn.ReLU() if not is_last else nn.Identity(),
        ], skip)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, cx, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(cx, self.norm(cx, input))
        qkv = qkv.rearrange('n (k n_head c_head) h w -> n k n_head (h w) c_head', k=3, n_head = self.n_head, c_head = c//self.n_head)
        q, k, v = [qkv[:, 0], qkv[:, 1], qkv[:, 2]]
        scale = k.shape[3]**-0.25
        # att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        att = jnp.einsum('nhlc,nhLc->nhlL', q*scale, k*scale)
        att = jax.nn.softmax(att, axis=3)
        y = jnp.einsum('nhlL,nhLc->nhlc', att, v)
        y = y.rearrange('n n_head (h w) c_head -> n (n_head c_head) h w', h=h, w=w)
        # y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(cx, self.out_proj(cx, y))

class CC12M1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 256, 256)
        self.clip_model = 'ViT-B/16'

        c = 128  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(512 + 128, 1024, 1024),
            ResLinearBlock(1024, 1024, 1024, is_last=True),
        )

        # with torch.no_grad():
        #     for param in self.mapping.parameters():
        #         param *= 0.5**0.5

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = AvgPool2d()
        self.up = nn.image.Upsample2d('linear')

        self.net = nn.Sequential(   # 256x256
            conv_block(3 + 16, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 128x128
                conv_block(cs[0], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 64x64
                    conv_block(cs[1], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,  # 32x32
                        conv_block(cs[2], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,  # 16x16
                            conv_block(cs[3], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            SkipBlock([
                                self.down,  # 8x8
                                conv_block(cs[4], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                SkipBlock([
                                    self.down,  # 4x4
                                    conv_block(cs[5], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SelfAttention2d(cs[6], cs[6] // 64),
                                    conv_block(cs[6], cs[6], cs[5]),
                                    SelfAttention2d(cs[5], cs[5] // 64),
                                    self.up,
                                ]),
                                conv_block(cs[5] * 2, cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[4]),
                                SelfAttention2d(cs[4], cs[4] // 64),
                                self.up,
                            ]),
                            conv_block(cs[4] * 2, cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[3]),
                            SelfAttention2d(cs[3], cs[3] // 64),
                            self.up,
                        ]),
                        conv_block(cs[3] * 2, cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[2]),
                        self.up,
                    ]),
                    conv_block(cs[2] * 2, cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[1]),
                    self.up,
                ]),
                conv_block(cs[1] * 2, cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[0]),
                self.up,
            ]),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 3, is_last=True),
        )

        # with torch.no_grad():
        #     for param in self.net.parameters():
        #         param *= 0.5**0.5

    def forward(self, cx, input, t, clip_embed):
        clip_embed = norm1(clip_embed) * clip_embed.shape[-1]**0.5
        mapping_timestep_embed = self.mapping_timestep_embed(cx, t[:, None])
        self.state['cond'] = self.mapping(cx, jnp.concatenate([clip_embed, mapping_timestep_embed], axis=1))
        timestep_embed = expand_to_planes(self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))
        self.state.clear()
        alphas, sigmas = cosine.to_alpha_sigma(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

cc12m_1_model = CC12M1Model()
cc12m_1_model.labeled_parameters_()

cc12m_1_wrap = make_cosine_model(cc12m_1_model)
cc12m_1_cfg_params = LazyParams.pt('https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth')

def cc12m_1_cfg_wrap(clip_embed, cfg_guidance_scale):
    return LerpModels([(cc12m_1_wrap(cc12m_1_cfg_params(), clip_embed=clip_embed), cfg_guidance_scale),
                       (cc12m_1_wrap(cc12m_1_cfg_params(), clip_embed=0*clip_embed), 1-cfg_guidance_scale)])

def cc12m_1_classifier_wrap(clip_embed):
    return LerpModels([(cc12m_1_wrap(cc12m_1_cfg_params(), clip_embed=clip_embed), 1.0),
                       (cc12m_1_wrap(cc12m_1_cfg_params(), clip_embed=0*clip_embed), -1.0)])
