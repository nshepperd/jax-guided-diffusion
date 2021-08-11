from abc import abstractmethod

import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
import jaxtorch.nn as nn
from jaxtorch.core import Module

class TimestepBlock(Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, cx, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, cx, x, emb):
        for layer in self.modules:
            if isinstance(layer, TimestepBlock):
                x = layer(cx, x, emb)
            else:
                x = layer(cx, x)
        return x

class QKVAttentionLegacy(Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, cx, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(3, axis=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = jnp.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = jax.nn.softmax(weight, axis=-1)
        a = jnp.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, width//3, length)

class Upsample2D(Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = jaxtorch.nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, cx, x):
        [b, c, h, w] = x.shape
        assert c == self.channels
        x = x.reshape([b, c, h, 1, w, 1])
        x = jnp.broadcast_to(x, [b, c, h, 2, w, 2])
        x = x.reshape([b, c, h*2, w*2])
        if self.use_conv:
            x = self.conv(cx, x)
        return x

class Downsample2D(Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if self.use_conv:
            self.op = jaxtorch.nn.Conv2d(
                self.channels, self.out_channels, 3, stride=2, padding=1
            )
        else:
            assert self.channels == self.out_channels

    def forward(self, cx, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            return self.op(cx, x)
        else:
            [b, c, h, w] = x.shape
            x = x.reshape(b, c, h//2, 2, w//2, 2)
            x = x.mean(axis=(3, 5))
            return x

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        assert dims == 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample2D(channels, False)
            self.x_upd = Upsample2D(channels, False)
        elif down:
            self.h_upd = Downsample2D(channels, False)
            self.x_upd = Downsample2D(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, zero_init=True)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, cx, x, emb):
        if self.updown:
            in_rest = nn.Sequential(*self.in_layers.modules[:-1])
            in_conv = self.in_layers.modules[-1]
            h = in_rest(cx, x)
            h = self.h_upd(cx, h)
            x = self.x_upd(cx, x)
            h = in_conv(cx, h)
        else:
            h = self.in_layers(cx, x)
        emb_out = self.emb_layers(cx, emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers.modules[0], nn.Sequential(*self.out_layers.modules[1:])
            scale, shift = jnp.split(emb_out, 2, axis=1)
            h = out_norm(cx, h) * (1 + scale) + shift
            h = out_rest(cx, h)
        else:
            h = h + emb_out
            h = self.out_layers(cx, h)
        return self.skip_connection(cx, x) + h
