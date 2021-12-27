from abc import abstractmethod

import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
import jaxtorch.nn as nn
from jaxtorch.core import Module
from functools import partial

from .unet import UNetModel

OPENAI_512_CONFIG = {
    'image_size': 512,
    'in_channels': 3,
    'model_channels': 256,
    'out_channels': 6,
    'num_res_blocks': 2,
    'attention_resolutions': (16, 32, 64),
    'dropout': 0.0,
    'channel_mult': (0.5, 1, 1, 2, 2, 4, 4),
    'num_classes': None,
    'use_checkpoint': False,
    'use_fp16': False,
    'num_heads': 4,
    'num_head_channels': 64,
    'num_heads_upsample': -1,
    'use_scale_shift_norm': True,
    'resblock_updown': True,
    'use_new_attention_order': False
}

OPENAI_256_CONFIG = {
    'image_size': 256,
    'in_channels': 3,
    'model_channels': 256,
    'out_channels': 6,
    'num_res_blocks': 2,
    'attention_resolutions': (8, 16, 32),
    'dropout': 0.0,
    'channel_mult': (1, 1, 2, 2, 4, 4),
    'num_classes': None,
    'use_checkpoint': False,
    'use_fp16': False,
    'num_heads': 4,
    'num_head_channels': 64,
    'num_heads_upsample': -1,
    'use_scale_shift_norm': True,
    'resblock_updown': True,
    'use_new_attention_order': False
}

def create_openai_256_model(use_checkpoint=False, **kwargs):
    return partial(UNetModel, **OPENAI_256_CONFIG)(use_checkpoint=use_checkpoint, **kwargs)

def create_openai_512_model(use_checkpoint=False, **kwargs):
    return partial(UNetModel, **OPENAI_512_CONFIG)(use_checkpoint=use_checkpoint, **kwargs)
