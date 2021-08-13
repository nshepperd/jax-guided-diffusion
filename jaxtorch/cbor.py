"""Wraps cbor2 with hooks for encoding and decoding tensors."""
import jax
import cbor2
import numpy as np
import functools

from cbor2 import CBORTag

# Standard tags for multidimensional arrays from RFC8746
# (little-endian, row-major).
TAG_FLOAT32 = 85
TAG_FLOAT64 = 86
TAG_INT32 = 78
TAG_INT64 = 79
TAG_ARRAY = 40

def encode_flat(arr):
    if arr.dtype == np.float32:
        return CBORTag(TAG_FLOAT32, arr.tobytes())
    if arr.dtype == np.int32:
        return CBORTag(TAG_INT32, arr.tobytes())
    else:
        raise NotImplemented

def default_encoder(encoder, value):
    if isinstance(value, jax.numpy.DeviceArray):
        encoder.encode(np.array(value))
    elif isinstance(value, np.ndarray):
        encoder.encode(CBORTag(TAG_ARRAY, [list(value.shape), encode_flat(value)]))
    else:
        raise NotImplemented

def tag_hook(decoder, tag, shareable_index=None):
    if tag.tag == TAG_ARRAY:
        [shape, value] = tag.value
        return value.reshape(shape)
    elif tag.tag == TAG_FLOAT32:
        return np.frombuffer(tag.value, dtype=np.float32)
    elif tag.tag == TAG_INT32:
        return np.frombuffer(tag.value, dtype=np.int32)
    elif tag.tag == TAG_INT64:
        return np.frombuffer(tag.value, dtype=np.int64)
    else:
        return tag

dumps = functools.partial(cbor2.dumps, default=default_encoder)
dump = functools.partial(cbor2.dump, default=default_encoder)

loads = functools.partial(cbor2.loads, tag_hook=tag_hook)
load = functools.partial(cbor2.load, tag_hook=tag_hook)
