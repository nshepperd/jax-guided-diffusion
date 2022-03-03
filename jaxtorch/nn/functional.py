import jax
import jax.numpy as jnp

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    (B, C1, iH, iW) = input.shape
    (C2, _, kH, kW) = weight.shape
    assert weight.shape[1] == C1//groups

    if type(padding) is int:
        padding = ((padding, padding), (padding, padding))
    elif type(padding) is str:
        padding = padding.upper()
    elif len(padding) == 2:
        padding = ((padding[0], padding[0]), (padding[1], padding[1]))

    if dilation > 1:
        dilation = (dilation, dilation)
    else:
        dilation = None

    output = jax.lax.conv_general_dilated(lhs=input,
                                          rhs=weight,
                                          window_strides=[stride, stride],
                                          padding=padding,
                                          lhs_dilation=None,
                                          rhs_dilation=dilation,
                                          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                                          feature_group_count=groups)
    if bias is not None:
        output = output + bias.reshape(bias.shape[0], 1, 1)
    return output


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    (B, C1, iH) = input.shape
    (C2, _, kH) = weight.shape
    assert weight.shape[1] == C1//groups

    if type(padding) is int:
        padding = [(padding, padding)]
    elif type(padding) is str:
        padding = padding.upper()
    elif len(padding) == 1:
        padding = ((padding[0], padding[0]),)

    if dilation > 1:
        dilation = (dilation,)
    else:
        dilation = None

    output = jax.lax.conv_general_dilated(lhs=input,
                                          rhs=weight,
                                          window_strides=[stride],
                                          padding=padding,
                                          lhs_dilation=None,
                                          rhs_dilation=dilation,
                                          dimension_numbers=('NCH', 'OIH', 'NCH'),
                                          feature_group_count=groups)
    if bias is not None:
        output = output + bias.reshape(bias.shape[0], 1)
    return output

def normalize(input, p=2.0, dim=1, eps=1e-12):
    if p != 2.0:
        raise NotImplementedError('only p=2.0 implemented so far')
    mag = input.square().sum(axis=dim, keepdims=True).sqrt()
    return input / mag.clamp(minval=eps)