"""Small FP32 geometry islands; the video backbone stays under AMP."""

from dataclasses import fields, is_dataclass, replace
from functools import wraps

import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _upcast(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{field.name: _upcast(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _upcast(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_upcast(item) for item in value)
    if isinstance(value, list):
        return [_upcast(item) for item in value]
    return value


def fp32_geometry(function):
    """Disable surrounding autocast and promote low-precision tensor inputs.

    Casting is differentiable; masks/indices and FP64 diagnostic inputs are
    preserved. Modules used inside the island must retain FP32 parameters.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        first = next(_tensors((args, kwargs)), None)
        if first is None:
            return function(*args, **kwargs)
        with torch.autocast(first.device.type, enabled=False):
            return function(*_upcast(args), **_upcast(kwargs))

    return wrapped
