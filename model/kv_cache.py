"""Preallocated, request-local inference cache (not model parameters)."""
import torch


class StaticLayerCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.key = self.value = None

    def append(self, key, value):
        end = self.length + key.shape[2]
        if end > self.capacity:
            raise ValueError('KV cache capacity exceeded; reset for a new prompt')
        if self.key is None:
            shape = (*key.shape[:2], self.capacity, key.shape[-1])
            self.key = key.new_empty(shape)
            self.value = value.new_empty(shape)
        for source, target in ((key, self.key), (value, self.value)):
            if source.shape[:2] != target.shape[:2] or source.shape[-1] != target.shape[-1]:
                raise ValueError('cache batch/head dimensions do not match')
            if source.dtype != target.dtype or source.device != target.device:
                raise ValueError('cache precision/device changed; create a new cache')
        self.key[:, :, self.length:end].copy_(key)
        self.value[:, :, self.length:end].copy_(value)
        self.length = end
        return self.key[:, :, :end], self.value[:, :, :end]


class StaticKVCache:
    """Allocate each layer once, lazily using its projected K/V dtype.

    Mutated in place; never share between concurrent requests. reset() reuses
    storage for a new prompt with the same batch size, device and dtype.
    """
    def __init__(self, num_layers, capacity):
        for number in (num_layers, capacity):
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError('num_layers and capacity must be positive integers')
        self.layers = [StaticLayerCache(capacity) for _ in range(num_layers)]
        self.capacity = capacity
        self.owner = None

    @property
    def length(self):
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise ValueError('inconsistent layer lengths; reset the cache')
        return next(iter(lengths))

    def reset(self):
        for layer in self.layers:
            layer.length = 0

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for layer in self.layers
                   for t in (layer.key, layer.value) if t is not None)
