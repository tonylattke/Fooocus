"""Gradio compatibility helpers for Fooocus.

Gradio 4 removed sketch Image / source+tool APIs. This module keeps the
asyncio timeout patch and component registry used by localization dumps,
and converts ImageEditor values to the legacy {image, mask} dict format.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import numpy as np

import gradio.routes
from gradio.blocks import Block

all_components: list[Any] = []

if not hasattr(Block, 'original_init'):
    Block.original_init = Block.__init__


def blk_ini(self, *args, **kwargs):
    all_components.append(self)
    return Block.original_init(self, *args, **kwargs)


Block.__init__ = blk_ini

gradio.routes.asyncio = importlib.reload(gradio.routes.asyncio)

if not hasattr(asyncio, 'original_wait_for'):
    asyncio.original_wait_for = asyncio.wait_for


def patched_wait_for(fut, timeout):
    # Fooocus jobs can run longer than Gradio's default wait timeouts.
    del timeout
    return asyncio.original_wait_for(fut, timeout=65535)


asyncio.wait_for = patched_wait_for
if hasattr(gradio.routes.asyncio, 'wait_for'):
    gradio.routes.asyncio.wait_for = patched_wait_for


def _as_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _layer_to_mask(layer: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    layer = np.asarray(layer)
    h, w = shape_hw
    if layer.ndim == 3 and layer.shape[2] == 4:
        alpha = layer[:, :, 3]
        # Also treat bright non-transparent RGB as painted (white brush).
        rgb = layer[:, :, :3]
        painted = np.maximum(alpha, np.max(rgb, axis=2))
        mask = painted
    elif layer.ndim == 3:
        mask = np.max(layer[:, :, :3], axis=2)
    else:
        mask = layer
    if mask.shape[:2] != (h, w):
        # Resize is uncommon; pad/crop defensively.
        out = np.zeros((h, w), dtype=np.uint8)
        hh, ww = min(h, mask.shape[0]), min(w, mask.shape[1])
        out[:hh, :ww] = mask[:hh, :ww]
        mask = out
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return np.stack([mask, mask, mask], axis=2)


def normalize_editor_to_sketch(value: Any) -> Any:
    """Convert Gradio 4 ImageEditor payloads to Gradio 3 sketch dicts.

    ImageEditor value: {background, layers, composite}
    Legacy sketch value: {image, mask}
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    if 'image' in value and 'mask' in value:
        return value
    if 'background' not in value and 'composite' not in value:
        return value

    background = value.get('background')
    if background is None:
        background = value.get('composite')
    if background is None:
        return None

    image = _as_rgb_uint8(background)
    h, w = image.shape[:2]
    layers = value.get('layers') or []
    mask = None
    for layer in layers:
        if layer is None:
            continue
        layer_mask = _layer_to_mask(layer, (h, w))
        mask = layer_mask if mask is None else np.maximum(mask, layer_mask)
    if mask is None:
        mask = np.zeros((h, w, 3), dtype=np.uint8)
    return {'image': image, 'mask': mask}


def editor_image_only(value: Any) -> Any:
    """Extract the RGB image from an ImageEditor or legacy sketch value."""
    if value is None:
        return None
    if isinstance(value, dict):
        if 'background' in value and value['background'] is not None:
            return _as_rgb_uint8(value['background'])
        if 'image' in value and value['image'] is not None:
            return _as_rgb_uint8(value['image'])
        if 'composite' in value and value['composite'] is not None:
            return _as_rgb_uint8(value['composite'])
    if isinstance(value, np.ndarray):
        return _as_rgb_uint8(value)
    return value
