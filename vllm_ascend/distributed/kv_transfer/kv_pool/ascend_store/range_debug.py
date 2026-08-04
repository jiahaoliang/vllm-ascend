from __future__ import annotations

import json
from collections.abc import Callable

from vllm.logger import logger

from vllm_ascend import envs

RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"


def _emit(payload_factory: Callable[[], dict[str, object]]) -> None:
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
            return
        logger.info(
            "%s %s",
            RANGE_DEBUG_PREFIX,
            json.dumps(payload_factory(), separators=(",", ":")),
        )
    except Exception:
        pass


def _build_range_payload(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> dict[str, object]:
    nested_sizes = [[int(size) for size in key_sizes] for key_sizes in sizes]
    return {
        "event": "range",
        "direction": direction,
        "layer_id": int(layer_id),
        "key_count": len(results),
        "requested_bytes": [sum(key_sizes) for key_sizes in nested_sizes],
        "sizes": nested_sizes,
        "object_offsets": [[int(offset) for offset in key_offsets] for key_offsets in object_offsets],
        "results": [int(result) for result in results],
    }


def emit_range_event(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> None:
    _emit(lambda: _build_range_payload(direction, layer_id, sizes, object_offsets, results))


def emit_commit_event(layer_id: int, key_count: int, results: list[int]) -> None:
    _emit(
        lambda: {
            "event": "commit",
            "layer_id": int(layer_id),
            "key_count": int(key_count),
            "results": [int(result) for result in results],
        }
    )


def emit_whole_key_event(direction: str, key_count: int) -> None:
    _emit(
        lambda: {
            "event": "whole_key",
            "direction": direction,
            "key_count": int(key_count),
        }
    )
