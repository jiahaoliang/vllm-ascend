from __future__ import annotations

import ctypes
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import (
    Backend,
    require_aligned_batch_results,
)

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerRangeReqMeta,
    LayerSaveTask,
    SharedBlockData,
    LayerTransferArrays,
    LayerTransferTask,
    LayerwisePreparation,
    ReqMeta,
    get_block_hashes,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import (
    LayerTransferArrayBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)

_H2D_STAGGER_SPIN_US = 50
_KVPOOL_RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"


def _build_range_debug_payload(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> dict[str, Any]:
    nested_sizes = [[int(size) for size in key_sizes] for key_sizes in sizes]
    return {
        "event": "range",
        "direction": direction,
        "layer_id": int(layer_id),
        "key_count": len(results),
        "requested_bytes": [sum(key_sizes) for key_sizes in nested_sizes],
        "sizes": nested_sizes,
        "object_offsets": [
            [int(offset) for offset in key_offsets]
            for key_offsets in object_offsets
        ],
        "results": [int(result) for result in results],
    }


def _emit_range_debug_event(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> None:
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
            return
        payload = _build_range_debug_payload(
            direction, layer_id, sizes, object_offsets, results
        )
        logger.info(
            "%s %s",
            _KVPOOL_RANGE_DEBUG_PREFIX,
            json.dumps(payload, separators=(",", ":")),
        )
    except Exception:
        pass


def _emit_commit_debug_event(
    layer_id: int,
    key_count: int,
    results: list[int],
) -> None:
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
            return
        payload = {
            "event": "commit",
            "layer_id": int(layer_id),
            "key_count": int(key_count),
            "results": [int(result) for result in results],
        }
        logger.info(
            "%s %s",
            _KVPOOL_RANGE_DEBUG_PREFIX,
            json.dumps(payload, separators=(",", ":")),
        )
    except Exception:
        pass


@dataclass(frozen=True)
class _LayerRevokeTask:
    keys: tuple[str, ...]


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


def _mark_last_transfer_tasks(layer_tasks: list[list[LayerTransferTask]], operation: str) -> None:
    """Assign request completion to its last actual layer transfer task."""
    last_task_by_req: dict[str, LayerTransferTask] = {}
    for tasks in layer_tasks:
        for task in tasks:
            task.finished_req_ids.clear()
            completion = task.completion
            if completion is None:
                raise RuntimeError(
                    f"Layerwise {operation} completion was not prepared for "
                    f"layer {task.layer_id}, group {task.group_id}"
                )
            if len(completion.req_ids) != len(completion.is_last_chunks):
                raise RuntimeError(
                    f"Mismatched {operation} completion metadata for layer {task.layer_id}, group {task.group_id}"
                )
            for req_id, is_last_chunk in zip(completion.req_ids, completion.is_last_chunks):
                if is_last_chunk:
                    last_task_by_req[req_id] = task

    # The final physical layer may already be resident in HBM. In that case,
    # finish the request on its last submitted transfer instead.
    for req_id, task in last_task_by_req.items():
        task.finished_req_ids.add(req_id)


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        num_layers: int,
        group_id: int = 0,
    ) -> None:
        self.my_key_index = my_key_index
        self.num_ranks_per_layer = num_ranks_per_layer
        self.page_size_bytes = page_size_bytes
        self.num_layers = num_layers
        self.group_id = group_id
        self._block_len_np = np.asarray(token_database.group_block_len[group_id], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[group_id],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(
            group_id,
            token_database.group_block_len[group_id],
        )
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        self._caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
        self._block_ids_buf: np.ndarray | None = None
        self._block_gvas_buf: np.ndarray | None = None

    def _ensure_buf(self, capacity: int) -> tuple[np.ndarray, np.ndarray]:
        if self._block_ids_buf is None or len(self._block_ids_buf) < capacity:
            self._block_ids_buf = np.empty(capacity, dtype=np.int64)
            self._block_gvas_buf = np.empty(capacity, dtype=np.int64)
        assert self._block_ids_buf is not None and self._block_gvas_buf is not None
        return self._block_ids_buf[:capacity], self._block_gvas_buf[:capacity]

    @staticmethod
    def _dedupe_transfer_blocks(
        block_ids_arr: np.ndarray,
        block_gvas_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_ids_arr.size <= 1:
            return block_ids_arr, block_gvas_arr

        block_transfer_array = np.column_stack((block_ids_arr, block_gvas_arr))
        _, unique_indices = np.unique(
            block_transfer_array,
            axis=0,
            return_index=True,
        )
        if unique_indices.size == block_ids_arr.size:
            return block_ids_arr, block_gvas_arr

        return (
            block_ids_arr[unique_indices],
            block_gvas_arr[unique_indices],
        )

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        caches_per_layer = self._caches_per_layer
        # group_* arrays are laid out flat as [layer0_caches..., layer1_caches...];
        # slice the per-layer window for ``layer_id``. Using the full length as the
        # stride (the old behaviour) overshoots for layer_id >= 1 and yields empty
        # slices -> broadcast errors.
        base_offset = layer_id * caches_per_layer
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset : base_offset + caches_per_layer]
        layer_block_len = self._block_len_np[base_offset : base_offset + caches_per_layer]
        layer_block_stride = self._block_stride_np[base_offset : base_offset + caches_per_layer]
        # Per-cache inner offsets within one layer's page: [0, len0, len0+len1, ...].
        layer_inner_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
        )
        rank_layer_offset = layer_id * self.page_size_bytes
        logger.debug(
            "[KVPOOL] build_transfer layer=%d page_size=%d caches_per_layer=%d "
            "rank_layer_offset=%d layer_block_len=%s layer_inner_offsets=%s "
            "base_gvas=%s",
            layer_id,
            self.page_size_bytes,
            caches_per_layer,
            rank_layer_offset,
            layer_block_len.tolist(),
            layer_inner_offsets.tolist(),
            base_gvas_arr.tolist(),
        )

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
        )

    def _require_request_arrays(
        self,
        block_range: LayerBlockRange,
        is_save: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        group_id = self.group_id
        block_ids_np: np.ndarray | None
        block_gvas_np: np.ndarray | None
        if is_save:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.block_gvas_np
        else:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.load_block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.load_block_gvas_np
        if block_ids_np is None or block_gvas_np is None:
            raise RuntimeError(
                f"ReqMeta {'save' if is_save else 'load'} block metadata"
                f" is not initialized for request {request.req_id}"
            )
        return block_ids_np, block_gvas_np

    @staticmethod
    def _request_block_keys(
        request: ReqMeta,
        is_save: bool,
    ) -> tuple[list[str | None], int, str | None]:
        if is_save:
            return (
                request.save_block_keys,
                request.save_key_block_offset,
                request.save_last_block_key,
            )
        return (
            request.load_block_keys,
            request.load_key_block_offset,
            request.load_last_block_key,
        )

    def _build_key_major_shared(
        self,
        task: LayerTransferTask,
        is_save: bool,
    ) -> SharedBlockData:
        block_ids: list[int] = []
        block_keys: list[str] = []
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_load_keys: list[str] = []
        seen_save_keys: set[str] = set()

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            all_load_keys.extend(request.load_keys)

            request_block_ids = (
                request.block_ids_np.tolist()
                if request.block_ids_np is not None
                else request.block_ids
            )
            if (
                block_range.start_block < 0
                or block_range.end_block < block_range.start_block
                or block_range.end_block > len(request_block_ids)
            ):
                raise RuntimeError(
                    f"ReqMeta block metadata does not cover requested block range "
                    f"[{block_range.start_block}, {block_range.end_block})"
                )

            request_keys, key_block_offset, last_block_key = self._request_block_keys(
                request,
                is_save,
            )
            key_start = block_range.start_block - key_block_offset
            key_end = block_range.end_block - key_block_offset
            if key_start < 0 or key_end > len(request_keys):
                raise RuntimeError(
                    f"ReqMeta {'save' if is_save else 'load'} block key metadata "
                    f"does not cover requested block range "
                    f"[{block_range.start_block}, {block_range.end_block}) "
                    f"with offset {key_block_offset}"
                )

            for block_id, key in zip(
                request_block_ids[block_range.start_block : block_range.end_block],
                request_keys[key_start:key_end],
                strict=True,
            ):
                if key is not None and (not is_save or key not in seen_save_keys):
                    block_ids.append(block_id)
                    block_keys.append(key)
                    if is_save:
                        seen_save_keys.add(key)

            if block_range.partial_block_index is not None:
                partial_block_index = block_range.partial_block_index
                if partial_block_index < 0 or partial_block_index >= len(request_block_ids):
                    raise RuntimeError(
                        f"ReqMeta block metadata does not cover partial block "
                        f"index {partial_block_index}"
                    )
                if last_block_key is not None and (
                    not is_save or last_block_key not in seen_save_keys
                ):
                    block_ids.append(request_block_ids[partial_block_index])
                    block_keys.append(last_block_key)
                    if is_save:
                        seen_save_keys.add(last_block_key)

        return SharedBlockData(
            block_ids_arr=np.asarray(block_ids, dtype=np.int64),
            block_gvas_arr=None,
            block_keys=block_keys,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            load_keys=all_load_keys,
        )

    def build_shared(self, task: LayerTransferTask, is_save: bool = True) -> SharedBlockData | None:
        """Pre-compute shared block data that is identical across all layers."""
        if not task.block_ranges:
            return None

        if task.use_key_major_ranges:
            return self._build_key_major_shared(task, is_save)

        total = 0
        for block_range in task.block_ranges:
            total += block_range.end_block - block_range.start_block
            if block_range.partial_block_index is not None:
                total += 1

        block_ids_arr, block_gvas_arr = self._ensure_buf(total)
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_load_keys: list[str] = []
        offset = 0

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            if request.load_keys:
                all_load_keys.extend(request.load_keys)
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range, is_save)
            gva_block_offset = request.gva_block_offset if is_save else request.load_gva_block_offset

            num_blocks = block_range.end_block - block_range.start_block
            if num_blocks > 0:
                gva_start = block_range.start_block - gva_block_offset
                gva_end = block_range.end_block - gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {gva_block_offset}"
                    )
                end = offset + num_blocks
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                assert request.last_block_gva is not None
                block_ids_arr[offset] = block_ids_np[block_range.partial_block_index]
                block_gvas_arr[offset] = request.last_block_gva
                offset += 1

        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(block_ids_arr[:offset], block_gvas_arr[:offset])

        logger.debug(
            "[KVPOOL] build_shared req_ids=%s block_gvas_arr=%s block_ids_arr=%s",
            req_ids,
            block_gvas_arr.tolist(),
            block_ids_arr.tolist(),
        )
        return SharedBlockData(
            block_ids_arr=block_ids_arr,
            block_gvas_arr=block_gvas_arr,
            block_keys=None,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            load_keys=all_load_keys,
        )

    def build_addrs(
        self,
        shared: SharedBlockData,
        layer_id: int,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta:
        """Compute per-layer addresses from pre-computed shared block data."""
        if shared.block_keys is not None:
            base_offset = layer_id * self._caches_per_layer
            layer_base_addrs = self._kv_caches_base_addr_np[
                base_offset : base_offset + self._caches_per_layer
            ]
            layer_block_len = self._block_len_np[
                base_offset : base_offset + self._caches_per_layer
            ]
            layer_block_stride = self._block_stride_np[
                base_offset : base_offset + self._caches_per_layer
            ]
            layer_inner_offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
            )
            offsets = (layer_id * self.page_size_bytes + layer_inner_offsets).tolist()
            sizes = layer_block_len.tolist()
            all_buffers = [
                (layer_base_addrs + block_id * layer_block_stride).tolist()
                for block_id in shared.block_ids_arr
            ]
            return LayerRangeReqMeta(
                req_ids=shared.req_ids,
                layer_id=layer_id,
                block_ids=shared.block_ids_arr.tolist(),
                keys=shared.block_keys,
                all_buffers=all_buffers,
                all_sizes=[sizes.copy() for _ in shared.block_ids_arr],
                all_offsets=[offsets.copy() for _ in shared.block_ids_arr],
                load_keys=shared.load_keys,
            )

        assert shared.block_gvas_arr is not None
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            shared.block_ids_arr, shared.block_gvas_arr, layer_id
        )

        return LayerBatchReqMeta(
            req_ids=shared.req_ids,
            layer_id=layer_id,
            is_last_chunks=shared.is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
            load_keys=shared.load_keys,
        )

    def build(
        self,
        task: LayerTransferTask,
        is_save: bool = True,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task, is_save)
        if shared is None:
            return None
        layer_index = task.layer_id if task.use_key_major_ranges else task.layer_idx_in_group
        return self.build_addrs(shared, layer_index)


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        name: str = "KVTransferThread",
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event or threading.Event()
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.num_addrs_per_block = len(token_database.group_block_len[0])
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._fatal_error: BaseException | None = None

    def prepare_layerwise_tasks(
        self,
        layer_tasks: list[list[LayerTransferTask]],
    ) -> None:
        """Prepare key-based metadata shared by all layers when needed."""

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta | LayerwisePreparation,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(
        self,
        req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            if req_ids is None:
                finished_requests = self.finished_requests.copy()
                self.finished_requests.clear()
            else:
                finished_requests = self.finished_requests & req_ids
                self.finished_requests -= finished_requests
        return finished_requests

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"{self.name} failed during asynchronous transfer preparation") from self._fatal_error

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts: np.ndarray = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices: np.ndarray = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        # direction: 0/SMEMB_COPY_L2G = save (write), 1/SMEMB_COPY_G2L = load (read)
        dir_name = "save(L2G)" if direction == 0 else "load(G2L)" if direction == 1 else f"dir{direction}"
        logger.debug(
            "[KVPOOL] batch_copy %s gvas=%d total_bytes=%d",
            dir_name,
            len(gvas),
            int(sizes.sum()) if len(sizes) else 0,
        )

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        assert self.m_store.store is not None
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[KVPOOL] batch_copy %s split_gvas=%s split_sizes=%s",
                    dir_name,
                    split_gvas.tolist(),
                    split_sizes.tolist(),
                )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self._set_os_thread_name()
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                self._fatal_error = e
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

    def _store_mask(self, req_meta: ReqMeta) -> tuple[list[bool], ...] | None:
        store_mask = getattr(self.token_database, "store_mask", None)
        if store_mask is None:
            return None
        try:
            return store_mask(req_meta.token_len_chunk, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_meta.req_id, exc)
            return None

    def _load_mask(self, req_meta: ReqMeta, token_len: int) -> tuple[list[bool], ...] | None:
        load_mask = getattr(self.token_database, "load_mask", None)
        if load_mask is None:
            return None
        return load_mask(req_meta.block_hashes, token_len)

    def _mask_allows_chunk(
        self,
        masks: tuple[list[bool], ...] | None,
        group_id: int,
        start: int,
    ) -> bool:
        mask_allows_chunk = getattr(self.token_database, "mask_allows_chunk", None)
        if mask_allows_chunk is None:
            return True
        return mask_allows_chunk(masks, group_id, start)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        put_step: int = 1,
        kv_role: str = "kv_producer",
        ready_event: threading.Event | None = None,
        group_uses_align_state: list[bool] | None = None,
        enable_kv_event: bool = False,
        worker: Any = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state or []
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}
        self.worker = worker

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def is_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return req_id in self.stored_requests

    def get_stored_request_count(self, req_id: str) -> int | None:
        with self.done_task_lock:
            return self.stored_requests.get(req_id)

    def get_stored_requests_snapshot(self) -> dict[str, int]:
        with self.done_task_lock:
            return dict(self.stored_requests)

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def mark_completed_events(self, event_id: int | None) -> None:
        if event_id is not None:
            with self.completed_events_lock:
                self.completed_events[event_id] = 1

    def get_completed_events(self):
        if not self.completed_events:
            return None
        with self.completed_events_lock:
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _handle_request(self, req_meta: ReqMeta):
        if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
            try:
                self.worker._store_kv_tp_mismatch(req_meta)
            finally:
                self.request_queue.task_done()
            return
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            if req_id not in self.stored_requests:
                self.request_queue.task_done()
                return

            store_masks = self._store_mask(req_meta)
            for group_id in req_meta.kv_cache_group_ids or [0]:
                starts = []
                ends = []
                keys = []
                block_hashes = []
                key_block_ids = []
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )

                for start, end, key, block_id in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    if not self._mask_allows_chunk(store_masks, group_id, start):
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())
                    block_hashes.append(group_block_hashes[start // group_block_size])
                    key_block_ids.append(block_id)

                if (
                    not self.dcp_size > 1
                    and not req_meta.disable_tp_key_sharding
                    and not self.group_uses_align_state[group_id]
                ):
                    starts = starts[self.tp_rank % self.put_step :: self.put_step]
                    ends = ends[self.tp_rank % self.put_step :: self.put_step]
                    keys = keys[self.tp_rank % self.put_step :: self.put_step]
                    block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]
                    key_block_ids = key_block_ids[self.tp_rank % self.put_step :: self.put_step]

                if not keys:
                    continue

                exists_states = self.lookup(keys)
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                if not missing_indices:
                    continue

                starts = [starts[index] for index in missing_indices]
                ends = [ends[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]
                block_hashes = [block_hashes[index] for index in missing_indices]
                key_block_ids = [key_block_ids[index] for index in missing_indices]

                logger.debug(
                    "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                    len(keys),
                    token_len // group_block_size,
                    len(missing_indices),
                    req_id,
                    group_id,
                )
                logger.debug(
                    "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                    req_id,
                    group_id,
                    token_len,
                    len(keys),
                    keys[:3],
                )

                addrs = []
                sizes = []
                stored_events: list[BlockStored] = []
                all_hashes = [maybe_convert_block_hash(bh) for bh in group_block_hashes]
                for index, start in enumerate(starts):
                    addr, size, _ = self._prepare_value(
                        start,
                        ends[index],
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=key_block_ids[index],
                    )
                    addrs.append(addr)
                    sizes.append(size)

                    # Create KV event
                    if self.enable_kv_event:
                        token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                        block_size = (
                            req_meta.original_block_size[group_id]
                            if isinstance(req_meta.original_block_size, list)
                            else req_meta.original_block_size
                        )
                        if block_size is not None:
                            block_idx = start // group_block_size
                            if block_idx >= len(all_hashes):
                                continue
                            current_hash = all_hashes[block_idx]
                            parent_hash = all_hashes[block_idx - 1] if block_idx > 0 else None
                            stored_event = BlockStored(
                                block_hashes=[current_hash],
                                parent_block_hash=parent_hash,
                                token_ids=token_ids,
                                block_size=block_size,
                                lora_id=None,
                                medium="cpu",
                                lora_name=None,
                            )
                            stored_events.append(stored_event)
                            logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                if self.kv_role == "kv_consumer":
                    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                        keys,
                        addrs,
                        sizes,
                        kv_cache_group_id=group_id,
                    )

                if current_event is not None:
                    current_event.synchronize()
                self.m_store.put(keys, addrs, sizes)

                # TODO Query specific replica info to update the event
                if self.enable_kv_event and stored_events is not None:
                    self.update_kv_event(stored_events)
        finally:
            # always free blocks
            self.mark_completed_events(req_meta.event_id)
        self.dec_stored_request(req_id)
        if self.stored_requests.get(req_id, -1) == 0:
            self.delete_finished_stored_request(req_id)
            self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int = 1,
        dcp_size: int = 1,
        ready_event: threading.Event | None = None,
        invalid_block_ids: set[int] | None = None,
        invalid_block_ids_lock: threading.Lock | None = None,
        worker: Any = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self._invalid_block_ids = invalid_block_ids if invalid_block_ids is not None else set()
        self._invalid_block_ids_lock = invalid_block_ids_lock or threading.Lock()
        self.worker = worker

    def _handle_request(self, req_meta: ReqMeta):
        try:
            load_spec = req_meta.load_spec
            req_id = req_meta.req_id
            if load_spec is None:
                logger.error("KV pool async recv request %s has no load spec; skip load.", req_id)
                self.set_finished_request(req_id)
                return

            token_len = load_spec.token_len
            if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                group_block_size = self._get_block_size(0)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                self.worker._load_kv_tp_mismatch(
                    req_meta.block_hashes,
                    req_meta.block_ids_by_group[0],
                    token_len,
                    mask_num,
                )
                self.set_finished_request(req_id)
                return

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            group_ids = req_meta.kv_cache_group_ids or [0]
            load_masks = self._load_mask(req_meta, token_len)
            for group_id in group_ids:
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                for start, end, key, block_id in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    if not self._mask_allows_chunk(load_masks, group_id, start):
                        continue
                    addr, size, block_id = self._prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=block_id,
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                self.set_finished_request(req_id)
                return
            key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
            addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
            size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
            block_id_list_c = (
                block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
            )
            logger.debug(
                "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
                key_list_c[:3],
            )
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            if ret is not None and any(r != 0 for r in ret):
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    ret,
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            elif ret is None:
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    [1] * len(block_id_list_c),
                )
                if len(req_meta.block_ids_by_group) == 1:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        req_id,
                        missing_block_ids,
                    )
            logger.debug(
                "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
                req_id,
                token_len,
                req_meta.kv_cache_group_ids or [0],
                len(key_list_c),
            )
            self.set_finished_request(req_id)
        finally:
            self.request_queue.task_done()


class KVCacheStoreKeyLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events

    def build_cached_process_tokens(self, task: LayerTransferTask) -> dict[int, list[tuple[int, int, list]]] | None:
        """Pre-compute process_tokens results for all layers (Key path).

        Returns a dict mapping block_range index to a list of
        (start, end, key_all_layers) tuples, where key_all_layers is the
        result of key.split_layers().
        """
        if not task.block_ranges:
            return None

        group_block_size = self._get_block_size(0)
        cache: dict[int, list[tuple[int, int, list]]] = {}

        for br_idx, block_range in enumerate(task.block_ranges):
            request = block_range.request
            mask_num = request.save_start_token // group_block_size * group_block_size
            entries = []
            for start, end, key in self.token_database.process_tokens(
                request.save_end_token,
                request.block_hashes,
                mask_num,
            ):
                block_index = start // group_block_size
                if block_index < block_range.start_block or block_index >= block_range.end_block:
                    continue
                key_all = key.split_layers(self.final_layer_id + 1)
                entries.append((start, end, key_all))
            cache[br_idx] = entries

        return cache

    def prepare_layerwise_tasks(
        self,
        layer_tasks: list[list[LayerTransferTask]],
    ) -> None:
        first_task = next((tasks[0] for tasks in layer_tasks if tasks), None)
        if first_task is None:
            return
        cached = self.build_cached_process_tokens(first_task)
        if cached is not None:
            for tasks in layer_tasks:
                for task in tasks:
                    task.cached_process_tokens = cached

    def add_request(  # type: ignore[override]
        self, req_meta: list[LayerTransferTask] | LayerwisePreparation
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask] | LayerwisePreparation
    ):
        if isinstance(transfer_tasks, LayerwisePreparation):
            transfer_tasks.ensure_ready()
            self.request_queue.task_done()
            return
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")

        transfer_task = transfer_tasks[0]
        if transfer_task.preparation is not None:
            transfer_task.preparation.ensure_ready()
        layer_id = transfer_task.layer_id
        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []

        # Reuse pre-computed process_tokens results if available
        cached_tokens = transfer_task.cached_process_tokens

        for br_idx, block_range in enumerate(transfer_task.block_ranges):
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            starts = []
            ends = []
            keys = []
            group_block_size = self._get_block_size(0)

            if cached_tokens is not None:
                # Fast path: reuse cached (start, end, key_all) tuples
                for start, end, key_all in cached_tokens[br_idx]:
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key_all[layer_id])
            else:
                mask_num = request.save_start_token // group_block_size * group_block_size
                for start, end, key in self.token_database.process_tokens(
                    request.save_end_token,
                    request.block_hashes,
                    mask_num,
                ):
                    block_index = start // group_block_size
                    if block_index < block_range.start_block or block_index >= block_range.end_block:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.split_layers(self.final_layer_id + 1)[layer_id])

            if not self.dcp_size > 1:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]

            for index, key in enumerate(keys):
                key_list.append(key.to_string())
                addr, size, _ = self.token_database.prepare_value_layer(
                    starts[index],
                    ends[index],
                    request.block_ids,
                    layer_id,
                )
                addr_list.append(addr)
                size_list.append(size)

        for req_id in req_ids:
            self.dec_stored_request(req_id)

        if key_list:
            exists_states = self.lookup(key_list)
            missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
            keys_to_put = [key_list[index] for index in missing_indices]
            addrs_to_put = [addr_list[index] for index in missing_indices]
            sizes_to_put = [size_list[index] for index in missing_indices]
            if keys_to_put:
                self.sync_save_events[layer_id].synchronize()
                self.m_store.put(keys_to_put, addrs_to_put, sizes_to_put)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk and self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)

        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        logger.debug("Key-based layer save event set: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()


class KVCacheStoreKeyLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreKeyLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1

    def add_request(  # type: ignore[override]
        self, req_meta: LayerLoadTask
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _wait_for_save(self, layer_id: int) -> None:
        while not self.layer_save_finished_events[layer_id].wait(timeout=10):
            logger.info("Layerwise %d save wait timed out, keep waiting before load", layer_id)
        logger.debug("Key-based layer save event cleared: layer %d", layer_id)
        self.layer_save_finished_events[layer_id].clear()

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        wait_for_save = data.wait_for_save_layer
        layer_id = data.layer_id
        if wait_for_save is not None:
            self._wait_for_save(wait_for_save)

        if data.attention_start_gate is not None:
            while not data.attention_start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_id)

        key_list = []
        addr_list = []
        size_list = []
        req_ids = []
        is_last_chunks = []
        if len(data.transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(data.transfer_tasks)}")
        if data.transfer_tasks:
            transfer_task = data.transfer_tasks[0]
            for block_range in transfer_task.block_ranges:
                request = block_range.request
                req_ids.append(request.req_id)
                is_last_chunks.append(request.is_last_chunk)
                for block_index in range(block_range.start_block, block_range.end_block):
                    if block_index >= len(request.block_hashes):
                        continue
                    block_hash = request.block_hashes[block_index]
                    chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
                    key = self.token_database._make_key_by_hash(
                        chunk_hash,
                    ).split_layers(self.final_layer_id + 1)[layer_id]
                    group_block_size = self._get_block_size(0)
                    start = block_index * group_block_size
                    end = start + group_block_size
                    addr, size, _ = self.token_database.prepare_value_layer(
                        start,
                        end,
                        request.block_ids,
                        layer_id,
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)

        if key_list:
            shift = (self.tp_rank * len(key_list)) // self.tp_size
            key_list_c = _circular_shift(key_list, shift)
            addr_list_c = _circular_shift(addr_list, shift)
            size_list_c = _circular_shift(size_list, shift)
            self.m_store.get(key_list_c, addr_list_c, size_list_c)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_ids, is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)

        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug("Key-based layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()
        data.transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_array_builders: list[LayerTransferArrayBuilder] | None = None,
        pd_transfer_waiter: Callable[[int], None] | None = None,
        sync_attn_events: list[torch.npu.Event] | None = None,
        layer_attn_recorded_events: list[threading.Event] | None = None,
        group_builders: list[LayerBatchBuilder] | None = None,
        put_started_keys: set[str] | None = None,
        put_started_keys_lock: threading.Lock | None = None,
        session_tracker: MooncakeSessionTracker | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.sync_attn_events = sync_attn_events
        self.layer_attn_recorded_events = layer_attn_recorded_events
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.group_array_builders = group_array_builders
        self.pd_transfer_waiter = pd_transfer_waiter
        if group_array_builders is not None:
            self.transfer_array_builder = group_array_builders[0]
        else:
            self.transfer_array_builder = LayerTransferArrayBuilder(
                token_database,
                num_layers,
                group_id=0,
            )
        self._put_started_keys = put_started_keys if put_started_keys is not None else set()
        self._put_started_keys_lock = put_started_keys_lock or threading.Lock()
        self._session_tracker = session_tracker
        self._active_put_keys: set[str] | None = None
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = None

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Build Mooncake range metadata shared by all transferred layers."""
        if self.group_builders is None:
            raise RuntimeError("Mooncake layer batch builders are not configured")
        return self.group_builders[task.group_id].build_shared(task, is_save=True)

    def prepare_layerwise_tasks(
        self,
        layer_tasks: list[list[LayerTransferTask]],
    ) -> None:
        _mark_last_transfer_tasks(layer_tasks, "save")
        last_task = None
        keys_by_group: dict[int, list[str]] = {}
        for tasks in layer_tasks:
            for task in tasks:
                task.write_finish_keys.clear()
                last_task = task
                if task.transfer_data is None:
                    raise RuntimeError(
                        f"Layerwise save data was not prepared for layer {task.layer_id}, group {task.group_id}"
                    )
                keys_by_group[task.group_id] = task.transfer_data.keys
        if last_task is not None:
            # One GVA contains every layer. Publish only after its final copy.
            last_task.write_finish_keys.extend(key for keys in keys_by_group.values() for key in keys)

    def _wait_attention_done(self, physical_layer: int) -> None:
        # slot_free also requires the compute stream to be past this layer's
        # attention. The threading flag guards against the npu event being a
        # no-op when synchronize() runs before record().
        if self.layer_attn_recorded_events is None or self.sync_attn_events is None:
            return
        while not self.layer_attn_recorded_events[physical_layer].wait(timeout=10):
            logger.info("Layerwise %d attention not recorded, keep waiting before slot_free", physical_layer)
        self.sync_attn_events[physical_layer].synchronize()

    def _set_slot_free(self, physical_layer: int) -> None:
        # slot_free = L2G copy done AND PD transfer done AND attention done.
        assert not self.layer_save_finished_events[physical_layer].is_set(), f"thread: {physical_layer} save failed "
        logger.debug("Layer save event set: layer %d", physical_layer)
        self.layer_save_finished_events[physical_layer].set()

    def add_request(  # type: ignore[override]
        self, req_meta: LayerSaveTask | LayerwisePreparation
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def add_revoke_request(self, keys: list[str]) -> None:
        deduplicated_keys = tuple(dict.fromkeys(keys))
        if deduplicated_keys:
            self.request_queue.put(_LayerRevokeTask(deduplicated_keys))

    def _remove_started_keys(self, keys: list[str]) -> None:
        with self._put_started_keys_lock:
            self._put_started_keys.difference_update(keys)

    def _revoke_range_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        try:
            results = require_aligned_batch_results(
                "batch_revoke", keys, self.m_store.batch_revoke(keys)
            )
            if any(result != 0 for result in results):
                logger.error("Layerwise revoke failed keys=%s results=%s", keys, results)
        except Exception as exc:
            logger.error("Layerwise revoke raised keys=%s error=%s", keys, exc)
        finally:
            # This tracker only gates future put-start calls. Drop keys after a
            # revoke attempt even when the remote cleanup fails; the Master TTL
            # owns any remaining PROCESSING session.
            self._remove_started_keys(keys)
            if self._session_tracker is not None:
                self._session_tracker.revoke_put_keys(keys)

    def _handle_range_request(self, req_meta: LayerRangeReqMeta) -> None:
        layer_id = req_meta.layer_id
        if self._active_put_keys is None or layer_id == 0:
            # This mutable set is scoped to one forward batch. Keep the shared
            # metadata immutable so later layers can filter failed keys safely.
            self._active_put_keys = set(req_meta.keys)
        active_indices = [
            index
            for index, key in enumerate(req_meta.keys)
            if key in self._active_put_keys
        ]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            if layer_id < len(self.sync_save_events):
                self.sync_save_events[layer_id].synchronize()
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results = require_aligned_batch_results(
                "batch_copy_put",
                active_keys,
                self.m_store.batch_copy_put(
                    active_keys,
                    active_buffers,
                    active_sizes,
                    active_offsets,
                ),
            )
            _emit_range_debug_event(
                "save", layer_id, active_sizes, active_offsets, results
            )
            failed_keys = [
                key
                for key, result in zip(active_keys, results, strict=True)
                if result < 0
            ]
            if failed_keys:
                # A ranged-write failure only invalidates this key; remaining
                # active keys continue copying their later layer ranges.
                self._revoke_range_keys(failed_keys)
                self._active_put_keys.difference_update(failed_keys)

        if layer_id == self.final_layer_id:
            # Only keys that completed every layer range may publish COMPLETE.
            active_keys = [
                key for key in req_meta.keys if key in self._active_put_keys
            ]
            if active_keys:
                try:
                    commit_results = require_aligned_batch_results(
                        "batch_commit",
                        active_keys,
                        self.m_store.batch_commit(active_keys),
                    )
                    _emit_commit_debug_event(
                        layer_id, len(active_keys), commit_results
                    )
                except Exception as exc:
                    logger.error(
                        "Layerwise commit raised keys=%s error=%s",
                        active_keys,
                        exc,
                    )
                    self._revoke_range_keys(active_keys)
                else:
                    failed_commit_keys = [
                        key
                        for key, result in zip(
                            active_keys, commit_results, strict=True
                        )
                        if result != 0
                    ]
                    if failed_commit_keys:
                        self._revoke_range_keys(failed_commit_keys)
                    committed_keys = [
                        key
                        for key, result in zip(
                            active_keys, commit_results, strict=True
                        )
                        if result == 0
                    ]
                    if self._session_tracker is not None:
                        self._session_tracker.commit_put_keys(committed_keys)
                    self._remove_started_keys(active_keys)
            self._active_put_keys = None

    def _handle_mooncake_range_save(
        self,
        transfer_tasks: list[LayerTransferTask],
        layer_id: int,
    ) -> None:
        shared: SharedBlockData | None = None
        try:
            try:
                if len(transfer_tasks) != 1:
                    raise ValueError(
                        f"Expected one Mooncake range task, got {len(transfer_tasks)}"
                    )
                task = transfer_tasks[0]
                shared = task.shared_block_data
                if shared is None:
                    raise RuntimeError(
                        f"Mooncake range metadata was not prepared for layer {layer_id}"
                    )
                if self.group_builders is None:
                    raise RuntimeError("Mooncake layer batch builders are not configured")
                builder = self.group_builders[task.group_id]
                req_meta = builder.build_addrs(shared, task.layer_id)
                if not isinstance(req_meta, LayerRangeReqMeta):
                    raise TypeError(
                        "Expected Mooncake range metadata, got "
                        f"{type(req_meta).__name__}"
                    )
                self._handle_range_request(req_meta)
            except Exception as exc:
                logger.error(
                    "Mooncake ranged save failed layer=%d error=%s",
                    layer_id,
                    exc,
                )
                if self._active_put_keys is not None:
                    keys_to_revoke = (
                        [
                            key
                            for key in shared.block_keys
                            if key in self._active_put_keys
                        ]
                        if shared is not None and shared.block_keys is not None
                        else sorted(self._active_put_keys)
                    )
                elif shared is not None and shared.block_keys is not None:
                    keys_to_revoke = list(dict.fromkeys(shared.block_keys))
                else:
                    keys_to_revoke = []
                self._revoke_range_keys(keys_to_revoke)
                # Later layers must stay inactive after any ranged-save failure.
                self._active_put_keys = set()

            if self.pd_transfer_waiter is not None:
                self.pd_transfer_waiter(layer_id)
            self._wait_attention_done(layer_id)

            req_ids = [
                block_range.request.req_id
                for task in transfer_tasks
                for block_range in task.block_ranges
            ]
            for req_id in req_ids:
                self.dec_stored_request(req_id)
                if self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)
            self._set_slot_free(layer_id)
        finally:
            self._finish_layer_save_task(transfer_tasks)

    def _finish_layer_save_task(
        self,
        transfer_tasks: list[LayerTransferTask],
    ) -> None:
        # Queue accounting is independent of transfer success. Request and
        # layer completion are published only by the successful path below.
        transfer_tasks.clear()
        self.request_queue.task_done()

    def _handle_request(  # type: ignore[override]
        self,
        request: LayerSaveTask | LayerwisePreparation | _LayerRevokeTask,
    ):
        if isinstance(request, _LayerRevokeTask):
            try:
                self._revoke_range_keys(list(request.keys))
            finally:
                self.request_queue.task_done()
            return
        if isinstance(request, LayerwisePreparation):
            try:
                request.ensure_ready()
            finally:
                self.request_queue.task_done()
            return
        physical_layer = request.layer_id
        transfer_tasks = request.transfer_tasks
        if transfer_tasks and any(task.use_key_major_ranges for task in transfer_tasks):
            self._handle_mooncake_range_save(transfer_tasks, physical_layer)
            return
        try:
            preparation = transfer_tasks[0].preparation if transfer_tasks else None
            if preparation is not None:
                preparation.ensure_ready()

            has_any_save = False
            all_gvas = []
            all_addrs = []
            all_sizes = []
            all_req_ids = []
            finished_req_ids: set[str] = set()
            write_finish_keys: list[str] = []
            for task in transfer_tasks:
                if task.layer_id != physical_layer:
                    raise RuntimeError(
                        f"Layerwise save request for layer {physical_layer} contains task for layer {task.layer_id}"
                    )
                transfer_data = task.transfer_data
                completion = task.completion
                if transfer_data is None or completion is None:
                    raise RuntimeError(
                        f"Layerwise save metadata was not prepared for layer {physical_layer}, group {task.group_id}"
                    )
                has_any_save = True
                builder = (
                    self.group_array_builders[task.group_id]
                    if self.group_array_builders
                    else self.transfer_array_builder
                )
                arrays = builder.build_addrs(transfer_data, task.layer_idx_in_group)
                all_req_ids.extend(completion.req_ids)
                finished_req_ids.update(task.finished_req_ids)
                write_finish_keys.extend(task.write_finish_keys)
                all_gvas.append(arrays.gvas_array)
                all_addrs.append(arrays.addr_array)
                all_sizes.append(arrays.size_array)

            if has_any_save:
                self.sync_save_events[physical_layer].synchronize()
                gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
                addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
                size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
                res = self._batch_copy_with_limits(
                    gvas_array,
                    addr_array,
                    size_array,
                    0,
                    self.max_transfer_blocks,
                    self.max_transfer_bytes,
                )
                if physical_layer <= 2 or res != 0:
                    logger.info(
                        "save_thread: layer=%d groups=%d blocks=%d res=%d",
                        physical_layer,
                        len(all_gvas),
                        len(gvas_array),
                        res,
                    )
                if res != 0:
                    raise RuntimeError(f"Layerwise {physical_layer} save batch_copy failed with return code {res}")
                if write_finish_keys:
                    finish_results = self.m_store.batch_write_finish(
                        write_finish_keys,
                        [0] * len(write_finish_keys),
                    )
                    if len(finish_results) != len(write_finish_keys) or any(
                        result != 0 for result in finish_results
                    ):
                        raise RuntimeError(
                            "Layerwise save batch_write_finish failed: "
                            f"expected={len(write_finish_keys)}, results={finish_results}"
                        )

            if self.pd_transfer_waiter is not None:
                self.pd_transfer_waiter(physical_layer)
            self._wait_attention_done(physical_layer)

            if has_any_save:
                for req_id in all_req_ids:
                    self.dec_stored_request(req_id)
                for req_id in finished_req_ids:
                    if self.try_finish_and_delete_stored_request(req_id):
                        self.set_finished_request(req_id)

            self._set_slot_free(physical_layer)
        except Exception as exc:
            logger.error("Layerwise save handler failed layer=%d error=%s", physical_layer, exc)
            raise
        finally:
            self._finish_layer_save_task(transfer_tasks)


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
        h2d_stagger_us: int = 0,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        group_array_builders: list[LayerTransferArrayBuilder] | None = None,
        load_lease_releaser: Callable[[set[str]], None] | None = None,
        group_builders: list[LayerBatchBuilder] | None = None,
        *,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
        load_abort_event: threading.Event | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1
        self.h2d_stagger_us = h2d_stagger_us
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock
        self.group_array_builders = group_array_builders
        self.load_lease_releaser = load_lease_releaser
        if group_array_builders is not None:
            self.transfer_array_builder = group_array_builders[0]
        else:
            self.transfer_array_builder = LayerTransferArrayBuilder(
                token_database,
                num_layers,
                group_id=0,
            )
        self._load_abort_event = load_abort_event or threading.Event()
        self._active_load_indices: set[int] | None = None
        self.group_builders: list[LayerBatchBuilder] | None = group_builders
        if group_builders is not None:
            self.layer_batch_builder = group_builders[0]
        else:
            self.layer_batch_builder = None

    def build_shared_data(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Build Mooncake range metadata shared by all transferred layers."""
        if self.group_builders is None:
            raise RuntimeError("Mooncake layer batch builders are not configured")
        return self.group_builders[task.group_id].build_shared(task, is_save=False)

    def prepare_layerwise_tasks(
        self,
        layer_tasks: list[list[LayerTransferTask]],
    ) -> None:
        _mark_last_transfer_tasks(layer_tasks, "load")

    def _set_layer_load_done(self, layer_id: int) -> None:
        assert not self.layer_load_finished_events[layer_id].is_set()
        logger.debug("Layer load event set: layer %d", layer_id)
        self.layer_load_finished_events[layer_id].set()

    def add_request(  # type: ignore[override]
        self, req_meta: LayerLoadTask | LayerwisePreparation
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _get_h2d_stagger_delay_us(self, layer_id: int) -> int:
        if self.h2d_stagger_us <= 0:
            return 0
        slot = (self.tp_rank + layer_id) % self.tp_size
        return slot * self.h2d_stagger_us

    def _stagger_h2d_submit(self, layer_id: int) -> None:
        delay_us = self._get_h2d_stagger_delay_us(layer_id)
        if delay_us <= 0:
            return

        deadline_ns = time.perf_counter_ns() + delay_us * 1_000
        sleep_us = delay_us - _H2D_STAGGER_SPIN_US
        if sleep_us > 0:
            time.sleep(sleep_us / 1_000_000)
        while time.perf_counter_ns() < deadline_ns:
            pass

    def _mark_invalid_transfer_task_blocks(
        self,
        transfer_tasks: list[LayerTransferTask],
    ) -> None:
        block_ids: set[int] = set()
        for task in transfer_tasks:
            for block_range in task.block_ranges:
                request_block_ids = block_range.request.block_ids
                block_ids.update(
                    request_block_ids[
                        block_range.start_block : block_range.end_block
                    ]
                )
                partial_block_index = block_range.partial_block_index
                if partial_block_index is not None and 0 <= partial_block_index < len(request_block_ids):
                    block_ids.add(request_block_ids[partial_block_index])
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def _mark_invalid_range_indices(
        self,
        req_meta: LayerRangeReqMeta,
        indices: list[int],
    ) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(
                req_meta.block_ids[index] for index in indices
            )

    def _handle_range_request(
        self,
        req_meta: LayerRangeReqMeta,
        shared: SharedBlockData,
    ) -> None:
        layer_id = req_meta.layer_id
        # Every layer is built from the same SharedBlockData row order, so a
        # row index identifies one key/local-block destination across layers.
        if self._active_load_indices is None or layer_id == 0:
            self._active_load_indices = set(range(len(req_meta.keys)))

        assert self._active_load_indices is not None
        active_indices = [
            index
            for index in range(len(req_meta.keys))
            if not self._load_abort_event.is_set()
            and index in self._active_load_indices
        ]
        active_keys = [req_meta.keys[index] for index in active_indices]
        if active_keys:
            self._stagger_h2d_submit(layer_id)
            active_buffers = [req_meta.all_buffers[index] for index in active_indices]
            active_sizes = [req_meta.all_sizes[index] for index in active_indices]
            active_offsets = [req_meta.all_offsets[index] for index in active_indices]
            results = require_aligned_batch_results(
                "batch_copy_get",
                active_keys,
                self.m_store.batch_copy_get(
                    active_keys,
                    active_buffers,
                    active_sizes,
                    active_offsets,
                ),
            )
            _emit_range_debug_event(
                "load", layer_id, active_sizes, active_offsets, results
            )
            failed_indices = [
                index
                for index, result in zip(active_indices, results, strict=True)
                if result < 0
            ]
            if failed_indices:
                self._mark_invalid_range_indices(req_meta, failed_indices)
                # A negative ranged result belongs to one destination row; it
                # does not by itself invalidate every row sharing the key.
                self._active_load_indices.difference_update(failed_indices)

        if layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(
                req_meta.req_ids, shared.is_last_chunks, strict=True
            ):
                if is_last_chunk:
                    self.set_finished_request(req_id)
            self._active_load_indices = None

    def _finish_layer_load_task(
        self,
        data: LayerLoadTask,
        layer_id: int,
        succeeded: bool,
    ) -> None:
        try:
            if succeeded:
                self._set_layer_load_done(layer_id)
                self.get_event.set()
        finally:
            data.transfer_tasks.clear()
            self.request_queue.task_done()

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask | LayerwisePreparation
    ):
        if isinstance(data, LayerwisePreparation):
            try:
                data.ensure_ready()
            finally:
                self.request_queue.task_done()
            return
        layer_id = data.layer_id
        succeeded = False
        transfer_tasks = data.transfer_tasks
        use_key_major_ranges = bool(transfer_tasks) and any(
            task.use_key_major_ranges for task in transfer_tasks
        )
        try:
            wait_for_save = data.wait_for_save_layer
            attention_start_gate = data.attention_start_gate

            if data.preparation is not None:
                data.preparation.ensure_ready()

            if len(transfer_tasks) == 0:
                if wait_for_save is not None:
                    while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                        logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
                    logger.debug("Layer save event cleared: layer %d", wait_for_save)
                    self.layer_save_finished_events[wait_for_save].clear()
                succeeded = True
                return

            range_meta: LayerRangeReqMeta | None = None
            range_shared: SharedBlockData | None = None
            task_arrays: list[tuple[LayerTransferTask, LayerTransferArrays]] = []
            if use_key_major_ranges:
                if len(transfer_tasks) != 1 or not all(
                    task.use_key_major_ranges for task in transfer_tasks
                ):
                    raise ValueError(
                        f"Expected one Mooncake range task, got {len(transfer_tasks)}"
                    )
                task = transfer_tasks[0]
                range_shared = task.shared_block_data
                if range_shared is None:
                    raise RuntimeError(
                        f"Mooncake range metadata was not prepared for layer {layer_id}"
                    )
                if self.group_builders is None:
                    raise RuntimeError("Mooncake layer batch builders are not configured")
                builder = self.group_builders[task.group_id]
                req_meta = builder.build_addrs(range_shared, task.layer_id)
                if not isinstance(req_meta, LayerRangeReqMeta):
                    raise TypeError(
                        "Expected Mooncake range metadata, got "
                        f"{type(req_meta).__name__}"
                    )
                range_meta = req_meta
            else:
                # Expand each group's block IDs and base GVAs into this layer's
                # copy arrays before waiting on the preceding save layer.
                for task in transfer_tasks:
                    transfer_data = task.transfer_data
                    builder = (
                        self.group_array_builders[task.group_id]
                        if self.group_array_builders
                        else self.transfer_array_builder
                    )
                    if transfer_data is None or task.completion is None:
                        raise RuntimeError(
                            f"Layerwise load metadata was not prepared for layer {layer_id}, group {task.group_id}"
                        )
                    arrays = builder.build_addrs(
                        transfer_data,
                        task.layer_idx_in_group,
                    )
                    task_arrays.append((task, arrays))

                if not task_arrays:
                    succeeded = True
                    return

            if wait_for_save is not None:
                while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                    logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
                logger.debug("Layer save event cleared: layer %d", wait_for_save)
                self.layer_save_finished_events[wait_for_save].clear()

            if attention_start_gate is not None:
                while not attention_start_gate.wait(timeout=10):
                    logger.info("Layerwise %d load waits for attention compute start", layer_id)

            if range_meta is not None:
                assert range_shared is not None
                self._handle_range_request(range_meta, range_shared)
                succeeded = True
                return

            finished_req_ids: set[str] = set()
            all_gvas = []
            all_addrs = []
            all_sizes = []
            for task, arrays in task_arrays:
                finished_req_ids.update(task.finished_req_ids)
                all_gvas.append(arrays.gvas_array)
                all_addrs.append(arrays.addr_array)
                all_sizes.append(arrays.size_array)

            self._stagger_h2d_submit(layer_id)
            gvas_array = np.concatenate(all_gvas) if len(all_gvas) > 1 else all_gvas[0]
            addr_array = np.concatenate(all_addrs) if len(all_addrs) > 1 else all_addrs[0]
            size_array = np.concatenate(all_sizes) if len(all_sizes) > 1 else all_sizes[0]
            res = self._batch_copy_with_limits(
                gvas_array,
                addr_array,
                size_array,
                1,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            )
            if layer_id <= 2 or res != 0:
                logger.info(
                    "load_thread: layer=%d groups=%d blocks=%d res=%d",
                    layer_id,
                    len(all_gvas),
                    len(gvas_array),
                    res,
                )
            if res != 0:
                raise RuntimeError(f"Layerwise {layer_id} load batch_copy failed with return code {res}")

            if finished_req_ids and self.load_lease_releaser is not None:
                self.load_lease_releaser(finished_req_ids)
            for req_id in finished_req_ids:
                self.set_finished_request(req_id)
            succeeded = True
        except Exception as exc:
            logger.error("Layerwise load handler failed layer=%d error=%s", layer_id, exc)
            self._mark_invalid_transfer_task_blocks(data.transfer_tasks)
            if use_key_major_ranges:
                if self._active_load_indices is not None:
                    self._active_load_indices.clear()
                # KVPoolWorker owns the exactly-once session cleanup. Waking
                # the layer waiter here only reports that the failed read has
                # drained; invalid-block state prevents consuming it as a hit.
                self._load_abort_event.set()
                succeeded = True
            else:
                raise
        finally:
            self._finish_layer_load_task(data, layer_id, succeeded)


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
