# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Preserve KV-cache group identity when handling load failures."""

from itertools import chain

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.group_block_id import (
    _decode_group_block_id,
)

_ORIGINAL_ATTR = "_ascend_group_block_failure_original"
_original_handle_invalid_blocks = getattr(
    Scheduler._handle_invalid_blocks,
    _ORIGINAL_ATTR,
    Scheduler._handle_invalid_blocks,
)


def _patched_handle_invalid_blocks(
    self: Scheduler,
    invalid_block_ids: set[int],
    num_scheduled_tokens: dict[str, int],
) -> set[str]:
    encoded_block_ids = {block_id for block_id in invalid_block_ids if block_id < 0}
    if not encoded_block_ids:
        return _original_handle_invalid_blocks(
            self,
            invalid_block_ids,
            num_scheduled_tokens,
        )

    affected_req_ids: set[str] = set()
    upstream_block_ids = invalid_block_ids - encoded_block_ids
    if upstream_block_ids:
        affected_req_ids.update(
            _original_handle_invalid_blocks(
                self,
                upstream_block_ids,
                num_scheduled_tokens,
            )
        )

    invalid_group_blocks = {_decode_group_block_id(block_id) for block_id in encoded_block_ids}
    async_load_reqs = (
        request for request in self.skipped_waiting if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    for request in chain(async_load_reqs, self.running):
        block_ids_by_group = self.kv_cache_manager.get_block_ids(request.request_id)
        if any(
            group_id < len(block_ids_by_group) and block_id in block_ids_by_group[group_id]
            for group_id, block_id in invalid_group_blocks
        ):
            affected_req_ids.add(request.request_id)

    return affected_req_ids


setattr(_patched_handle_invalid_blocks, _ORIGINAL_ATTR, _original_handle_invalid_blocks)
Scheduler._handle_invalid_blocks = _patched_handle_invalid_blocks
