# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.group_block_id import (
    _encode_group_block_id,
)
from vllm_ascend.patch.platform import patch_group_block_failures


class _InheritedRecomputeScheduler(RecomputeScheduler):
    pass


class _GroupBlockTable:
    def __init__(self, block_ids_by_request: dict[str, tuple[list[int], ...]]):
        self.block_ids_by_request = block_ids_by_request

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.block_ids_by_request[request_id]

    def evict_blocks(self, block_ids: set[int]) -> None:
        pytest.fail(f"encoded group/block failures must not evict blocks: {block_ids}")


def _request(
    request_id: str,
    *,
    status: RequestStatus = RequestStatus.RUNNING,
    num_computed_tokens: int = 64,
):
    return SimpleNamespace(
        request_id=request_id,
        status=status,
        num_computed_tokens=num_computed_tokens,
    )


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        Scheduler,
        AsyncScheduler,
        RecomputeScheduler,
        AsyncRecomputeScheduler,
        _InheritedRecomputeScheduler,
    ],
)
def test_scheduler_method_matches_encoded_failure_against_exact_group_without_mutation(
    scheduler_cls,
):
    target = _request("target", num_computed_tokens=96)
    shared = _request("shared", num_computed_tokens=80)
    same_id_other_group = _request("same-id-other-group", num_computed_tokens=72)
    unaffected = _request("unaffected", num_computed_tokens=48)
    async_target = _request(
        "async-target",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        num_computed_tokens=32,
    )
    requests = [target, shared, same_id_other_group, unaffected, async_target]
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.running = [target, shared, same_id_other_group, unaffected]
    scheduler.skipped_waiting = [async_target]
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = _GroupBlockTable(
        {
            "target": ([100], [7, 8]),
            "shared": ([101], [7]),
            "same-id-other-group": ([7], [9]),
            "unaffected": ([102], [10]),
            "async-target": ([103], [7]),
        }
    )
    original_num_computed_tokens = {request.request_id: request.num_computed_tokens for request in requests}

    affected = scheduler._handle_invalid_blocks(
        {_encode_group_block_id(1, 7)},
        {"target": 16, "shared": 16, "same-id-other-group": 16},
    )

    assert affected == {"target", "shared", "async-target"}
    assert {request.request_id: request.num_computed_tokens for request in requests} == original_num_computed_tokens


def test_scheduler_patch_reload_keeps_single_upstream_delegate():
    original = getattr(
        Scheduler._handle_invalid_blocks,
        patch_group_block_failures._ORIGINAL_ATTR,
    )

    importlib.reload(patch_group_block_failures)

    assert Scheduler._handle_invalid_blocks is patch_group_block_failures._patched_handle_invalid_blocks
    assert (
        getattr(
            Scheduler._handle_invalid_blocks,
            patch_group_block_failures._ORIGINAL_ATTR,
        )
        is original
    )


def test_scheduler_method_delegates_nonnegative_ids_unchanged(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    invalid_block_ids = {3, 5}
    num_scheduled_tokens = {"legacy": 16}
    calls = []

    def upstream_handle_invalid_blocks(self, block_ids, scheduled_tokens):
        calls.append((self, block_ids, scheduled_tokens))
        return {"legacy"}

    monkeypatch.setattr(
        patch_group_block_failures,
        "_original_handle_invalid_blocks",
        upstream_handle_invalid_blocks,
    )

    affected = scheduler._handle_invalid_blocks(
        invalid_block_ids,
        num_scheduled_tokens,
    )

    assert affected == {"legacy"}
    assert calls == [(scheduler, invalid_block_ids, num_scheduled_tokens)]
    assert calls[0][1] is invalid_block_ids
    assert calls[0][2] is num_scheduled_tokens


def test_scheduler_method_delegates_nonnegative_subset_and_matches_encoded_subset(
    monkeypatch,
):
    target = _request("target")
    other_group = _request("other-group")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running = [target, other_group]
    scheduler.skipped_waiting = []
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = _GroupBlockTable(
        {
            "target": ([11], [21]),
            "other-group": ([21], [22]),
        }
    )
    delegated = []

    def upstream_handle_invalid_blocks(self, block_ids, scheduled_tokens):
        delegated.append((self, block_ids, scheduled_tokens))
        return {"legacy"}

    monkeypatch.setattr(
        patch_group_block_failures,
        "_original_handle_invalid_blocks",
        upstream_handle_invalid_blocks,
    )
    num_scheduled_tokens = {"target": 16, "other-group": 16}

    affected = scheduler._handle_invalid_blocks(
        {5, _encode_group_block_id(1, 21)},
        num_scheduled_tokens,
    )

    assert affected == {"legacy", "target"}
    assert delegated == [(scheduler, {5}, num_scheduled_tokens)]
