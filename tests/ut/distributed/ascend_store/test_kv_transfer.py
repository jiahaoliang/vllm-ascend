#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import json
import os
import threading
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    kv_transfer as kv_transfer_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    range_debug as range_debug_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    GroupBatchPlan,
    GroupTransferData,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LayerRangeReqMeta,
    LayerSaveTask,
    LayerTransferArrays,
    LayerTransferTask,
    LayerwisePreparation,
    LoadSpec,
    PoolKey,
    ReqMeta,
    TransferCompletion,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _mark_last_transfer_tasks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import (
    LayerTransferArrayBuilder,
    LayerwiseTransferPreparer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []
        self.copy_put_calls = []
        self.copy_get_calls = []
        self.commit_calls = []
        self.revoke_calls = []
        self.copy_put_results: list[list[int]] = []
        self.copy_get_results: list[list[int]] = []
        self.commit_results: list[list[int]] = []
        self.revoke_results: list[list[int]] = []
        self.commit_error: Exception | None = None
        self.revoke_error: Exception | None = None

    def set_device(self):
        pass

    def exists(self, keys):
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))

    def batch_copy_put(self, keys, all_buffers, all_sizes, all_offsets):
        self.copy_put_calls.append((list(keys), list(all_buffers), list(all_sizes), list(all_offsets)))
        return self.copy_put_results.pop(0) if self.copy_put_results else [0] * len(keys)

    def batch_copy_get(self, keys, all_buffers, all_sizes, all_offsets):
        self.copy_get_calls.append((list(keys), list(all_buffers), list(all_sizes), list(all_offsets)))
        return self.copy_get_results.pop(0) if self.copy_get_results else [0] * len(keys)

    def batch_commit(self, keys):
        self.commit_calls.append(list(keys))
        if self.commit_error is not None:
            raise self.commit_error
        return self.commit_results.pop(0) if self.commit_results else [0] * len(keys)

    def batch_revoke(self, keys):
        self.revoke_calls.append(list(keys))
        if self.revoke_error is not None:
            raise self.revoke_error
        return self.revoke_results.pop(0) if self.revoke_results else [0] * len(keys)


class FakeKey:
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return self._val


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.group_block_len = [[block_size, block_size]]
        self.group_kv_caches_base_addr = [[0, block_size]]
        self.group_block_stride = {0: [block_size, block_size]}

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, h in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num:
                continue
            yield start, end, PoolKey(meta, f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = block_ids[start // self.block_size]
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start], block_id

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class MaskedFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self, block_size=16, masks=([True],)):
        super().__init__(block_size)
        self.masks = masks

    def store_mask(self, token_len, num_prompt_tokens=None):
        return self.masks

    def load_mask(self, block_hashes, token_len):
        return self.masks

    def mask_allows_chunk(self, masks, kv_cache_group_id, start):
        if masks is None:
            return True
        block_idx = start // self.block_size
        return block_idx < len(masks[kv_cache_group_id]) and masks[kv_cache_group_id][block_idx]


class TestLayerTransferArrayBuilderCompactGvas(unittest.TestCase):
    def _make_builder(self, group_id=1):
        db = MagicMock()
        db.group_block_len = {0: [16], 1: [16]}
        db.group_kv_caches_base_addr = {0: [1000], 1: [2000]}
        db.group_block_stride = {0: [16], 1: [16]}
        return LayerTransferArrayBuilder(
            token_database=db,
            num_layers=1,
            group_id=group_id,
        )

    def test_layer_gvas_are_base_gvas_plus_vectorized_offsets(self):
        db = MagicMock()
        db.group_block_len = {0: [4, 6, 4, 6]}
        db.group_kv_caches_base_addr = {0: [100, 200, 300, 400]}
        db.group_block_stride = {0: [10, 10, 10, 10]}
        builder = LayerTransferArrayBuilder(
            token_database=db,
            num_layers=2,
        )

        addrs, sizes, gvas = builder._build_transfer_arrays(
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([1000, 2000], dtype=np.int64),
            layer_id=1,
        )

        np.testing.assert_array_equal(addrs, [320, 420, 330, 430])
        np.testing.assert_array_equal(sizes, [4, 6, 4, 6])
        np.testing.assert_array_equal(gvas, [1010, 1014, 2010, 2014])

    def test_variable_cache_legs_use_compact_layer_offsets(self):
        db = MagicMock()
        # Layer 0 has main K/V + indexer K; layer 1 has only main K/V.
        db.group_block_len = {0: [4, 6, 2, 4, 6]}
        db.group_kv_caches_base_addr = {0: [100, 200, 250, 300, 400]}
        db.group_block_stride = {0: [10, 10, 10, 10, 10]}
        db.group_layer_offsets = {0: [0, 3, 5]}
        builder = LayerTransferArrayBuilder(
            token_database=db,
            num_layers=2,
        )

        block_ids = np.asarray([2], dtype=np.int64)
        base_gvas = np.asarray([1000], dtype=np.int64)
        layer0 = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=0)
        layer1 = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=1)

        np.testing.assert_array_equal(layer0[0], [120, 220, 270])
        np.testing.assert_array_equal(layer0[1], [4, 6, 2])
        np.testing.assert_array_equal(layer0[2], [1000, 1004, 1010])
        np.testing.assert_array_equal(layer1[0], [320, 420])
        np.testing.assert_array_equal(layer1[1], [4, 6])
        np.testing.assert_array_equal(layer1[2], [1012, 1016])

    def test_build_addrs_only_consumes_block_ids_and_base_gvas(self):
        data = GroupTransferData(
            block_ids_arr=np.asarray([10, 11], dtype=np.int64),
            base_gvas_arr=np.asarray([800, 900], dtype=np.int64),
        )

        arrays = self._make_builder().build_addrs(data, layer_id=0)

        np.testing.assert_array_equal(arrays.addr_array, [2160, 2176])
        np.testing.assert_array_equal(arrays.gvas_array, [800, 900])


class TestLayerwiseTransferPreparer(unittest.TestCase):
    @staticmethod
    def _make_preparer():
        preparer = LayerwiseTransferPreparer(
            MagicMock(),
            model_name="model",
            head_or_tp_rank=0,
            hash_block_size=16,
            enabled=True,
            can_allocate=True,
            num_groups=1,
        )
        preparer.configure_layout(
            group_block_len={0: [16]},
        )
        return preparer

    def test_prepares_once_and_attaches_to_save_tasks(self):
        plans = []
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        layer_tasks = [[task]]
        transfer_data = MagicMock()
        completion = MagicMock()
        preparer = self._make_preparer()
        prepare_tasks = MagicMock()
        with patch.object(
            preparer,
            "resolve_save_groups",
            return_value={0: (transfer_data, completion)},
        ) as resolve:
            preparation = preparer.create_save_preparation(
                plans,
                layer_tasks,
                prepare_tasks,
            )

            self.assertIs(task.preparation, preparation)
            resolve.assert_not_called()
            prepare_tasks.assert_not_called()

            preparation.ensure_ready()
            preparation.ensure_ready()

            resolve.assert_called_once()
            self.assertIs(task.transfer_data, transfer_data)
            self.assertIs(task.completion, completion)
            prepare_tasks.assert_called_once_with(layer_tasks)

    def test_save_finishes_on_last_actual_transfer_task(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        transfer_data = MagicMock()
        completion = TransferCompletion(["r1"], [True])
        preparer = self._make_preparer()
        with patch.object(
            preparer,
            "resolve_save_groups",
            return_value={0: (transfer_data, completion)},
        ):
            preparation = preparer.create_save_preparation(
                [],
                [[task], []],
                lambda tasks: _mark_last_transfer_tasks(tasks, "save"),
            )
            preparation.ensure_ready()

        self.assertEqual(task.finished_req_ids, {"r1"})

    def test_load_preparation_does_not_mutate_transfer_tasks(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        preparation = self._make_preparer().create_load_preparation(
            [],
            [[task]],
        )

        self.assertIsNone(task.preparation)
        self.assertIsNotNone(preparation)

    def test_load_finishes_on_last_actual_transfer_task(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        transfer_data = MagicMock()
        completion = TransferCompletion(["r1"], [True])
        preparer = self._make_preparer()
        with patch.object(
            preparer,
            "resolve_load_groups",
            return_value={(0, False): (transfer_data, completion)},
        ):
            preparation = preparer.create_load_preparation(
                [],
                [[task], []],
                lambda tasks: _mark_last_transfer_tasks(tasks, "load"),
            )
            preparation.ensure_ready()

        self.assertEqual(task.finished_req_ids, {"r1"})

    @staticmethod
    def _make_load_plan(block_hashes):
        requests = [
            ReqMeta(
                req_id=f"request-{index}",
                token_len_chunk=16,
                block_ids_by_group=[[index + 1]],
                block_hashes=[block_hash],
                is_last_chunk=True,
            )
            for index, block_hash in enumerate(block_hashes)
        ]
        return GroupBatchPlan(
            group_id=0,
            block_size=16,
            full_load_ranges=[LayerBlockRange(request=request, start_block=0, end_block=1) for request in requests],
        )

    def test_rejects_invalid_gva_before_acquiring_lease(self):
        preparer = self._make_preparer()
        key_info = MagicMock()
        key_info.size.return_value = 1
        key_info.gva_list.return_value = []
        preparer.m_store.batch_get_key_info.return_value = [key_info]

        with self.assertRaisesRegex(RuntimeError, "invalid GVA metadata"):
            preparer.resolve_load_groups([self._make_load_plan(["aa"])])

        preparer.m_store.batch_add_lease.assert_not_called()
        self.assertEqual(preparer.load_lease_keys_by_request, {})

    def test_rolls_back_successful_lease_after_partial_failure(self):
        preparer = self._make_preparer()
        key_infos = []
        for gva in (1000, 2000):
            key_info = MagicMock()
            key_info.size.return_value = 1
            key_info.gva_list.return_value = [gva]
            key_infos.append(key_info)
        preparer.m_store.batch_get_key_info.return_value = key_infos
        preparer.m_store.batch_add_lease.return_value = [0, -3102]
        preparer.m_store.batch_remove_lease.return_value = 0

        with self.assertRaisesRegex(RuntimeError, "lease acquisition failed"):
            preparer.resolve_load_groups([self._make_load_plan(["aa", "bb"])])

        first_key = preparer.make_gva_key(0, "aa")
        preparer.m_store.batch_remove_lease.assert_called_once_with([first_key])
        self.assertEqual(preparer.load_lease_keys_by_request, {})


class TestLayerwiseTaskPreparation(unittest.TestCase):
    def test_key_send_reuses_cached_process_tokens(self):
        thread = object.__new__(KVCacheStoreKeyLayerSendingThread)
        cached = {0: [(0, 16, [MagicMock()])]}
        thread.build_cached_process_tokens = MagicMock(return_value=cached)
        tasks = [
            [LayerTransferTask(layer_id=0, block_ranges=[])],
            [LayerTransferTask(layer_id=1, block_ranges=[])],
        ]

        thread.prepare_layerwise_tasks(tasks)

        self.assertIs(tasks[0][0].cached_process_tokens, cached)
        self.assertIs(tasks[1][0].cached_process_tokens, cached)
        thread.build_cached_process_tokens.assert_called_once_with(tasks[0][0])


class RangeBatchFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self):
        super().__init__()
        self.group_block_len = [[64, 32, 64, 32, 64, 32]]
        self.group_kv_caches_base_addr = [[100, 200, 500, 600, 1000, 2000]]
        self.group_block_stride = {0: [100, 50, 100, 50, 100, 50]}


class TestLayerBatchBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = LayerBatchBuilder(
            RangeBatchFakeTokenDatabase(),
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=96,
            num_layers=3,
        )

    def _build(self, block_ids, block_keys):
        request = ReqMeta(
            req_id="r1",
            block_ids=block_ids,
            save_block_keys=block_keys,
        )
        task = LayerTransferTask(
            layer_id=2,
            block_ranges=[LayerBlockRange(request, 0, len(block_ids))],
            use_key_major_ranges=True,
        )
        return self.builder.build(task)

    def test_builds_memcache_shared_data_without_block_keys(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            block_ids_np=np.asarray([1], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 1)],
        )

        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        self.assertIsNone(shared.block_keys)
        np.testing.assert_array_equal(shared.block_ids_arr, [1])
        np.testing.assert_array_equal(shared.block_gvas_arr, [0x1000])

    def test_builds_key_major_range_batch_for_two_blocks_and_two_segments(self):
        batch = self._build([1, 2], ["key-1", "key-2"])

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertIsInstance(batch, LayerRangeReqMeta)
        self.assertEqual(batch.req_ids, ["r1"])
        self.assertEqual(batch.layer_id, 2)
        self.assertEqual(batch.block_ids, [1, 2])
        self.assertEqual(batch.keys, ["key-1", "key-2"])
        self.assertEqual(batch.all_buffers, [[1100, 2050], [1200, 2100]])
        self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])
        self.assertEqual(batch.all_offsets, [[192, 256], [192, 256]])

    def test_reuses_key_major_shared_data_across_layers(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1, 2],
            save_block_keys=["key-1", "key-2"],
        )
        task = LayerTransferTask(
            layer_id=2,
            block_ranges=[LayerBlockRange(request, 0, 2)],
            use_key_major_ranges=True,
        )
        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        layer_0_batch = self.builder.build_addrs(shared, 0)
        layer_2_batch = self.builder.build_addrs(shared, 2)

        for batch in (layer_0_batch, layer_2_batch):
            self.assertIsInstance(batch, LayerRangeReqMeta)
            self.assertEqual(batch.block_ids, [1, 2])
            self.assertEqual(batch.keys, ["key-1", "key-2"])
            self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])

        self.assertEqual(layer_0_batch.all_buffers, [[200, 250], [300, 300]])
        self.assertEqual(layer_0_batch.all_offsets, [[0, 64], [0, 64]])
        self.assertEqual(layer_2_batch.all_buffers, [[1100, 2050], [1200, 2100]])
        self.assertEqual(layer_2_batch.all_offsets, [[192, 256], [192, 256]])

    def test_partial_only_failed_session_stays_an_empty_key_major_batch(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            save_last_block_key=None,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 0, partial_block_index=0)],
            use_key_major_ranges=True,
        )

        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        self.assertEqual(shared.block_keys, [])
        np.testing.assert_array_equal(shared.block_ids_arr, [])
        batch = self.builder.build_addrs(shared, 0)
        self.assertIsInstance(batch, LayerRangeReqMeta)
        self.assertEqual(batch.keys, [])
        self.assertEqual(batch.all_buffers, [])

    def test_filters_none_key_with_its_aligned_block_and_range_values(self):
        batch = self._build([1, 2, 3], ["key-1", None, "key-3"])

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.block_ids, [1, 3])
        self.assertEqual(batch.keys, ["key-1", "key-3"])
        self.assertEqual(batch.all_buffers, [[1100, 2050], [1300, 2150]])
        self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])
        self.assertEqual(batch.all_offsets, [[192, 256], [192, 256]])


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        return t, store

    def test_add_request(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

    def test_get_and_clear_finished_requests(self):
        t, _ = self._make_thread()
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        finished = t.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup_all_exist(self):
        t, _ = self._make_thread([1, 1, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, True, True])

    def test_lookup_partial(self):
        t, _ = self._make_thread([1, 0, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_exception(self):
        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        result = t.lookup(["k1"])
        self.assertEqual(result, [False])

    def test_update_and_get_kv_events(self):
        t, _ = self._make_thread()
        event1 = BlockStored(
            block_hashes=["h1"],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["h2"],
            parent_block_hash="h1",
            token_ids=[4, 5, 6],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        t.update_kv_event([event1, event2])
        events = t.get_kv_events()
        self.assertEqual(len(events), 2)
        # After get, events should be cleared
        self.assertEqual(len(t.get_kv_events()), 0)

    def test_handle_request_base_noop(self):
        t, _ = self._make_thread()
        # Base class _handle_request does nothing
        t._handle_request(MagicMock())


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)

    def test_handle_request_all_exist_no_put(self):
        t, store = self._make_thread([1, 1])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_in_stored(self):
        t, store = self._make_thread([0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_consumer_role(self):
        t, store = self._make_thread([0], kv_role="kv_consumer")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_dec_nonexistent_request(self):
        t, _ = self._make_thread()
        t.dec_stored_request("nonexist")  # should not raise

    def test_delete_nonexistent_request(self):
        t, _ = self._make_thread()
        t.delete_finished_stored_request("nonexist")  # should not raise

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=event,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore([0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)

    def test_handle_request_applies_store_mask(self):
        store = FakeStore([0, 0])
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_applies_load_mask(self):
        store = FakeStore()
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.get_calls[0]
        self.assertEqual(len(keys), 1)


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class _DeprecatedKVCacheStoreLayerSendingThreadTests(unittest.TestCase):
    __test__ = False

    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[],
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
            token_ids=list(range(num_keys * 16)),
            original_block_size=16,
            block_hashes=[f"h{i}".encode() for i in range(num_keys)],
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        _meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=event,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_layerwise_kv_event_published_on_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])
        self.assertEqual(events[0].token_ids, list(range(16)))
        self.assertEqual(events[0].block_size, 16)

    def test_layerwise_kv_event_not_published_before_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=0, is_last_chunk=False, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.get_kv_events(), [])

    def test_layerwise_kv_event_uses_missing_blocks_from_previous_layers(self):
        t, store = self._make_thread([0], num_layers=2)
        first_layer_req = self._make_layer_req(layer_id=0, is_last_chunk=True, num_keys=1)
        t.add_stored_request(first_layer_req.req_id)
        t.request_queue.put(first_layer_req)
        t._handle_request(first_layer_req)
        t.m_store.exists_result = [1]
        final_layer_req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(final_layer_req)
        t._handle_request(final_layer_req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])


class TestGVALayerSendingThread(unittest.TestCase):
    def _make_thread(self, copy_result=0, builders=None, pd_transfer_waiter=None):
        store = MagicMock()
        store.store.batch_copy.return_value = copy_result
        db = MagicMock()
        db.group_block_len = {0: [16]}
        layer_finished = threading.Event()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[layer_finished],
            sync_save_events=[MagicMock()],
            group_array_builders=builders,
            pd_transfer_waiter=pd_transfer_waiter,
        )
        return thread, store, layer_finished

    def test_merges_group_data_and_completes_after_success(self):
        builders = [MagicMock(), MagicMock()]
        builders[0].build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        builders[1].build_addrs.return_value = LayerTransferArrays(
            np.asarray([20]),
            np.asarray([16]),
            np.asarray([200]),
        )
        thread, store, layer_finished = self._make_thread(builders=builders)
        store.batch_write_finish.return_value = [0, 0]
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=GroupTransferData(
                    block_ids_arr=np.asarray([group_id]),
                    base_gvas_arr=np.asarray([100 + group_id]),
                    keys=[f"k{group_id}"],
                ),
                completion=TransferCompletion(["r1"], [True]),
                group_id=group_id,
            )
            for group_id in range(2)
        ]
        thread.prepare_layerwise_tasks([tasks])
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        thread._handle_request(request)

        store.store.batch_copy.assert_called_once_with([100, 200], [10, 20], [16, 16], 0)
        store.batch_write_finish.assert_called_once_with(["k0", "k1"], [0, 0])
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})
        self.assertTrue(layer_finished.is_set())

    def test_write_finish_failure_does_not_complete_request_or_layer(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        thread, store, layer_finished = self._make_thread(builders=[builder])
        store.batch_write_finish.return_value = [-3102]
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=GroupTransferData(
                block_ids_arr=np.asarray([0]),
                base_gvas_arr=np.asarray([100]),
                keys=["k0"],
            ),
            completion=TransferCompletion(["r1"], [True]),
        )
        thread.prepare_layerwise_tasks([[task]])
        thread.add_stored_request("r1")
        request = LayerSaveTask(layer_id=0, transfer_tasks=[task])
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "batch_write_finish failed"):
            thread._handle_request(request)

        self.assertEqual(thread.stored_requests["r1"], 1)
        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(layer_finished.is_set())

    def test_missing_group_data_is_fatal(self):
        thread, _, layer_finished = self._make_thread(builders=[MagicMock()])
        tasks = [LayerTransferTask(layer_id=0, block_ranges=[], group_id=0)]
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "save metadata was not prepared"):
            thread._handle_request(request)

        self.assertFalse(layer_finished.is_set())

    def test_copy_failure_does_not_complete_request_or_layer(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        thread, _, layer_finished = self._make_thread(copy_result=1, builders=[builder])
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=MagicMock(),
                completion=TransferCompletion(["r1"], [True]),
            )
        ]
        thread.add_stored_request("r1")
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "save batch_copy failed"):
            thread._handle_request(request)

        self.assertEqual(thread.stored_requests["r1"], 1)
        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(layer_finished.is_set())

    def test_pd_read_finishes_before_layer_save_completion(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        call_order = []

        def wait_for_pd(layer_id):
            call_order.append(("pd", layer_id))

        thread, store, layer_finished = self._make_thread(
            builders=[builder],
            pd_transfer_waiter=wait_for_pd,
        )
        store.store.batch_copy.side_effect = lambda *_args: call_order.append(("save", 0)) or 0
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=MagicMock(),
                completion=TransferCompletion([], []),
            )
        ]
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        thread._handle_request(request)

        self.assertEqual(call_order, [("save", 0), ("pd", 0)])
        self.assertTrue(layer_finished.is_set())

    def test_pd_read_failure_does_not_release_layer_save_gate(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )

        def fail_pd_read(_layer_id):
            raise RuntimeError("PD read failed")

        thread, _, layer_finished = self._make_thread(
            builders=[builder],
            pd_transfer_waiter=fail_pd_read,
        )
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=MagicMock(),
                completion=TransferCompletion([], []),
            )
        ]
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "PD read failed"):
            thread._handle_request(request)

        self.assertFalse(layer_finished.is_set())


class TestGVALayerSendingThreadEventSplit(unittest.TestCase):
    """PD and attention completion gate physical slot reuse."""

    def _make_thread(self, copy_result=0, builders=None, pd_transfer_waiter=None, attn_recorded=True):
        store = MagicMock()
        store.store.batch_copy.return_value = copy_result
        db = MagicMock()
        db.group_block_len = {0: [16]}
        layer_finished = threading.Event()
        attn_flag = threading.Event()
        if attn_recorded:
            attn_flag.set()
        sync_attn = MagicMock()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[layer_finished],
            sync_save_events=[MagicMock()],
            group_array_builders=builders,
            pd_transfer_waiter=pd_transfer_waiter,
            sync_attn_events=[sync_attn],
            layer_attn_recorded_events=[attn_flag],
        )
        return thread, store, layer_finished, sync_attn

    def _make_builder(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        return builder

    def _make_task(self):
        return LayerSaveTask(
            layer_id=0,
            transfer_tasks=[
                LayerTransferTask(
                    layer_id=0,
                    block_ranges=[],
                    transfer_data=MagicMock(),
                    completion=TransferCompletion([], []),
                )
            ],
        )

    def test_copy_runs_before_pd_and_slot_free(self):
        call_order = []

        def wait_for_pd(_layer_id):
            call_order.append("pd")

        thread, store, layer_finished, _ = self._make_thread(
            builders=[self._make_builder()],
            pd_transfer_waiter=wait_for_pd,
        )
        store.store.batch_copy.side_effect = lambda *_a: call_order.append("copy") or 0
        request = self._make_task()
        thread.request_queue.put(request)

        thread._handle_request(request)

        self.assertEqual(call_order, ["copy", "pd"])
        self.assertTrue(layer_finished.is_set())

    def test_slot_free_waits_for_attention_done(self):
        # Attention not yet recorded: the send thread must block on the
        # threading flag, so spin it up and release the flag shortly after.
        thread, store, layer_finished, sync_attn = self._make_thread(
            builders=[self._make_builder()],
            attn_recorded=False,
        )
        attn_flag = thread.layer_attn_recorded_events[0]
        request = self._make_task()
        thread.request_queue.put(request)

        done = threading.Event()

        def run():
            thread._handle_request(request)
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        # Give the worker a moment to reach the attention-done wait.
        self.assertFalse(done.wait(timeout=0.2))
        store.store.batch_copy.assert_called_once()
        self.assertFalse(layer_finished.is_set())  # slot_free still gated on (c)
        attn_flag.set()
        self.assertTrue(done.wait(timeout=5))
        worker.join(timeout=5)
        sync_attn.synchronize.assert_called_once()
        self.assertTrue(layer_finished.is_set())

    def test_control_only_save_waits_for_pd_and_attention_before_slot_free(self):
        call_order = []

        def wait_for_pd(_layer_id):
            call_order.append("pd")

        thread, store, layer_finished, sync_attn = self._make_thread(
            builders=[self._make_builder()],
            pd_transfer_waiter=wait_for_pd,
            attn_recorded=False,
        )
        request = LayerSaveTask(layer_id=0, transfer_tasks=[])
        thread.request_queue.put(request)
        done = threading.Event()

        def run():
            thread._handle_request(request)
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        self.assertFalse(done.wait(timeout=0.2))
        self.assertEqual(call_order, ["pd"])
        self.assertFalse(layer_finished.is_set())
        thread.layer_attn_recorded_events[0].set()
        self.assertTrue(done.wait(timeout=5))
        worker.join(timeout=5)
        sync_attn.synchronize.assert_called_once()
        self.assertTrue(layer_finished.is_set())
        store.store.batch_copy.assert_not_called()


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class _DeprecatedKVCacheStoreLayerRecvingThreadTests(unittest.TestCase):
    __test__ = False

    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


class TestGVALayerRecvingThread(unittest.TestCase):
    def test_h2d_stagger_sleeps_before_short_final_spin(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._get_h2d_stagger_delay_us = MagicMock(return_value=100)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.perf_counter_ns",
                side_effect=[0, 100_000],
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.sleep") as sleep,
        ):
            thread._stagger_h2d_submit(layer_id=0)

        sleep.assert_called_once_with(50 / 1_000_000)

    def test_short_h2d_stagger_does_not_sleep(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._get_h2d_stagger_delay_us = MagicMock(return_value=25)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.perf_counter_ns",
                side_effect=[0, 25_000],
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.sleep") as sleep,
        ):
            thread._stagger_h2d_submit(layer_id=0)

        sleep.assert_not_called()

    def test_layer_transfer_releases_load_leases_after_copy(self):
        store = MagicMock()
        store.batch_add_lease.return_value = [0, 0]
        store.store.batch_copy.return_value = 0
        load_lease_releaser = MagicMock()
        db = MagicMock()
        db.group_block_len = {0: [16]}
        db.group_kv_caches_base_addr = {0: [1000]}
        db.group_block_stride = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            addr_array=np.asarray([1000], dtype=np.int64),
            size_array=np.asarray([16], dtype=np.int64),
            gvas_array=np.asarray([2000], dtype=np.int64),
        )
        finished_events = [threading.Event()]
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=finished_events,
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            group_array_builders=[builder],
            load_lease_releaser=load_lease_releaser,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        preparation_callback = MagicMock()
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=0,
            preparation=LayerwisePreparation(preparation_callback),
        )

        thread.request_queue.put(load_task)
        thread._handle_request(load_task)

        preparation_callback.assert_called_once_with()
        store.batch_add_lease.assert_not_called()
        store.batch_remove_lease.assert_not_called()
        store.store.batch_copy.assert_called_once()
        load_lease_releaser.assert_called_once_with({"r1"})
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})

    def test_last_actual_transfer_can_finish_before_final_physical_layer(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 0
        db = MagicMock()
        db.group_block_len = {0: [16, 16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([1000]),
            np.asarray([16]),
            np.asarray([2000]),
        )
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=[threading.Event(), threading.Event()],
            layer_save_finished_events=[threading.Event(), threading.Event()],
            num_layers=2,
            group_array_builders=[builder],
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(None, [task], layer_id=0)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})

    def test_copy_failure_does_not_complete_request_or_layer(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 1
        db = MagicMock()
        db.group_block_len = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([1000]),
            np.asarray([16]),
            np.asarray([2000]),
        )
        layer_finished = threading.Event()
        load_lease_releaser = MagicMock()
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=[layer_finished],
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            group_array_builders=[builder],
            load_lease_releaser=load_lease_releaser,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(None, [task], layer_id=0)
        thread.request_queue.put(load_task)

        with self.assertRaisesRegex(RuntimeError, "load batch_copy failed"):
            thread._handle_request(load_task)

        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(layer_finished.is_set())
        load_lease_releaser.assert_not_called()


class TestKVCacheStoreLayerFinalization(unittest.TestCase):
    @staticmethod
    def _make_send_thread():
        store = MagicMock()
        db = MagicMock()
        db.group_block_len = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.side_effect = RuntimeError("probe failed")
        layer_finished = threading.Event()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[layer_finished],
            sync_save_events=[MagicMock()],
            group_array_builders=[builder],
        )
        request = ReqMeta(req_id="r1", block_ids=[1])
        task = LayerTransferTask(
            0,
            [LayerBlockRange(request, 0, 1)],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
        )
        layer_request = LayerSaveTask(layer_id=0, transfer_tasks=[task])
        return thread, layer_request, layer_finished

    @staticmethod
    def _make_recv_thread():
        store = MagicMock()
        db = MagicMock()
        db.group_block_len = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.side_effect = RuntimeError("probe failed")
        invalid_block_ids: set[int] = set()
        get_event = threading.Event()
        layer_finished = threading.Event()
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            layer_load_finished_events=[layer_finished],
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            group_array_builders=[builder],
            invalid_block_ids=invalid_block_ids,
            invalid_block_ids_lock=threading.Lock(),
        )
        request = ReqMeta(req_id="r1", block_ids=[3])
        task = LayerTransferTask(
            0,
            [LayerBlockRange(request, 0, 1)],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
        )
        data = LayerLoadTask(None, [task], 0)
        return thread, data, invalid_block_ids, get_event, layer_finished

    def test_save_exception_finishes_queue_without_completing_request_or_layer(self):
        thread, request, layer_finished = self._make_send_thread()
        thread.add_stored_request("r1")
        thread.request_queue.put(request)

        with (
            patch.object(thread.request_queue, "task_done", wraps=thread.request_queue.task_done) as task_done,
            self.assertRaisesRegex(RuntimeError, "probe failed"),
        ):
            thread._handle_request(request)

        self.assertEqual(task_done.call_count, 1)
        self.assertFalse(layer_finished.is_set())
        self.assertEqual(thread.stored_requests["r1"], 1)
        self.assertEqual(thread.get_and_clear_finished_requests(), set())

    def test_load_exception_finishes_queue_and_marks_blocks_invalid(self):
        thread, data, invalid_block_ids, get_event, layer_finished = self._make_recv_thread()
        thread.request_queue.put(data)

        with (
            patch.object(thread.request_queue, "task_done", wraps=thread.request_queue.task_done) as task_done,
            self.assertRaisesRegex(RuntimeError, "probe failed"),
        ):
            thread._handle_request(data)

        self.assertEqual(task_done.call_count, 1)
        self.assertFalse(layer_finished.is_set())
        self.assertFalse(get_event.is_set())
        self.assertEqual(invalid_block_ids, {3})


class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    @staticmethod
    def _make_thread(num_layers=2):
        store = FakeStore()
        started_keys = {"key-1", "key-2"}
        token_database = RangeBatchFakeTokenDatabase()
        range_builder = LayerBatchBuilder(
            token_database,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=96,
            num_layers=num_layers,
        )
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=token_database,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[MagicMock() for _ in range(num_layers)],
            group_builders=[range_builder],
            put_started_keys=started_keys,
            put_started_keys_lock=threading.Lock(),
        )
        thread._put_revoke_pending_keys = set()
        thread._put_revoke_queued_keys = set()
        thread._put_revoke_inflight_keys = set()
        return thread, store, started_keys

    @staticmethod
    def _make_task(thread, layer_id):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1, 2],
            save_block_keys=["key-1", "key-2"],
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id,
            [LayerBlockRange(request, 0, 2)],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)
        return [task]

    @staticmethod
    def _run_task(thread, tasks):
        for block_range in tasks[0].block_ranges:
            thread.add_stored_request(block_range.request.req_id)
        request = LayerSaveTask(layer_id=tasks[0].layer_id, transfer_tasks=tasks)
        thread.request_queue.put(request)
        thread._handle_request(request)

    def test_positive_results_keep_active_keys_and_commit_on_final_layer(self):
        thread, store, started_keys = self._make_thread()
        store.copy_put_results = [[96, -1], [96]]
        store.revoke_results = [[0]]

        for layer_id in range(2):
            self._run_task(thread, self._make_task(thread, layer_id))

        self.assertEqual(store.copy_put_calls[0][0], ["key-1", "key-2"])
        self.assertEqual(store.copy_put_calls[1][0], ["key-1"])
        self.assertEqual(
            store.copy_put_calls[1],
            (
                ["key-1"],
                [[650, 1100, 2050]],
                [[32, 64, 32]],
                [[96, 128, 192]],
            ),
        )
        self.assertEqual(store.revoke_calls, [["key-2"]])
        self.assertEqual(store.commit_calls, [["key-1"]])
        self.assertEqual(started_keys, set())

    def test_range_debug_records_physical_save_layers_and_final_commit(self):
        thread, store, _ = self._make_thread()
        store.copy_put_results = [[160, 160], [128, 128]]

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug_module.logger, "info") as log_info,
        ):
            for layer_id in range(2):
                self._run_task(thread, self._make_task(thread, layer_id))

        events = [
            json.loads(call.args[2])
            for call in log_info.call_args_list
            if call.args[:2] == ("%s %s", "[KVPOOL_RANGE_DEBUG]")
        ]
        self.assertEqual([event["event"] for event in events], ["range", "range", "commit"])
        self.assertEqual([event["layer_id"] for event in events], [0, 1, 1])
        for event, expected_sizes, expected_bytes in zip(
            events[:2],
            ([[64, 32, 64], [32, 64, 32]]),
            (160, 128),
            strict=True,
        ):
            self.assertEqual(event["direction"], "save")
            self.assertEqual(event["key_count"], 2)
            self.assertEqual(event["requested_bytes"], [expected_bytes] * 2)
            self.assertEqual(event["sizes"], [expected_sizes] * 2)
            self.assertEqual(event["results"], [expected_bytes] * 2)
            self.assertNotIn("keys", event)
        self.assertEqual(events[-1]["results"], [0, 0])

    def test_range_debug_disabled_skips_payload_builder(self):
        thread, store, _ = self._make_thread(num_layers=1)

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "0"}),
            patch.object(
                range_debug_module,
                "_build_range_payload",
                side_effect=AssertionError("payload builder must not run"),
            ) as build_payload,
        ):
            self._run_task(thread, self._make_task(thread, 0))

        build_payload.assert_not_called()
        self.assertEqual(store.commit_calls, [["key-1", "key-2"]])

    def test_range_debug_serialization_failure_preserves_save_cleanup(self):
        thread, store, started_keys = self._make_thread(num_layers=1)
        store.copy_put_results = [[288, -1]]
        store.revoke_results = [[0]]

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(
                range_debug_module.json,
                "dumps",
                side_effect=RuntimeError("serialize failed"),
            ),
        ):
            self._run_task(thread, self._make_task(thread, 0))

        self.assertEqual(store.revoke_calls, [["key-2"]])
        self.assertEqual(store.commit_calls, [["key-1"]])
        self.assertEqual(started_keys, set())

    def test_malformed_results_revoke_all_keys_and_finish_request(self):
        for results in ([96], [96, 96, 96], ["invalid", 96]):
            with self.subTest(results=results):
                thread, store, started_keys = self._make_thread()
                store.copy_put_results = [results]
                store.revoke_results = [[0, 0]]
                self._run_task(thread, self._make_task(thread, 0))

                self.assertEqual(store.revoke_calls, [["key-1", "key-2"]])
                self.assertEqual(started_keys, set())
                self.assertEqual(dict(thread.stored_requests), {})
                self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})
                self.assertEqual(thread.get_kv_events(), [])

    def test_first_layer_exception_revokes_shared_keys_and_stops_later_layers(self):
        for failure in ("builder", "sync", "backend"):
            with self.subTest(failure=failure):
                thread, store, started_keys = self._make_thread()
                first_layer_tasks = self._make_task(thread, 0)

                if failure == "builder":
                    with patch.object(
                        thread.layer_batch_builder,
                        "build_addrs",
                        side_effect=RuntimeError("metadata failed"),
                    ):
                        self._run_task(thread, first_layer_tasks)
                elif failure == "sync":
                    thread.sync_save_events[0].synchronize.side_effect = RuntimeError("sync failed")
                    self._run_task(thread, first_layer_tasks)
                    thread.sync_save_events[0].synchronize.side_effect = None
                else:
                    with patch.object(
                        store,
                        "batch_copy_put",
                        side_effect=RuntimeError("transfer failed"),
                    ):
                        self._run_task(thread, first_layer_tasks)

                self.assertEqual(store.revoke_calls, [["key-1", "key-2"]])
                self.assertEqual(started_keys, set())
                self.assertTrue(thread.layer_save_finished_events[0].is_set())
                self.assertEqual(thread.request_queue.unfinished_tasks, 0)

                self._run_task(thread, self._make_task(thread, 1))

                self.assertEqual(store.copy_put_calls, [])
                self.assertEqual(store.commit_calls, [])
                self.assertEqual(store.revoke_calls, [["key-1", "key-2"]])
                self.assertTrue(thread.layer_save_finished_events[1].is_set())
                self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_empty_key_major_batch_finishes_without_copy_or_commit(self):
        thread, store, started_keys = self._make_thread()
        started_keys.clear()
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            save_last_block_key=None,
            is_last_chunk=True,
        )

        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id=layer_id,
                block_ranges=[
                    LayerBlockRange(
                        request,
                        0,
                        0,
                        partial_block_index=0,
                    )
                ],
                use_key_major_ranges=True,
            )
            task.shared_block_data = thread.build_shared_data(task)
            self._run_task(thread, [task])

        self.assertEqual(store.copy_put_calls, [])
        self.assertEqual(store.commit_calls, [])
        self.assertEqual(store.revoke_calls, [])
        self.assertEqual(started_keys, set())
        self.assertTrue(all(event.is_set() for event in thread.layer_save_finished_events))
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_commit_failure_revokes_only_failed_key(self):
        thread, store, started_keys = self._make_thread(num_layers=1)
        store.commit_results = [[0, -1]]
        store.revoke_results = [[0]]

        self._run_task(thread, self._make_task(thread, 0))

        self.assertEqual(store.commit_calls, [["key-1", "key-2"]])
        self.assertEqual(store.revoke_calls, [["key-2"]])
        self.assertEqual(started_keys, set())
        self.assertEqual(thread._put_revoke_pending_keys, set())

    def test_commit_promotes_only_successful_keys_to_future_loads(self):
        thread, store, _ = self._make_thread(num_layers=1)
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("key-1", 0), ("key-2", 1)])
        thread._session_tracker = tracker
        store.commit_results = [[0, -1]]
        store.revoke_results = [[0]]

        self._run_task(thread, self._make_task(thread, 0))

        self.assertEqual(
            tracker.prepare_load_entries("r1", []),
            [("key-1", 0)],
        )

    def test_commit_error_results_revoke_all_active_keys_and_clear_tracker(self):
        cases = (
            ("raises", None),
            ("too_short", [0]),
            ("too_long", [0, 0, 0]),
            ("non_integer", ["invalid", 0]),
        )
        for name, results in cases:
            with self.subTest(name=name):
                thread, store, started_keys = self._make_thread(num_layers=1)
                store.revoke_results = [[0, 0]]
                if results is None:
                    store.commit_error = RuntimeError("commit failed")
                else:
                    store.commit_results = [results]

                self._run_task(thread, self._make_task(thread, 0))

                self.assertEqual(store.commit_calls, [["key-1", "key-2"]])
                self.assertEqual(store.revoke_calls, [["key-1", "key-2"]])
                self.assertEqual(started_keys, set())
                self.assertTrue(thread.layer_save_finished_events[0].is_set())

    def test_revoke_error_results_retain_pending_key_and_finish_layer(self):
        cases = (
            ("raises", None),
            ("too_short", []),
            ("too_long", [0, 0]),
            ("non_integer", ["invalid"]),
        )
        for name, results in cases:
            with self.subTest(name=name):
                thread, store, started_keys = self._make_thread()
                store.copy_put_results = [[96, -1]]
                if results is None:
                    store.revoke_error = RuntimeError("revoke failed")
                else:
                    store.revoke_results = [results, results, results]

                self._run_task(thread, self._make_task(thread, 0))

                self.assertEqual(store.revoke_calls, [["key-2"]] * 3)
                self.assertEqual(started_keys, {"key-1"})
                self.assertEqual(thread._put_revoke_pending_keys, {"key-2"})
                self.assertTrue(thread.layer_save_finished_events[0].is_set())
                self.assertEqual(dict(thread.stored_requests), {})
                self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})

    def test_revoke_retries_only_failed_keys_and_releases_successes(self):
        thread, store, started_keys = self._make_thread()
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("key-1", 0), ("key-2", 1)])
        thread._session_tracker = tracker
        store.revoke_results = [[0, -600], [-600], [-600]]

        with patch.object(kv_transfer_module.time, "sleep") as sleep:
            thread._revoke_range_keys(["key-1", "key-2"])

        self.assertEqual(store.revoke_calls, [["key-1", "key-2"], ["key-2"], ["key-2"]])
        self.assertEqual(sleep.call_args_list, [call(0.1), call(0.5)])
        self.assertEqual(started_keys, set())
        self.assertEqual(thread._put_revoke_pending_keys, {"key-2"})
        tracker.commit_put_keys(["key-1", "key-2"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("key-2", 1)])

    def test_revoke_nonzero_then_success_clears_pending_ownership(self):
        thread, store, started_keys = self._make_thread()
        store.revoke_results = [[-600], [0]]

        with patch.object(kv_transfer_module.time, "sleep") as sleep:
            thread._revoke_range_keys(["key-1"])

        self.assertEqual(store.revoke_calls, [["key-1"], ["key-1"]])
        sleep.assert_called_once_with(0.1)
        self.assertEqual(started_keys, {"key-2"})
        self.assertEqual(thread._put_revoke_pending_keys, set())

    def test_pending_key_is_not_written_by_ranged_save(self):
        thread, store, started_keys = self._make_thread(num_layers=1)
        started_keys.remove("key-2")
        thread._put_revoke_pending_keys.add("key-2")
        store.copy_put_results = [[288]]

        self._run_task(thread, self._make_task(thread, 0))

        self.assertEqual(store.copy_put_calls[0][0], ["key-1"])
        self.assertEqual(store.commit_calls, [["key-1"]])
        self.assertEqual(thread._put_revoke_pending_keys, {"key-2"})

    def test_control_revoke_runs_on_sending_thread_and_clears_trackers(self):
        thread, store, started_keys = self._make_thread()
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("key-1", 0)])
        thread._session_tracker = tracker
        store.revoke_results = [[0]]

        thread.add_revoke_request(["key-1", "key-1"])
        request = thread.request_queue.get_nowait()
        thread._handle_request(request)

        self.assertEqual(store.revoke_calls, [["key-1"]])
        self.assertEqual(started_keys, {"key-2"})
        self.assertEqual(thread._put_revoke_pending_keys, set())
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_control_revoke_failure_retains_pending_key_for_later_retry(self):
        thread, store, started_keys = self._make_thread()
        store.revoke_error = RuntimeError("revoke failed")

        thread.add_revoke_request(["key-1"])
        request = thread.request_queue.get_nowait()
        thread._handle_request(request)

        self.assertEqual(store.revoke_calls, [["key-1"]] * 3)
        self.assertEqual(started_keys, {"key-2"})
        self.assertEqual(thread._put_revoke_pending_keys, {"key-1"})
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

        store.revoke_error = None
        thread.add_revoke_request(["key-1"])
        retry = thread.request_queue.get_nowait()
        thread._handle_request(retry)

        self.assertEqual(store.revoke_calls[-1], ["key-1"])
        self.assertEqual(thread._put_revoke_pending_keys, set())

    def test_control_revoke_deduplicates_queued_and_inflight_keys(self):
        thread, _, started_keys = self._make_thread()

        thread.add_revoke_request(["key-1", "key-1"])
        thread.add_revoke_request(["key-1"])

        self.assertEqual(thread.request_queue.qsize(), 1)
        self.assertEqual(thread._put_revoke_queued_keys, {"key-1"})
        self.assertEqual(thread._put_revoke_pending_keys, {"key-1"})
        self.assertEqual(started_keys, {"key-2"})

        thread.request_queue.get_nowait()
        thread._put_revoke_queued_keys.clear()
        thread._put_revoke_inflight_keys.add("key-1")
        thread.add_revoke_request(["key-1"])
        self.assertEqual(thread.request_queue.qsize(), 0)
        thread.request_queue.task_done()

    def test_control_revoke_queue_failure_keeps_pending_and_unqueues_key(self):
        thread, _, started_keys = self._make_thread()

        with (
            patch.object(thread.request_queue, "put", side_effect=RuntimeError("queue failed")),
            self.assertRaisesRegex(RuntimeError, "queue failed"),
        ):
            thread.add_revoke_request(["key-1"])

        self.assertEqual(started_keys, {"key-2"})
        self.assertEqual(thread._put_revoke_pending_keys, {"key-1"})
        self.assertEqual(thread._put_revoke_queued_keys, set())

    def test_stale_pending_retry_does_not_reacquire_released_ownership(self):
        thread, _, started_keys = self._make_thread()
        started_keys.remove("key-1")
        thread._put_revoke_pending_keys.add("key-1")

        thread._put_revoke_pending_keys.remove("key-1")
        thread.add_revoke_request(["key-1"])

        self.assertEqual(thread.request_queue.qsize(), 0)
        self.assertEqual(thread._put_revoke_pending_keys, set())
        self.assertEqual(started_keys, {"key-2"})

    def test_duplicate_save_key_is_written_and_committed_once(self):
        thread, store, _ = self._make_thread(num_layers=1)
        first = ReqMeta(req_id="r1", block_ids=[1], save_block_keys=["key-1"])
        second = ReqMeta(req_id="r2", block_ids=[2], save_block_keys=["key-1"])
        task = LayerTransferTask(
            0,
            [LayerBlockRange(first, 0, 1), LayerBlockRange(second, 0, 1)],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)

        self._run_task(thread, [task])

        assert task.shared_block_data is not None
        self.assertEqual(task.shared_block_data.block_keys, ["key-1"])
        self.assertEqual(store.copy_put_calls[0][0], ["key-1"])
        self.assertEqual(store.commit_calls[0], ["key-1"])


class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    @staticmethod
    def _make_thread(num_layers=2):
        store = FakeStore()
        invalid_block_ids: set[int] = set()
        load_abort_event = threading.Event()
        get_event = threading.Event()
        token_database = RangeBatchFakeTokenDatabase()
        range_builder = LayerBatchBuilder(
            token_database,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=96,
            num_layers=num_layers,
        )
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=token_database,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            layer_load_finished_events=[threading.Event() for _ in range(num_layers)],
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            num_layers=num_layers,
            group_builders=[range_builder],
            invalid_block_ids=invalid_block_ids,
            invalid_block_ids_lock=threading.Lock(),
            load_abort_event=load_abort_event,
        )
        return thread, store, invalid_block_ids, get_event, load_abort_event

    @staticmethod
    def _make_load_task(thread, layer_id, block_ids=None, block_keys=None):
        block_ids = [3, 4] if block_ids is None else block_ids
        block_keys = ["key-3", "key-4"] if block_keys is None else block_keys
        request = ReqMeta(
            req_id="r1",
            block_ids=block_ids,
            load_block_keys=block_keys,
            load_keys=list(dict.fromkeys(block_keys)),
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id,
            [LayerBlockRange(request, 0, len(block_ids))],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)
        return LayerLoadTask(None, [task], layer_id)

    @staticmethod
    def _make_partial_load_task(thread, include_full_block):
        block_ids = [3, 42] if include_full_block else [42]
        full_block_count = 1 if include_full_block else 0
        request = ReqMeta(
            req_id="r1",
            block_ids=block_ids,
            load_block_keys=["key-3"] if include_full_block else [],
            load_last_block_key="key-tail",
            load_keys=["key-3", "key-tail"] if include_full_block else ["key-tail"],
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[
                LayerBlockRange(
                    request,
                    0,
                    full_block_count,
                    partial_block_index=len(block_ids) - 1,
                )
            ],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)
        return LayerLoadTask(None, [task], 0)

    @staticmethod
    def _make_concurrent_load_task(thread):
        requests = [
            ReqMeta(
                req_id=f"r{index}",
                block_ids=[block_id],
                load_block_keys=[f"key-{block_id}"],
                load_keys=[f"key-{block_id}"],
                is_last_chunk=True,
            )
            for index, block_id in enumerate((3, 4, 5), start=1)
        ]
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 1) for request in requests],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)
        return LayerLoadTask(None, [task], 0)

    @staticmethod
    def _run_task(thread, data):
        thread.request_queue.put(data)
        thread._handle_request(data)

    def test_negative_read_marks_exact_block_and_filters_later_layers(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread()
        store.copy_get_results = [[96, -1], [96]]

        for layer_id in range(2):
            self._run_task(thread, self._make_load_task(thread, layer_id))

        self.assertEqual(store.copy_get_calls[0][0], ["key-3", "key-4"])
        self.assertEqual(
            store.copy_get_calls[1],
            (
                ["key-3"],
                [[750, 1300, 2150]],
                [[32, 64, 32]],
                [[96, 128, 192]],
            ),
        )
        self.assertEqual(invalid_block_ids, {4})
        self.assertFalse(load_abort_event.is_set())

    def test_distinct_shared_layouts_keep_independent_active_rows(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread()
        store.copy_get_results = [[96], [96] * 5]

        self._run_task(
            thread,
            self._make_load_task(
                thread,
                layer_id=0,
                block_ids=[4],
                block_keys=["key-tail"],
            ),
        )
        self._run_task(
            thread,
            self._make_load_task(
                thread,
                layer_id=1,
                block_ids=[0, 1, 2, 3, 4],
                block_keys=[f"key-{index}" for index in range(5)],
            ),
        )

        self.assertEqual(store.copy_get_calls[0][0], ["key-tail"])
        self.assertEqual(
            store.copy_get_calls[1][0],
            ["key-0", "key-1", "key-2", "key-3", "key-4"],
        )
        self.assertEqual(invalid_block_ids, set())
        self.assertFalse(load_abort_event.is_set())

    def test_range_debug_records_physical_load_layers(self):
        thread, store, invalid_block_ids, _, _ = self._make_thread()
        store.copy_get_results = [[160, 160], [128, 128]]

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug_module.logger, "info") as log_info,
        ):
            for layer_id in range(2):
                self._run_task(thread, self._make_load_task(thread, layer_id))

        events = [
            json.loads(call.args[2])
            for call in log_info.call_args_list
            if call.args[:2] == ("%s %s", "[KVPOOL_RANGE_DEBUG]")
        ]
        self.assertEqual([event["event"] for event in events], ["range", "range"])
        self.assertEqual([event["direction"] for event in events], ["load", "load"])
        self.assertEqual([event["layer_id"] for event in events], [0, 1])
        self.assertEqual(events[0]["requested_bytes"], [160, 160])
        self.assertEqual(events[0]["results"], [160, 160])
        self.assertEqual(invalid_block_ids, set())

    def test_range_debug_logger_failure_preserves_load_failure(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread(num_layers=1)
        store.copy_get_results = [[288, -1]]

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(
                kv_transfer_module.logger,
                "info",
                side_effect=RuntimeError("logger failed"),
            ),
        ):
            self._run_task(thread, self._make_load_task(thread, 0))

        self.assertEqual(invalid_block_ids, {4})
        self.assertFalse(load_abort_event.is_set())

    def test_duplicate_remote_key_failure_only_filters_failed_row(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread()
        store.copy_get_results = [[96, -1], [96]]

        for layer_id in range(2):
            self._run_task(
                thread,
                self._make_load_task(
                    thread,
                    layer_id,
                    block_ids=[3, 4],
                    block_keys=["shared-key", "shared-key"],
                ),
            )

        self.assertEqual(
            store.copy_get_calls[1],
            (
                ["shared-key"],
                [[750, 1300, 2150]],
                [[32, 64, 32]],
                [[96, 128, 192]],
            ),
        )
        self.assertEqual(invalid_block_ids, {4})
        self.assertFalse(load_abort_event.is_set())

    def test_duplicate_remote_key_all_failures_filter_all_rows(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread()
        store.copy_get_results = [[-1, -1]]

        for layer_id in range(2):
            self._run_task(
                thread,
                self._make_load_task(
                    thread,
                    layer_id,
                    block_ids=[3, 4],
                    block_keys=["shared-key", "shared-key"],
                ),
            )

        self.assertEqual(len(store.copy_get_calls), 1)
        self.assertEqual(invalid_block_ids, {3, 4})
        self.assertFalse(load_abort_event.is_set())

    def test_request_exception_does_not_stop_later_range_subgroups(self):
        thread, store, invalid_block_ids, get_event, load_abort_event = self._make_thread(num_layers=1)
        data = self._make_concurrent_load_task(thread)
        thread.request_queue.put(data)

        with patch.object(
            store,
            "batch_copy_get",
            side_effect=([96], RuntimeError("request failed"), [96]),
        ) as copy_get:
            thread._handle_request(data)

        self.assertEqual(
            [call.args[0] for call in copy_get.call_args_list],
            [["key-3"], ["key-4"], ["key-5"]],
        )
        self.assertEqual(invalid_block_ids, {4})
        self.assertFalse(load_abort_event.is_set())
        self.assertTrue(thread.layer_load_finished_events[0].is_set())
        self.assertTrue(get_event.is_set())

    def test_request_failure_is_local_for_every_subgroup_and_result_shape(self):
        failures = (
            RuntimeError("request failed"),
            [],
            [96, 96],
            ["invalid"],
        )
        for failed_index in range(3):
            for failure in failures:
                with self.subTest(failed_index=failed_index, failure=failure):
                    thread, store, invalid_block_ids, _, load_abort_event = self._make_thread(num_layers=1)
                    data = self._make_concurrent_load_task(thread)
                    thread.request_queue.put(data)
                    responses: list[object] = [[96], [96], [96]]
                    responses[failed_index] = failure

                    with patch.object(
                        store,
                        "batch_copy_get",
                        side_effect=responses,
                    ) as copy_get:
                        thread._handle_request(data)

                    self.assertEqual(
                        [call.args[0] for call in copy_get.call_args_list],
                        [["key-3"], ["key-4"], ["key-5"]],
                    )
                    self.assertEqual(invalid_block_ids, {3 + failed_index})
                    self.assertFalse(load_abort_event.is_set())

    def test_malformed_read_results_invalidate_request_without_batch_abort(self):
        for results in ([96], [96, 96, 96], ["invalid", 96]):
            with self.subTest(results=results):
                thread, store, invalid_block_ids, get_event, load_abort_event = self._make_thread()
                store.copy_get_results = [results]

                self._run_task(thread, self._make_load_task(thread, 0))

                self.assertEqual(invalid_block_ids, {3, 4})
                self.assertFalse(load_abort_event.is_set())
                self.assertTrue(thread.layer_load_finished_events[0].is_set())
                self.assertTrue(get_event.is_set())

    def test_copy_get_exception_invalidates_request_and_finishes_layer(self):
        thread, store, invalid_block_ids, get_event, load_abort_event = self._make_thread()
        data = self._make_load_task(thread, 0)
        thread.request_queue.put(data)

        with (
            patch.object(store, "batch_copy_get", side_effect=RuntimeError("transfer failed")),
            patch.object(
                thread.request_queue,
                "task_done",
                wraps=thread.request_queue.task_done,
            ) as task_done,
        ):
            thread._handle_request(data)

        self.assertEqual(invalid_block_ids, {3, 4})
        self.assertFalse(load_abort_event.is_set())
        self.assertTrue(thread.layer_load_finished_events[0].is_set())
        self.assertTrue(get_event.is_set())
        self.assertEqual(task_done.call_count, 1)
        self.assertEqual(
            thread._inactive_load_rows,
            {("r1", 3, "key-3"), ("r1", 4, "key-4")},
        )

    def test_exception_fallback_marks_full_and_partial_blocks(self):
        for include_full_block in (False, True):
            expected_invalid = {3, 42} if include_full_block else {42}
            for failure in ("builder", "backend", "malformed"):
                with self.subTest(
                    include_full_block=include_full_block,
                    failure=failure,
                ):
                    thread, store, invalid_block_ids, get_event, load_abort_event = self._make_thread()
                    data = self._make_partial_load_task(thread, include_full_block)
                    thread.request_queue.put(data)
                    if failure == "builder":
                        thread.layer_batch_builder.build_addrs = MagicMock(side_effect=RuntimeError("metadata failed"))
                    elif failure == "backend":
                        store.batch_copy_get = MagicMock(side_effect=RuntimeError("transfer failed"))
                    else:
                        store.copy_get_results = [[96] if include_full_block else []]

                    with patch.object(
                        thread.request_queue,
                        "task_done",
                        wraps=thread.request_queue.task_done,
                    ) as task_done:
                        thread._handle_request(data)

                    self.assertEqual(invalid_block_ids, expected_invalid)
                    self.assertEqual(load_abort_event.is_set(), failure == "builder")
                    self.assertTrue(thread.layer_load_finished_events[0].is_set())
                    self.assertTrue(get_event.is_set())
                    self.assertEqual(task_done.call_count, 1)
                    self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_duplicate_remote_key_loads_into_both_local_blocks(self):
        thread, store, invalid_block_ids, _, _ = self._make_thread(num_layers=1)
        data = self._make_load_task(
            thread,
            0,
            block_ids=[3, 4],
            block_keys=["shared-key", "shared-key"],
        )

        self._run_task(thread, data)

        self.assertEqual(store.copy_get_calls[0][0], ["shared-key", "shared-key"])
        self.assertNotEqual(
            store.copy_get_calls[0][1][0],
            store.copy_get_calls[0][1][1],
        )
        self.assertEqual(invalid_block_ids, set())

    def test_concurrent_requests_use_separate_ranged_batches(self):
        thread, store, invalid_block_ids, _, _ = self._make_thread(num_layers=1)
        first = ReqMeta(
            req_id="r1",
            block_ids=[3, 4],
            load_block_keys=["shared-key", "r1-key"],
            load_keys=["shared-key", "r1-key"],
            is_last_chunk=True,
        )
        second = ReqMeta(
            req_id="r2",
            block_ids=[5, 6],
            load_block_keys=["shared-key", "r2-key"],
            load_keys=["shared-key", "r2-key"],
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            0,
            [
                LayerBlockRange(first, 0, 2),
                LayerBlockRange(second, 0, 2),
            ],
            use_key_major_ranges=True,
        )
        task.shared_block_data = thread.build_shared_data(task)

        self._run_task(thread, LayerLoadTask(None, [task], 0))

        self.assertEqual(
            [call[0] for call in store.copy_get_calls],
            [["shared-key", "r1-key"], ["shared-key", "r2-key"]],
        )
        self.assertEqual(invalid_block_ids, set())

    def test_concurrent_request_failure_filters_only_its_row(self):
        thread, store, invalid_block_ids, _, load_abort_event = self._make_thread()
        store.copy_get_results = [
            [96, -1],
            [96, 96],
            [96],
            [96, 96],
        ]
        first = ReqMeta(
            req_id="r1",
            block_ids=[3, 4],
            load_block_keys=["shared-key", "r1-key"],
            load_keys=["shared-key", "r1-key"],
            is_last_chunk=True,
        )
        second = ReqMeta(
            req_id="r2",
            block_ids=[5, 6],
            load_block_keys=["shared-key", "r2-key"],
            load_keys=["shared-key", "r2-key"],
            is_last_chunk=True,
        )

        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id,
                [
                    LayerBlockRange(first, 0, 2),
                    LayerBlockRange(second, 0, 2),
                ],
                use_key_major_ranges=True,
            )
            task.shared_block_data = thread.build_shared_data(task)
            self._run_task(thread, LayerLoadTask(None, [task], layer_id))

        self.assertEqual(
            [call[0] for call in store.copy_get_calls],
            [
                ["shared-key", "r1-key"],
                ["shared-key", "r2-key"],
                ["shared-key"],
                ["shared-key", "r2-key"],
            ],
        )
        self.assertEqual(invalid_block_ids, {4})
        self.assertFalse(load_abort_event.is_set())


class TestKVTransferTpMismatchDispatch(unittest.TestCase):
    """TP-mismatch worker dispatch wiring for Sending/Recving threads."""

    def _make_sending(self, worker=None, exists_result=None):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=False,
            worker=worker,
        )
        return t, store

    def _make_recving(self, worker=None):
        store = FakeStore([0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
            worker=worker,
        )
        return t, store

    def test_sending_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_sending(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._store_kv_tp_mismatch.assert_called_once_with(req)

    def test_sending_normal_path_when_worker_none(self):
        # worker=None -> tp_mismatch dispatch skipped, normal store path runs.
        t, store = self._make_sending(worker=None, exists_result=[1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)  # normal path executed

    def test_recving_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_called_once()
        args = worker._load_kv_tp_mismatch.call_args.args
        # (block_hashes, block_ids, token_len, mask_num)
        self.assertEqual(args[2], 16)  # token_len
        self.assertEqual(args[3], 0)  # mask_num

    def test_recving_tp_mismatch_missing_load_spec_finishes(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_not_called()
        self.assertEqual(t.get_and_clear_finished_requests(), {"r1"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_recving_tp_mismatch_task_done_on_exception(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        worker._load_kv_tp_mismatch.side_effect = RuntimeError("load failed")
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        with self.assertRaises(RuntimeError):
            t._handle_request(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
