#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
import unittest
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import range_debug


class TestRangeDebug(unittest.TestCase):
    def test_emitters_share_the_existing_json_event_contract(self):
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug.logger, "info") as log_info,
        ):
            range_debug.emit_range_event("load", 2, [[16]], [[32]], [16])
            range_debug.emit_commit_event(2, 1, [0])
            range_debug.emit_whole_key_event("put", 3)

        self.assertEqual(
            [json.loads(call.args[2]) for call in log_info.call_args_list],
            [
                {
                    "event": "range",
                    "direction": "load",
                    "layer_id": 2,
                    "key_count": 1,
                    "requested_bytes": [16],
                    "sizes": [[16]],
                    "object_offsets": [[32]],
                    "results": [16],
                },
                {
                    "event": "commit",
                    "layer_id": 2,
                    "key_count": 1,
                    "results": [0],
                },
                {
                    "event": "whole_key",
                    "direction": "put",
                    "key_count": 3,
                },
            ],
        )

    def test_disabled_emission_does_not_build_or_log_payload(self):
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "0"}),
            patch.object(
                range_debug,
                "_build_range_payload",
                side_effect=AssertionError("payload builder must not run"),
            ) as build_payload,
            patch.object(range_debug.logger, "info") as log_info,
        ):
            range_debug.emit_range_event("load", 2, [[16]], [[32]], [16])

        build_payload.assert_not_called()
        log_info.assert_not_called()

    def test_payload_coercion_and_serialization_failures_are_best_effort(self):
        class InvalidInteger:
            def __int__(self):
                raise ValueError("cannot convert")

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug.logger, "info") as log_info,
        ):
            range_debug.emit_range_event("load", 2, [[InvalidInteger()]], [[32]], [16])  # type: ignore[list-item]
        log_info.assert_not_called()

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug.json, "dumps", side_effect=TypeError("cannot serialize")),
        ):
            range_debug.emit_commit_event(2, 1, [0])

    def test_logger_failures_never_escape_emitters(self):
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_KVPOOL_RANGE_DEBUG": "1"}),
            patch.object(range_debug.logger, "info", side_effect=RuntimeError("logger failed")) as log_info,
        ):
            range_debug.emit_range_event("load", 2, [[16]], [[32]], [16])
            range_debug.emit_commit_event(2, 1, [0])
            range_debug.emit_whole_key_event("get", 3)

        self.assertEqual(log_info.call_count, 3)


if __name__ == "__main__":
    unittest.main()
