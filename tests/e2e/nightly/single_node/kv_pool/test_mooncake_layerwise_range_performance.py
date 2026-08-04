from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vllm_ascend import envs

_SOURCE_PATTERN_MODULUS = 251


@dataclass(frozen=True)
class _BenchmarkShape:
    device_index: int
    layer_count: int
    request_count: int
    rows_per_request: int
    segments_per_row: int
    segment_bytes: int
    warmup_iterations: int
    measured_iterations: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _BenchmarkShape:
        section = config.get("range_performance")
        if not isinstance(section, dict):
            pytest.fail("Mooncake nightly config requires a range_performance object")
        field_names = tuple(cls.__dataclass_fields__)
        missing = [name for name in field_names if name not in section]
        if missing:
            pytest.fail("range_performance is missing fields: " + ", ".join(missing))
        invalid = [
            name for name in field_names if isinstance(section[name], bool) or not isinstance(section[name], int)
        ]
        if invalid:
            pytest.fail("range_performance fields must be integers: " + ", ".join(invalid))
        values = {name: section[name] for name in field_names}
        if values["device_index"] < 0:
            pytest.fail("range_performance.device_index must be non-negative")
        non_positive = [name for name, value in values.items() if name != "device_index" and value <= 0]
        if non_positive:
            pytest.fail("range_performance fields must be positive: " + ", ".join(non_positive))
        return cls(**values)

    @property
    def row_count(self) -> int:
        return self.request_count * self.rows_per_request

    @property
    def row_bytes(self) -> int:
        return self.segments_per_row * self.segment_bytes

    @property
    def object_bytes(self) -> int:
        return self.layer_count * self.row_bytes


@dataclass(frozen=True)
class _DirectionMetrics:
    throughput_gbps: float
    p50_ms: float
    p95_ms: float
    call_count: int
    transferred_bytes: int


def _required_positive_float(raw_value: str | None, name: str) -> float:
    if raw_value is None:
        pytest.fail(f"configured Mooncake nightly run requires {name}")
    try:
        value = float(raw_value)
    except ValueError as exc:
        pytest.fail(f"{name} must be a number: {exc}")
    if not math.isfinite(value) or value <= 0:
        pytest.fail(f"{name} must be a positive finite number")
    return value


def _require_successful_range_results(results: list[int], expected_count: int) -> None:
    assert len(results) == expected_count, (len(results), expected_count)
    assert all(isinstance(result, int) and not isinstance(result, bool) for result in results), results
    assert all(result >= 0 for result in results), results


def _percentile_ms(latencies_s: list[float], percentile: float) -> float:
    ordered = sorted(latencies_s)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index] * 1000


def _run_ranges(
    operation: Callable[
        [list[str], list[list[int]], list[list[int]], list[list[int]]],
        list[int],
    ],
    keys: list[str],
    buffers: list[Any],
    shape: _BenchmarkShape,
    row_batches: list[list[int]],
    iterations: int,
    synchronize: Callable[[], None],
) -> _DirectionMetrics:
    latencies_s: list[float] = []
    transferred_bytes = 0
    for _ in range(iterations):
        for layer_index in range(shape.layer_count):
            object_layer_offset = layer_index * shape.row_bytes
            for row_indices in row_batches:
                batch_keys = [keys[index] for index in row_indices]
                all_buffers = [
                    [
                        buffers[index].data_ptr() + object_layer_offset + segment_index * shape.segment_bytes
                        for segment_index in range(shape.segments_per_row)
                    ]
                    for index in row_indices
                ]
                all_sizes = [[shape.segment_bytes] * shape.segments_per_row for _ in row_indices]
                all_offsets = [
                    [
                        object_layer_offset + segment_index * shape.segment_bytes
                        for segment_index in range(shape.segments_per_row)
                    ]
                    for _ in row_indices
                ]
                synchronize()
                started_at = time.perf_counter()
                results = operation(
                    batch_keys,
                    all_buffers,
                    all_sizes,
                    all_offsets,
                )
                synchronize()
                elapsed_s = time.perf_counter() - started_at
                _require_successful_range_results(results, len(batch_keys))
                latencies_s.append(elapsed_s)
                transferred_bytes += shape.row_bytes * len(batch_keys)

    elapsed_s = sum(latencies_s)
    assert elapsed_s > 0
    return _DirectionMetrics(
        throughput_gbps=transferred_bytes / elapsed_s / 1_000_000_000,
        p50_ms=_percentile_ms(latencies_s, 0.50),
        p95_ms=_percentile_ms(latencies_s, 0.95),
        call_count=len(latencies_s),
        transferred_bytes=transferred_bytes,
    )


def _remove_test_keys(store: Any, keys: list[str]) -> None:
    batch_remove = getattr(store, "batch_remove", None)
    if callable(batch_remove):
        results = list(batch_remove(keys, True))
    else:
        remove = getattr(store, "remove", None)
        if not callable(remove):
            pytest.fail("Mooncake Client does not expose batch_remove or remove for benchmark cleanup")
        results = [int(remove(key, True)) for key in keys]
    assert results == [0] * len(keys)


def test_ranged_success_contract_accepts_zero_and_positive_results() -> None:
    for results in ([0], [128], [0, 128]):
        _require_successful_range_results(results, len(results))


def test_ranged_success_contract_rejects_malformed_or_negative_results() -> None:
    for results, expected_count in (([-1], 1), ([True], 1), ([1.0], 1), ([0], 2)):
        with pytest.raises(AssertionError):
            _require_successful_range_results(results, expected_count)  # type: ignore[arg-type]


def test_mooncake_layerwise_ranged_transfer_performance(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    config_value = envs.VLLM_ASCEND_NIGHTLY_MOONCAKE_CONFIG
    if not config_value:
        pytest.skip("set VLLM_ASCEND_NIGHTLY_MOONCAKE_CONFIG to run the Mooncake/NPU performance gate")

    config_path = Path(config_value)
    if not config_path.is_file():
        pytest.fail(f"VLLM_ASCEND_NIGHTLY_MOONCAKE_CONFIG does not name a file: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        pytest.fail("Mooncake nightly config root must be an object")

    shape = _BenchmarkShape.from_config(config)
    min_gbps = _required_positive_float(
        envs.VLLM_ASCEND_NIGHTLY_KVPOOL_MIN_GBPS,
        "VLLM_ASCEND_NIGHTLY_KVPOOL_MIN_GBPS",
    )
    max_p95_ms = _required_positive_float(
        envs.VLLM_ASCEND_NIGHTLY_KVPOOL_MAX_P95_MS,
        "VLLM_ASCEND_NIGHTLY_KVPOOL_MAX_P95_MS",
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))

    import torch
    from vllm.config import ParallelConfig

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
        MooncakeBackend,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.fail("configured Mooncake nightly performance gate requires an available NPU")
    torch.npu.set_device(shape.device_index)
    device = f"npu:{shape.device_index}"
    source_buffers = [torch.empty(shape.object_bytes, dtype=torch.uint8, device=device) for _ in range(shape.row_count)]
    target_buffers = [torch.empty(shape.object_bytes, dtype=torch.uint8, device=device) for _ in range(shape.row_count)]
    for index, source in enumerate(source_buffers):
        source.fill_((index % _SOURCE_PATTERN_MODULUS) + 1)
    for target in target_buffers:
        target.zero_()

    backend = MooncakeBackend(ParallelConfig())
    backend.validate_layerwise_support()
    all_buffers = source_buffers + target_buffers
    backend.register_buffer(
        [buffer.data_ptr() for buffer in all_buffers],
        [buffer.numel() * buffer.element_size() for buffer in all_buffers],
    )
    keys = [f"vllm-ascend-nightly-range-{uuid.uuid4().hex}-{index}" for index in range(shape.row_count)]
    all_rows = [list(range(shape.row_count))]
    request_rows = [
        list(range(start, start + shape.rows_per_request))
        for start in range(0, shape.row_count, shape.rows_per_request)
    ]
    open_put_keys: list[str] = []
    open_get_keys: list[str] = []
    committed_keys: list[str] = []
    try:
        put_start_results = backend.batch_put_start(
            keys,
            [shape.object_bytes] * len(keys),
        )
        open_put_keys = [key for key, result in zip(keys, put_start_results, strict=True) if result == 0]
        assert put_start_results == [0] * len(keys)
        _run_ranges(
            backend.batch_copy_put,
            keys,
            source_buffers,
            shape,
            all_rows,
            shape.warmup_iterations,
            torch.npu.synchronize,
        )
        save_metrics = _run_ranges(
            backend.batch_copy_put,
            keys,
            source_buffers,
            shape,
            all_rows,
            shape.measured_iterations,
            torch.npu.synchronize,
        )
        commit_results = backend.batch_commit(keys)
        committed_keys = [key for key, result in zip(keys, commit_results, strict=True) if result == 0]
        open_put_keys = [key for key, result in zip(keys, commit_results, strict=True) if result != 0]
        assert commit_results == [0] * len(keys)

        get_start_results = backend.batch_get_start(keys)
        open_get_keys = [key for key, result in zip(keys, get_start_results, strict=True) if result == 0]
        assert get_start_results == [0] * len(keys)
        _run_ranges(
            backend.batch_copy_get,
            keys,
            target_buffers,
            shape,
            request_rows,
            shape.warmup_iterations,
            torch.npu.synchronize,
        )
        load_metrics = _run_ranges(
            backend.batch_copy_get,
            keys,
            target_buffers,
            shape,
            request_rows,
            shape.measured_iterations,
            torch.npu.synchronize,
        )
        assert backend.batch_get_end(keys) == 0
        open_get_keys = []
        assert all(torch.equal(source, target) for source, target in zip(source_buffers, target_buffers, strict=True))

        metrics = {"save": save_metrics, "load": load_metrics}
        for direction, result in metrics.items():
            record_property(f"{direction}_throughput_gbps", result.throughput_gbps)
            record_property(f"{direction}_p50_ms", result.p50_ms)
            record_property(f"{direction}_p95_ms", result.p95_ms)
            assert result.throughput_gbps >= min_gbps
            assert result.p95_ms <= max_p95_ms
        record_property("layer_count", shape.layer_count)
        record_property("request_count", shape.request_count)
        record_property("rows_per_request", shape.rows_per_request)
        record_property("save_batch_rows", shape.row_count)
        record_property("load_batch_rows", shape.rows_per_request)
        print(
            json.dumps(
                {
                    "shape": shape.__dict__,
                    "save": save_metrics.__dict__,
                    "load": load_metrics.__dict__,
                    "min_gbps": min_gbps,
                    "max_p95_ms": max_p95_ms,
                },
                sort_keys=True,
            )
        )
    finally:
        try:
            if open_get_keys:
                assert backend.batch_get_end(open_get_keys) == 0
            if open_put_keys:
                assert backend.batch_revoke(open_put_keys) == [0] * len(open_put_keys)
            if committed_keys:
                assert backend.store is not None
                _remove_test_keys(backend.store, committed_keys)
        finally:
            close = getattr(backend.store, "close", None)
            if callable(close):
                close()
