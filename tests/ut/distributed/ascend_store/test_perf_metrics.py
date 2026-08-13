from __future__ import annotations

import json
import sys
import threading
import unittest
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.perf_metrics import (
    KVPoolPerfMetrics,
    PerfMetricConfig,
)


class TestPerfMetricConfig(unittest.TestCase):
    def test_defaults_are_disabled(self) -> None:
        self.assertEqual(
            PerfMetricConfig.from_env({}),
            PerfMetricConfig(enabled=False, interval_seconds=10),
        )

    def test_explicit_values(self) -> None:
        self.assertEqual(
            PerfMetricConfig.from_env(
                {
                    "VLLM_ASCEND_KVPOOL_PERF_METRICS": "1",
                    "VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS": "2",
                }
            ),
            PerfMetricConfig(enabled=True, interval_seconds=2),
        )

    def test_invalid_values_fail_closed(self) -> None:
        for enabled in ("", "2", "true", "-1"):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                PerfMetricConfig.from_env(
                    {"VLLM_ASCEND_KVPOOL_PERF_METRICS": enabled}
                )
        for interval in ("", "0", "-1", "1.5", "01"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                PerfMetricConfig.from_env(
                    {
                        "VLLM_ASCEND_KVPOOL_PERF_METRICS_INTERVAL_SECONDS": interval
                    }
                )


class TestKVPoolPerfMetrics(unittest.TestCase):
    def test_disabled_metrics_are_noops(self) -> None:
        emitted: list[str] = []
        metrics = KVPoolPerfMetrics(
            PerfMetricConfig(False, 10), emit_fn=emitted.append
        )

        metrics.record("copy", 1_000_000, bytes_count=64)
        with metrics.measure("wait"):
            pass

        self.assertEqual(metrics.snapshot()["events"], {})
        self.assertIsNone(metrics.emit())
        self.assertEqual(emitted, [])

    def test_disabled_measurement_reuses_one_null_context(self) -> None:
        metrics = KVPoolPerfMetrics(PerfMetricConfig(False, 10))

        first = metrics.measure("first")
        second = metrics.measure("second", bytes_count=64, layer_id=2)

        self.assertIs(first, second)
        with first:
            pass
        with second:
            pass

    def test_snapshot_reports_count_bytes_and_percentiles(self) -> None:
        metrics = KVPoolPerfMetrics(
            PerfMetricConfig(True, 10), labels={"rank": 1}
        )
        for milliseconds in (1, 2, 3, 4, 100):
            metrics.record(
                "batch_copy_get",
                milliseconds * 1_000_000,
                bytes_count=128,
                layer_id=2,
            )

        snapshot = metrics.snapshot()
        event = snapshot["events"]["batch_copy_get|layer=2"]
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual(snapshot["labels"], {"rank": 1})
        self.assertEqual(event["count"], 5)
        self.assertEqual(event["bytes"], 640)
        self.assertEqual(event["sum_ms"], 110)
        self.assertEqual(event["exclusive_sum_ms"], 110)
        self.assertEqual(event["p50_ms"], 3)
        self.assertEqual(event["p95_ms"], 100)
        self.assertEqual(event["max_ms"], 100)

    def test_emit_is_json_and_resets_interval(self) -> None:
        emitted: list[str] = []
        metrics = KVPoolPerfMetrics(
            PerfMetricConfig(True, 10), emit_fn=emitted.append
        )
        metrics.record("batch_put_start", 2_000_000)

        line = metrics.emit(reset=True)

        self.assertIsNotNone(line)
        assert line is not None
        payload = json.loads(line.removeprefix("KVPOOL_PERF_METRICS "))
        self.assertEqual(payload["events"]["batch_put_start"]["count"], 1)
        self.assertEqual(emitted, [line])
        self.assertEqual(metrics.snapshot()["events"], {})

    def test_concurrent_recording_is_not_lost(self) -> None:
        metrics = KVPoolPerfMetrics(PerfMetricConfig(True, 10))

        def record() -> None:
            for _ in range(200):
                metrics.record("wait_for_layer_load", 1_000)

        threads = [threading.Thread(target=record) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        event = metrics.snapshot()["events"]["wait_for_layer_load"]
        self.assertEqual(event["count"], 800)
        self.assertEqual(event["sample_count"], 800)

    def test_nested_measurements_report_exclusive_time(self) -> None:
        metrics = KVPoolPerfMetrics(PerfMetricConfig(True, 10))

        module = sys.modules[KVPoolPerfMetrics.__module__]
        with patch.object(
            module.time,
            "perf_counter_ns",
            side_effect=(
                0,
                10_000_000,
                30_000_000,
                35_000_000,
                50_000_000,
            ),
        ), metrics.measure("outer"), metrics.measure("inner"):
            pass

        events = metrics.snapshot()["events"]
        self.assertEqual(events["inner"]["sum_ms"], 20)
        self.assertEqual(events["inner"]["exclusive_sum_ms"], 20)
        self.assertEqual(events["outer"]["sum_ms"], 50)
        self.assertEqual(events["outer"]["exclusive_sum_ms"], 25)

    def test_negative_measurements_are_rejected(self) -> None:
        metrics = KVPoolPerfMetrics(PerfMetricConfig(True, 10))
        with self.assertRaises(ValueError):
            metrics.record("bad", -1)
        with self.assertRaises(ValueError):
            metrics.record("bad", 1, bytes_count=-1)
        with self.assertRaises(ValueError):
            metrics.record("bad", 1, exclusive_elapsed_ns=-1)
        with self.assertRaises(ValueError):
            metrics.record("bad", 1, exclusive_elapsed_ns=2)


if __name__ == "__main__":
    unittest.main()
