"""Offline contracts and real tiny-CPU production forward; no remote execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import execution_shape_probe as P

MANIFEST = ROOT / "results/processed/execution_shape_manifest_20261003.json"


def small_entry():
    entry = copy.deepcopy(P.checked_manifest(MANIFEST)["entries"][0])
    items = [P.ExecutionItem("p", 2, 0, "prefill", False),
             P.ExecutionItem("q", 3, 2, "prefill", True),
             P.ExecutionItem("d", 1, 5, "decode", True)]
    entry["execution_shape"] = P.ExecutionShape(tuple(items)).describe()
    entry["execution_shape_key"] = P.execution_shape_key(entry["execution_shape"])
    entry["prime_requests"] = {"p": {"prompt_len": 5, "gen_len": 5},
                               "q": {"prompt_len": 5, "gen_len": 5},
                               "d": {"prompt_len": 4, "gen_len": 5}}
    entry["warmup_s"] = 0.1
    entry["training"]["records_per_shape_per_process"] = 3
    return entry


class ManifestTests(unittest.TestCase):
    def _bad(self, change):
        obj = P.checked_manifest(MANIFEST)
        change(obj)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                P.checked_manifest(path)

    def test_actual_plan_has_six_shapes_three_processes_thirty_records(self):
        plan = P.dry_plan(P.checked_manifest(MANIFEST))
        self.assertEqual(6, len(plan["points"]))
        self.assertEqual({1, 2, 3}, {p["n_forwards"] for p in plan["points"]})
        self.assertEqual({3}, {p["independent_processes_required"] for p in plan["points"]})
        self.assertEqual({30}, {p["records"] for p in plan["points"]})
        self.assertIsNone(plan["duration_seconds"])

    def test_empty_manifest_rejected(self):
        self._bad(lambda m: m.update(entries=[]))

    def test_wrong_signature_rejected(self):
        self._bad(lambda m: m["entries"][0]["execution_signature"].update(logits_last_only=False))

    def test_shape_key_rejected(self):
        self._bad(lambda m: m["entries"][0].update(execution_shape_key="wrong"))

    def test_duplicate_shape_rejected(self):
        self._bad(lambda m: m["entries"].append(copy.deepcopy(m["entries"][0])))

    def test_group_mismatch_rejected(self):
        self._bad(lambda m: m["entries"][0]["execution_shape"].update(n_forwards=99))

    def test_unknown_kv_rejected(self):
        self._bad(lambda m: m["entries"][0]["execution_shape"]["items"][0].update(kv_len=None))

    def test_prompt_contract_rejected(self):
        def flip(m):
            item = m["entries"][0]["execution_shape"]["items"][0]
            item["need_token"] = not item["need_token"]
        self._bad(flip)

    def test_zero_warmup_rejected(self):
        self._bad(lambda m: m["entries"][0].update(warmup_s=0))

    def test_one_session_rejected(self):
        self._bad(lambda m: m["entries"][0].update(repeats=1))

    def test_bool_not_integer(self):
        with self.assertRaises(ValueError):
            P.positive_int(True, "records")

    def test_nan_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                P.read_json(path)

    def test_cli_default_does_not_load_hardware(self):
        with patch.object(P, "execute", side_effect=AssertionError("hardware reached")), patch("builtins.print"):
            self.assertEqual(0, P.main(["--manifest", str(MANIFEST)]))

    def test_execute_refuses_non_linux_without_creating_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(P.platform, "system", return_value="Windows"):
            target = Path(directory) / "new.json"
            with self.assertRaisesRegex(ValueError, "Linux"):
                P.execute(P.checked_manifest(MANIFEST), MANIFEST, target, "training", Path("missing"))
            self.assertFalse(target.exists())

    def test_validation_requires_frozen_before_hardware(self):
        with patch.object(P.platform, "system", return_value="Linux"):
            with self.assertRaisesRegex(ValueError, "frozen-estimator"):
                P.execute(P.checked_manifest(MANIFEST), MANIFEST, Path("unused"), "validation", Path("missing"))


class ProductionCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import _tiny_support as T
        if T.torch is None or T.Qwen2ForCausalLM is None:
            raise AssertionError(f"required local CPU dependencies missing: {T.TORCH_MISSING_REASON}")
        cls.T = T
        T.torch.set_num_threads(1)

    def driver(self):
        drv = self.T.make_driver(self.T.make_model(), logits_last_only=True, sample_needed_only=True)
        drv.load()
        return drv

    def test_exact_zero_and_heterogeneous_kv_then_whole_step(self):
        drv, entry = self.driver(), small_entry()
        batch = P.prime(drv, entry)
        self.assertNotIn("p", drv._slots)  # kv=0 has no synthetic past tensor
        self.assertEqual(2, drv._slots["q"].past_len)
        self.assertEqual(5, drv._slots["d"].past_len)
        self.assertEqual([2, 3, 1], [i.n_new_tokens for i in batch])
        row = P.measure_once(drv, entry)
        self.assertEqual(3, row["n_forwards"])
        self.assertAlmostEqual(row["total_ms"], row["fwd_ms"] + row["host_ms"])
        self.assertEqual([2, 5, 6], [drv._slots[r].past_len for r in ("p", "q", "d")])
        # A second record must recreate the SAME starting KV, not grow the first record.
        P.measure_once(drv, entry)
        self.assertEqual([2, 5, 6], [drv._slots[r].past_len for r in ("p", "q", "d")])

    def test_warmup_excluded_and_records_exact(self):
        drv, entry = self.driver(), small_entry()
        ticks = iter([0, 0.01, 0.02, 0.2, 0.21])
        row = P.collect_entry(drv, entry, "training", clock=lambda: next(ticks))
        self.assertEqual(2, row["warmup"]["calls"])
        self.assertTrue(row["warmup"]["excluded"])
        self.assertEqual([0, 1, 2], [r["record_index"] for r in row["records"]])
        self.assertFalse(drv._slots)

    def test_cpu_cannot_be_labeled_measured(self):
        drv = self.driver()
        with self.assertRaisesRegex(ValueError, "CPU/TEST_DATA"):
            P.runtime_identity(drv, P.load_config(ROOT / "configs/mthreads/prereg_batch/cost_single.yaml"), {})


if __name__ == "__main__":
    unittest.main()
