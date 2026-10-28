"""TEST_DATA-only temporary models exercise preparation; no measured model is made.

The positive fixture deliberately supplies the MEASURED field to exercise its
parser. It is fabricated test input, stored only in a temporary TEST_DATA path.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_execution_pilot_tested",
                                             ROOT / "scripts/prepare_execution_pilot.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def synthetic_model():
    """Not evidence: a temporary positive parser fixture with invented costs."""
    signature = P.execution_signature(P.load_config(ROOT / P.SOURCE_DIR / "cost_single.yaml").runtime)
    key = P.ExecutionShape((P.ExecutionItem("r0", 512, 0, "prefill", False),)).key()
    code_digest = P.compute_tree_digest(ROOT, scope=("src", "benchmark"))
    source = {"git_commit": "a" * 40, "tree_digest": "b" * 64, "git_dirty": False,
              "execution_code_digest": code_digest}
    inputs = {"manifest_sha256": "c" * 64, "frozen_estimator_sha256": "d" * 64}
    for split in ("training", "validation"):
        inputs[split] = [{"path": f"TEST_DATA/{split}-{i}.json", "session_id": f"{split}-{i}",
                          "sha256": hashlib.sha256(f"TEST_DATA-{split}-{i}".encode()).hexdigest()}
                         for i in range(3)]
    return {"protocol_version": "execution-shape-v1", "data_label": "MEASURED",
            "execution_timing": P.TIMING, "execution_signature": signature,
            "source": source, "inputs": inputs, "safety_margin": 0,
            "execution_code_digest": code_digest,
            "coefficients": {}, "calibrated_range": {},
            "fingerprint": {"hardware": {"device_model": "TEST_DATA S4000", "driver_version": "TEST_DATA"},
                            "software": {key: "TEST_DATA" for key in ("torch", "torch_musa", "transformers", "python")},
                            "model": {"id": signature["model"], "revision": signature["model_revision"]},
                            "precision": signature["precision"]},
            "execution_shape_table": {key: {"fwd_ms": 100.0, "host_ms": 2.0}},
            "validation": {"model_holds": True, "thresholds": {"TEST_DATA_error_max": 0.10}}}


class TestPreparation(unittest.TestCase):
    def setUp(self):
        # Resolve and verify the temporary target under configs before allowing
        # TemporaryDirectory cleanup; no deletion targets any existing source.
        self.tmp = tempfile.TemporaryDirectory(prefix="TEST_DATA_prepare_", dir=ROOT / "configs")
        self.base = Path(self.tmp.name).resolve()
        self.assertTrue(self.base.is_relative_to((ROOT / "configs").resolve()))
        self.addCleanup(self.tmp.cleanup)
        self.model_path = self.base / "TEST_DATA_model.json"
        self.cfg_dir = (self.base / "derived").relative_to(ROOT).as_posix()
        self.model_rel = self.model_path.relative_to(ROOT).as_posix()
        self.write_model(synthetic_model())
        self.source_bytes = {arm: (ROOT / P.SOURCE_DIR / f"{arm}.yaml").read_bytes() for arm in P.ARMS}

    def write_model(self, model):
        self.model_path.write_text(json.dumps(model, allow_nan=False), encoding="utf-8")

    def call(self, **kwargs):
        return P.prepare(model_path=self.model_rel, cfg_dir=self.cfg_dir, tag="PILOT_TEST_DATA", **kwargs)

    def test_positive_temporary_fixture_changes_only_model_and_legacy_host(self):
        report = self.call()
        destination = ROOT / self.cfg_dir
        self.assertEqual(report["status"], "PREPARED_NOT_RUN")
        self.assertTrue(report["matrix_dryplan_ok"])
        self.assertEqual(report["plan"]["planned_runs"],
                         ["baseline#r0", "cost_single#r0", "cost_dual#r0"])
        self.assertFalse(report["pilot_executed"])
        self.assertFalse(report["remote_environment_verified"])
        self.assertEqual((destination / "static_small.yaml").read_bytes(), self.source_bytes["static_small"])
        for arm in P.ARMS[1:]:
            expected = yaml.safe_load(self.source_bytes[arm])
            expected["controller"]["cost_model_path"] = self.model_rel
            expected["controller"].pop("host_overhead_ms", None)
            expected["controller"].pop("host_overhead_fit", None)
            got = yaml.safe_load((destination / f"{arm}.yaml").read_text(encoding="utf-8"))
            self.assertEqual(got, expected)
        for arm in P.ARMS:
            self.assertEqual((ROOT / P.SOURCE_DIR / f"{arm}.yaml").read_bytes(), self.source_bytes[arm])

    def test_refuses_test_data_failed_unvalidated_models_before_output(self):
        for change in (lambda m: m.update(data_label="TEST_DATA"),
                       lambda m: m["validation"].update(model_holds=False),
                       lambda m: m.pop("validation"),
                       lambda m: m.pop("execution_shape_table")):
            with self.subTest(change=change):
                model = synthetic_model()
                change(model)
                self.write_model(model)
                with self.assertRaises(ValueError):
                    self.call()
                self.assertFalse((ROOT / self.cfg_dir).exists())

    def test_refuses_missing_or_reused_identity(self):
        for change in (lambda m: m.pop("source"),
                       lambda m: m["source"].pop("tree_digest"),
                       lambda m: m["source"].pop("execution_code_digest"),
                       lambda m: m.update(execution_code_digest="f" * 64),
                       lambda m: m["fingerprint"]["software"].pop("torch_musa"),
                       lambda m: m["inputs"].pop("frozen_estimator_sha256"),
                       lambda m: m["inputs"].update(validation=m["inputs"]["training"]),
                       lambda m: m["inputs"].update(validation=m["inputs"]["validation"][:1])):
            with self.subTest(change=change):
                model = synthetic_model()
                change(model)
                self.write_model(model)
                with self.assertRaises(ValueError):
                    self.call()
                self.assertFalse((ROOT / self.cfg_dir).exists())

    def test_refuses_signature_and_timing_mismatch(self):
        for change in (lambda m: m["execution_signature"].update(sample_needed_only=False),
                       lambda m: m.update(execution_timing="per_forward_times_n"),
                       lambda m: m["fingerprint"]["model"].update(revision="b" * 40)):
            with self.subTest(change=change):
                model = synthetic_model()
                change(model)
                self.write_model(model)
                with self.assertRaises(ValueError):
                    self.call()

    def test_rejects_invalid_cost_records(self):
        for value in (0, -1, True, None):
            with self.subTest(value=value):
                model = synthetic_model()
                next(iter(model["execution_shape_table"].values()))["fwd_ms"] = value
                self.write_model(model)
                with self.assertRaises(ValueError):
                    self.call()

    def test_refuses_existing_directory_without_writing(self):
        destination = ROOT / self.cfg_dir
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.call()
        self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_protected_and_escape_paths_are_rejected(self):
        for value in (P.SOURCE_DIR, f"{P.SOURCE_DIR}/new", "../outside", "/absolute",
                      "C:/absolute", "results/raw/new", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.prepare(model_path=self.model_rel, cfg_dir=value)

    def test_failed_real_matrix_check_never_marks_ready(self):
        with patch.object(P.pilot, "cross_check_with_matrix", return_value=(False, "TEST_DATA mismatch")):
            with self.assertRaisesRegex(ValueError, "dry-plan failed"):
                self.call()
        destination = ROOT / self.cfg_dir
        self.assertFalse((destination / "preparation.json").exists())
        failed = json.loads((destination / "preparation_failed.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["status"], "FAILED_NOT_READY")
        self.assertFalse(failed["matrix_dryplan_ok"])

    def test_execution_code_change_during_preparation_never_marks_ready(self):
        digest = synthetic_model()["execution_code_digest"]
        with patch.object(P, "compute_tree_digest", side_effect=[digest, "f" * 64]), \
                patch.object(P.pilot, "cross_check_with_matrix", return_value=(True, "TEST_DATA 3 items")):
            with self.assertRaisesRegex(ValueError, "code changed during preparation"):
                self.call()
        self.assertFalse((ROOT / self.cfg_dir / "preparation.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
