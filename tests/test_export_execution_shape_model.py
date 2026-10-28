"""Synthetic contract tests only; never produce a real model or GPU evidence."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import execution_shape_probe as P
import export_execution_shape_model as E


def artifact_fixture():
    """Mock device declarations exercise validators; not measured evidence."""
    manifest = P.checked_manifest(ROOT / "results/processed/execution_shape_manifest_20261003.json")
    signature = manifest["entries"][0]["execution_signature"]
    fingerprint = {"hardware": {"device_model": "MTT S4000", "driver_version": "mock"},
                   "software": {"torch": "mock", "torch_musa": "mock", "transformers": "mock", "python": "mock"},
                   "model": {"id": signature["model"], "revision": signature["model_revision"]},
                   "precision": signature["precision"]}
    runtime = {"hardware_backed": True, "actual_device_name": "MTT S4000",
               "driver": {"kind": "transformers", "device": "musa:0", "model_revision_resolved": signature["model_revision"]},
               "fingerprint": fingerprint}
    return {"_TEST_FIXTURE_ONLY": True, "schema": P.SCHEMA, "complete": True,
            "data_label": "MEASURED", "split": "training", "execution_timing": P.TIMING,
            "session_id": "TEST_SESSION", "process_id": 1, "started_utc": "2026-10-03T00:00:00+00:00",
            "runtime": runtime, "runtime_end": runtime, "execution_signature": signature,
            "source": {"git_commit": "test", "tree_digest": "a"*64, "execution_code_digest": "b"*64},
            "environment": {"hostname": "test", "fingerprint_matches_host": True,
                            "fingerprint_sha256": "c"*64, "fingerprint": {"available": True}},
            "configs": {e["config_path"]: {"text": "TEST_DATA", "sha256": hashlib.sha256(b"TEST_DATA").hexdigest()}
                        for e in manifest["entries"]},
            "manifest": manifest, "manifest_sha256": "d"*64,
            "points": [{"execution_shape_key": e["execution_shape_key"], "execution_shape": e["execution_shape"],
                        "warmup": {"excluded": True, "calls": 1, "seconds": e["warmup_s"]},
                        "records": [{"record_index": n, "fwd_ms": 90., "host_ms": 10., "total_ms": 100.,
                                     "driver_elapsed_ms": 100., "n_forwards": e["execution_shape"]["n_forwards"]}
                                    for n in range(30)]} for e in manifest["entries"]]}


class ArtifactContractTests(unittest.TestCase):
    def test_valid_mock_structure_only(self):
        identity, medians = E.validate_artifact(artifact_fixture(), "training")
        self.assertEqual(6, len(medians))
        self.assertEqual({100.}, {row["total_ms"] for row in medians.values()})
        # Only in-memory validator output; no freeze/export call or file output.

    def reject(self, mutation):
        artifact = artifact_fixture()
        mutation(artifact)
        with self.assertRaises((ValueError, KeyError)):
            E.validate_artifact(artifact, "training")

    def test_test_data_rejected(self):
        self.reject(lambda a: a.update(data_label="TEST_DATA"))

    def test_cpu_rejected(self):
        self.reject(lambda a: a["runtime"].update(hardware_backed=False))

    def test_incomplete_rejected(self):
        self.reject(lambda a: a.update(complete=False))

    def test_missing_shape_rejected(self):
        self.reject(lambda a: a["points"].pop())

    def test_missing_record_rejected(self):
        self.reject(lambda a: a["points"][0]["records"].pop())

    def test_duplicate_record_rejected(self):
        self.reject(lambda a: a["points"][0]["records"][1].update(record_index=0))

    def test_nan_rejected(self):
        self.reject(lambda a: a["points"][0]["records"][0].update(fwd_ms=float("nan")))

    def test_host_subsum_instead_of_elapsed_rejected(self):
        self.reject(lambda a: a["points"][0]["records"][0].update(driver_elapsed_ms=101.))

    def test_source_missing_rejected(self):
        self.reject(lambda a: a["source"].pop("execution_code_digest"))

    def test_unknown_fingerprint_rejected(self):
        self.reject(lambda a: a["runtime"]["fingerprint"]["software"].update(torch=None))

    def test_warmup_not_excluded_rejected(self):
        self.reject(lambda a: a["points"][0]["warmup"].update(excluded=False))

    def test_wrong_revision_rejected(self):
        self.reject(lambda a: a["runtime"]["driver"].update(model_revision_resolved="other"))

    def test_too_few_sessions_rejected(self):
        with self.assertRaisesRegex(ValueError, "three"):
            E.read_sessions([], "training")

    def test_same_artifact_cannot_supply_three_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock_contract_only.json"
            E.write_new(path, artifact_fixture())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                E.read_sessions([path, path, path], "training")

    def test_new_session_names_do_not_make_one_pid_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for n in range(3):
                artifact = artifact_fixture()
                artifact["session_id"] = f"TEST_NAME_{n}"
                artifact["started_utc"] = f"2026-10-03T00:00:0{n}+00:00"
                path = Path(directory) / f"mock_contract_only_{n}.json"
                E.write_new(path, artifact)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                E.read_sessions(paths, "training")

    def test_cli_test_data_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path, out = Path(directory) / "cpu.json", Path(directory) / "frozen.json"
            fixture = artifact_fixture()
            fixture["data_label"] = "TEST_DATA"
            E.write_new(path, fixture)
            with patch("builtins.print"):
                self.assertEqual(2, E.main(["--freeze", str(path), str(path), str(path), "--out", str(out)]))
            self.assertFalse(out.exists())

    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                E.write_new(path, {"data_label": "TEST_DATA"})
            self.assertEqual("old", path.read_text())


class EstimatorTests(unittest.TestCase):
    def test_components_stay_paired_even_median(self):
        records = [{"fwd_ms": 1., "host_ms": 8., "total_ms": 9.},
                   {"fwd_ms": 9., "host_ms": 2., "total_ms": 11.}]
        row = E.paired_median(records)
        self.assertEqual({"fwd_ms": 5., "host_ms": 5., "total_ms": 10.}, row)

    def candidate(self, predicted=100., actual=100.):
        frozen = {"data_label": "TEST_DATA", "execution_shape_table": {"test": {"fwd_ms": predicted*.9, "host_ms": predicted*.1}},
                  "training_session_summaries": [{"session_id": f"TEST_T{i}", "medians": {"test": {"total_ms": predicted}}} for i in range(3)]}
        validation = [{"session_id": f"TEST_V{i}", "medians": {"test": {"total_ms": actual}}} for i in range(3)]
        return frozen, validation

    def test_equal_cost_passes_in_memory_test_builder(self):
        frozen, rows = self.candidate()
        report = E.evaluate(frozen, rows)
        self.assertTrue(report["model_holds"])
        self.assertEqual(3, report["independent_validation_processes"])
        self.assertEqual("TEST_DATA", frozen["data_label"])

    def test_bad_model_fails_without_retuning(self):
        frozen, rows = self.candidate(100., 150.)
        before = copy.deepcopy(frozen)
        report = E.evaluate(frozen, rows)
        self.assertFalse(report["model_holds"])
        self.assertEqual(before, frozen)

    def test_large_overestimate_fails_absolute_gate(self):
        frozen, rows = self.candidate(200., 100.)
        self.assertFalse(E.evaluate(frozen, rows)["model_holds"])


if __name__ == "__main__":
    unittest.main()
