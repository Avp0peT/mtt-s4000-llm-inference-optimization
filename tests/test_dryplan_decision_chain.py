"""Offline fixture and production-shape manifest contracts (no device access)."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dryplan_decision_chain.py"
spec = importlib.util.spec_from_file_location("dryplan_under_test", SCRIPT)
dryplan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dryplan)


class TestDerivedConfigCLI(unittest.TestCase):
    def test_uses_explicit_derived_config_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "configs", prefix="TEST_DATA_replay_") as tmp:
            directory = Path(tmp)
            # Unique arm name is absent from the old default directory.
            (directory / "derived_only.yaml").write_bytes((dryplan.CFG_DIR / "cost_single.yaml").read_bytes())
            out = directory / "replay.json"
            r = subprocess.run([sys.executable, str(SCRIPT), "--cfg-dir", str(directory),
                                "--arms", "derived_only", "--json", str(out)], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               env={**os.environ, "PYTHONUTF8": "1"})
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["arms"][0]["arm_name"], "derived_only")
            self.assertIn("不可判", r.stdout)


class TestQueueFixtures(unittest.TestCase):
    def setUp(self):
        self.reqs = dryplan._make_requests()

    def snapshot(self, name, **updates):
        st = copy.deepcopy(next(s for s in dryplan.STATES if s["name"] == name))
        st.update(updates)
        return dryplan._snapshot_for(st, self.reqs, max_num_seqs=8)

    def test_partitions_are_disjoint_and_exhaust_arrivals(self):
        for st in dryplan.STATES:
            with self.subTest(state=st["name"]):
                s = dryplan._snapshot_for(st, self.reqs, max_num_seqs=8)
                phases = [[r.req_id for r in s[k]] for k in
                          ("waiting", "decoding", "completed")]
                phases.append([r.req_id for r, _ in s["prefilling"]])
                all_ids = sum(phases, [])
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), {r.req_id for r in s["arrived"]})
                self.assertLessEqual(len(s["decoding"]) + len(s["prefilling"]), 8)

    def test_regression_mixed_does_not_decode_waiting_requests(self):
        s = self.snapshot("mixed_admit")
        self.assertEqual([r.req_id for r in s["decoding"]], [f"r{i:02d}" for i in range(8)])
        self.assertEqual([r.req_id for r in s["waiting"]], [f"r{i:02d}" for i in range(8, 16)])

    def test_regression_steady_decodes_arrived_first_eight(self):
        s = self.snapshot("steady_decode")
        self.assertFalse(s["waiting"])
        self.assertEqual([r.req_id for r in s["decoding"]], [f"r{i:02d}" for i in range(8)])
        self.assertTrue(all(r.arrival_ms <= s["now_ms"] for r in s["decoding"]))

    def test_progress_records_cache_before_next_forward(self):
        s = self.snapshot("contended_prefill", generated_tokens={"r00": 1, "r01": 7})
        self.assertEqual(s["kv_lengths"]["r00"], 1536)
        self.assertEqual(s["kv_lengths"]["r01"], 1542)
        self.assertEqual(s["kv_lengths"]["r04"], 768)
        self.assertEqual(s["kv_lengths"]["r06"], 0)
        self.assertEqual(s["progress"]["r04"]["generated_tokens"], 0)

    def test_completed_requests_do_not_consume_active_slots(self):
        s = self.snapshot("contended_prefill", completed=[6, 7])
        self.assertEqual(len(s["waiting"]), 8)
        self.assertEqual(s["progress"]["r06"]["generated_tokens"], 32)
        self.assertNotIn("r06", s["kv_lengths"])

    def test_reject_overlapping_states(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.snapshot("contended_prefill", prefilling=[0, 5])

    def test_reject_not_arrived_active_request(self):
        with self.assertRaisesRegex(ValueError, "not arrived"):
            self.snapshot("steady_decode", decoding_indices=list(range(8, 16)))

    def test_reject_sequence_overcommit(self):
        with self.assertRaisesRegex(ValueError, "max_num_seqs"):
            self.snapshot("mixed_admit", prefilling=[8])

    def test_reject_impossible_partial_progress(self):
        for remaining in (0, 1536, -1):
            with self.subTest(remaining=remaining), self.assertRaisesRegex(ValueError, "remaining"):
                self.snapshot("contended_prefill", prefill_remaining={4: remaining})

    def test_reject_invalid_generated_tokens(self):
        for generated in (0, 32, -1):
            with self.subTest(generated=generated), self.assertRaisesRegex(ValueError, "generated"):
                self.snapshot("steady_decode", generated_tokens={"r00": generated})

    def test_reject_arrival_time_mismatch(self):
        with self.assertRaisesRegex(ValueError, "arrival timestamps"):
            self.snapshot("steady_decode", now_ms=15)

    def test_reject_duplicate_request_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            dryplan._snapshot_for(dryplan.STATES[0], self.reqs + [self.reqs[0]], max_num_seqs=8)

    def test_fixture_runs_do_not_inherit_controller_state(self):
        cfg = dryplan.CFG_DIR / "cost_dual.yaml"
        full = dryplan.run_arm(cfg)
        alone = dryplan.run_arm(cfg, state_names=["steady_decode"])
        self.assertEqual(next(s for s in full["states"] if s["state"] == "steady_decode"),
                         alone["states"][0])

    def test_fixture_refuses_changed_workload_instead_of_silently_using_plan(self):
        cfg = dryplan.load_config(dryplan.CFG_DIR / "cost_single.yaml")
        raw = copy.deepcopy(cfg.raw)
        raw["workload"]["prompt_len_min"] = 2048
        with self.assertRaisesRegex(ValueError, "differs from config"):
            dryplan._validate_plan_cell(SimpleNamespace(raw=raw))


class TestDryplanRealChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.artifact = Path(cls.tmp.name) / "dryplan.json"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        cls.process = subprocess.run(
            [sys.executable, str(SCRIPT), "--json", str(cls.artifact)], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        if cls.process.returncode:
            raise AssertionError(cls.process.stderr[-4000:])
        cls.data = json.loads(cls.artifact.read_text(encoding="utf-8"),
                              parse_constant=lambda token: (_ for _ in ()).throw(
                                  ValueError(f"non-standard JSON {token}")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_explicit_evidence_boundary_and_lf(self):
        self.assertEqual(self.data["evidence_class"], "TEST_DATA")
        self.assertNotIn(b"\r", self.artifact.read_bytes())
        for arm in self.data["arms"]:
            for st in arm["states"]:
                self.assertEqual(st["state_source"], "independent_fixture")
                self.assertIsNone(st["reachable_from_production"])
                self.assertEqual(st["evidence_class"], "TEST_DATA")

    def test_budgets_come_from_configs_and_only_prefill_differs(self):
        by = {a["arm_name"]: a for a in self.data["arms"]}
        single, dual = (by[n]["controller"] for n in ("cost_single", "cost_dual"))
        for name, arm in by.items():
            cfg = dryplan.load_config(dryplan.CFG_DIR / f"{name}.yaml")
            self.assertEqual(arm["controller"]["step_deadline_ms"], cfg.controller.target_tpot_ms)
        self.assertIsNone(single["prefill_deadline_ms"])
        self.assertIsNotNone(dual["prefill_deadline_ms"])
        self.assertEqual({k: v for k, v in single.items() if k != "prefill_deadline_ms"},
                         {k: v for k, v in dual.items() if k != "prefill_deadline_ms"})

    def test_candidates_record_decisions_and_execution_shape(self):
        for arm in self.data["arms"]:
            for st in arm["states"]:
                self.assertTrue(st["candidates"])
                for candidate in st["candidates"]:
                    for field in ("predicted_forward_ms", "budget_kind", "budget_ms",
                                  "feasible", "reject_reason", "execution_shape"):
                        self.assertIn(field, candidate)
                    self.assertEqual(bool(candidate["reject_reason"]), not candidate["feasible"])

    def test_coverage_manifest_not_fixed_nine_points(self):
        gap = self.data["coverage_gap"]
        self.assertEqual(gap["origin"], "actual_candidate_execution_groups")
        self.assertIn("input_bug_corrections", gap)
        self.assertTrue(gap["entries"])
        self.assertEqual(gap["unique_execution_shapes"], len(gap["entries"]))
        for row in gap["entries"]:
            self.assertTrue(row["sources"])
            self.assertTrue(row["execution_groups"])
            self.assertIn("independent_validation", row)
            self.assertIn("S01", row["execution_semantics"])
        self.assertNotIn("n_new_points_estimate", gap)

    def test_manifest_sources_are_exactly_the_candidate_shapes(self):
        gap = self.data["coverage_gap"]
        candidates = {(arm["arm_name"], state["state"], index): candidate
                      for arm in self.data["arms"] for state in arm["states"]
                      for index, candidate in enumerate(state["candidates"])}
        emitted_keys = set()
        for entry in gap["entries"]:
            emitted_keys.add(entry["execution_shape_key"])
            self.assertEqual(entry["execution_shape_key"],
                             dryplan.execution_shape_key(entry["execution_shape"]))
            for source in entry["sources"]:
                candidate = candidates[(source["arm"], source["state"], source["candidate_index"])]
                self.assertEqual(entry["execution_shape_key"],
                                 dryplan.execution_shape_key(candidate["execution_shape"]))
        self.assertEqual(emitted_keys, {dryplan.execution_shape_key(c["execution_shape"])
                                        for c in candidates.values()})

    def test_regression_eight_admitted_is_one_forward_item_and_kv_stays_zero(self):
        state = self.data["arms"][0]["states"][0]
        candidate = next(c for c in state["candidates"]
                         if c["chunk"] == 512 and c["max_tokens"] == 32768)
        self.assertEqual(candidate["legacy_batch"], 8)
        self.assertEqual(candidate["batch"], 1)
        self.assertEqual(candidate["execution_shape"]["items"], [{
            "req_id": "r00", "n_new_tokens": 512, "kv_len": 0,
            "role": "prefill", "need_token": False}])
        self.assertEqual(candidate["coverage_status"], "execution_contract_unvalidated")

    def test_already_covered_shapes_do_not_require_unnecessary_calibration(self):
        arms = copy.deepcopy(self.data["arms"])
        for arm in arms:
            for state in arm["states"]:
                for candidate in state["candidates"]:
                    candidate["coverage_status"] = "exact_execution_key"
                    candidate["coverage_reason"] = "TEST_DATA: simulated coverage only"
        manifest = dryplan._coverage_gap_analysis(arms)
        self.assertTrue(manifest["entries"])
        self.assertTrue(all(not entry["requires_new_calibration"] for entry in manifest["entries"]))
        self.assertTrue(all(entry["sampling_purpose"] == "optional_identity_anchor"
                            for entry in manifest["entries"]))

    def test_reserved_legacy_output_cannot_be_overwritten(self):
        old = ROOT / "results/processed/dryplan_decision_chain.json"
        before = old.read_bytes() if old.exists() else None
        with self.assertRaisesRegex(ValueError, "new versioned path"):
            dryplan._write_json(str(old), self.data)
        self.assertEqual(old.read_bytes() if old.exists() else None, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
