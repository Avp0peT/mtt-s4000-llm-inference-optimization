"""TEST_DATA: bind a prediction to one actual next step, never a whole cycle."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "benchmark", "tests"):
    sys.path.insert(0, str(ROOT / part))

import real_backend as rb  # noqa: E402
from ascend_opt.execution_shape import plan_execution_shape  # noqa: E402
from ascend_opt.step_planner import plan_step  # noqa: E402
from ascend_opt.workload import Request  # noqa: E402
from test_real_backend import _cfg  # noqa: E402


def _decision(**kw):
    """An explicit TEST_DATA prediction; production CostPolicy is tested elsewhere."""
    plan = plan_step(scheduler=kw["scheduler"], chunk_policy=kw["chunk_policy"],
                     cache=kw["cache"], waiting=[s.req for s in kw["waiting"]],
                     decoding=[s.req for s in kw["decoding"]],
                     prefilling=[(s.req, s.prefill_remaining) for s in kw["prefilling"]],
                     cur_chunk=kw["cur_chunk"], cur_max_tokens=kw["cur_max_tokens"],
                     max_num_seqs=kw["max_num_seqs"], chunk_enabled=kw["chunk_enabled"])
    shape = plan_execution_shape(
        plan, waiting=[s.req for s in kw["waiting"]], decoding=[s.req for s in kw["decoding"]],
        prefilling=[(s.req, s.prefill_remaining) for s in kw["prefilling"]],
        per_request_kv_lengths={s.req.req_id: s.req.prompt_len + s.req.gen_len - s.decode_remaining - 1
                                for s in kw["decoding"]}).describe()
    candidate = {"chunk": kw["cur_chunk"], "max_tokens": kw["cur_max_tokens"],
                 "predicted_ms": 1.0, "host_ms": 0.5, "execution_shape": shape}
    entry = {"step": kw["step"], "cost_policy": {
        "changed": True, "chunk": kw["cur_chunk"], "max_tokens": kw["cur_max_tokens"],
        "fallback_reason": None, "candidates": [candidate]}}
    return kw["cur_chunk"], kw["cur_max_tokens"], entry


def _run(*, new_arrival=False, shadow=False, fallback=False, disable_observation=False,
         fail_on_step=None):
    cfg = _cfg()
    cfg = replace(cfg, run=replace(cfg.run, warmup_s=0, control_interval_steps=4),
                  controller=replace(cfg.controller, enabled=True, kind="cost", shadow=shadow))
    requests = [Request(req_id="r0", prompt_len=8, gen_len=3, arrival_ms=0)]
    if new_arrival:
        # Arrives during step 0, after the queue snapshot used by its decision.
        requests.append(Request(req_id="r1", prompt_len=4, gen_len=1, arrival_ms=2))
    clock = rb.VirtualClock()
    driver = rb.FakeDriver(clock=clock, fixed_ms=0., per_seq_ms=0., per_token_ms=1.,
                           fail_on_step=fail_on_step)

    def decide(**kw):
        chunk, tokens, entry = _decision(**kw)
        if fallback:
            entry["cost_policy"].update(changed=False, fallback_reason="TEST_DATA missing_model")
        return chunk, tokens, entry

    pending = rb._pending_execution_prediction
    with patch.object(rb, "build_arm", return_value=object()), \
            patch.object(rb, "apply_arm", side_effect=decide), \
            patch.object(rb, "_pending_execution_prediction",
                         side_effect=(lambda *a, **k: None) if disable_observation else pending):
        result = rb.RealBackend(cfg, driver=driver, clock=clock.now,
                                sleep=clock.advance, allow_non_hardware_driver=True).run(requests)
    rows = [entry["prediction_execution_match"] for entry in result.controller_trace
            if "prediction_execution_match" in entry]
    return result, driver, rows


class TestNextStepPredictionMatching(unittest.TestCase):
    def test_matching_prediction_is_consumed_only_by_immediate_next_step(self):
        result, drv, rows = _run()
        self.assertEqual(len(drv.calls), 3)
        self.assertEqual(len(rows), 1, "control period 4 must not reuse prediction on step 2")
        row = rows[0]
        self.assertEqual((row["decision_step"], row["target_step"], row["actual_step"]), (0, 1, 1))
        self.assertTrue(row["shape_match"])
        self.assertTrue(row["n_forwards_match"])
        self.assertEqual(row["actual_shape"]["items"][0]["kv_len"], 8)
        self.assertEqual(row["actual_timing"]["fwd_ms"], 1.)
        self.assertFalse(row["comparison_eligible"], "FakeDriver never becomes measured model error")
        self.assertIn("nonhardware_execution", row["ineligible_reasons"])

    def test_new_arrival_changes_shape_and_cannot_be_used_as_prediction_error(self):
        _, _, rows = _run(new_arrival=True)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["shape_match"])
        self.assertFalse(row["comparison_eligible"])
        self.assertIn("execution_shape_changed", row["ineligible_reasons"])
        self.assertEqual([i["req_id"] for i in row["predicted_shape"]["items"]], ["r0"])
        self.assertEqual([i["req_id"] for i in row["actual_shape"]["items"]], ["r1", "r0"])

    def test_fallback_and_shadow_emit_no_invented_predictions(self):
        for args in ({"fallback": True}, {"shadow": True}):
            with self.subTest(args=args):
                self.assertEqual(_run(**args)[2], [])

    def test_failed_next_forward_is_explicitly_ineligible(self):
        result, _, rows = _run(fail_on_step=2)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["actual_timing"])
        self.assertFalse(rows[0]["comparison_eligible"])
        self.assertIn("driver_failed", rows[0]["ineligible_reasons"])

    def test_observation_does_not_change_schedule_or_request_metrics(self):
        observed, drv_a, _ = _run(new_arrival=True)
        unobserved, drv_b, _ = _run(new_arrival=True, disable_observation=True)
        self.assertEqual(drv_a.calls, drv_b.calls)
        self.assertEqual(observed.metrics, unobserved.metrics)
        self.assertEqual(observed.wall_ms, unobserved.wall_ms)

    def test_no_selected_candidate_or_no_prediction_is_not_pending(self):
        for candidate in ({"predicted_ms": None}, {"execution_shape": None}):
            entry = {"cost_policy": {"changed": True, "chunk": 1, "max_tokens": 1,
                      "candidates": [{"chunk": 1, "max_tokens": 1, **candidate}]}}
            self.assertIsNone(rb._pending_execution_prediction(entry, decision_step=0, shadow=False))
        self.assertIsNone(rb._pending_execution_prediction(None, decision_step=0, shadow=False))


class TestObservationEligibility(unittest.TestCase):
    """Synthetic guard inputs, explicitly distinct from actual hardware tests."""
    def test_host_matches_full_driver_elapsed_contract_and_group_count_is_checked(self):
        row = _run()[2][0]
        pending = {k: row[k] for k in ("decision_step", "target_step", "predicted_shape",
                                       "predicted_fwd_ms", "predicted_host_ms")}
        timing = rb.StepTiming(n_seqs=1, n_tokens=1, n_forwards=1, fwd_ms=6.,
                               pack_ms=1., readback_ms=1., started_s=0., finished_s=.01)
        out = rb._prediction_execution_observation(pending, actual_step=1,
            actual_shape=row["actual_shape"], timing=timing, hardware_backed=True)
        self.assertTrue(out["comparison_eligible"], "TEST_DATA eligible guard case only")
        self.assertEqual(out["actual_timing"]["host_ms"], 4.)
        self.assertEqual(out["actual_timing"]["unmodeled_driver_ms"], 2.)
        mismatch = rb._prediction_execution_observation(pending, actual_step=1,
            actual_shape=row["actual_shape"], timing=replace(timing, n_forwards=2), hardware_backed=True)
        self.assertFalse(mismatch["comparison_eligible"])
        self.assertIn("forward_group_count_unconfirmed_or_changed", mismatch["ineligible_reasons"])
        expired = rb._prediction_execution_observation(pending, actual_step=2,
            actual_shape=row["actual_shape"], timing=timing, hardware_backed=True)
        self.assertFalse(expired["comparison_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
