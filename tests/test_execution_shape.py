"""Production execution shape versus independently specified requests and CPU tensors.

All constructed workloads are TEST_DATA. The CPU probe checks the production
TransformerDriver, but supplies no MUSA performance or hardware evidence.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "benchmark", "tests"):
    if str(_ROOT / _part) not in sys.path:
        sys.path.insert(0, str(_ROOT / _part))

from ascend_opt.execution_shape import (  # noqa: E402
    execution_shape_key, group_by_new_tokens, plan_execution_shape, plan_forward_specs,
)
from ascend_opt.step_planner import StepPlan, plan_step  # noqa: E402
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
from ascend_opt.chunk_policy import ChunkConfig, FixedChunkPolicy  # noqa: E402
from ascend_opt.cache_policy import PrefixCacheConfig, PrefixCachePolicy  # noqa: E402
from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.workload import Request  # noqa: E402
from real_backend import ForwardItem, TransformerDriver  # noqa: E402


def _req(name: str, prompt: int) -> Request:
    return Request(req_id=name, prompt_len=prompt, gen_len=8)


def _vectors(shape):
    return [(i.req_id, i.n_new_tokens, i.kv_len, i.role, i.need_token)
            for i in shape.items]


def _initial_plan(waiting, chunk):
    return plan_step(scheduler=ContinuousBatchingScheduler(),
                     chunk_policy=FixedChunkPolicy(ChunkConfig(enabled=True)),
                     cache=PrefixCachePolicy(PrefixCacheConfig(enabled=False)),
                     waiting=waiting, decoding=[], prefilling=[],
                     cur_chunk=chunk, cur_max_tokens=32768,
                     max_num_seqs=8, chunk_enabled=True)


class TestExecutionShapeContract(unittest.TestCase):
    def test_eight_admitted_but_only_one_executes(self):
        waiting = [_req(f"r{i}", 1536) for i in range(8)]
        plan = _initial_plan(waiting, 512)
        self.assertEqual(plan.prefill_alloc, {"r0": 512})
        shape = plan_execution_shape(plan, waiting=waiting, decoding=[], prefilling=[])
        self.assertEqual(len(plan.new_start_ids), 8)
        self.assertEqual(_vectors(shape), [("r0", 512, 0, "prefill", False)])
        self.assertEqual(shape.describe()["batch"], 1)
        self.assertEqual(shape.describe()["n_forwards"], 1)

    def test_unequal_prefills_are_not_one_prefill_plus_decode(self):
        waiting = [_req(f"r{i}", 1536) for i in range(8)]
        plan = _initial_plan(waiting, 2048)
        self.assertEqual(plan.prefill_alloc, {"r0": 1536, "r1": 512})
        shape = plan_execution_shape(plan, waiting=waiting, decoding=[], prefilling=[])
        self.assertEqual(_vectors(shape), [("r0", 1536, 0, "prefill", True),
                                          ("r1", 512, 0, "prefill", False)])
        groups = shape.describe()["groups"]
        self.assertEqual([(g["n_new_tokens"], g["batch"]) for g in groups],
                         [(1536, 1), (512, 1)])

    def test_continued_prefill_is_counted_without_new_admissions(self):
        cont = _req("cont", 12)
        plan = StepPlan(prefill_alloc={"cont": 4}, tokens_prefilled=4)
        shape = plan_execution_shape(plan, waiting=[], decoding=[], prefilling=[(cont, 4)])
        self.assertEqual(_vectors(shape), [("cont", 4, 8, "prefill", True)])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            plan_execution_shape(plan, waiting=[], decoding=[], prefilling=[(cont, 4)],
                                 per_request_kv_lengths={"cont": 9})

    def test_unknown_decode_kv_is_not_filled_with_zero(self):
        dec = _req("decode", 12)
        plan = StepPlan(decode_ids=("decode",))
        unknown = plan_execution_shape(plan, waiting=[], decoding=[dec], prefilling=[])
        self.assertEqual(_vectors(unknown), [("decode", 1, None, "decode", True)])
        self.assertIsNone(unknown.describe()["groups"][0]["past_max"])
        known = plan_execution_shape(plan, waiting=[], decoding=[dec], prefilling=[],
                                     per_request_kv_lengths={"decode": 15})
        self.assertEqual(_vectors(known), [("decode", 1, 15, "decode", True)])
        self.assertNotEqual(unknown.key(), known.key())
        pred, reason, _ = CostModel(payload={}).predict_execution(
            unknown.describe(), execution_signature={})
        self.assertIsNone(pred)
        self.assertEqual(reason, "unknown_kv")

    def test_prefix_cached_start_uses_cache_length_and_immediate_is_excluded(self):
        a, b = _req("partial", 8), _req("immediate", 8)
        plan = StepPlan(new_start_ids=(a.req_id, b.req_id), immediate_ids=(b.req_id,),
                        cached_tokens={a.req_id: 4, b.req_id: 8},
                        prefill_alloc={a.req_id: 4}, tokens_prefilled=4)
        shape = plan_execution_shape(plan, waiting=[a, b], decoding=[], prefilling=[])
        self.assertEqual(_vectors(shape), [("partial", 4, 4, "prefill", True)])

    def test_invalid_queue_overlap_and_negative_kv_are_rejected(self):
        req = _req("same", 8)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            plan_forward_specs(StepPlan(), waiting=[req], decoding=[req], prefilling=[])
        for kv in (-1, True, 1.5):
            with self.subTest(kv=kv), self.assertRaisesRegex(ValueError, "invalid KV"):
                plan_execution_shape(StepPlan(decode_ids=(req.req_id,)), waiting=[],
                                     decoding=[req], prefilling=[],
                                     per_request_kv_lengths={req.req_id: kv})

    def test_overallocation_is_rejected(self):
        req = _req("partial", 8)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            plan_forward_specs(StepPlan(prefill_alloc={req.req_id: 5}), waiting=[],
                               decoding=[], prefilling=[(req, 4)])

    def test_grouping_is_descending_and_stable_within_group(self):
        items = [ForwardItem(_req(name, 8), n, want)
                 for name, n, want in (("a", 1, True), ("b", 4, False),
                                       ("c", 4, True), ("d", 2, False))]
        self.assertEqual([(n, [i.req.req_id for i in group])
                          for n, group in group_by_new_tokens(items)],
                         [(4, ["b", "c"]), (2, ["d"]), (1, ["a"])])
        self.assertEqual(TransformerDriver._shape_groups(items), group_by_new_tokens(items))

    def test_transformer_driver_calls_the_shared_grouping_function(self):
        import real_backend as backend
        items = [ForwardItem(_req("a", 4), 4, True), ForwardItem(_req("b", 2), 2, True)]
        with patch.object(backend, "group_by_new_tokens", wraps=group_by_new_tokens) as shared:
            self.assertEqual([n for n, _ in TransformerDriver._shape_groups(items)], [4, 2])
        shared.assert_called_once_with(items)

    def test_real_backend_passes_shared_plan_specs_to_driver(self):
        """Exercise the real scheduling loop with a labelled nonhardware driver."""
        import real_backend as backend
        from run_benchmark import load_config

        cfg = load_config(_ROOT / "configs/mthreads/exec_opt/s01_both.yaml")
        cfg = replace(cfg, run=replace(cfg.run, warmup_s=0))
        clock = backend.VirtualClock()
        drv = backend.FakeDriver(clock=clock)
        runner = backend.RealBackend(cfg, driver=drv, clock=clock.now,
                                     sleep=lambda _s: None, allow_non_hardware_driver=True)
        planned = []
        sent = []

        def shared_specs(*args, **kwargs):
            specs = plan_forward_specs(*args, **kwargs)
            planned.append([(s.req.req_id, s.n_new_tokens, s.need_token) for s in specs])
            return specs

        original_forward = drv.forward

        def forward(items):
            sent.append([(i.req.req_id, i.n_new_tokens, i.need_token) for i in items])
            return original_forward(items)

        with patch.object(backend, "plan_forward_specs", side_effect=shared_specs) as shared, \
                patch.object(drv, "forward", side_effect=forward):
            result = runner.run([_req("a", 6), _req("b", 8)])
        self.assertFalse(drv.hardware_backed)
        self.assertGreater(shared.call_count, 0)
        self.assertEqual(sent, [step for step in planned if step])
        self.assertEqual(len(result.metrics), 2)

    def test_exact_key_excludes_ids_but_preserves_per_row_semantics(self):
        def shape(rid, *, kv=0, need=True):
            return {"items": [{"req_id": rid, "n_new_tokens": 1,
                               "kv_len": kv, "role": "prefill", "need_token": need}]}
        key = execution_shape_key(shape("a"))
        self.assertEqual(key, execution_shape_key(shape("other")))
        self.assertNotEqual(key, execution_shape_key(shape("a", kv=None)))
        self.assertNotEqual(key, execution_shape_key(shape("a", need=False)))


def run_cpu_shape_probe():
    """Manual expected vectors/tensors versus a real CPU model's actual calls."""
    import _tiny_support as T

    if T.torch is None or T.Qwen2ForCausalLM is None:
        raise RuntimeError(f"CPU shape verification requires local dependencies: {T.TORCH_MISSING_REASON}")
    model = T.make_model(seed=23)
    calls = []

    def capture(_model, _args, kwargs):
        # Copy the actual inputs before forward/cache mutation, without changing them.
        calls.append({k: kwargs[k].detach().cpu().tolist()
                      for k in ("input_ids", "attention_mask", "position_ids")})

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    drv = T.make_driver(model, logits_last_only=True, sample_needed_only=True)
    drv.load()
    a, b, c, d = (_req("a", 6), _req("b", 7), _req("c", 4), _req("d", 3))
    drv.forward([ForwardItem(b, 2, False), ForwardItem(d, 3, True)])
    calls.clear()
    expected = [
        [("b", 4, 2, "prefill", False), ("a", 4, 0, "prefill", False),
         ("c", 4, 0, "prefill", True), ("d", 1, 3, "decode", True)],
        [("b", 1, 6, "prefill", True), ("a", 2, 4, "prefill", True),
         ("c", 1, 4, "decode", True), ("d", 1, 4, "decode", True)],
    ]
    # These tensors are specified independently of shape/grouping helpers.
    expected_calls = [
        {"position_ids": [[2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2, 3]],
         "attention_mask": [[1]*6, [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]]},
        {"position_ids": [[3]], "attention_mask": [[1]*4]},
        {"position_ids": [[4, 5]], "attention_mask": [[1]*6]},
        {"position_ids": [[6], [4], [4]],
         "attention_mask": [[1]*7, [0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]]},
    ]
    steps = [
        (StepPlan(new_start_ids=("a", "c"), decode_ids=("d",),
                  prefill_alloc={"b": 4, "a": 4, "c": 4}, tokens_prefilled=12),
         [a, c], [d], [(b, 5)], {"d": 3}),
        (StepPlan(decode_ids=("c", "d"), prefill_alloc={"b": 1, "a": 2},
                  tokens_prefilled=3), [], [c, d], [(b, 1), (a, 2)], {"c": 4, "d": 4}),
    ]
    evidence = []
    for i, (plan, waiting, decoding, prefilling, kv) in enumerate(steps):
        shape = plan_execution_shape(plan, waiting=waiting, decoding=decoding,
                                     prefilling=prefilling, per_request_kv_lengths=kv)
        assert _vectors(shape) == expected[i], (_vectors(shape), expected[i])
        specs = plan_forward_specs(plan, waiting=waiting, decoding=decoding, prefilling=prefilling)
        before = {s.req_id: s.past_len for s in drv._slots.values()}
        out = drv.forward([ForwardItem(s.req, s.n_new_tokens, s.need_token) for s in specs])
        after = {s.req_id: s.past_len for s in drv._slots.values()}
        assert out.timing.n_forwards == 2
        assert [s.req_id for s in out.slots] == [row[0] for row in expected[i]]
        for slot, row in zip(out.slots, expected[i]):
            rid, n, past, _role, need = row
            assert before.get(rid, 0) == past
            assert after[rid] == past + n
            assert (slot.token_id is not None) == need
        evidence.append({"shape": shape.describe(), "expected_vectors": expected[i],
                         "kv_before": before, "kv_after": after,
                         "n_forwards": out.timing.n_forwards,
                         "slots": [{"req_id": s.req_id, "token_id": s.token_id}
                                   for s in out.slots]})
    handle.remove()
    assert len(calls) == len(expected_calls)
    for actual, wanted in zip(calls, expected_calls):
        assert actual["position_ids"] == wanted["position_ids"], (actual, wanted)
        assert actual["attention_mask"] == wanted["attention_mask"], (actual, wanted)
    return {"evidence_class": "LOCAL_CPU_MICRO", "data_label": "TEST_DATA",
            "hardware_backed": drv.hardware_backed, "steps": evidence,
            "actual_calls": calls, "independent_expected_calls": expected_calls,
            "ok": True}


class TestExecutionShapeRealCpu(unittest.TestCase):
    def test_real_driver_masks_positions_and_kv_match_manual_two_step_plan(self):
        evidence = run_cpu_shape_probe()
        self.assertTrue(evidence["ok"])
        self.assertFalse(evidence["hardware_backed"])
        self.assertEqual(len(evidence["actual_calls"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
