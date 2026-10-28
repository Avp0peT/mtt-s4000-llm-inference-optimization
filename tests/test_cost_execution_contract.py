"""TEST_DATA only: exact whole-step model and production candidate wiring."""
import copy
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "benchmark"):
    sys.path.insert(0, str(p))

from ascend_opt.execution_shape import execution_shape_key
from ascend_opt.cost_model import CostModel
from ascend_opt.arm import apply_arm
from arm_factory import build_arm
from run_benchmark import load_config, simulate
from tests.test_cost_policy import decide, make_requests
from real_backend import FakeDriver, RealBackend, VirtualClock


def policy():
    cfg = load_config(ROOT / "configs/mthreads/prereg_batch/cost_single.yaml")
    cfg = replace(cfg, controller=replace(cfg.controller,
                                         cost_model_fingerprint_source="declared"))
    return build_arm(cfg)


def exact_fixture():
    p = policy()
    requests = make_requests(8, prompt_len=1536)
    initial = decide(p, waiting=requests, chunk_ceiling=512, budget_ceiling=32768)
    shape = initial.candidates[0].execution_shape
    payload = {"data_label": "TEST_DATA", "coefficients": {}, "calibrated_range": {},
               "safety_margin": 0., "execution_signature": p.config.execution_signature,
               "execution_timing": "whole_step_fwd_plus_host_v1",
               "execution_shape_table": {execution_shape_key(shape): {"fwd_ms": 100., "host_ms": 25.}}}
    p.model = CostModel(payload)
    return p, requests, shape


def test_factory_always_requires_execution_contract_and_corrects_batch():
    p = policy()
    result = decide(p, waiting=make_requests(8, prompt_len=1536),
                    chunk_ceiling=512, budget_ceiling=32768)
    assert not result.changed
    c = next(c for c in result.candidates if c.max_tokens == 32768)
    assert (c.legacy_batch, c.batch) == (8, 1)
    assert c.execution_shape["items"][0]["kv_len"] == 0
    assert c.predicted_ms is None
    assert c.coverage_status == "execution_contract_unvalidated"


def test_exact_table_host_is_added_once_not_old_host_fit():
    p, requests, _ = exact_fixture()
    p.config = replace(p.config, host_overhead_ms={1: 9000.}, step_deadline_ms=126.)
    d = decide(p, waiting=requests, chunk_ceiling=512, budget_ceiling=32768)
    assert d.changed and all(c.feasible for c in d.candidates)
    assert all(c.host_ms == 25. for c in d.candidates)
    p.config = replace(p.config, step_deadline_ms=124.)
    d = decide(p, waiting=requests, chunk_ceiling=512, budget_ceiling=32768)
    assert d.degraded and not any(c.feasible for c in d.candidates)


def test_signature_missing_shape_and_unknown_kv_have_distinct_reasons():
    p, _, shape = exact_fixture()
    signature = p.config.execution_signature
    pred, status, _ = p.model.predict_execution(shape, execution_signature=signature)
    assert pred is not None and status == "exact_execution_key"
    wrong = dict(signature, sample_needed_only=False)
    assert p.model.predict_execution(shape, execution_signature=wrong)[1] == "execution_signature_mismatch"
    changed = copy.deepcopy(shape)
    changed["items"][0]["need_token"] = True
    assert p.model.predict_execution(changed, execution_signature=signature)[1] == "missing_execution_key"
    changed["items"][0]["kv_len"] = None
    assert p.model.predict_execution(changed, execution_signature=signature)[1] == "unknown_kv"


def test_multigroup_not_predicted_by_adding_single_group_records():
    p, _, shape = exact_fixture()
    multiple = copy.deepcopy(shape)
    multiple["items"].append(dict(req_id="second", n_new_tokens=1, kv_len=1536,
                                  role="decode", need_token=True))
    multiple["n_forwards"] = 2
    assert p.model.predict_execution(multiple, execution_signature=p.config.execution_signature)[0] is None


def test_degraded_choice_minimizes_total_step_not_forward_alone():
    p, requests, _ = exact_fixture()
    p.config = replace(p.config, candidate_chunks=(512, 1024), step_deadline_ms=50.)
    shapes = decide(p, waiting=requests, chunk_ceiling=1024).candidates
    for c in shapes:
        p.model.payload["execution_shape_table"][execution_shape_key(c.execution_shape)] = {
            "fwd_ms": 100. if c.chunk == 512 else 110.,
            "host_ms": 100. if c.chunk == 512 else 0.}
    d = decide(p, waiting=requests, chunk_ceiling=1024)
    assert d.degraded and d.chunk == 1024
    assert "110.0ms > 50.0ms" in d.reason


def test_unlimited_budget_still_serializes_strict_json():
    p, requests, _ = exact_fixture()
    p.config = replace(p.config, step_deadline_ms=None)
    d = decide(p, waiting=requests, chunk_ceiling=512, budget_ceiling=32768)
    json.dumps(d.describe(), allow_nan=False)
    assert all(c.describe()["budget_unbounded"] for c in d.candidates)


def test_observed_loader_rejects_test_data_unvalidated_or_other_source():
    from tests.test_cost_policy import GOOD_FP
    from ascend_opt.cost_model import CostModelError
    p, _, _ = exact_fixture()
    payload = copy.deepcopy(p.model.payload)
    payload.update(fingerprint=GOOD_FP, data_label="MEASURED",
                   validation={"model_holds": True}, execution_code_digest="test-digest")
    observed = dict(GOOD_FP, execution_code_digest="test-digest")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "TEST_DATA_fixture.json"
        for updates in ({"data_label": "TEST_DATA"}, {"validation": {"model_holds": False}},
                        {"execution_code_digest": "changed"}, {"execution_code_digest": None}):
            path.write_text(json.dumps(dict(payload, **updates)), encoding="utf-8")
            try:
                CostModel.load(path, observed)
            except CostModelError:
                pass
            else:
                raise AssertionError(f"invalid exact model loaded: {updates}")
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert CostModel.load(path, observed).payload["execution_shape_table"]


def test_decode_cache_length_excludes_last_sampled_token():
    req = make_requests(1, prompt_len=1536, gen_len=8)[0]
    p = policy()
    from tests.test_cost_policy import parts
    from ascend_opt.chunk_policy import ChunkLimits
    scheduler, cp, cache = parts()
    _, _, entry = apply_arm(kind="cost", policy=p, shadow=True, step=0, now_ms=1.,
        cur_chunk=512, cur_max_tokens=8192, waiting=[], prefilling=[],
        decoding=[SimpleNamespace(req=req, decode_remaining=7, last_emit_ms=0.)],
        scheduler=scheduler, chunk_policy=cp, cache=cache, limits=ChunkLimits(),
        max_num_seqs=8, chunk_enabled=True)
    assert entry["cost_policy"]["candidates"][0]["execution_shape"]["items"][0]["kv_len"] == 1536


def test_new_request_kv_comes_from_plan_cache_not_waiting_zero():
    p = policy()
    from tests.test_cost_policy import parts
    from ascend_opt.chunk_policy import ChunkLimits
    scheduler, cp, _ = parts()
    req = make_requests(1, prompt_len=8)[0]
    cache = SimpleNamespace(lookup=lambda req: 4)
    _, _, entry = apply_arm(kind="cost", policy=p, shadow=True, step=0, now_ms=1.,
        cur_chunk=512, cur_max_tokens=8192, waiting=[SimpleNamespace(req=req)],
        prefilling=[], decoding=[], scheduler=scheduler, chunk_policy=cp, cache=cache,
        limits=ChunkLimits(), max_num_seqs=8, chunk_enabled=True)
    assert entry["cost_policy"]["candidates"][0]["execution_shape"]["items"][0]["kv_len"] == 4


def test_both_backends_pass_post_step_phase_lists():
    from tests.test_real_backend import _cfg, _requests
    cfg = _cfg({"num_requests": 3, "prompt_len_min": 4, "prompt_len_max": 4,
                "gen_len_min": 2, "gen_len_max": 2, "concurrency": 2})
    cfg = replace(cfg, controller=replace(cfg.controller, enabled=True))
    spec, requests = _requests(cfg)
    calls = []
    def check(**kw):
        for key, phase in (("waiting", "waiting"), ("prefilling", "prefill"), ("decoding", "decode")):
            assert all(s.phase == phase for s in kw[key]), (key, [s.phase for s in kw[key]])
        calls.append(kw)
        return kw["cur_chunk"], kw["cur_max_tokens"], None
    for module in ("real_backend", "run_benchmark"):
        with patch(module + ".build_arm", return_value=object()), patch(module + ".apply_arm", side_effect=check):
            if module == "real_backend":
                clock = VirtualClock()
                RealBackend(cfg, driver=FakeDriver(clock=clock), clock=clock.now,
                            sleep=lambda _: None, allow_non_hardware_driver=True).run(requests, spec=spec)
            else:
                simulate(cfg, requests, spec=spec)
    assert calls
