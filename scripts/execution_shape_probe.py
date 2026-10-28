#!/usr/bin/env python
"""Exact production whole-step calibration. Default is a hardware-free dry plan.

Each --execute invocation is ONE independent process/session and ONE split.
Repetitions within that process are records, not independent experiments.
No old calibration artifacts/models are modified or silently reused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "benchmark", ROOT / "scripts"):
    sys.path.insert(0, str(_p))

from ascend_opt.execution_shape import ExecutionItem, ExecutionShape, execution_shape_key, execution_signature
from ascend_opt.workload import Request
from real_backend import ForwardItem, build_driver
from run_benchmark import load_config

SCHEMA = "execution_shape_probe_v1"
TIMING = "whole_step_fwd_plus_host_v1"


def read_json(path):
    def invalid(value):
        raise ValueError(f"non-standard JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name, minimum=0.0, *, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"invalid {name}: {value}")
    return float(value)


def checked_manifest(path):
    manifest = read_json(path)
    if manifest.get("schema") != "execution_shape_manifest_v1":
        raise ValueError("unsupported manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest has no entries")
    keys, point_ids = set(), set()
    for entry in entries:
        point_id = entry.get("point_id")
        if not isinstance(point_id, str) or not point_id or point_id in point_ids:
            raise ValueError("missing/duplicate point_id")
        point_ids.add(point_id)
        path_part = Path(entry["config_path"])
        if path_part.is_absolute() or ".." in path_part.parts:
            raise ValueError("config_path must be repository-relative")
        cfg = load_config(ROOT / path_part)
        signature = execution_signature(cfg.runtime)
        if entry.get("execution_signature") != signature:
            raise ValueError(f"{point_id}: manifest/config execution signature mismatch")
        revision = signature.get("model_revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("model_revision must be a pinned 40-character commit")
        if entry.get("model_revision") != revision:
            raise ValueError("entry model revision mismatch")
        shape = entry["execution_shape"]
        items = shape.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("empty whole-step shape")
        ids, typed = set(), []
        for item in items:
            rid = item["req_id"]
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("invalid/duplicate request ID")
            ids.add(rid)
            n = positive_int(item["n_new_tokens"], "n_new_tokens")
            kv = positive_int(item["kv_len"], "kv_len", minimum=0)
            role, need = item["role"], item["need_token"]
            if role not in ("prefill", "decode") or type(need) is not bool:
                raise ValueError("invalid role/need_token")
            req = entry["prime_requests"][rid]
            prompt = positive_int(req["prompt_len"], "prompt_len")
            gen = positive_int(req["gen_len"], "gen_len")
            if role == "prefill":
                if kv + n > prompt or need != (kv + n == prompt):
                    raise ValueError("prefill vector disagrees with original prompt")
            elif n != 1 or not need or kv < prompt or kv - prompt + 1 >= gen:
                raise ValueError("decode vector disagrees with original request")
            typed.append(ExecutionItem(rid, n, kv, role, need))
        if shape != ExecutionShape(tuple(typed)).describe():
            raise ValueError("whole-step shape/groups differ from production grouping")
        key = execution_shape_key(shape)
        if key != entry.get("execution_shape_key") or key in keys:
            raise ValueError("shape key mismatch/duplicate")
        keys.add(key)
        positive_int(entry["repeats"], "training independent processes", 3)
        positive_int(entry["validation_repeats"], "validation independent processes", 3)
        finite(entry["warmup_s"], "warmup_s", strict=True)
        for section, repeats in (("training", "repeats"), ("independent_validation", "validation_repeats")):
            design = entry[section]
            if design["independent_processes"] != entry[repeats]:
                raise ValueError("inconsistent independent process counts")
            positive_int(design["records_per_shape_per_process"], "records per shape", 3)
    if len({json.dumps(e["execution_signature"], sort_keys=True) for e in entries}) != 1:
        raise ValueError("one invocation requires one execution signature")
    return manifest


def dry_plan(manifest, split="training"):
    section = "training" if split == "training" else "independent_validation"
    rows = [{"point_id": e["point_id"], "execution_shape_key": e["execution_shape_key"],
             "n_forwards": e["execution_shape"]["n_forwards"],
             "warmup_s": e["warmup_s"], "records": e[section]["records_per_shape_per_process"],
             "independent_processes_required": e[section]["independent_processes"]}
            for e in manifest["entries"]]
    return {"status": "PLAN_ONLY", "schema": SCHEMA, "split": split, "points": rows,
            "one_invocation": "one independent process; all shapes; no GPU call in dry-plan",
            "execution_timing": TIMING, "prime_and_warmup_outside_measurement": True,
            "training_and_validation": "separate invocations; freeze training estimator before validation",
            "duration_seconds": None, "coverage": "exact listed shapes only; no shape generalization"}


def prime(drv, entry):
    """Recreate each exact KV state through real production forward calls."""
    shape = entry["execution_shape"]
    ids = [item["req_id"] for item in shape["items"]]
    drv.retire(ids)
    reqs = {rid: Request(req_id=rid, **entry["prime_requests"][rid]) for rid in ids}
    for item in shape["items"]:
        req, kv = reqs[item["req_id"]], item["kv_len"]
        if item["role"] == "prefill":
            if kv:
                drv.forward([ForwardItem(req, kv, False)])
        else:
            drv.forward([ForwardItem(req, req.prompt_len, True)])
            for _ in range(kv - req.prompt_len):
                drv.forward([ForwardItem(req, 1, True)])
        slot = drv._slots.get(req.req_id)
        if kv == 0:
            if slot is not None and (slot.past_len != 0 or slot.past):
                raise ValueError("KV=0 must have no past tensors")
        elif slot is None or slot.past_len != kv:
            raise ValueError("priming failed to construct exact KV")
    return [ForwardItem(reqs[i["req_id"]], i["n_new_tokens"], i["need_token"])
            for i in shape["items"]]


def measure_once(drv, entry):
    batch = prime(drv, entry)  # outside the actual whole-step measurement
    before = time.perf_counter()
    outcome = drv.forward(batch)
    wall_ms = (time.perf_counter() - before) * 1000.0
    timing = outcome.timing
    shape = entry["execution_shape"]
    if (timing.n_forwards != shape["n_forwards"] or timing.n_seqs != len(batch)
            or timing.n_tokens != sum(i.n_new_tokens for i in batch)):
        raise ValueError("production forward did not execute the requested complete shape")
    slots = {s.req_id: s for s in outcome.slots}
    if len(slots) != len(batch) or set(slots) != {i.req.req_id for i in batch}:
        raise ValueError("production returned missing/duplicate slots")
    for item in batch:
        if (slots[item.req.req_id].token_id is not None) != item.need_token:
            raise ValueError("need_token semantics changed")
    fwd = finite(timing.fwd_ms, "fwd_ms", strict=True)
    pack = finite(timing.pack_ms, "pack_ms")
    readback = finite(timing.readback_ms, "readback_ms")
    elapsed = finite(timing.elapsed_ms, "driver_elapsed_ms", strict=True)
    host = finite(elapsed - fwd, "host_ms")
    return {"fwd_ms": fwd, "pack_ms": pack, "readback_ms": readback,
            "host_ms": host, "total_ms": elapsed,
            "driver_elapsed_ms": elapsed,
            "call_wall_ms": finite(wall_ms, "call_wall_ms", strict=True),
            "n_forwards": timing.n_forwards}


def collect_entry(drv, entry, split, *, clock=time.monotonic):
    start = clock()
    warmups = 0
    while clock() - start < entry["warmup_s"]:
        measure_once(drv, entry)
        warmups += 1
    warmup_elapsed = clock() - start
    section = "training" if split == "training" else "independent_validation"
    records = [dict(record_index=n, **measure_once(drv, entry))
               for n in range(entry[section]["records_per_shape_per_process"])]
    drv.retire([i["req_id"] for i in entry["execution_shape"]["items"]])
    return {"point_id": entry["point_id"], "execution_shape": entry["execution_shape"],
            "execution_shape_key": entry["execution_shape_key"], "records": records,
            "warmup": {"seconds": warmup_elapsed, "calls": warmups, "excluded": True},
            "priming": "fresh production KV before every record; excluded from measurement"}


def runtime_identity(drv, cfg, environment):
    from ascend_opt.provenance import parse_mthreads_software
    desc = drv.describe()
    torch, _ = drv._torch()
    musa = getattr(torch, "musa", None)
    if (not getattr(drv, "hardware_backed", False) or desc.get("kind") != "transformers"
            or not str(desc.get("device", "")).startswith("musa")
            or musa is None or not musa.is_available()):
        raise ValueError("real MUSA TransformerDriver required; CPU/TEST_DATA cannot be measured")
    device_name = musa.get_device_name(drv.device)
    if "S4000" not in str(device_name).upper().replace(" ", ""):
        raise ValueError("actual device is not MTT S4000")
    if desc.get("model_revision_resolved") != cfg.runtime["model_revision"]:
        raise ValueError("loaded model revision does not match pinned revision")
    opt = desc["execution_optimizations"]
    for flag in ("logits_last_only", "sample_needed_only"):
        if opt[flag]["effective"] != bool(cfg.runtime.get(flag, False)):
            raise ValueError(f"{flag} not effective")
    device = environment["fingerprint"].get("device") or {}
    if "S4000" not in str(device.get("model", "")).upper().replace(" ", ""):
        raise ValueError("environment device model disagrees with actual S4000")
    versions = parse_mthreads_software(device.get("mthreads_gmi_info"))
    import torch_musa
    import transformers
    fingerprint = {"hardware": {"device_model": device.get("model"),
                    "driver_version": versions.get("driver_version")},
                   "software": {"torch": torch.__version__, "torch_musa": torch_musa.__version__,
                    "transformers": transformers.__version__, "python": platform.python_version()},
                   "model": {"id": desc["model"], "revision": desc["model_revision_resolved"]},
                   "precision": desc["precision"]}
    if any(not v for group in ("hardware", "software", "model") for v in fingerprint[group].values()):
        raise ValueError("incomplete observed runtime fingerprint")
    return {"driver": desc, "hardware_backed": True, "actual_device_name": device_name, "fingerprint": fingerprint}


def execute(manifest, manifest_path, out, split, env_file, frozen_estimator=None):
    from verify_s01_numeric_acceptance import _git_identity, _environment_snapshot
    from ascend_opt.provenance import compute_tree_digest
    if platform.system() != "Linux":
        raise ValueError("--execute requires the remote Linux MUSA host")
    frozen_hash = None
    if split == "validation":
        if frozen_estimator is None:
            raise ValueError("validation requires --frozen-estimator created after all training")
        frozen = read_json(frozen_estimator)
        if (frozen.get("schema") != "execution_shape_frozen_v1" or frozen.get("data_label") != "MEASURED"
                or frozen.get("manifest_sha256") != sha256(manifest_path)):
            raise ValueError("invalid frozen estimator/manifest identity")
        frozen_hash = sha256(frozen_estimator)
    elif frozen_estimator is not None:
        raise ValueError("training must not consume a validation estimator")
    environment = _environment_snapshot(env_file)
    env = environment["fingerprint"]
    if environment["fingerprint_matches_host"] is not True or env.get("available") is not True:
        raise ValueError("same-host available environment fingerprint required; run inspect_env first (freshness not verified here)")
    if env.get("available_source") != "declared":
        raise ValueError("environment availability must be explicitly collected")
    source = _git_identity()
    source["execution_code_digest"] = compute_tree_digest(ROOT, scope=("src", "benchmark"))
    if not source.get("tree_digest") or not source.get("git_commit"):
        raise ValueError("source identity unavailable")
    cfg_path = ROOT / manifest["entries"][0]["config_path"]
    cfg = load_config(cfg_path)
    # Reserve new artifact before loading a large model; never replace any output.
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        artifact = {"schema": SCHEMA, "session_id": uuid.uuid4().hex,
                    "process_id": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "split": split, "data_label": "NOT_MEASURED", "complete": False,
                    "frozen_estimator_sha256": frozen_hash,
                    "execution_timing": TIMING, "execution_signature": execution_signature(cfg.runtime),
                    "manifest_sha256": sha256(manifest_path), "manifest": manifest,
                    "source": source, "environment": environment, "points": [],
                    "configs": {e["config_path"]: {"sha256": sha256(ROOT / e["config_path"]),
                                "text": (ROOT / e["config_path"]).read_bytes().decode("utf-8")}
                                for e in manifest["entries"]}}
        def save():
            handle.seek(0)
            handle.truncate()
            json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        save()
        try:
            drv = build_driver(cfg)
            torch, _ = drv._torch()
            if not drv._device_available(torch):
                raise ValueError("MUSA unavailable before model load")
            drv.load()
            artifact["runtime"] = runtime_identity(drv, cfg, environment)
            if split == "validation":
                if (frozen.get("source") != source
                        or frozen.get("fingerprint") != artifact["runtime"]["fingerprint"]
                        or frozen.get("configs") != artifact["configs"]
                        or frozen.get("execution_signature") != artifact["execution_signature"]):
                    raise ValueError("frozen training identity differs from current validation runtime/source/config")
            for entry in manifest["entries"]:
                artifact["points"].append(collect_entry(drv, entry, split))
                save()
            artifact["runtime_end"] = runtime_identity(drv, cfg, environment)
            if _git_identity()["tree_digest"] != source["tree_digest"]:
                raise ValueError("source changed during probe")
            if sha256(env_file) != environment["fingerprint_sha256"]:
                raise ValueError("environment fingerprint changed during probe")
            artifact.update(complete=True, data_label="MEASURED")
        except Exception as exc:
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        save()
    return artifact


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--dry-plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    ap.add_argument("--split", choices=("training", "validation"), default="training")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--frozen-estimator", type=Path)
    args = ap.parse_args(argv)
    try:
        manifest = checked_manifest(args.manifest)
        if not args.execute:
            print(json.dumps(dry_plan(manifest, args.split), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.out is None or args.env_file is None:
            raise ValueError("--execute requires new --out and current --env-file")
        artifact = execute(manifest, args.manifest, args.out, args.split, args.env_file, args.frozen_estimator)
        print(json.dumps({"complete": artifact["complete"], "data_label": artifact["data_label"], "out": str(args.out)}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
