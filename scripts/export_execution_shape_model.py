#!/usr/bin/env python
"""Freeze training first; validate held-out processes; export only passing real data.

--freeze TRAIN_A TRAIN_B TRAIN_C --out NEW_FROZEN.json
--validate FROZEN.json --validation VAL_A VAL_B VAL_C --out NEW_MODEL.json --report NEW_REPORT.json
CPU/TEST_DATA artifacts are never accepted by either CLI operation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from execution_shape_probe import ROOT, SCHEMA, TIMING, finite, read_json, sha256
from ascend_opt.cost_model import FINGERPRINT_LEAVES, FINGERPRINT_SCALARS
from ascend_opt.execution_shape import execution_shape_key
from costcalib_fit import verdict


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stamp(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs a timezone")
    return parsed


def write_new(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def paired_median(records):
    """Sort on total; average the central pair's components for an even count."""
    rows = sorted(records, key=lambda r: r["total_ms"])
    if not rows:
        raise ValueError("empty records")
    middle = [rows[len(rows) // 2]] if len(rows) % 2 else rows[len(rows) // 2 - 1:len(rows) // 2 + 1]
    fwd = statistics.mean(r["fwd_ms"] for r in middle)
    host = statistics.mean(r["host_ms"] for r in middle)
    return {"fwd_ms": fwd, "host_ms": host, "total_ms": fwd + host}


def validate_artifact(artifact, split):
    if artifact.get("schema") != SCHEMA or artifact.get("complete") is not True:
        raise ValueError("incomplete or unknown probe artifact")
    if artifact.get("data_label") != "MEASURED" or artifact.get("split") != split:
        raise ValueError("only complete MEASURED artifacts from the correct split are eligible")
    if artifact.get("execution_timing") != TIMING:
        raise ValueError("wrong measurement timing")
    if not artifact.get("session_id") or not artifact.get("process_id"):
        raise ValueError("independent process identity missing")
    stamp(artifact["started_utc"])
    runtime = artifact["runtime"]
    if (runtime.get("hardware_backed") is not True or runtime["driver"].get("kind") != "transformers"
            or not str(runtime["driver"].get("device", "")).startswith("musa")
            or "S4000" not in str(runtime.get("actual_device_name", "")).upper().replace(" ", "")):
        raise ValueError("CPU/TEST_DATA or missing actual MUSA identity")
    fingerprint = runtime["fingerprint"]
    for group, leaf in FINGERPRINT_LEAVES:
        if not (fingerprint.get(group) or {}).get(leaf):
            raise ValueError(f"missing fingerprint {group}.{leaf}")
    for leaf in FINGERPRINT_SCALARS:
        if not fingerprint.get(leaf):
            raise ValueError(f"missing fingerprint {leaf}")
    signature = artifact["execution_signature"]
    if (fingerprint["model"] != {"id": signature["model"], "revision": signature["model_revision"]}
            or fingerprint["precision"] != signature["precision"]
            or runtime["driver"].get("model_revision_resolved") != signature["model_revision"]):
        raise ValueError("model/runtime identity mismatch")
    if artifact["runtime_end"]["fingerprint"] != fingerprint:
        raise ValueError("runtime fingerprint changed")
    source = artifact["source"]
    if not source.get("tree_digest") or not source.get("git_commit") or not source.get("execution_code_digest"):
        raise ValueError("source identity missing")
    environment = artifact["environment"]
    if (environment.get("fingerprint_matches_host") is not True
            or environment["fingerprint"].get("available") is not True
            or not environment.get("fingerprint_sha256")):
        raise ValueError("environment identity missing/unavailable")
    configs = artifact["configs"]
    for cfg in configs.values():
        if hashlib.sha256(cfg["text"].encode("utf-8")).hexdigest() != cfg["sha256"]:
            raise ValueError("config text/hash mismatch")
    manifest = artifact["manifest"]
    if manifest.get("schema") != "execution_shape_manifest_v1" or not artifact.get("manifest_sha256"):
        raise ValueError("missing manifest identity")
    entries = manifest["entries"]
    expected = {e["execution_shape_key"]: e for e in entries}
    if not entries or len(expected) != len(entries):
        raise ValueError("empty/duplicate manifest shapes")
    points = artifact["points"]
    if len(points) != len(entries) or {p["execution_shape_key"] for p in points} != set(expected):
        raise ValueError("not all exact shapes measured exactly once")
    medians = {}
    section = "training" if split == "training" else "independent_validation"
    for point in points:
        key = point["execution_shape_key"]
        entry = expected[key]
        if (point["execution_shape"] != entry["execution_shape"]
                or execution_shape_key(point["execution_shape"]) != key
                or entry["execution_signature"] != signature):
            raise ValueError("point/manifest execution identity mismatch")
        if entry["config_path"] not in configs:
            raise ValueError("point config snapshot missing")
        if entry[section]["independent_processes"] != 3:
            raise ValueError("this frozen design requires exactly three independent processes per split")
        rows = point["records"]
        if len(rows) != entry[section]["records_per_shape_per_process"] or len(rows) != 30:
            raise ValueError("each process must contain exactly 30 records per shape")
        if [r["record_index"] for r in rows] != list(range(len(rows))):
            raise ValueError("record indices incomplete/duplicate")
        warm = point["warmup"]
        if warm.get("excluded") is not True or warm.get("calls", 0) < 1 or warm.get("seconds", 0) < entry["warmup_s"]:
            raise ValueError("warmup incomplete or included in records")
        for row in rows:
            fwd = finite(row["fwd_ms"], "fwd_ms", strict=True)
            host = finite(row["host_ms"], "host_ms")
            total = finite(row["total_ms"], "total_ms", strict=True)
            elapsed = finite(row["driver_elapsed_ms"], "driver_elapsed_ms", strict=True)
            if not math.isclose(total, fwd + host, rel_tol=1e-12) or not math.isclose(total, elapsed, rel_tol=1e-12):
                raise ValueError("whole-step total is not driver elapsed/fwd+host")
            if row["n_forwards"] != entry["execution_shape"]["n_forwards"]:
                raise ValueError("wrong actual forward count")
        medians[key] = paired_median(rows)
    identity = {"fingerprint": fingerprint, "source": source, "execution_signature": signature,
                "execution_timing": TIMING, "manifest_sha256": artifact["manifest_sha256"],
                "manifest": manifest, "configs": configs,
                "hostname": environment["hostname"]}
    return identity, medians


def read_sessions(paths, split):
    if len(paths) != 3:
        raise ValueError("exactly three independent process artifacts required per split")
    identities, summaries, refs, session_ids, process_keys, digests = [], [], [], set(), set(), set()
    for path in paths:
        artifact = read_json(path)
        digest = sha256(path)
        identity, summary = validate_artifact(artifact, split)
        sid = artifact["session_id"]
        # One CLI invocation is one process. Conservatively refuse PID reuse
        # within this six-session batch rather than infer independence.
        process_key = (identity["hostname"], artifact["process_id"])
        if sid in session_ids or digest in digests or process_key in process_keys:
            raise ValueError("duplicate session/process/artifact")
        session_ids.add(sid); digests.add(digest); process_keys.add(process_key)
        identities.append(identity)
        summaries.append({"session_id": sid, "medians": summary})
        refs.append({"path": str(path), "sha256": digest, "session_id": sid,
                     "process_key": list(process_key), "started_utc": artifact["started_utc"],
                     "frozen_estimator_sha256": artifact.get("frozen_estimator_sha256")})
    if any(i != identities[0] for i in identities[1:]):
        raise ValueError("source/config/manifest/fingerprint/signature changed between sessions")
    return identities[0], summaries, refs


def freeze(paths):
    identity, summaries, refs = read_sessions(paths, "training")
    if any(r["frozen_estimator_sha256"] is not None for r in refs):
        raise ValueError("training unexpectedly consumed frozen estimator")
    table = {}
    for key in summaries[0]["medians"]:
        chosen = max(summaries, key=lambda s: s["medians"][key]["total_ms"])
        pair = chosen["medians"][key]
        table[key] = {"fwd_ms": pair["fwd_ms"], "host_ms": pair["host_ms"],
                      "selected_training_session": chosen["session_id"]}
    return {"schema": "execution_shape_frozen_v1", "data_label": "MEASURED",
            "created_utc": utc_now(), **identity, "execution_shape_table": table,
            "safety_margin": 0.0, "training": refs, "training_session_summaries": summaries,
            "estimator": "within-process median total with paired components; maximum of three training medians",
            "threshold_design": "reuse costcalib_fit thresholds for same-shape held-out sessions; abs-rel rule was historically posthoc; frozen for this protocol"}


def quantile(values, q):
    rows = sorted(values)
    if not rows:
        raise ValueError("empty metric")
    return rows[min(len(rows) - 1, int(q * len(rows)))]


def evaluate(frozen, summaries):
    errors, under, details = [], [], []
    for session in summaries:
        per_session = []
        for key, row in session["medians"].items():
            pair = frozen["execution_shape_table"][key]
            predicted = pair["fwd_ms"] + pair["host_ms"]
            relative = (predicted - row["total_ms"]) / row["total_ms"]
            errors.append(abs(relative))
            if relative < 0:
                under.append(-relative)
            per_session.append({"key": key, "predicted_ms": predicted,
                                "actual_session_median_ms": row["total_ms"], "relative_error": relative})
        details.append({"session_id": session["session_id"], "points": per_session})
    drifts = []
    for key in frozen["execution_shape_table"]:
        vals = [s["medians"][key]["total_ms"] for s in frozen["training_session_summaries"] + summaries]
        drifts.append((max(vals) - min(vals)) / statistics.median(vals))
    up95 = quantile(under, .95) if under else 0.0
    abs90, drift90 = quantile(errors, .9), quantile(drifts, .9)
    result = verdict(up95, drift90, abs90)
    result.update(underest_p95=up95, abs_rel_p90=abs90, drift_p90=drift90,
                  independent_validation_processes=len(summaries), per_session=details,
                  statistical_unit="independent process; shape/session residual quantiles descriptive only; no token-level independence",
                  generalization="held-out sessions of the SAME exact shapes, not unseen shapes",
                  estimator_frozen=True, safety_margin=0.0)
    return result


def validate(frozen_path, paths):
    frozen = read_json(frozen_path)
    if frozen.get("schema") != "execution_shape_frozen_v1" or frozen.get("data_label") != "MEASURED":
        raise ValueError("not a frozen real-data estimator")
    # Rebuild from training inputs: editing a frozen table cannot launder a new fit.
    rebuilt = freeze([r["path"] for r in frozen["training"]])
    for key in ("execution_shape_table", "training_session_summaries", "training", "fingerprint",
                "source", "execution_signature", "manifest_sha256", "manifest", "configs", "safety_margin"):
        if frozen[key] != rebuilt[key]:
            raise ValueError(f"frozen estimator no longer matches training inputs: {key}")
    identity, summaries, refs = read_sessions(paths, "validation")
    for key, value in identity.items():
        if frozen.get(key) != value:
            raise ValueError(f"training/validation identity mismatch: {key}")
    digest = sha256(frozen_path)
    old_ids = {r["session_id"] for r in frozen["training"]}
    old_hashes = {r["sha256"] for r in frozen["training"]}
    old_processes = {tuple(r["process_key"]) for r in frozen["training"]}
    for ref in refs:
        if (ref["session_id"] in old_ids or ref["sha256"] in old_hashes
                or tuple(ref["process_key"]) in old_processes
                or ref["frozen_estimator_sha256"] != digest
                or stamp(ref["started_utc"]) <= stamp(frozen["created_utc"])):
            raise ValueError("validation was reused or not collected after this estimator was frozen")
    report = evaluate(frozen, summaries)
    model = None
    if report["model_holds"]:
        model = {"protocol_version": "execution-shape-v1", "data_label": "MEASURED",
                 "created_utc": utc_now(), "fingerprint": frozen["fingerprint"], "source": frozen["source"],
                 "execution_code_digest": frozen["source"]["execution_code_digest"],
                 "execution_signature": frozen["execution_signature"], "execution_timing": TIMING,
                 "execution_shape_table": frozen["execution_shape_table"], "safety_margin": 0.0,
                 "coefficients": {}, "calibrated_range": {}, "validation": report,
                 "inputs": {"manifest_sha256": frozen["manifest_sha256"], "frozen_estimator_sha256": digest,
                            "training": frozen["training"], "validation": refs},
                 "evidence_restrictions": ["exact listed shapes only; unknown keys must fall back",
                    "three independent training and three independent validation processes",
                    "no unseen-shape generalization and no end-to-end performance conclusion",
                    "synthetic deterministic production prompts; shape coverage is not workload reachability"]}
    return report, model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", nargs="+", type=Path)
    group.add_argument("--validate", type=Path)
    ap.add_argument("--validation", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("refusing to overwrite output")
        if args.freeze:
            write_new(args.out, freeze(args.freeze))
            print("FROZEN: calibration estimator only; independent validation still required")
            return 0
        if not args.validation or args.report is None or args.report == args.out or args.report.exists():
            raise ValueError("validation needs independent artifacts and distinct new --report")
        report, model = validate(args.validate, args.validation)
        write_new(args.report, report)
        if model is None:
            print("FAIL: no model exported", file=sys.stderr)
            return 1
        write_new(args.out, model)
        print("PASS: new exact-shape model exported")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        if (args.validate and args.report is not None and args.report != args.out
                and not args.report.exists()):
            write_new(args.report, {"status": "UNKNOWN", "model_holds": None,
                                    "reason": f"{type(exc).__name__}: {exc}"})
        print(f"UNKNOWN/REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
