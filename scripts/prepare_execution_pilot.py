#!/usr/bin/env python
"""Derive a NEW three-arm pilot after a measured execution model passes validation.

This does not run a pilot or contact a server. A fresh config directory is
created only after model/identity checks, and the real matrix dry-plan must
confirm baseline + cost_single + cost_dual, each exactly once.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts", ROOT / "benchmark"):
    sys.path.insert(0, str(directory))

from ascend_opt.cost_model import CostModel, FINGERPRINT_LEAVES, FINGERPRINT_SCALARS
from ascend_opt.execution_shape import ExecutionItem, ExecutionShape, execution_signature
from ascend_opt.provenance import compute_tree_digest
import pilot_execution_plan as pilot
from run_benchmark import load_config

SOURCE_DIR = "configs/mthreads/prereg_batch"
ARMS = ("static_small", "cost_single", "cost_dual")
TIMING = "whole_step_fwd_plus_host_v1"


def _read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"non-standard JSON number: {value}")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(data, dict):
        raise ValueError("model must be a JSON object")
    return data


def _repo_path(value: str, *, config_directory: bool = False) -> Path:
    path = PurePosixPath(value)
    if (not value or value in (".", "..") or path.is_absolute() or
            any(part in (".", "..") for part in value.split("/")) or
            any(ord(c) < 32 for c in value) or "\\" in value or ":" in value):
        raise ValueError("paths must be repository-relative POSIX paths without '..'")
    if config_directory and (len(path.parts) < 2 or path.parts[0] != "configs"):
        raise ValueError("cfg-dir must name a new directory below configs/")
    resolved = (ROOT / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("resolved path escapes repository")
    if not config_directory and resolved.is_relative_to((ROOT / "results/raw").resolve()):
        raise ValueError("deployable exported models must not be read from raw results")
    return resolved


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(value, name):
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"missing model identity: {name}")
    return value


def validate_model(model: dict, signatures: list[dict]) -> None:
    """Fail closed before generating any configuration files."""
    if model.get("data_label") != "MEASURED":
        raise ValueError("pilot requires a MEASURED model; TEST_DATA/NOT_MEASURED are not deployable")
    if model.get("protocol_version") != "execution-shape-v1":
        raise ValueError("model is not a validated execution-shape-v1 export")
    if (model.get("validation") or {}).get("model_holds") is not True:
        raise ValueError("independent validation.model_holds must be true")
    signature = _required(model.get("execution_signature"), "execution_signature")
    if any(signature != expected for expected in signatures):
        raise ValueError("execution_signature differs from the pilot runtime/model")
    if model.get("execution_timing") != TIMING:
        raise ValueError("unsupported execution_timing")
    fingerprint = _required(model.get("fingerprint"), "fingerprint")
    for group, leaf in FINGERPRINT_LEAVES:
        value = _required((fingerprint.get(group) or {}).get(leaf), f"fingerprint.{group}.{leaf}")
        if not isinstance(value, str):
            raise ValueError(f"invalid fingerprint identity: {group}.{leaf}")
    for field in FINGERPRINT_SCALARS:
        _required(fingerprint.get(field), f"fingerprint.{field}")
    if (fingerprint["model"]["id"] != signature["model"] or
            fingerprint["model"]["revision"] != signature["model_revision"] or
            fingerprint["precision"] != signature["precision"]):
        raise ValueError("fingerprint model identity differs from execution_signature")
    _verify_export_identity(model)
    table = model.get("execution_shape_table")
    if not isinstance(table, dict) or not table:
        raise ValueError("execution_shape_table must contain measured whole-step records")
    cost_model = CostModel(model)
    for key, value in table.items():
        vectors = json.loads(key)
        if not isinstance(vectors, list) or not vectors:
            raise ValueError("invalid execution shape key")
        items = []
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != 4:
                raise ValueError("invalid execution shape vector")
            new, kv, role, need = vector
            if (type(new) is not int or new <= 0 or type(kv) is not int or kv < 0 or
                    role not in ("prefill", "decode") or type(need) is not bool):
                raise ValueError("invalid execution shape values")
            if role == "decode" and (new != 1 or not need):
                raise ValueError("invalid decode execution vector")
            items.append(ExecutionItem(f"item{index}", new, kv, role, need))
        shape = ExecutionShape(tuple(items))
        if shape.key() != key:
            raise ValueError("non-canonical execution shape key")
        if not isinstance(value, dict):
            raise ValueError("invalid measured execution record")
        predicted, status, reason = cost_model.predict_execution(
            shape.describe(), execution_signature=signature)
        if predicted is None or status != "exact_execution_key":
            raise ValueError(f"execution record rejected: {status}: {reason}")


def _verify_export_identity(model: dict) -> None:
    """Validate the exporter's source and independent-input bindings, not live HW."""
    def digest(value, length, field):
        if (not isinstance(value, str) or len(value) != length or
                any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"missing/invalid model identity: {field}")

    source = model.get("source") or {}
    digest(source.get("git_commit"), 40, "source.git_commit")
    digest(source.get("tree_digest"), 64, "source.tree_digest")
    digest(source.get("execution_code_digest"), 64, "source.execution_code_digest")
    digest(model.get("execution_code_digest"), 64, "execution_code_digest")
    if source["execution_code_digest"] != model["execution_code_digest"]:
        raise ValueError("export execution code identities disagree")
    if model["execution_code_digest"] != compute_tree_digest(ROOT, scope=("src", "benchmark")):
        raise ValueError("model execution code differs from current src/benchmark")
    inputs = model.get("inputs") or {}
    for field in ("manifest_sha256", "frozen_estimator_sha256"):
        digest(inputs.get(field), 64, f"inputs.{field}")
    sessions, hashes = set(), set()
    for split in ("training", "validation"):
        records = inputs.get(split)
        if not isinstance(records, list) or len(records) != 3:
            raise ValueError(f"inputs.{split} needs exactly three independent sessions")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"invalid inputs.{split} record")
            sid = _required(record.get("session_id"), f"inputs.{split}.session_id")
            if not isinstance(sid, str):
                raise ValueError("session identity must be a string")
            _required(record.get("path"), f"inputs.{split}.path")
            digest(record.get("sha256"), 64, f"inputs.{split}.sha256")
            if sid in sessions or record["sha256"] in hashes:
                raise ValueError("training/validation inputs must be independent and disjoint")
            sessions.add(sid)
            hashes.add(record["sha256"])
    if not isinstance((model.get("validation") or {}).get("thresholds"), dict) or not model["validation"]["thresholds"]:
        raise ValueError("validated model lacks its frozen validation thresholds")


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(*, model_path: str, cfg_dir: str, tag: str | None = None) -> dict:
    model_file = _repo_path(model_path)
    destination = _repo_path(cfg_dir, config_directory=True)
    if destination.is_relative_to((ROOT / SOURCE_DIR).resolve()):
        raise ValueError("cfg-dir must not alter the frozen prereg configuration tree")
    if destination.exists():
        raise ValueError("cfg-dir already exists; refusing to overwrite any configuration")
    sources = {arm: ROOT / SOURCE_DIR / f"{arm}.yaml" for arm in ARMS}
    source_bytes = {arm: path.read_bytes() for arm, path in sources.items()}
    configs = {arm: yaml.safe_load(data.decode("utf-8")) for arm, data in source_bytes.items()}
    signatures = [execution_signature(load_config(sources[arm]).runtime) for arm in ARMS]
    if configs["static_small"]["controller"].get("enabled") is not False:
        raise ValueError("static_small baseline controller must stay disabled")
    model_digest = _sha(model_file)
    model = _read_json(model_file)
    if _sha(model_file) != model_digest:
        raise ValueError("model changed while reading")
    validate_model(model, signatures)
    model_rel = model_file.relative_to(ROOT.resolve()).as_posix()
    plan = pilot.build_plan(tag=tag, cfg_dir=destination.relative_to(ROOT.resolve()).as_posix())
    output_bytes = {"static_small": source_bytes["static_small"]}
    for arm in ARMS[1:]:
        config = copy.deepcopy(configs[arm])
        if config["controller"].get("kind") != "cost" or config["controller"].get("enabled") is not True:
            raise ValueError(f"{arm} is not an enabled cost arm")
        config["controller"]["cost_model_path"] = model_rel
        for old_host in ("host_overhead_ms", "host_overhead_fit"):
            config["controller"].pop(old_host, None)
        header = ("# Derived by scripts/prepare_execution_pilot.py; not executed.\n"
                  f"# Source: {SOURCE_DIR}/{arm}.yaml\n"
                  f"# Measured execution model: {model_rel}; sha256={model_digest}\n"
                  "# Legacy host estimates removed: exact whole-step records contain host cost.\n")
        output_bytes[arm] = (header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False)).encode("utf-8")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"schema": "execution_pilot_preparation_v1", "evidence_class": "PLAN_ONLY",
              "status": "PREPARING", "model_path": model_rel, "model_sha256": model_digest,
              "execution_signature": model["execution_signature"],
              "model_validation": model["validation"], "source_config_sha256": {
                  arm: hashlib.sha256(data).hexdigest() for arm, data in source_bytes.items()},
              "plan": plan, "matrix_dryplan_ok": False, "pilot_executed": False,
              "remote_environment_verified": False}
    try:
        for arm in ARMS:
            with (destination / f"{arm}.yaml").open("xb") as handle:
                handle.write(output_bytes[arm])
            generated = load_config(destination / f"{arm}.yaml")
            if execution_signature(generated.runtime) != model["execution_signature"]:
                raise ValueError(f"generated {arm} execution signature mismatch")
        problems = pilot.validate(plan)
        if problems:
            raise ValueError("; ".join(problems))
        ok, note = pilot.cross_check_with_matrix(plan)
        report["matrix_dryplan_note"] = note
        if not ok:
            raise ValueError(f"real matrix dry-plan failed: {note}")
        if _sha(model_file) != model_digest:
            raise ValueError("model changed during preparation")
        if compute_tree_digest(ROOT, scope=("src", "benchmark")) != model["execution_code_digest"]:
            raise ValueError("execution code changed during preparation")
        report.update(status="PREPARED_NOT_RUN", matrix_dryplan_ok=True,
                      generated_config_sha256={arm: _sha(destination / f"{arm}.yaml") for arm in ARMS})
        _write_json_exclusive(destination / "preparation.json", report)
    except Exception as exc:
        report.update(status="FAILED_NOT_READY", error=f"{type(exc).__name__}: {exc}")
        _write_json_exclusive(destination / "preparation_failed.json", report)
        raise
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="repository-relative exported measured model JSON")
    parser.add_argument("--cfg-dir", required=True, help="NEW repository-relative directory below configs/")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    try:
        report = prepare(model_path=args.model, cfg_dir=args.cfg_dir, tag=args.tag)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
