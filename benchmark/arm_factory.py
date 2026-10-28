"""策略臂的**构造**：模拟器与真机**共用**的唯一实现（第 12/13 步）。

与 :mod:`ascend_opt.arm`（决策与应用）配套：本模块管"从配置造出策略对象"，
那个模块管"每步怎么用它"。两处都只有一份实现 —— 本仓库为"两条路径各写一份"
吃过亏（见 `step_planner` 的 docstring），第 13 步又实测踩到一次：
只把臂接进模拟器 ⇒ 真机上 `slo_weave` 决策数 = 0、两臂产物逐项相同。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ascend_opt.execution_shape import execution_signature

from ascend_opt import (
    CostModel,
    CostPolicy,
    CostPolicyConfig,
    SLOWeaveConfig,
    SLOWeavePolicy,
)

__all__ = ["build_arm", "observed_cost_fingerprint"]


def observed_cost_fingerprint(cfg: Any) -> dict[str, Any]:
    """**实测**指纹（真机路径用）。

    与标定产物里 `fingerprint.observed` **同形**；软件侧取本进程环境，
    硬件侧取 `scripts/inspect_env.sh` 预先写好的探针文件（与遥测同一个来源）。
    取不到的项**保持 `None`** —— `CostModel.load` 会把"未测"当**未确认**并拒绝，
    这是刻意的（"没确认"与"确认相符"是两件事）。
    """
    import platform as _platform

    from ascend_opt.provenance import parse_mthreads_software, compute_tree_digest
    from ascend_opt.telemetry import load_hardware_env

    root = Path(__file__).resolve().parents[1]
    hw = load_hardware_env([root / "results" / "raw" / "_env" / "env.json",
                            "/root/ascend_probe/results/raw/_env/env.json"])
    dev = hw.get("device") if isinstance(hw.get("device"), dict) else {}
    drv = parse_mthreads_software(dev.get("mthreads_gmi_info"))

    def _mod(name: str) -> str | None:
        try:
            return __import__(name).__version__
        except Exception:                                   # noqa: BLE001
            return None

    return {
        "hardware": {"device_model": dev.get("model"),
                     "driver_version": drv.get("driver_version")},
        "software": {"torch": _mod("torch"), "torch_musa": _mod("torch_musa"),
                     "transformers": _mod("transformers"),
                     "python": _platform.python_version()},
        "model": {"id": (cfg.runtime or {}).get("model"),
                  "revision": (cfg.runtime or {}).get("model_revision")},
        "precision": (cfg.runtime or {}).get("precision"),
        "execution_code_digest": compute_tree_digest(root, scope=("src", "benchmark")),
    }


def build_arm(cfg: Any) -> Any:
    """按 `controller.kind` 造策略对象；`threshold` 或 `enabled=False` ⇒ `None`。

    ⚠️ 指纹不符/未测时 `CostModel.load` **直接抛**，不让不可信的模型进到测量里。
    离线（`declared`）走 `load_offline`：**明说"没做指纹校验"**并记进产物，
    **不**造一个"声明指纹"去比对（那是假装校验过了）。
    """
    if cfg.controller.kind == "threshold" or not cfg.controller.enabled:
        return None
    root = Path(__file__).resolve().parents[1]
    model_path = (cfg.controller.cost_model_path
                  or str(root / "configs" / "cost_calib" / "model_v1.json"))
    if cfg.controller.cost_model_fingerprint_source == "observed":
        model = CostModel.load(model_path, observed_cost_fingerprint(cfg))
    else:
        model = CostModel.load_offline(model_path)

    if cfg.controller.kind == "cost":
        # 候选分块：**显式列表优先**；未给则旧行为（只取 min/max 两个点）。
        # ⚠️ 越界一律报错，不静默夹取 —— 夹取会让"我改了候选集"这件事**没有发生**，
        # 而两次对照的差异就被悄悄抹掉了（那正是"改了个不生效的参数"）。
        explicit = getattr(cfg.controller, "candidate_chunks", None)
        if explicit:
            lo, hi = int(cfg.controller.chunk_size_min), int(cfg.controller.chunk_size_max)
            oob = sorted({int(c) for c in explicit if not (lo <= int(c) <= hi)})
            if oob:
                raise ValueError(
                    f"candidate_chunks 越界：{oob} 不在 [{lo}, {hi}] 内。\n"
                    f"  ⇒ 若想让更小的块真的可动，必须**同时**把 chunk_size_min 降到那个值"
                    f"（运行期上限的唯一权威是 ChunkLimits，它会把你写的候选夹回去）。")
            cand_chunks = tuple(sorted({int(c) for c in explicit}))
        else:
            cand_chunks = tuple(sorted({cfg.controller.chunk_size_min,
                                        cfg.controller.chunk_size_max})) or (1024,)
        return CostPolicy(
            CostPolicyConfig(
                enabled=True,
                execution_signature=execution_signature(cfg.runtime or {}),
                step_deadline_ms=cfg.controller.target_tpot_ms,
                # ★ 双预算：由**配置**表达（默认 None ⇒ 不启用 ⇒ 既有行为不变）。
                #   不在这里传的话，`ControllerConfig.prefill_deadline_ms` 就只是个
                #   写进产物、谁也影响不到的死字段。
                prefill_deadline_ms=getattr(cfg.controller, "prefill_deadline_ms", None),
                # 候选必须落在运行期的上下限里（`ChunkLimits` 仍是唯一权威，
                # 这里只是把配置的 min/max 转成候选集）。
                candidate_chunks=cand_chunks,
                candidate_max_tokens=tuple(sorted(
                    {cfg.controller.max_batched_tokens_min,
                     cfg.controller.max_batched_tokens_max})) or (8192,),
                # 预算判据是"**整步** ≤ 预算"⇒ host 侧开销必须显式计入
                # （模型只预测前向；不加上它就是把两个口径混着比）。
                host_overhead_ms=getattr(cfg.controller, "host_overhead_ms", None),
                host_overhead_fit=(
                    tuple(cfg.controller.host_overhead_fit)
                    if getattr(cfg.controller, "host_overhead_fit", None) else None),
                # 第 15 步消融"去掉误差裕量"：必须由**配置**表达（默认 True = 既有行为）
                apply_safety_margin=bool(getattr(cfg.controller, "apply_safety_margin", True)),
            ),
            model,
        )
    # 第 13 步的最接近方法对照：`D` 不另立一个数，直接取已标定的 TPOT 目标。
    return SLOWeavePolicy(
        SLOWeaveConfig(
            enabled=True,
            tpot_target_ms=(cfg.controller.slo_weave_tpot_target_ms
                            if cfg.controller.slo_weave_tpot_target_ms is not None
                            else cfg.controller.target_tpot_ms),
            c_max=cfg.controller.slo_weave_c_max,
        ),
        model,
    )
