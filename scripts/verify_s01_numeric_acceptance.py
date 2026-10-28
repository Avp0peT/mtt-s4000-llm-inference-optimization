#!/usr/bin/env python
"""S01 复核的**数值正确性验收**（真机短跑；本机也可用**真实 CPU 微型驱动**自证）。

★ 为什么 S01 需要它、而 `--dry-run` 不够
------------------------------------------
S01 的两个开关（`runtime.logits_last_only` / `runtime.sample_needed_only`）改的是
**"取哪些 logits、对哪些行采样"** ⇒ 它们**可能改变产出的 token**。
判据因此是：

    同一负载、两个臂，**每个请求的 token 序列必须逐位相同**。

⚠️ `--dry-run` **验不了**：配置头里就写着"这些开关在 --dry-run 下不生效"
（模拟器不碰这条路）⇒ 用模拟路径做数值验收等于没验。

⚠️ 为什么不能只比 `summary.output_tokens`（个数的和）：**个数相同**完全兼容
"内容换了"。所以本脚本比**逐 token 的 id 序列**。

★★ 2026-10-03 第二轮重写（用户指定**固定、合法的请求执行步骤**）
-----------------------------------------------------------------
旧版用"循环补齐到 GEN"的**自适应**步骤 —— 那样每步喂几条、喂多少由运行时状态决定，
两次运行之间**不能保证逐步同形**，于是"逐步检查 slot 数量/唯一性"这类断言失去基准。
本版改为**显式、固定、可预先算清**的 6 组步骤（见 `SCHEDULE`），于是：

* 每请求产出的 token 数**可以先算出来**（= 8）并与实测对账；
* 每个前向组喂了哪些行、哪些行**不该**出 token，都是**声明式**的；
* 逐步断言：**slot 数量**、**req_id 唯一性**、**请求集合**、**输出位置**。

**固定步骤（3 请求 s0/s1/s2）—— 按用户原文逐条落成"组"**

| 组 | 内容 | need_token | 用户原文条目 |
| --- | --- | --- | --- |
| 1 | s0 完整 prefill 64；s2 完整 prefill 80 | T / T | ① |
| 2 | s1 **先** prefill 70（**单独一组**） | **F** | ② |
| 3 | 同一前向组：s0 decode1；s1 prompt1；s2 decode1 | T / **F** / T | ③ ★ **唯一"真实混合组"** |
| 4×6 | 固定 **6 轮**三请求 decode | T / T / T | ⑤ |
| 5 | s1 最后 prompt1 | T | ④ |
| 6 | s1 再 decode1 | T | ⑤ |

⚠️ **为什么把 s1 的"先 prefill"单独成组**：用户明确点名"**False 行 2 个、真实混合组 1 个**"。
若把 s0/s1/s2 的首块 prefill 合成一组，它自身就是"混合组"⇒ 会数出 **2** 个混合组，
与用户给出的 1 不符。⇒ 用户要的"混合组"**专指**条目 ③ 那一组（decode 与 prompt 交错），
条目 ② 是**单独一行**的 False 步。本表按此落。

⇒ 每请求 **8** 个 token、总 **24**；`need_token=False` 行 **2** 个；**真实混合组 1 个**。

★「真实 CPU 微型驱动」是**默认**的本地自证路径
--------------------------------------------------
`--driver cpu`（默认）：用 `tests/_tiny_support.py` 的**真 `TransformerDriver`** 子类
（`hardware_backed=False`）跑一个小 Qwen2，走的**就是生产代码** `TransformerDriver.forward`
里那两个开关的分支 —— 不是假驱动、不是模拟器。
⇒ 它能在**本机**证明"判据本身会 FAIL"（见回归），但**性能无效**、且**不能**替代真机标定。
⇒ `--driver fake` 只在跑"脚本自身逻辑"的极端场合用，**默认不用**：假驱动与真分支是两条
   代码路径，用它通过说明不了生产路径对。

⚠️ **产物禁止覆盖**：`--json` 目标已存在 ⇒ 直接拒绝（exit 2），不静默覆盖历史证据。

用法::

    # 本机自证（默认 CPU 微型驱动）—— 证据等级 LOCAL_CPU_MICRO
    python scripts/verify_s01_numeric_acceptance.py \
        --config-a configs/mthreads/exec_opt/s01_reference.yaml \
        --config-b configs/mthreads/exec_opt/s01_both.yaml \
        --driver cpu \
        --json results/processed/s01_numeric_<唯一编号>.json

    # 真机（MUSA，最终执行目录，环境已激活）—— ★ **必须显式 --driver auto**
    #   先 source /root/musa-envs/qwen310/activate_musa.sh
    bash -c 'source /root/musa-envs/qwen310/activate_musa.sh && \
      python scripts/verify_s01_numeric_acceptance.py \
        --config-a configs/mthreads/exec_opt/s01_reference.yaml \
        --config-b configs/mthreads/exec_opt/s01_both.yaml \
        --driver auto --require-real-hw \
        --json results/processed/s01_numeric-<唯一编号>.json'

★★ **为什么真机必须显式写 `--driver auto`**（不是"好看"）：
   ① `--driver` 的默认值是 `cpu` ⇒ 在真机上**漏写**会静默跑 CPU 微型驱动，
      产物标成 `LOCAL_CPU_MICRO` ⇒ **把本机证据当成真机证据**（最危险的形态）；
   ② `--driver auto` 会**额外**核验三件事（`_hardware_problems`）：
      **实际 MUSA 设备** / **模型 resolved revision** / **优化真的生效**；
      判据是**拒绝式**的 —— 任一条不成立即不得 PASS。

★ **证据等级不得升级**：`LOCAL_CPU_MICRO` 证明的是**接口与判据语义**，
  **不是**硬件数值行为；**不得**在报告里写成"S4000 已验证"。

⚠️ **产物禁止覆盖**（`--json` 目标已存在 ⇒ exit 2）：数值证据必须能自证是哪次调用写的。

退出码：0 = 通过；1 = 不通过；2 = 输入/环境问题（含"产物已存在、拒绝覆盖"）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "benchmark"), str(_ROOT / "src"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.workload import Request  # noqa: E402
from ascend_opt.provenance import (  # noqa: E402
    compute_tree_digest, detect_source_identity, read_source_stamp,
)
from ascend_opt.telemetry import load_hardware_env  # noqa: E402
from real_backend import ForwardItem, build_driver  # noqa: E402
from run_benchmark import load_config  # noqa: E402

# --------------------------------------------------------------------------- #
# 固定计划（**声明式**：期望值从这里派生，不手写）
# --------------------------------------------------------------------------- #

#: 三条请求的 prompt 长度。**长度各不相同**，这样"按行映射"若有位置假设就会暴露。
PLEN = {"s0": 64, "s1": 72, "s2": 80}
#: s1 的**分块预填充**：先喂 70（不出 token），之后再补 **1**（出 token）。
#: ⚠️ 补的块**只能是 1**：驱动的分块预填充守卫要求"本次要喂的 token ≤ prompt 剩余"
#:    （实测：喂 2 会被拒绝 —— `prompt 只剩 1 个（past_len=71, prompt_len=72）`）。
#:    这不是 bug，是驱动在**正确地**拒绝一个非法的分块计划。
S1_FIRST_CHUNK = 70
S1_TOPUP = 1
_S1_TOTAL = S1_FIRST_CHUNK + S1_TOPUP           # = 71
assert PLEN["s1"] == 72, "s1 的 prompt 必须是 72（先 70 + 再补，补后达 71，末位置出首 token）"

GEN = 8                     # 每请求预期产出的 token 数
N_DECODE_ROUNDS = 6         # 固定 6 轮三请求 decode

#: 每一步：(名字, [(req_id, n_new_tokens, need_token), ...])
#: 顺序即执行顺序；`SCHEDULE` 是**唯一来源**，期望值全部由它派生。
#:
#: ⚠️ s1 的 prompt 共 72，分**三**次喂完：70（单独 F 组）→ 1（混组里的 prompt1，F）
#:    → 1（topup，T，末位置产首 token）。三次加起来正好 72，
#:    且**两次 False、一次 True** ⇒ 正好对应"False 行 2 个"。
SCHEDULE: list[tuple[str, list[tuple[str, int, bool]]]] = [
    # ① 完整 prefill（两行都出首 token；**不含** s1，避免多出一个"混合组"）
    ("g1.prefill_full",
     [("s0", PLEN["s0"], True), ("s2", PLEN["s2"], True)]),
    # ② s1 先 prefill 70，单独一组，不出 token
    ("g2.s1_prefill_chunk",
     [("s1", S1_FIRST_CHUNK, False)]),
    # ③ ★ **唯一"真实混合组"**：decode1 / prompt1 / decode1 交错
    ("g3.mixed_decode_prompt",
     [("s0", 1, True), ("s1", 1, False), ("s2", 1, True)]),
    # ④ s1 收尾补 1（末位置产首 token）
    ("g4.s1_topup", [("s1", S1_TOPUP, True)]),
]
for _i in range(N_DECODE_ROUNDS):
    SCHEDULE.append((f"g5.decode_r{_i}",
                     [("s0", 1, True), ("s1", 1, True), ("s2", 1, True)]))
SCHEDULE.append(("g6.s1_final_decode", [("s1", 1, True)]))

assert S1_FIRST_CHUNK + 1 + S1_TOPUP == PLEN["s1"], (
    "s1 的三次喂入（首块 + 混组里的 1 + 收尾）必须恰好等于它的 prompt 长度")


def _derive_expectations() -> dict[str, Any]:
    """从 `SCHEDULE` 派生期望值 —— 手写期望值是最容易出错的地方。"""
    n_tok = {k: 0 for k in PLEN}
    false_rows: list[tuple[str, str]] = []          # (step, req)
    mixed_groups: list[str] = []
    for step, items in SCHEDULE:
        trues = [it for it in items if it[2]]
        falses = [it for it in items if not it[2]]
        if trues and falses:
            mixed_groups.append(step)
        for rid, _n, want in items:
            if want:
                n_tok[rid] += 1
            else:
                false_rows.append((step, rid))
    return {
        "tokens_per_req": n_tok,
        "total": sum(n_tok.values()),
        "false_rows": false_rows,
        "n_false_rows": len(false_rows),
        "mixed_groups": mixed_groups,
        "n_mixed_groups": len(mixed_groups),
        "n_steps": len(SCHEDULE),
    }


EXPECT = _derive_expectations()


# --------------------------------------------------------------------------- #
# 身份：复用既有口径
# --------------------------------------------------------------------------- #

def _git_identity() -> dict[str, Any]:
    """复用部署身份规则，另算当前树摘要；未知 dirty 保留 None。"""
    commit, source, dirty, _ = detect_source_identity(_ROOT)
    return {"git_commit": commit or None, "git_source": source,
            "git_dirty": dirty, "tree_digest": compute_tree_digest(_ROOT),
            "tree_digest_basis": "computed_on_host",
            "stamp": read_source_stamp(_ROOT)}


def _environment_snapshot(env_path: Path) -> dict[str, Any]:
    """只读现有采集器的指纹；异机缓存不能充当当前主机版本。"""
    try:
        fingerprint = load_hardware_env([env_path])
    except (OSError, TypeError, AttributeError) as exc:
        fingerprint = {"available": None, "reason": f"无法读取指纹：{exc}"}
    hostname = platform.node()
    same_host = (fingerprint.get("hostname") == hostname
                 if fingerprint.get("hostname") else None)
    values = (fingerprint.get("discovery") or {}).get("values") or {}
    return {
        "hostname": hostname, "python": sys.version.split()[0],
        "executable": sys.executable, "platform": platform.platform(),
        "fingerprint_path": str(env_path), "fingerprint_sha256": _file_sha256(env_path),
        "fingerprint": fingerprint, "fingerprint_matches_host": same_host,
        # 同机不等于新采集：保留原始采集时间，部署验收须先 inspect_env。
        "fingerprint_freshness": "not_verified",
        "driver_version": values.get("MTT_DRIVER_VERSION") if same_host else None,
        "musa_version": values.get("MUSA_VERSION") if same_host else None,
        "versions_basis": "same_host_fingerprint" if same_host else "unavailable",
    }


def _env_identity(cfg) -> dict[str, Any]:
    """环境/模型身份：从配置与运行时能拿到什么记什么。"""
    rt = (cfg.raw.get("runtime") or {}) if hasattr(cfg, "raw") else {}
    return {"model_revision_config": rt.get("model_revision"),
            "precision": rt.get("precision"),
            "python": sys.version.split()[0]}


def _file_sha256(p: Path) -> str | None:
    """配置快照的字节级身份（续跑/复用判据要用**逐字节**而不是"看着一样"）。"""
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def _requested_revision(cfg) -> str | None:
    """配置里**请求**的模型 revision（≠ 实际 resolved 的那个）。"""
    return ((cfg.raw.get("runtime") or {}).get("model_revision"))


def sanitize_json(obj: Any, *, path: str = "$") -> tuple[Any, list[str]]:
    """把**非有限浮点**（NaN / ±Inf）换成 `null`，并**逐条记下原因**。

    ⚠️ 为什么要这样而不是直接 `json.dumps(..., allow_nan=False)`：
      `allow_nan=False` 会**抛异常** ⇒ 产物**写不出来**，而那正是最需要产物的时刻
      （数据已经坏了）。直接沿用默认（`allow_nan=True`）更糟：会写出 `NaN`/`Infinity`
      这种**非标准 JSON**，任何严格解析器都拒收，而写的一方**看不到任何异常**。
    ⇒ 本函数：保留结构、坏值置 `null`、成因写进 `_json_notes` ⇒ 产物**永远合法**。
    """
    notes: list[str] = []

    def walk(o: Any, p: str) -> Any:
        if isinstance(o, bool) or o is None:
            return o
        if isinstance(o, float):
            if math.isfinite(o):
                return o
            notes.append(f"{p} = {o!r} 不是有限数 ⇒ 已置 `null`（非有限值不可作诊断依据）")
            return None
        if isinstance(o, dict):
            return {k: walk(v, f"{p}.{k}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(o)]
        return o

    return walk(obj, path), notes


def _optimization_problems(arms: list[dict]) -> list[str]:
    """固定 A=对照关、B=处理开；计数不能用同一条件验收两臂。"""
    probs: list[str] = []
    if len(arms) != 2:
        return [f"要求对照/处理两臂，实际 {len(arms)} 臂"]
    for arm, role, enabled in zip(arms, ("对照", "处理"), (False, True)):
        label = f"{role} {arm.get('config')}"
        ex = arm.get("execution_optimizations") or {}
        for key in ("sample_needed_only", "logits_last_only"):
            opt = ex.get(key) or {}
            if opt.get("requested") is not enabled or opt.get("effective") is not enabled:
                probs.append(f"{label}: {key} requested/effective 必须为 {enabled} "
                             "⇒ 开关角色不符或优化未真正生效")
        sn = ex.get("sample_needed_only") or {}
        skipped = sn.get("n_sampling_rows_skipped")
        want = EXPECT["n_false_rows"] if enabled else 0
        if type(skipped) is not int or skipped != want:
            probs.append(f"{label}: n_sampling_rows_skipped={skipped}，固定排程要求 {want} "
                         "⇒ 角色不符或优化未真正生效")
        calls = (ex.get("logits_last_only") or {}).get("n_calls_last_only")
        if type(calls) is not int or (calls <= 0 if enabled else calls != 0):
            probs.append(f"{label}: n_calls_last_only={calls}，"
                         f"要求 {'> 0' if enabled else '= 0'} ⇒ last-only 未按角色执行")
    return probs


def _hardware_problems(drv, cfg, arms: list[dict], *, driver: str) -> list[str]:
    """★★ **仅真机路径**（`--driver auto`）的设备/模型/优化核验。

    用户 2026-10-03 点名四项：
      ① **核验实际 MUSA 设备**（不能只看配置里写着 `musa:0`）；
      ② **核验模型 resolved revision**（配置写的 revision 只是**请求**，
         真正加载到的是 `model_revision_resolved`；两者不等 ⇒ 跑的不是同一份权重）；
      ③ **核验优化生效**（S01 开关要真的改变了执行，不是"配了没走"）；
      ④ **保留 CPU 证据等级**（`LOCAL_CPU_MICRO` 不得被读成 S4000 证据）。

    ⚠️ 全部是**拒绝式**判据：任一条不成立 ⇒ 不得判定 PASS。
    """
    probs: list[str] = []
    if driver != "auto":
        return probs        # CPU 微型驱动没有真设备，这些判据**不适用**（不是"通过"）

    # ---- ① 实际设备必须是真的 MUSA，且不是 CPU ----
    kinds = {a.get("driver_identity", {}).get("kind") for a in arms}
    devices = {str(a.get("driver_identity", {}).get("device")) for a in arms}
    hw = {a.get("hardware_backed") for a in arms}
    if hw != {True}:
        probs.append(f"`--driver auto` 但 `hardware_backed` 不为 True（={hw}）"
                     " ⇒ 跑的不是真机设备")
    for dev in devices:
        if "musa" not in dev.lower():
            probs.append(f"实际设备 device={dev!r} 不含 'musa' ⇒ 不是 MUSA 设备"
                         "（配置写 musa:0 不等于真的跑在 MUSA 上）")
    if len(devices) != 1:
        probs.append(f"两臂实际设备不一致：{devices}")
    if kinds and kinds != {"transformers"}:
        probs.append(f"驱动 kind={kinds} != {{'transformers'}} ⇒ 与真机路径不符")

    # ---- ② 模型 resolved revision 必须与配置请求的一致 ----
    want = (cfg.raw.get("runtime") or {}).get("model_revision")
    if not want:
        probs.append("配置缺少 model_revision ⇒ 无法核对固定权重")
    for a in arms:
        got = a.get("driver_identity", {}).get("model_revision_resolved")
        if not got:
            probs.append(f"{a.get('config')}: `model_revision_resolved` 缺失 ⇒ "
                         "无法证明加载的是钉死的那份权重")
        elif want and got != want:
            probs.append(f"{a.get('config')}: resolved revision {got} != 配置请求 {want}"
                         " ⇒ 跑的不是同一份权重")

    # ---- ③ 对照应关闭，只有处理臂必须行使优化 ----
    probs.extend(_optimization_problems(arms))
    return probs


# --------------------------------------------------------------------------- #
# 驱动选择
# --------------------------------------------------------------------------- #

def _build_cpu_driver(cfg_path: Path):
    """**真实 CPU 微型驱动**：真 `TransformerDriver` 子类，跑小 Qwen2。

    ⚠️ 它走的是**生产代码** `TransformerDriver.forward` 里那两个开关的分支
    —— 这是"本地能自证判据"的前提。代价：`hardware_backed=False` ⇒ **性能无效**。
    """
    try:
        import _tiny_support as T  # noqa: E402  (tests/ 已在 sys.path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"无法导入 tests/_tiny_support.py：{type(exc).__name__}: {exc}") from exc
    if T.torch is None:
        raise RuntimeError(f"本机缺少 torch/transformers：{T.TORCH_MISSING_REASON}")

    cfg = load_config(cfg_path)
    rt = cfg.raw.get("runtime") or {}
    model = T.make_model()
    drv = T.make_driver(
        model,
        logits_last_only=bool(rt.get("logits_last_only", False)),
        sample_needed_only=bool(rt.get("sample_needed_only", False)),
    )
    return drv, cfg


def _build_driver_for(cfg_path: Path, *, driver: str):
    """按 `--driver` 造驱动。

    * `cpu`   —— 真实 CPU 微型驱动（默认，本机自证生产路径）
    * `auto`  —— 真机路径：`build_driver(cfg)`（MUSA / transformers）
    * `fake`  —— 仅用于跑脚本自身逻辑（**不作为数值证据**）
    """
    if driver == "cpu":
        return _build_cpu_driver(cfg_path)
    cfg = load_config(cfg_path)
    if driver == "fake":
        from real_backend import FakeDriver

        return FakeDriver(cfg, seed=0), cfg
    drv = build_driver(cfg)
    return drv, cfg


# --------------------------------------------------------------------------- #
# 执行固定计划
# --------------------------------------------------------------------------- #

def _run_arm(cfg_path: Path, *, driver: str) -> dict[str, Any]:
    """按 `SCHEDULE` 逐组执行，并在**每一步**做结构性断言。"""
    drv, cfg = _build_driver_for(cfg_path, driver=driver)
    drv.load()

    reqs = {rid: Request(req_id=rid, prompt_len=n, gen_len=GEN)
            for rid, n in PLEN.items()}
    tokens: dict[str, list[int]] = {rid: [] for rid in PLEN}
    slots_seen: set[str] = set()
    no_token_positions: list[list[Any]] = []            # [step, req, pos_at_that_moment]
    logits_digests: list[str | None] = []
    violations: list[str] = []
    per_step: list[dict[str, Any]] = []

    for step, plan in SCHEDULE:
        items = [ForwardItem(reqs[rid], n_new_tokens=n, need_token=want)
                 for rid, n, want in plan]
        want_ids = [it[0] for it in plan]
        want_order = list(want_ids)
        expect_no_token = {rid for rid, _n, want in plan if not want}

        out = drv.forward(items)
        got_ids = [s.req_id for s in out.slots]

        step_probs: list[str] = []
        # ① slot 数量必须恰好等于本组喂进去的行数
        if len(out.slots) != len(items):
            step_probs.append(
                f"{step}: slot 数量 {len(out.slots)} != 喂入行数 {len(items)}")
        # ② req_id 必须唯一（重复 = 同一请求在一次前向里出现两次）
        if len(set(got_ids)) != len(got_ids):
            dup = sorted({r for r in got_ids if got_ids.count(r) > 1})
            step_probs.append(f"{step}: req_id 重复 {dup}")
        # ③ 返回的请求集合必须**恰好**等于本组喂进去的那一组
        if set(got_ids) != set(want_ids):
            step_probs.append(
                f"{step}: 请求集合不符 —— 缺 {sorted(set(want_ids) - set(got_ids))}；"
                f"多 {sorted(set(got_ids) - set(want_ids))}")
        # ④ 顺序必须与喂入一致（按行映射的前提；顺序错 = 行映射错位）
        if got_ids != want_order:
            step_probs.append(f"{step}: 返回顺序 {got_ids} != 喂入顺序 {want_order}")

        # ⑤ 输出位置：need_token=False 的行**不得**有 token；True 的必须有
        for slot in out.slots:
            rid = slot.req_id
            slots_seen.add(rid)
            if rid in expect_no_token:
                if slot.token_id is not None:
                    step_probs.append(
                        f"{step}: {rid} need_token=False 却产出 token_id={slot.token_id}")
                else:
                    no_token_positions.append([step, rid, len(tokens.get(rid, []))])
                continue
            if slot.token_id is None:
                step_probs.append(f"{step}: {rid} need_token=True 却没有 token_id")
                continue
            if len(tokens.get(rid, [])) >= GEN:
                step_probs.append(
                    f"{step}: {rid} 已满 {GEN} 个 token 却仍在产出（计划溢出）")
                continue
            tokens.setdefault(rid, []).append(int(slot.token_id))
            d = getattr(slot, "logits_digest", None)
            if isinstance(d, dict):
                logits_digests.append(d.get("sha256"))
            elif d is not None:
                logits_digests.append(str(d))

        violations.extend(step_probs)
        per_step.append({
            "step": step,
            "n_items": len(items),
            "n_slots": len(out.slots),
            "returned_ids": got_ids,
            "expected_ids": want_order,
            "expect_no_token": sorted(expect_no_token),
            "problems": step_probs,
        })

    desc = drv.describe() if hasattr(drv, "describe") else {}
    exopt = (desc.get("execution_optimizations") or {})
    return {
        "config": str(cfg_path),
        "driver_kind": desc.get("kind"),
        "hardware_backed": getattr(drv, "hardware_backed", None),
        "tokens": tokens,
        "slots_seen": sorted(slots_seen),
        "expected_req_ids": sorted(PLEN),
        "token_digest": hashlib.sha256(
            json.dumps({k: list(v) for k, v in sorted(tokens.items())}).encode()
        ).hexdigest(),
        "logits_digests": logits_digests,
        "n_steps_logits": len(logits_digests),
        "no_token_positions": no_token_positions,
        "per_step": per_step,
        "violations": violations,
        "execution_optimizations": exopt,
        "driver_identity": {k: desc.get(k) for k in
                            ("kind", "model_revision", "model_revision_resolved",
                             "device", "precision", "sampling")},
    }


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #

def _check_expected_set(arm: dict, *, label: str, probs: list[str]) -> None:
    """要求 ①：驱动**真的返回过**的请求集合恰好等于预期。

    ⚠️ 用 `slots_seen` 而非 `set(tokens)` —— 后者预填全部 req_id，恒真。
    """
    got = set(arm.get("slots_seen") or arm["tokens"])
    want = set(arm["expected_req_ids"])
    if got != want:
        probs.append(f"{label}: 请求集合不符 —— 缺 {sorted(want - got)}；多 {sorted(got - want)}")


def _check_lengths(arm: dict, *, label: str, probs: list[str]) -> None:
    """要求 ②：每请求 token 数 == 由 `SCHEDULE` 派生的预期，且总数对账。"""
    for rid, exp in EXPECT["tokens_per_req"].items():
        n = len(arm["tokens"].get(rid, []))
        if n != exp:
            probs.append(f"{label}: {rid} 产出 {n} 个 token（预期 {exp}）")
    total = sum(len(v) for v in arm["tokens"].values())
    if total != EXPECT["total"]:
        probs.append(f"{label}: token 总数 {total} != 预期 {EXPECT['total']}")


def _check_no_token_places(arm: dict, *, label: str, probs: list[str]) -> None:
    """要求 ③④：不应产出 token 的位置必须真的没有 token。

    ⚠️ 判据不能拿"最终序列长度"比记录位置 —— False 之后还会继续 decode，
    序列当然更长。记录的是**当时那一刻**的长度，正确判据是 `pos < 最终长度`
    （等于 ⇒ 那之后一个都没产出 ⇒ 疑似截断；大于 ⇒ 记录不自洽）。
    """
    if len(arm["no_token_positions"]) != EXPECT["n_false_rows"]:
        probs.append(
            f"{label}: 记到 {len(arm['no_token_positions'])} 处 no-token 位置，"
            f"预期 {EXPECT['n_false_rows']} 处 ⇒ 要求 ③④ 覆盖不足/过量")
    for step, rid, pos in arm["no_token_positions"]:
        seq = arm["tokens"].get(rid, [])
        if pos > len(seq):
            probs.append(f"{label}: {step} {rid} 记录位置 {pos} 超过最终长度 {len(seq)} ⇒ 不自洽")
        elif pos == len(seq):
            probs.append(f"{label}: {step} {rid} 在位置 {pos} 之后**再未产出 token**（疑似截断）")


def _check_mixed_group(arm: dict, *, label: str, probs: list[str]) -> None:
    """要求 ④：**真实混合组**必须存在且被走到（一组里既有 True 又有 False）。"""
    n_mixed = sum(1 for s in arm["per_step"] if s["expect_no_token"] and
                  len(s["expect_no_token"]) < s["n_items"])
    if n_mixed != EXPECT["n_mixed_groups"]:
        probs.append(f"{label}: 走到 {n_mixed} 个混合组，预期 {EXPECT['n_mixed_groups']} 个")


def _check_skip_count(arm: dict, *, label: str, probs: list[str]) -> None:
    """要求 ⑤：`n_sampling_rows_skipped` 必须 > 0（否则这条路没走到）。"""
    sn = (arm.get("execution_optimizations") or {}).get("sample_needed_only") or {}
    skipped = sn.get("n_sampling_rows_skipped")
    if skipped is None:
        probs.append(f"{label}: 驱动未报告 `n_sampling_rows_skipped` ⇒ 无法证明跳过发生")
    elif not isinstance(skipped, int) or skipped <= 0:
        probs.append(f"{label}: `n_sampling_rows_skipped = {skipped}`（要求 > 0）"
                     "⇒ sample_needed_only 未真正生效")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-a", required=True, help="对照臂（S01 关闭）")
    ap.add_argument("--config-b", required=True, help="处理臂（S01 开启）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--env-fingerprint", type=Path,
                    default=_ROOT / "results" / "raw" / "_env" / "env.json",
                    help="当前执行主机预先采集的环境指纹（只读，保存摘要与采集时间）")
    ap.add_argument("--run-id", default=None,
                    help="数值产物编号（默认按时间戳生成，保证唯一）")
    ap.add_argument("--driver", choices=("cpu", "auto", "fake"), default="cpu",
                    help="cpu=真实 CPU 微型驱动（默认，本机自证）；"
                         "**auto=真机（真机必须显式写，否则会静默跑 CPU 微型驱动）**；"
                         "fake=仅跑脚本逻辑")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="**默认拒绝覆盖**已存在的产物；仅显式给出时才覆盖")
    ap.add_argument("--require-real-hw", action="store_true",
                    help="★ 要求 `--driver auto`；否则直接拒绝"
                         "（防『漏写真机』⇒ 把本机证据当 S4000 证据）")
    args = ap.parse_args()

    if args.require_real_hw and args.driver != "auto":
        print(f"✗ --require-real-hw 要求 --driver auto，当前是 {args.driver!r}\n"
              f"  默认值是 'cpu' ⇒ 漏写会让本机微型驱动的结果被当成真机证据。",
              file=sys.stderr)
        return 2
    if args.require_real_hw and (not args.json or args.allow_overwrite):
        print("✗ --require-real-hw 必须指定唯一 --json 产物且禁止 --allow-overwrite",
              file=sys.stderr)
        return 2

    cfg_a, cfg_b = Path(args.config_a), Path(args.config_b)
    for p in (cfg_a, cfg_b):
        if not p.is_file():
            print(f"✗ 配置不存在：{p}", file=sys.stderr)
            return 2

    ra = load_config(cfg_a).raw["runtime"]
    rb = load_config(cfg_b).raw["runtime"]
    if bool(ra.get("logits_last_only")) or bool(ra.get("sample_needed_only")):
        print("✗ 对照臂的 S01 开关**不是关闭**的 ⇒ 这次比较没有意义", file=sys.stderr)
        return 2
    if not (bool(rb.get("logits_last_only")) and bool(rb.get("sample_needed_only"))):
        print("✗ 处理臂的 S01 开关**不是全开** ⇒ 这次比较没有意义", file=sys.stderr)
        return 2
    if ra.get("model_revision") != rb.get("model_revision"):
        print("✗ 两臂的 model_revision 不同 ⇒ 多了一个变量", file=sys.stderr)
        return 2

    # ★ 产物禁止覆盖：历史证据不得被静默改写
    outp = Path(args.json) if args.json else None
    if outp is not None and outp.exists() and not args.allow_overwrite:
        print(f"✗ 产物已存在，拒绝覆盖：{outp}\n"
              f"  换一个 --run-id/路径，或显式加 --allow-overwrite。", file=sys.stderr)
        return 2

    run_id = args.run_id or f"s01num-{time.strftime('%Y%m%d-%H%M%S')}"
    # 运行前取一次，四类身份和兼容字段复用同一快照。
    source_identity = _git_identity()
    environment = _environment_snapshot(args.env_fingerprint)
    environment["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[numeric] run_id = {run_id}（driver={args.driver}）")
    print(f"[numeric] 对照 = {cfg_a.name}（S01 关）")
    print(f"[numeric] 处理 = {cfg_b.name}（S01 开）")
    print(f"[numeric] 计划：{EXPECT['n_steps']} 组；每请求 {GEN} token、总 {EXPECT['total']}；"
          f"False 行 {EXPECT['n_false_rows']}；混合组 {EXPECT['n_mixed_groups']}")

    try:
        a = _run_arm(cfg_a, driver=args.driver)
        b = _run_arm(cfg_b, driver=args.driver)
    except Exception as exc:
        print(f"✗ 执行驱动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    probs: list[str] = []
    print("\n-- ① 请求集合（恰好等于预期）--")
    _check_expected_set(a, label="对照", probs=probs)
    _check_expected_set(b, label="处理", probs=probs)
    print(f"  预期集合 = {a['expected_req_ids']}")

    print("\n-- ② token 序列逐位比较 --")
    ids = sorted(set(a["tokens"]) | set(b["tokens"]))
    for rid in ids:
        ta, tb = a["tokens"].get(rid), b["tokens"].get(rid)
        same = ta == tb
        if not same:
            probs.append(f"token 序列不同：{rid} 对照 {ta} / 处理 {tb}")
        print(f"  {'✓' if same else '✗'} {rid}: 对照 {ta} / 处理 {tb}")

    print("\n-- ③ 每请求长度与总数对账（**两臂各自**，不只看两臂相等）--")
    _check_lengths(a, label="对照", probs=probs)
    _check_lengths(b, label="处理", probs=probs)
    print(f"  预期每请求 {EXPECT['tokens_per_req']}；总 {EXPECT['total']}")
    print(f"  对照 {sum(len(v) for v in a['tokens'].values())} / "
          f"处理 {sum(len(v) for v in b['tokens'].values())}")

    print("\n-- ④ 不应产出 token 的位置与混合组 --")
    _check_no_token_places(a, label="对照", probs=probs)
    _check_no_token_places(b, label="处理", probs=probs)
    _check_mixed_group(a, label="对照", probs=probs)
    _check_mixed_group(b, label="处理", probs=probs)
    for st in a["per_step"]:
        mark = "混合" if st["expect_no_token"] and len(st["expect_no_token"]) < st["n_items"] else ""
        print(f"  {st['step']:<20} 行{st['n_items']} slot{st['n_slots']} {mark}")

    print("\n-- ⑤ 逐步结构违反项（slot 数量/唯一性/集合/顺序/输出位置）--")
    for label, arm in (("对照", a), ("处理", b)):
        if arm["violations"]:
            for v in arm["violations"]:
                probs.append(f"{label}: {v}")
                print(f"  ✗ {label}: {v}")
        else:
            print(f"  ✓ {label}: 无（{len(arm['per_step'])} 组逐步检查全过）")

    print("\n-- ⑥ 跳过计数（sample_needed_only 是否真的生效）--")
    _check_skip_count(b, label="处理", probs=probs)
    if args.driver != "auto":
        probs.extend(_optimization_problems([a, b]))
    for label, arm in (("对照", a), ("处理", b)):
        sn = ((arm.get("execution_optimizations") or {}).get("sample_needed_only") or {})
        print(f"  {label}: requested={sn.get('requested')} effective={sn.get('effective')} "
              f"n_sampled_rows={sn.get('n_sampled_rows')} "
              f"n_sampling_rows_skipped={sn.get('n_sampling_rows_skipped')}")

    print("\n-- ⑥b 真机设备 / 模型 revision / 优化生效（**仅 --driver auto**）--")
    if args.driver != "auto":
        print("  ⚠️ 本项**不适用**（非真机驱动）⇒ **不构成通过**，也不得据此说『设备已核验』")
        print(f"  当前证据等级 = {('LOCAL_CPU_MICRO' if args.driver == 'cpu' else 'SCRIPT_LOGIC_ONLY')}"
              " ⇒ **不得**升级为 S4000 证据")
    else:
        hw_probs = _hardware_problems(drv=None, cfg=load_config(cfg_b),
                                      arms=[a, b], driver=args.driver)
        for p in hw_probs:
            probs.append(p)
            print(f"  ✗ {p}")
        if not hw_probs:
            print("  ✓ 实际设备为 MUSA、resolved revision 与配置一致、优化真的生效")
        for label, arm in (("对照", a), ("处理", b)):
            di = arm.get("driver_identity") or {}
            print(f"  {label}: kind={di.get('kind')} device={di.get('device')} "
                  f"rev_req={di.get('model_revision')} "
                  f"rev_resolved={di.get('model_revision_resolved')}")

    print("\n-- ⑦ 末位置 logits 摘要（若驱动开了 capture_logits_digest）--")
    da, db = a["logits_digests"], b["logits_digests"]
    if not da and not db:
        logits_status = "未验证（两臂都未采集 logits 摘要）"
        print("  （两臂都未采集 ⇒ 本项**未验证**，不是通过）")
    else:
        same_lg = da == db
        if not same_lg:
            probs.append(f"logits 摘要不同（对照 n={len(da)} / 处理 n={len(db)}）")
        logits_status = "PASS" if same_lg else "FAIL"
        print(f"  对照 n={len(da)} 处理 n={len(db)} ⇒ {'✓ 相同' if same_lg else '✗ 不同'}")

    print("\n-- ⑧ 整体摘要（sha256，比摘要而非个数）--")
    print(f"  对照 {a['token_digest']}")
    print(f"  处理 {b['token_digest']}")
    same_digest = a["token_digest"] == b["token_digest"]
    if not same_digest:
        probs.append("token 序列整体 sha256 不同")
    print(f"  {'✓ 相同' if same_digest else '✗ 不同'}")

    print("\n-- ⑨ 证据等级提示 --")
    hw = a.get("hardware_backed")
    if hw is False or args.driver in ("cpu", "fake"):
        print("  ⚠️ 本次为**本机 CPU 微型驱动**（hardware_backed=False）⇒ 证明的是"
              "**接口与判据语义**，**不是**硬件数值行为，**不得**当性能/标定证据。")
    else:
        print("  ✓ 硬件后端执行（真机数值验收）。")

    ok = not probs
    ev = ("LOCAL_CPU_MICRO" if args.driver == "cpu"
          else "SCRIPT_LOGIC_ONLY" if args.driver == "fake"
          else "REAL_HARDWARE")
    payload = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "exit_code": 0 if ok else 1,
        "driver_mode": args.driver,
        "config_a": str(cfg_a), "config_b": str(cfg_b),
        # ★ 身份绑定：源码 / 配置 / 模型 / 环境 四类同源记录（用户点名）
        "identity": {
            "source": source_identity,
            "config": {"a": str(cfg_a), "b": str(cfg_b),
                       "a_sha256": _file_sha256(cfg_a),
                       "b_sha256": _file_sha256(cfg_b)},
            "model": {"revision_requested": _requested_revision(load_config(cfg_b)),
                      "revision_resolved_a": (a.get("driver_identity") or {}).get(
                          "model_revision_resolved"),
                      "revision_resolved_b": (b.get("driver_identity") or {}).get(
                          "model_revision_resolved")},
            "environment": {**environment, "a": _env_identity(load_config(cfg_a)),
                            "b": _env_identity(load_config(cfg_b)),
                            "device": (b.get("driver_identity") or {}).get("device"),
                            "driver_kind": (b.get("driver_identity") or {}).get("kind"),
                            "hardware_backed": b.get("hardware_backed")},
        },
        "source_identity": source_identity,
        "env_identity_a": _env_identity(load_config(cfg_a)),
        "env_identity_b": _env_identity(load_config(cfg_b)),
        "expected": EXPECT,
        "schedule": [{"step": s, "items": [list(it) for it in plan]} for s, plan in SCHEDULE],
        "arm_a": a, "arm_b": b,
        "logits_status": logits_status,
        "problems": probs,
        "ok": ok,
        "evidence_class": ev,
        "evidence_scope": ("本机 CPU 微型驱动 ⇒ 只证明**接口与判据语义**，"
                           "**不是**硬件数值行为；**不得**升级为 S4000 证据"
                           if ev == "LOCAL_CPU_MICRO" else
                           "真机 MUSA 执行 ⇒ 可作硬件数值证据"
                           if ev == "REAL_HARDWARE" else
                           "假驱动 ⇒ 只证明脚本自身逻辑，**不是**证据"),
        "verdict": "PASS 两臂数值逐位相同" if ok else "FAIL 见 problems",
    }
    if outp is not None:
        outp.parent.mkdir(parents=True, exist_ok=True)
        # ★★ 非有限诊断值 → `null` + 错误说明，并**严格**输出 JSON（用户 2026-10-03 点名）。
        clean, jnotes = sanitize_json(payload)
        clean["_json_notes"] = jnotes
        outp.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"\n写出 {outp}")
        if jnotes:
            print(f"  ⚠️ 有 {len(jnotes)} 个非有限诊断值已置 `null`（详见产物 `_json_notes`）")
            for n in jnotes[:5]:
                print(f"      · {n}")

    print(f"\n★ 数值验收 = {'PASS' if ok else 'FAIL'}")
    for p in probs:
        print(f"  ✗ {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
