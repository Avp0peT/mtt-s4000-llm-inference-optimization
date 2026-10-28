#!/usr/bin/env python3
"""设备侧成本标定探针（协议 **`cost-calib-v1`**）—— 第 11 步。

与既有探针的关系（**在 v5 基础上改，不另写一份**）
------------------------------------------------
形状构造与计时原语**逐字复用** `results/raw/_probe/20260924/scripts/fwd_faststate_probe.py`
（v5）里的 `build_past_for_hf` / `build_past` / `make_batch` / `make_fwd` /
`async_chunk` / `sync_series` / `submit_cost`。
那份探针的注释已写明它"结构上与 harness 完全同构"—— 那是本探针能用的前提，不要重写。

本探针相对 v5 新增的四件事（都是 **pre-registration 要求的**）
------------------------------------------------------------
1. **网格驱动**：采样点来自 `configs/cost_calib/shape_grid_v1.yaml`（取值点**先定**），
   而不是写死在代码里。v5 只覆盖纯 decode（`n_new=1`），本探针要覆盖 `prefill_tok>0`
   与"一步多个串行前向"。
2. **两个块分开采、分开报**：
   * 块 A = `async_chunk`（连续提交 + 块尾 sync，占空比 ~100%）⇒ **只作性能下界参考**；
   * 块 B = **逐步 sync + KV 真实生长**（v5 的 S5 MIMIC 协议）⇒ **被拟合的是这个**。
   验收明确禁止"只拟合快态最小值"，所以两者的角色写在产物里（`role` 字段），
   并由 `scripts/costcalib_grid.py` 的守卫兜住。
3. **协议版本 / 会话 id / 指纹**落进产物（含**驱动版本**——R14 刚把它接成字段）。
4. **保留原始序列**（每个 call 的耗时），不只存汇总 ⇒ 事后可重算。

⚠️ 纪律（继承 v5，**不要改**）
------------------------------
**这不是 harness 产物**（无 `meta.json`、无标签盖章）⇒
**结论不得进 `EXPERIMENTS.md` 的结论行，只能进 `RESEARCH_LOG`**。

用法（在远端跑）::

    source /root/musa-envs/qwen310/activate_musa.sh
    python fwd_costcalib_probe.py --grid configs/cost_calib/shape_grid_v1.yaml \\
        --out /root/ascend_probe/adhoc/costcalib_v1_<session>.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from costcalib_grid import Shape, expand, load_grid, validate  # noqa: E402

T0 = time.monotonic()  # 进程时间线原点：在任何重活之前

#: harness 的上下文上限（prompt 2048 + gen 128）。块 B 的 KV 生长封顶到这里 ——
#: 实际服务里 KV 也只长到这里，再往上就不是被测系统的形状了。
KV_CAP = 2176

# ⚠️ MUSA 相关的一切都**延迟到运行期**加载：本模块的纯逻辑部分要能被离线单测覆盖，
# 而测试机上没有 torch_musa。
musa = None            # type: ignore[assignment]
torch = None           # type: ignore[assignment]
torch_musa = None      # type: ignore[assignment]


def _load_runtime() -> None:
    """真机上才加载设备栈。v5 是模块级 import；这里挪进函数，换取"纯逻辑可测"。"""
    global musa, torch, torch_musa
    import torch as _torch
    import torch_musa as _torch_musa

    torch = _torch
    torch_musa = _torch_musa
    musa = _torch.musa


# --------------------------------------------------------------------------- #
# 纯逻辑（**不碰设备**）：汇总与产物组装 —— 这部分是单元测试的落点
# --------------------------------------------------------------------------- #


def summarize_block_a(blocks_of_calls: Sequence[Sequence[float]]) -> dict[str, Any]:
    """块 A（快态平台）：取**块最小值**为该格值（`COST_PROFILE` §4）。

    取最小值不是"挑好看的"：占空比 ~100% 时设备锁在快态，**最小值就是快态平台**；
    平均值会把"偶发被打回慢档"的块混进来。多块的**极差**同时报出来，
    它是"这条协议是否真的钉住了快态"的判据。
    """
    per_block = [min(b) for b in blocks_of_calls if b]
    if not per_block:
        return {"role": "lower_bound_only", "n_blocks": 0,
                "per_block_min_ms": [], "platform_min_ms": None,
                "block_spread_ms": None}
    return {
        "role": "lower_bound_only",          # ⚠️ 只作下界参考，**不得用于拟合**
        "n_blocks": len(per_block),
        "per_block_min_ms": [round(v, 3) for v in per_block],
        "platform_min_ms": round(min(per_block), 3),
        "block_spread_ms": round(max(per_block) - min(per_block), 3),
    }


def summarize_block_b(calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """块 B（服务协议）：逐步墙钟的分布 —— **被拟合的量**。

    报 `p50 / p90 / max`**和原始序列**：单看中位会把"偶发的慢步"抹掉，而
    SLO 违约往往正是那些慢步造成的（与 `stall` 的教训同源）。
    """
    ms = [c["ms"] for c in calls if c.get("ms") is not None]
    if not ms:
        return {"role": "fitted", "n_calls": 0, "p50_ms": None, "p90_ms": None,
                "max_ms": None, "calls": []}
    s = sorted(ms)
    return {
        "role": "fitted",                    # ⚠️ 这个才是拟合对象
        "n_calls": len(ms),
        "p50_ms": round(statistics.median(s), 3),
        "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
        "max_ms": round(s[-1], 3),
        "calls": calls,                      # 原始序列：事后可重算，不只留汇总
    }


def fingerprint(runtime_env: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    """组装指纹：**预期值来自网格文件**，实测值来自运行期环境。

    两者都留着，是为了"失配可判定"—— 只看实测值没法知道它是否还是标定时的那个环境。
    """
    expected = grid.get("fingerprint") or {}
    return {
        "expected": expected,
        "observed": runtime_env,
        "grid_protocol_version": grid.get("protocol_version"),
    }


def build_artifact(*, session_id: str, t0_monotonic_s: float, grid: dict[str, Any],
                   grid_path: str, points: list[dict[str, Any]],
                   runtime_env: dict[str, Any], complete: bool,
                   submit_cost_ms: list[float] | None = None,
                   notes: str = "") -> dict[str, Any]:
    """把逐点结果组装成产物（纯函数）。

    ⚠️ 两个块的 `role` 由 :func:`summarize_block_a` / :func:`summarize_block_b` 带着走，
    这里**不再改**它们 —— 一旦"哪个块是拟合对象"能被组装层改写，
    验收第 2 条（不得只拟合快态最小值）就守不住了。
    """
    return {
        "probe": "fwd_costcalib_probe",
        "protocol_version": grid.get("protocol_version"),
        "session_id": session_id,
        # ⚠️ 单调时钟值**不是** epoch 时间：叫它 epoch 会让人拿它去和别的日志对时。
        "t0_monotonic_s": round(t0_monotonic_s, 3),
        "elapsed_s": round(time.monotonic() - T0, 1),
        "grid_path": grid_path,
        "grid_sha256": hashlib.sha256(
            json.dumps(grid, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16],
        "fingerprint": fingerprint(runtime_env, grid),
        "session_policy": grid.get("session") or {},
        "submit_cost_ms": submit_cost_ms or [],
        "points": points,
        "complete": complete,
        "note": ("设备侧探针，无 harness 盖章；结论只能进 RESEARCH_LOG，"
                 "不得进 EXPERIMENTS.md 结论行"),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# 形状构造与计时原语：**逐字复用 v5**（见模块 docstring 的出处）
# --------------------------------------------------------------------------- #


def cache_needs_object() -> bool:
    try:
        from transformers import cache_utils
    except Exception:
        return False
    dyn = getattr(cache_utils, "DynamicCache", None)
    return dyn is not None and hasattr(dyn, "from_legacy_cache")


def build_past_for_hf(kv):
    if not kv:
        return ()
    legacy = tuple((k, v) for k, v in kv)
    if cache_needs_object():
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def build_past(model, batch: int, kv_lens: list[int], dtype, device):
    """复刻 `_pack_past` 的形状：逐层 (key, value)，左补位到 max(kv_lens)。【v5 原文】"""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    past_max = max(kv_lens)
    layers = []
    for _ in range(n_layers):
        rows_k, rows_v = [], []
        for L in kv_lens:
            k = torch.zeros(1, n_kv, L, head_dim, dtype=dtype, device=device)
            v = torch.zeros(1, n_kv, L, head_dim, dtype=dtype, device=device)
            if L < past_max:
                pad = past_max - L
                k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                v = torch.nn.functional.pad(v, (0, 0, pad, 0))
            rows_k.append(k)
            rows_v.append(v)
        layers.append((torch.cat(rows_k, dim=0), torch.cat(rows_v, dim=0)))
        del rows_k, rows_v
    return layers, past_max


def batch_layout(kv_lens: list[int], n_new_spec) -> tuple[int, int, list[tuple[int, int]]]:
    """纯逻辑：算出 `(每序列的新 token 数最大值, 总列数, [(左 pad, 该序列的有效长度)])`。

    抽出来是为了**不依赖 torch 也能测**（真实批量要求矩形张量 ⇒ 短的那几条要右侧 padding）。

    ``n_new_spec`` 可以是 **int**（旧口径：所有序列相同）或 **list[int]**（逐序列不同）。
    """
    n_list = ([int(n_new_spec)] * len(kv_lens) if isinstance(n_new_spec, int)
              else [int(x) for x in n_new_spec])
    if len(n_list) != len(kv_lens):
        raise ValueError(f"n_new 列表长度 {len(n_list)} != kv_lens 长度 {len(kv_lens)}")
    if min(n_list) < 1:
        raise ValueError(f"每条序列至少要 1 个新 token（收到 {min(n_list)}）")
    past_max = max(kv_lens)
    max_n = max(n_list)
    rows = [(past_max - L, L + n) for L, n in zip(kv_lens, n_list)]
    return max_n, past_max + max_n, rows


def make_batch(model, batch: int, kv_lens: list[int], n_new, dtype, device):
    """构造一步的输入。

    ⚠️ **`n_new` 的口径**（第 14 步实测的关键）：
    * ``int`` ⇒ 旧的"所有序列都做这么多"（v1..v5 的 `mixed_serving` 用的是这个）；
    * ``list[int]`` ⇒ **真实结构**：一步里**一条**序列承担 prefill，其余各只做 1 个 token。
      实测证据：遥测 `note` 里 `prefill_tok=512 decode=7` ⇒ 512 是**全步总量**、给**一条**请求，
      而旧口径把它当成"每条都做 512" ⇒ 差约 `batch` 倍。
    """
    n_list = ([int(n_new)] * batch if isinstance(n_new, int) else [int(x) for x in n_new])
    max_n, total, rows = batch_layout(kv_lens, n_list)
    ids = torch.randint(1, 1000, (batch, max_n), dtype=torch.long, device=device)
    mask = torch.zeros((batch, total), dtype=torch.long, device=device)
    for i, (left, valid) in enumerate(rows):
        # 有效区间 = [左 pad, 左 pad + 该序列的 (KV + 新 token))；右侧超出 max_n 的部分保持 0
        mask[i, left:left + valid] = 1
    pos = torch.stack([
        torch.cat([torch.arange(L, L + n, dtype=torch.long, device=device),
                   torch.zeros(max_n - n, dtype=torch.long, device=device)])
        for L, n in zip(kv_lens, n_list)
    ])
    return ids, mask, pos


def make_fwd(model, ids, mask, pos, past):
    def fn():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask,
                         position_ids=pos,
                         past_key_values=build_past_for_hf(past),
                         use_cache=True)
    return fn


def async_chunk(fn, n: int) -> float:
    """连续提交 n 次前向、块尾 sync 一次 ⇒ 占空比 ~100%。【v5 原文】**块 A**。"""
    musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    musa.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def sync_series(fn, n: int) -> list[dict[str, Any]]:
    """逐步 sync（含 python 间隙）⇒ harness 的节拍。**块 B 的计时部分**。【v5 原文】"""
    musa.synchronize()
    out = []
    for i in range(n):
        wall = round(time.monotonic() - T0, 2)
        t0 = time.perf_counter()
        fn()
        musa.synchronize()
        out.append({"i": i, "t": wall, "ms": round((time.perf_counter() - t0) * 1000.0, 3)})
    return out


def submit_cost(fn, n: int = 10) -> list[float]:
    """主机侧提交成本：每次 fn() 的 host 耗时（不同步）。【v5 原文】"""
    musa.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(round((time.perf_counter() - t0) * 1000.0, 3))
    musa.synchronize()
    return ts


# --------------------------------------------------------------------------- #
# 运行期
# --------------------------------------------------------------------------- #


def _kb(shape: Shape) -> str:
    return f"b{shape.batch}_pf{shape.prefill_tok}_kv{shape.kv_len}_nf{shape.n_forwards}"


def grown_kv_lens(kv: int, step: int, n_list: Sequence[int], cap: int) -> list[int]:
    """「服务协议」块里第 `step` 步的 **每条序列** 的 KV 长度。

    ⚠️ 为什么必须**按序列**算（这是一个真实事故的修复）
    ---------------------------------------------------
    旧实现是 `L = min(kv_lens[0] + i * n_new, KV_CAP)`，然后 `[L] * b`。
    在 `n_new` 是 `int`（`uniform` 口径）时它是对的；但 `real_mixed` 把 `n_new`
    变成 **list**（一条做 `prefill_tok`、其余各 1 个）⇒ `i * n_new` 成了 list、
    再 `int + list` ⇒ **`TypeError`**。

    **后果（实测）**：`real_mixed` 这条路径**从未跑通过** ——
    v6 标定的 96 个 `real_mixed` 点**失败 84 个**（`mixed_serving` 全灭），
    D1 pilot 的 16 个小 chunk 点**失败 12 个**（`batch>1` 的全灭，只有 `batch=1` 能过）。
    而标定脚本**照常把产物写出来**、覆盖率闸门只看到"覆盖率 53.57% 不足"
    ⇒ 表现成"模型没建好"，而不是"代码有 bug"。

    ⇒ 按序列各自增长（这本来就是真实语义：每条的 KV 按自己新生成的 token 长）。

    ⚠️ **`uniform` 下与旧实现逐值相同**（`[kv + step*n]*b`）—— 由用例反面钉住，
    免得"修 real_mixed"顺手改掉了旧口径、让 v1..v5 的可比性失效。
    """
    return [max(min(int(kv) + int(step) * int(n), int(cap)), 1) for n in n_list]


def measure_point(model, shape: Shape, dtype, device, blocks: dict[str, Any]) -> dict[str, Any]:
    """采一个采样点的两个块。

    形状约定：
    * `n_new` = 本步的新 token 数 = `prefill_tok`（纯 decode 点该值为 0 ⇒ 取 1）；
    * 块 A 用**固定** `kv_len`（测快态平台，不该让形状在块内变化）；
    * 块 B **让 KV 真实生长**（`min(kv_len + i*n_new, KV_CAP)`）—— 这是"服务协议"的关键，
      harness 里 KV 就是随步数长的。
    """
    b, kv = shape.batch, shape.kv_len
    # ⚠️ **新 token 的分配方式**（第 14 步实测的关键口径）：
    #   `uniform`   = 每条序列都做 `prefill_tok` 个（v1..v5 的 mixed_serving 用的）；
    #   `real_mixed`= **一条**序列承担 `prefill_tok` 个、其余各只做 **1** 个
    #                 —— 这才是 harness 的真实结构（`note` 里 `prefill_tok=512 decode=7`）。
    # 两者差约 `batch` 倍，是本项目一个**跨口径的系统误差**来源。
    _pf = int(shape.prefill_tok)
    _mode = str(getattr(shape, "n_new_mode", None) or "uniform")
    if _mode == "real_mixed" and _pf > 0 and b > 1:
        n_new = [_pf] + [1] * (b - 1)
    else:
        n_new = _pf if _pf > 0 else 1
    kv_lens = [max(kv, 1)] * b

    # ---- 块 A：快态平台（固定形状、连续提交）----
    past, _ = build_past(model, b, kv_lens, dtype, device)
    ids, mask, pos = make_batch(model, b, kv_lens, n_new, dtype, device)
    fn_a = make_fwd(model, ids, mask, pos, past)
    nn = shape.n_forwards
    blocks_a: list[list[float]] = []
    for _ in range(blocks["A"]["n_blocks"]):
        if nn == 1:
            blocks_a.append([async_chunk(fn_a, blocks["A"]["block_n"])])
        else:
            # 一步多个**串行**前向：块内的 per-call 成本按"整组前向"记，再除以 nn。
            musa.synchronize()
            t0 = time.perf_counter()
            for _ in range(blocks["A"]["block_n"]):
                for _ in range(nn):
                    fn_a()
            musa.synchronize()
            per_call = (time.perf_counter() - t0) * 1000.0 / blocks["A"]["block_n"]
            blocks_a.append([per_call / nn])
    del past, ids, mask, pos
    gc.collect()

    # ---- 块 B：服务协议（逐步 sync + KV 真实生长）----
    # ⚠️ 按**序列**各自增长（`real_mixed` 下各序列增长速度不同）；见 `grown_kv_lens` 的注释：
    #    旧写法 `kv_lens[0] + i * n_new` 在 `n_new` 为 list 时抛 `TypeError`，
    #    导致 `real_mixed` 路径**从未跑通**（v6 失败 84/96、D1 pilot 失败 12/16）。
    n_list = ([int(n_new)] * b) if isinstance(n_new, int) else [int(x) for x in n_new]
    calls: list[dict[str, Any]] = []
    for i in range(blocks["B"]["n_calls"]):
        kv_lens_i = grown_kv_lens(kv, i, n_list, KV_CAP)
        past_i, _ = build_past(model, b, kv_lens_i, dtype, device)
        ids_i, mask_i, pos_i = make_batch(model, b, kv_lens_i, n_new, dtype, device)
        fn_i = make_fwd(model, ids_i, mask_i, pos_i, past_i)
        calls.extend(sync_series(fn_i, 1))
        del past_i, ids_i, mask_i, pos_i
    gc.collect()

    return {"shape": shape.__dict__, "key": _kb(shape),
            "A": summarize_block_a(blocks_a), "B": summarize_block_b(calls)}


def select_shapes(shapes: Sequence[Shape], only_group: Sequence[str] | None,
                  limit: int | None) -> list[Shape]:
    """从**已展开**的点里选出本轮要采的那些（纯函数，便于离线测试）。

    存在理由：`expand()` 返回的是**按 `as_key` 排序**的列表，
    **不是**网格文件里的顺序 ⇒ `--limit N` 取到的永远是排序最前的点（`pure_decode`），
    **当不了分组过滤**。实测踩到过（我以为"把新组放文件第一位"就能用 `--limit 16` 采它）。

    顺序：**先按组过滤，再取前 N**（`--limit` 只用于冒烟，别指望它选点）。
    组名不存在时**抛 `ValueError`**（含可选组名清单）—— 静默选出 0 点、或选出别的组，
    都会让"我采了这一组"这句话变成假的。
    """
    out = list(shapes)
    if only_group:
        want = set(only_group)
        known = {s.group for s in out}
        unknown = sorted(want - known)
        if unknown:
            raise ValueError(f"网格里没有这些组：{unknown}；可选：{sorted(known)}")
        out = [s for s in out if s.group in want]
    if limit:
        out = out[: int(limit)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/cost_calib/shape_grid_v1.yaml")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-id", default=None,
                    help="不填则自动生成 <UTC>-<host前8>-<序号>")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个点（冒烟用）")
    #: ★ 真正的**按组过滤**。为什么需要：`expand()` 返回的是**按 key 排序**的列表
    #: （`sorted(out.values(), key=lambda s: s.as_key())`），**不是网格文件里的顺序**
    #: ⇒ `--limit N` 取到的永远是排序最前的那些点（`pure_decode`），**当不了分组过滤**。
    #: 实测踩到：本以为"把新组放到文件第一位"就能用 `--limit 16` 采它 —— 错。
    #: （`--limit` 仍保留，但只用于冒烟；要选点请用本开关。）
    ap.add_argument("--only-group", nargs="+", default=None,
                    help="只采这些组（按 `id`）。子集信息会如实写进产物，"
                         "避免 `complete=True` 被读成'整个网格都采完了'")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    problems = validate(grid)
    if problems:
        for x in problems:
            print(f"✗ 网格校验失败：{x}", file=sys.stderr)
        return 2
    shapes_all = expand(grid)
    try:
        shapes = select_shapes(shapes_all, args.only_group, args.limit)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    if not shapes:
        print("✗ 选出来 0 个点（过滤器写错了？）", file=sys.stderr)
        return 2
    blocks = grid["blocks"]

    import datetime as _dt
    session_id = args.session_id or (
        f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{platform.node()[:8]}-{int(T0) % 100000}"
    )
    print(f"[costcalib] protocol={grid['protocol_version']} session={session_id} "
          f"points={len(shapes)}", flush=True)

    _load_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16
    device = torch.device("musa")
    t_load = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    AutoTokenizer.from_pretrained(args.model)
    print(f"[costcalib] 模型载入 {time.monotonic()-t_load:.1f}s", flush=True)

    runtime_env = {
        "device_name": musa.get_device_name(0),
        "torch": torch.__version__,
        "torch_musa": getattr(torch_musa, "__version__", None),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    import transformers
    runtime_env["transformers"] = transformers.__version__

    points: list[dict[str, Any]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush(complete: bool, notes: str = "") -> None:
        art = build_artifact(session_id=session_id, t0_monotonic_s=T0, grid=grid,
                             grid_path=args.grid, points=points,
                             runtime_env=runtime_env, complete=complete, notes=notes)
        out_path.write_text(json.dumps(art, ensure_ascii=False, indent=1),
                            encoding="utf-8", newline="\n")

    # 主机提交成本（async 解释的锚点）
    past0, _ = build_past(model, 1, [1024], dtype, device)
    ids0, mask0, pos0 = make_batch(model, 1, [1024], 1, dtype, device)
    sub = submit_cost(make_fwd(model, ids0, mask0, pos0, past0), 10)

    for i, s in enumerate(shapes, 1):
        print(f"[costcalib] {i}/{len(shapes)} {_kb(s)} t={time.monotonic()-T0:.0f}s", flush=True)
        try:
            points.append(measure_point(model, s, dtype, device, blocks))
        except Exception as e:  # noqa: BLE001
            import traceback
            points.append({"shape": s.__dict__, "key": _kb(s),
                           "error": repr(e), "traceback": traceback.format_exc()})
        flush(complete=False)

    # 收尾时把 submit_cost 一起写进去（build_artifact 支持，但要在最后组装）
    art = build_artifact(session_id=session_id, t0_monotonic_s=T0, grid=grid,
                         grid_path=args.grid, points=points, runtime_env=runtime_env,
                         complete=True, submit_cost_ms=sub)
    # ⚠️ **子集必须显形**：`complete=True` 的含义是"**被选中的点**都测完了"，
    # 而不是"整个网格都测完了"。不写这几个字段，子集产物与全量产物在形状上无法区分
    # —— 而"这轮采的是不是整个网格"正是后面能不能把它当一次完整会话的依据。
    art["subset"] = {
        "only_group": list(args.only_group) if args.only_group else None,
        "limit": args.limit,
        "n_points_selected": len(shapes),
        "n_points_in_grid": len(shapes_all),
        "is_full_grid": (not args.only_group and not args.limit),
    }
    out_path.write_text(json.dumps(art, ensure_ascii=False, indent=1),
                        encoding="utf-8", newline="\n")
    print(f"[costcalib] 完成 -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
