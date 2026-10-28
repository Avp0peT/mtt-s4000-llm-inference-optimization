"""策略臂的**决策与应用**：两条执行路径**共用**的唯一实现（第 12/13 步）。

⚠️ 为什么必须共用（而不是各写一份）
------------------------------------
本仓库已经因为"两条路径各写一份"吃过亏 —— 见 `step_planner` 的模块 docstring：
*"再写一遍会带来一种不会报错的失效：以后修了模拟器里的某一处闸门，真实路径不会跟着修。"*

**实测也踩到了**：第 13 步第一版只把臂接进模拟器（`simulate()`），真机路径（`real_backend`）
没有 ⇒ 真机跑出来 `slo_weave` 决策数 = **0**、两臂产物**逐项相同**，
而配置里明明写着 `controller.kind: slo_weave`。
那批产物**看起来完全正常**（`MEASURED`、戳齐全、账目自洽）—— 这正是"不会报错的失效"。

因此本模块是臂决策的**唯一**落点：两条路径都调 :func:`apply_arm`，传入各自的
状态对象（两边都提供 `req` / `prefill_remaining` / `last_emit_ms` / `decode_remaining`）。

本模块纯逻辑，不导入任何硬件/框架库。
"""

from __future__ import annotations

from typing import Any, Sequence

from .chunk_policy import ChunkLimits
from .cost_policy import CostPolicy
from .slo_weave import SLOWeavePolicy

__all__ = ["apply_arm", "arm_state"]


def arm_state(arm: Any) -> dict[str, Any]:
    """策略臂自己的计数器（`CostPolicyState.describe()` 等），供落盘。

    为什么必须落盘、且**两条路径共用同一份**
    ----------------------------------------
    `_controller_effect()` 式的汇总读的是**采样**与**轨迹**，而
    `n_degraded`（"有预测但全部超预算 ⇒ 取最省那个"）与
    `n_prefill_starved`（"有 prefill 积压、而所有候选本步都推进不了 prefill"）
    是策略对象的**内部状态**，两者都不在采样里。

    ⚠️ 实测踩到（2026-09-26）：没有这两个计数时，事后只能拿 D0 可行率去**猜**动作方向 ——
    我据此推出"结构上无法发 prefill"，而遥测 `note` 显示 **183/292 步仍在发
    `prefill_tok=1024`**（因为全无预测时走 `_fallback()` 回退到阈值控制器，它照常发）。
    ⇒ 纪律：**"数不出来"不等于"没发生"**；缺计数时只能靠猜，而猜会错。

    没有 `state` 的臂（如静态臂）返回 `{}`，**不虚构**键。
    `describe()` 抛错时把错误**显形**在产物里（不静默吞掉）——
    度量自己坏掉必须能被看见，否则与"没这一项"无法区分。
    """
    st = getattr(arm, "state", None)
    fn = getattr(st, "describe", None)
    if st is None or not callable(fn):
        return {}
    try:
        out = fn()
    except Exception as exc:                                    # noqa: BLE001
        return {"_error": f"arm.state.describe() 失败：{exc!r}"}
    return dict(out) if isinstance(out, dict) else {"_error": "describe() 未返回 dict"}


def apply_arm(
    *, kind: str, policy: Any, shadow: bool, step: int, now_ms: float,
    cur_chunk: int, cur_max_tokens: int,
    waiting: Sequence[Any], decoding: Sequence[Any], prefilling: Sequence[Any],
    scheduler: Any, chunk_policy: Any, cache: Any,
    limits: ChunkLimits, max_num_seqs: int, chunk_enabled: bool,
) -> tuple[int, int, dict[str, Any] | None]:
    """算一次臂决策并（非 shadow 时）应用到 `cur_chunk` / `cur_max_tokens`。

    返回 `(新的 cur_chunk, 新的 cur_max_tokens, 要落盘的轨迹条目或 None)`。

    ⚠️ `shadow=True` ⇒ **只记录、不生效**（验收："shadow 仅证明预测/决策可记录，
    闭环才验证动作实际生效"）。
    """
    if kind == "threshold":
        return cur_chunk, cur_max_tokens, None

    # 代表上下文长度：取在解码请求里**最大**的（prompt + 已产 token）。
    # 用 `gen_len - decode_remaining` 推，**不另加字段**（`getattr(..., 0)` 那种写法在
    # 字段不存在时会静默只给 prompt 长度，让模型恒收到偏小的 kv）。
    kv_len = 0
    if decoding:
        kv_len = max(int(s.req.prompt_len)
                     + (int(s.req.gen_len) - int(s.decode_remaining)) for s in decoding)

    if kind == "slo_weave":
        # `ℓ_i` 用现成的 `last_emit_ms`（每次产 token 都更新）；只对**在解码**的请求取
        # （该字段仅在首个 token 之后有意义）。
        deadlines = [s.last_emit_ms + policy.config.tpot_target_ms for s in decoding]
        pending = [(s.req.req_id, int(s.prefill_remaining)) for s in prefilling]
        pending += [(s.req.req_id, int(s.req.prompt_len)) for s in waiting]
        d = policy.decide(now_ms=now_ms, deadlines=deadlines,
                          pending_prefill=pending, kv_len=kv_len)
        entry = {"step": step, "slo_weave": d.describe()}
        if shadow:
            return cur_chunk, cur_max_tokens, entry
        if d.decode_only:
            # ★ 论文明写的 decode-only 迭代（Algorithm 1 第 13 行）：用**既有机制**表达
            # ——把本步 token 预算压到"只够解码"（每条 1 token）⇒ prefill 容量为 0。
            # 只在本步生效，下一步重新决策。
            return cur_chunk, max(1, len(decoding)), entry
        if d.chunk > 0:
            # ⚠️ **设值而不是取 min**：论文的 `c_t*` 就是这次迭代**实际用的** chunk，
            # 不是"上限"。取 min 会让该臂永远无法放大（实测踩到过：它想 2048、阈值在 1024
            # ⇒ min 恒为无操作，于是"两臂相同"是接线造成的假象）。
            return max(1, min(d.chunk, limits.chunk_size or d.chunk)), cur_max_tokens, entry
        return cur_chunk, cur_max_tokens, entry

    if kind == "cost":
        kv_lengths = {s.req.req_id: int(s.req.prompt_len - s.prefill_remaining)
                      for s in prefilling}
        # New requests derive their KV from plan.cached_tokens, not a guessed 0.
        # Last sampled token is consumed by the NEXT forward, not cached yet.
        kv_lengths.update({s.req.req_id: int(s.req.prompt_len + s.req.gen_len
                                             - s.decode_remaining - 1)
                           for s in decoding})
        d = policy.decide(
            scheduler=scheduler, chunk_policy=chunk_policy, cache=cache,
            waiting=[s.req for s in waiting], decoding=[s.req for s in decoding],
            prefilling=[(s.req, s.prefill_remaining) for s in prefilling],
            limits=limits, chunk_ceiling=cur_chunk, budget_ceiling=cur_max_tokens,
            max_num_seqs=max_num_seqs, chunk_enabled=chunk_enabled, kv_len=kv_len, step=step,
            per_request_kv_lengths=kv_lengths)
        entry = {"step": step, "cost_policy": d.describe()}
        if shadow or not d.changed:
            return cur_chunk, cur_max_tokens, entry
        return d.chunk, d.max_tokens, entry

    raise ValueError(f"未知的策略臂 kind={kind!r}")
