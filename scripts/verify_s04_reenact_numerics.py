#!/usr/bin/env python3
"""S04 常驻 KV 快路径：**独立的强制重入数值验证**（真机 / 真模型）。

为什么单独写一个脚本，而不是再跑一遍 `tests/test_optim_s04_persistent_kv.py`
--------------------------------------------------------------------------
那个文件用的是 **CPU 微型随机模型**（`tests/_tiny_support.make_model`）。它证明了
"实现自洽"，但**证明不了真机上的同一性质**：真机走 MUSA + fp16 + 真权重，
dtype/设备/内存布局都不同；而本缺陷（读过期缓冲）恰恰是**形状合法、不报错**的
一类 —— 微型模型上没暴露，不等于真机上没有。

★ 2026-10-03 修复：这个脚本此前**根本跑不起来**（如实登记）
----------------------------------------------------------------
上一轮把它标成"已就绪"，但实测第一次调用就崩：

    drv = build_driver(cfg)
    AttributeError: 'dict' object has no attribute 'runtime'

**已确认的缺陷只有一个**（"确认"= 有实测报错为证）：

1. 用 `yaml.safe_load` 直接读文件，把 **dict** 递给 `build_driver`；而真驱动读的是
   `cfg.runtime`（dataclass 属性）⇒ 立刻 AttributeError。
   ⇒ 改为项目的 `load_config()`（**顺带得到它的校验**：平台一致性、必填段）。

⚠️ **我一度把"接口对不上"报成了第二个缺陷，那是错的（自行更正）**：
我先入为主地以为 `forward()` 返回内部的 `_GroupOutcome`（字段 `produced`），
据此把脚本里原来的 `out.slots` 判成"照假驱动写的"。核对公开签名的实际结果：

    TransformerDriver.forward(batch) -> StepOutcome        # 真驱动，公开接口
    StepOutcome.slots: list[SlotResult]                    # req_id / token_id / ready_s
    _GroupOutcome                                          # 仅内部中间产物

⇒ **原来的 `out.slots` 是对的**。这条更正很重要：错误地把"对的东西"改掉，
会引入一个**新的**缺陷，而它同样以"更早的地方崩过所以没跑到"的方式藏起来。
（这正是本仓反复记的「先怀疑自己的对比方法，再怀疑代码」。）

2. **确实缺**的是：完全没有 logits 对照（`SlotResult` 原本只回传 token id）。
   ⇒ 新增 `SlotResult.logits_digest`（默认 `None`，既有字段语义不变）+
   `runtime.capture_logits_digest` 开关（**默认关** ⇒ 既有配置逐字节不变）。

教训：**"脚本写完了"不等于"脚本能跑"**。一个从未在**真实入口**上被执行过的脚本，
会以 `AttributeError` 这种最简单的方式失败——而它此前被读成了"前置已就绪"。
⇒ 这就是下面那条 `--plan-only` 与 `tests/test_verify_s04_cli.py` 存在的理由。

★ 判据（**预先冻结**，不看数据再定）
------------------------------------
必须**同时**成立，否则本项判定为 `FAIL`：

0. ★★ **采集完整性**（2026-10-03 加，**先判它**）—— 每个臂逐项核对：
   预期槽位集合、**模型层数**（对着模型自己核，不是"非空即可"）、
   K/V 张量形状 `(1, n_kv_heads, L, head_dim)` 与 dtype（与遥测声明一致）、
   每个槽位都有 token、logits **采集点集合**覆盖每一步且每个槽位都被覆盖过。
   ⇒ 为什么排在第一条：`{} == {}` 与"全都采到且相同"在后文几条判据下**同分**。
   之前的做法是在每处 `==` 旁边各加一个非空守卫（防守）；这一条是**正面验收**，
   而且能挡住只靠非空挡不住的那些缺法（只采到 1 层、只有 1 个槽位有值、
   logits 全挤在一步上、精度悄悄变了）。
1. `n_fastpath_steps > 0` —— 快路径真的被走过（否则下面的相等什么也没证）；
2. `n_fallback_steps > 0` 且 `unequal_kv_len` 出现在 `fallback_reasons` 里
   —— 回退是**混长**造成的（不是"新序列"之类的无关回退）；
3. `n_resyncs > 0` —— **重入**真的发生了。这条最关键：没有它，本脚本等于在验
   "纯快路径"，而缺陷只在重入时出现；
4. 全序列 **token 逐一相同**（与基线），**且两侧都非空**
   （两侧都空时 `==` 恒真 ⇒ 已显式加非空守卫，并由第 0 条正面覆盖）；
5. **逐层** KV 在**重入那一刻**与基线**逐位相同**（`(a==b).all()`，**不是**近似），
   且两侧都非空；层列表为空也判失败；
6. **逐层** KV 在**结束时刻**与基线**逐位相同**；
7. `tokens_out` **对账**（**同口径**）：`measured_window.tokens_out_measured`
   （**正式段**）== 脚本实际观察到的 token 数，且两臂相等。
   ⚠️ 只比两臂相等不够（两臂**同样漏记**时照样相等）；
   ⚠️★ 也**不得**拿含预热的 `tokens_out` 去比 —— 启用预热后两者本来就不该相等
   （EXECOPT5 实测 `11,609` vs `7,865`），那是**口径混比**、会报假的不符。
   有预热时额外核 `总量 - 正式段 == 预热段` 且两臂预热段一致。
8. **logits 摘要**逐步逐行相同（`sha256` 逐位判据）——比 token 更早暴露偏差，
   且**两侧都必须有采集点**（空对空不算相同）。
9. 口径：`measured_window` 可读，且 `pkv_*_measured` 与 `pkv_*_total` 分开。

⚠️ **预热口径（必须如实）**：本脚本默认**不预热**。此时 `measured_window.basis`
是 `no_warmup`，即"总量本身就是正式段口径"——这**不是**"预热扣除已验证"。
脚本据此把 `warmup_deduction_verified` 记为 `false` 并在产物里写明；
只有传了 `--warmup-s > 0` 且 `basis == "total_minus_warmup"`、
`n_forwards_measured < n_forwards`（扣除真的发生了）时才记 `true`。

⚠️ **本脚本不回答**性能问题（`pack_ms` 降多少）——那是性能复核要做的，
判据与预算另行冻结。
⚠️ 也**不**回答过载 / SLO —— 本脚本不涉及。
⚠️ **不覆盖**"快路径 → 回退"之后**未重入**的尾部：那是另一条路径，见
`tests/test_optim_s04_persistent_kv.py::TestFallbackKeepsResultsCorrect`。

输出纪律
--------
* `--out` **必须不存在**（拒绝覆盖旧结果，`rc=3`）。不给 `--out` 时自动生成
  `results/raw/_s04_verify/<UTC 时间戳>-<源码摘要前 12 位>.json`（天然唯一）。
* 产物里记录**源码 / 两臂生效配置 / 模型 / 环境**四类身份。
* 失败也会写出产物（判据逐条留在里面）⇒ **失败证据不会丢**；
  而"旧 PASS 被当成本次结果"在唯一路径 + 拒绝覆盖下不可能发生。
* `--plan-only` 只做"离线可验"的那部分（读配置、派生两臂、采集身份、定输出路径、
  校验单变量性），**不装载模型**。CLI 离线集成测试走这条路。

用法（远端）::

    source /root/musa-envs/qwen310/activate_musa.sh
    python scripts/verify_s04_reenact_numerics.py \
        --config configs/mthreads/exec_opt/s01_both.yaml \
        --out /root/s04_reenact_verify.json

本地离线自检（不装载模型）::

    python scripts/verify_s04_reenact_numerics.py --plan-only \
        --config configs/mthreads/exec_opt/s01_both.yaml --out /tmp/plan.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from real_backend import ForwardItem, build_driver  # noqa: E402
from run_benchmark import load_config  # noqa: E402

#: 两臂的差集必须**正好**是这几个键（单变量性）。
#: `persistent_kv_capacity` 在 `persistent_kv_slots=False` 时**不生效**
#: （`build_driver` 把它读成 `None`），所以它与开关键同属"一个逻辑变量"。
_PKV_RUNTIME_KEYS = ("persistent_kv_slots", "persistent_kv_capacity")


# --------------------------------------------------------------------------- #
# 配置：两臂都由**项目 load_config**派生
# --------------------------------------------------------------------------- #

def build_arm_configs(cfg_path: str | Path, *, capacity: int,
                      capture_logits_digest: bool = True):
    """返回 `(baseline_cfg, pkv_cfg)`。

    ★ 用 `load_config()` 而不是 `yaml.safe_load`：前者会**校验**（平台一致性、
    必填段、`max_num_seqs >= 1`…），后者把一份坏配置原样放行到"跑起来才炸"。
    ★ 用 `dataclasses.replace` 而不是改 `cfg.__dict__`：`Config` 是 dataclass，
    就地改会让"两臂"共享同一个 runtime 字典，第二臂会污染第一臂。
    """
    base = load_config(cfg_path)

    def _arm(*, pkv: bool):
        runtime = dict(base.runtime)
        runtime["persistent_kv_slots"] = bool(pkv)
        runtime["persistent_kv_capacity"] = (int(capacity) if pkv else None)
        runtime["capture_logits_digest"] = bool(capture_logits_digest)
        return replace(base, runtime=runtime)

    return _arm(pkv=False), _arm(pkv=True)


def arm_runtime_delta(base_cfg: Any, pkv_cfg: Any) -> dict[str, tuple[Any, Any]]:
    """两臂 runtime 的差异（键 -> (基线值, pkv 值)）。单变量性判据用它。"""
    keys = sorted(set(base_cfg.runtime) | set(pkv_cfg.runtime))
    return {
        k: (base_cfg.runtime.get(k), pkv_cfg.runtime.get(k))
        for k in keys
        if base_cfg.runtime.get(k) != pkv_cfg.runtime.get(k)
    }


# --------------------------------------------------------------------------- #
# 场景：强制重入
# --------------------------------------------------------------------------- #

def _make_request(name: str, prompt_len: int, gen_len: int):
    """构造一个请求。字段名以 `ascend_opt.workload.Request` 为准（**现读**，不凭记忆）。"""
    from ascend_opt.workload import Request

    return Request(req_id=name, prompt_len=prompt_len, gen_len=gen_len, arrival_ms=0.0)


def _kv_snapshot(drv, names: Sequence[str]) -> dict[str, list]:
    """逐层克隆这些槽位的 KV。**先 detach 再 clone**，避免把计算图带出来。"""
    snap: dict[str, list] = {}
    for n in names:
        slot = drv._slots.get(n)
        if slot is None:
            continue
        snap[n] = [(k.detach().clone(), v.detach().clone()) for k, v in slot.past]
    return snap


#: 场景里出现的**全部**槽位名（token / logits 覆盖判据用它）。
SCENARIO_SLOTS: tuple[str, ...] = ("a", "b", "c")
#: 需要逐层 KV 快照的槽位名。
#: ⚠️ 与 `_drive` **共用这一个常量**（两处各写一份的话，改一处就会让
#: "预期槽位"判据开始检查一个场景里不存在的名字，而那种失败看起来像代码缺陷）。
KV_SNAPSHOT_SLOTS: tuple[str, ...] = ("a", "b")


def _drive(drv, *, plen: int, gen: int, n_steady: int, n_reenter: int, plen_long: int):
    """推进"快路径 → 混长回退 → 成员退出 → **重入**"。

    返回 `tokens / logits / KV@重入 / KV@结束` 四样。
    形态对齐真机 EXECOPT4 观测：**成员退出后剩下的重新等长**是最常见的重入来源。

    ⚠️ 接口按**真驱动**核对过：`TransformerDriver.forward()` 返回 `StepOutcome`，
    字段 `slots: list[SlotResult]`（`_GroupOutcome` 是它内部的中间产物）。
    （本脚本上一版因为传错配置类型而崩在更早的地方，所以这行从未被执行到——
    这也是"离线集成测试"必须存在的原因。）
    """
    tokens: dict[str, list[int]] = {}
    logits: dict[str, dict[str, Any]] = {}
    step_no = {"i": 0}

    def step(batch: list[ForwardItem]) -> None:
        out = drv.forward(batch)
        i = step_no["i"]
        step_no["i"] = i + 1
        for slot in out.slots:
            if slot.token_id is not None:
                tokens.setdefault(slot.req_id, []).append(int(slot.token_id))
            if slot.logits_digest is not None:
                logits[f"{i}:{slot.req_id}"] = slot.logits_digest

    def item(name: str, n_new: int, plen_: int = plen) -> ForwardItem:
        return ForwardItem(_make_request(name, plen_, gen), n_new_tokens=n_new,
                           need_token=True)

    # ① 两条等长序列：各自一次性吃掉 prompt（prefill），随后进入纯 decode。
    step([item("a", plen)])
    step([item("b", plen)])
    # ② 等长纯 decode ⇒ **快路径**。
    for _ in range(n_steady):
        step([item("a", 1), item("b", 1)])
    # ③ 插入一条**更长**的序列 ⇒ 历史不等长 ⇒ **回退**（应有 `unequal_kv_len`）。
    step([item("c", plen_long, plen_long)])
    for _ in range(2):
        step([item("a", 1), item("b", 1), item("c", 1, plen_long)])
    # ④ ★ c 退出 ⇒ a/b 重新等长 ⇒ **重入快路径**（过期缓冲在此处被读到）。
    drv.retire(["c"])
    step([item("a", 1), item("b", 1)])
    kv_at_reentry = _kv_snapshot(drv, list(KV_SNAPSHOT_SLOTS))
    for _ in range(max(0, n_reenter - 1)):
        step([item("a", 1), item("b", 1)])
    return tokens, logits, kv_at_reentry, _kv_snapshot(drv, list(KV_SNAPSHOT_SLOTS))


def _run_arm(cfg: Any, *, warmup_s: float, **scenario):
    """跑一臂。`cfg` 是**已派生的 `Config`**（不是 dict）。"""
    drv = build_driver(cfg)
    drv.load()
    warm_info: dict[str, Any] = {}
    if warmup_s > 0:
        warm_info = drv.warmup(
            float(warmup_s),
            batch=2,
            prompt_len=int(scenario["plen"]),
            gen_len=int(scenario["gen"]),
        )
    tokens, logits, kv_re, kv_fin = _drive(drv, **scenario)
    # ★ 模型层数**从模型自己取**（`_model_n_layers()` 取不到时会抛错）——
    #   完整性判据要拿它核对"每层都采到了"，而"非空即可"会放过"只采到 1 层"。
    try:
        n_layers = int(drv._model_n_layers())
    except Exception:                                              # noqa: BLE001
        n_layers = None
    return {
        "tokens": tokens, "logits": logits,
        "kv_at_reentry": kv_re, "kv_final": kv_fin,
        "stats": drv.stats(), "driver": drv.describe(),
        "warmup_info": warm_info,
        "n_layers": n_layers,
    }


def _compare_kv(base: dict, opt: dict, *, label: str = "KV") -> tuple[bool, list[str]]:
    """逐层逐位对照（**不是**近似）。返回 `(全部相同, 差异描述)`。

    ★★ 2026-10-03 补"共同缺失"守卫
    -------------------------------
    旧实现只比 `sorted(base) == sorted(opt)`。**两侧都空**时它成立 ⇒ 返回
    `(True, [])` —— 于是一条"没取到任何 KV"的运行会被读成"两臂 KV 逐位相同"。
    这是典型的**恒定值假通过**：**缺失**伪装成**相等**。

    触发路径很现实：`_kv_snapshot(drv, names)` 里 `drv._slots.get(n)` 取不到就
    `continue` ⇒ 若槽位名写错/提前被回收，两侧都会得到 `{}`，而脚本会**报通过**。
    ⇒ 现在：两侧都空、或两侧的层列表都为空 ⇒ **判失败**（"没取到" ≠ "相同"）。
    """
    if not base and not opt:
        return False, [f"{label}：两臂**都是空的**（没取到任何槽位）"
                       "⇒ 对照无效；**缺失不等于相同**"]
    if not any(len(v) > 0 for v in base.values()) and not any(
            len(v) > 0 for v in opt.values()):
        return False, [f"{label}：两臂的逐层列表**都为空**（层数为 0）"
                       "⇒ 对照无效；**缺失不等于相同**"]
    if sorted(base) != sorted(opt):
        return False, [f"槽位集合不同：{sorted(base)} vs {sorted(opt)}"]
    notes: list[str] = []
    for name in sorted(base):
        b_layers, o_layers = base[name], opt[name]
        if len(b_layers) != len(o_layers):
            notes.append(f"{name}: 层数不同 {len(b_layers)} vs {len(o_layers)}")
            continue
        for li, ((bk, bv), (ok_, ov)) in enumerate(zip(b_layers, o_layers)):
            for tag, b, o in (("K", bk, ok_), ("V", bv, ov)):
                if tuple(b.shape) != tuple(o.shape):
                    notes.append(f"{name} L{li} {tag}: 形状 {tuple(b.shape)} vs {tuple(o.shape)}")
                    continue
                if not bool((b == o).all()):
                    d = float((b.float() - o.float()).abs().max())
                    notes.append(f"{name} L{li} {tag}: **数值不同**（最大差 {d:.3e}）")
    return (not notes), notes


def _kv_digest(kv: dict) -> dict[str, Any]:
    """把逐层 KV 压成**轻量摘要**（每层 K/V 的 sha256 + 形状 + 极值）。

    ★ 为什么要落盘摘要（2026-10-03 加）：
    原来的产物只写"逐层 KV 相同 / 不同"这个**结论**，不写任何可复算的量。
    于是事后想回答"到底哪一层的哪个值不一样"、或想与**将来**一次运行对照，
    都必须**重跑真机**——而真机机时是本项目最贵的资源。
    ⇒ 摘要**轻**（每层几十字节）却足以：
      ① 事后定位差异落在哪一层；
      ② 与未来同类运行**直接比对**（不必重算基线）；
      ③ 自证"这次真的取到了 KV"（形状/层数非空），堵住上面的"共同缺失"。

    只存 `sha256` 与标量，**不存张量**——否则产物会随层数×序列长度膨胀。
    """
    import hashlib

    out: dict[str, Any] = {}
    for name, layers in sorted(kv.items()):
        rows: list[dict[str, Any]] = []
        for li, pair in enumerate(layers):
            try:
                k, v = pair
            except (TypeError, ValueError):
                rows.append({"layer": li, "error": f"不是 (K,V) 二元组：{type(pair).__name__}"})
                continue
            rows.append({
                "layer": li,
                "K": _tensor_digest(k),
                "V": _tensor_digest(v),
            })
        out[name] = {"n_layers": len(layers), "layers": rows}
    return out


def _tensor_digest(t: Any) -> dict[str, Any]:
    """单个张量的轻量摘要：sha256（按**原始 dtype 的字节**）+ shape + dtype + 标量。

    ★ `dtype` 必须记（2026-10-03 加）：`sha256` 是对**字节**算的，而同一组数值
    在 fp16/fp32 下字节不同 ⇒ 只见摘要、不见 dtype 时，**无法判断两次运行是不是
    同一精度**。而"精度变了"恰恰是最容易悄悄发生、又最先毁掉可比性的一种变化。
    ⚠️ 同时给 `float` 视图的 sha：它把 dtype 归一化掉，于是"同值不同精度"
    与"值真的变了"**可以分开**（两个 sha 一起看才分得清）。
    """
    import hashlib

    try:
        raw_t = t.detach().to("cpu").contiguous()
        raw = raw_t.numpy().tobytes()
        host = raw_t.float()
        fraw = host.numpy().tobytes()
        return {
            "shape": list(raw_t.shape),
            "dtype": str(raw_t.dtype).replace("torch.", ""),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_as_float32": hashlib.sha256(fraw).hexdigest(),
            "sum": round(float(host.sum().item()), 6),
            "min": round(float(host.min().item()), 6),
            "max": round(float(host.max().item()), 6),
        }
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tokens_digest(tokens: dict[str, list[int]]) -> dict[str, Any]:
    """token 序列的轻量摘要（每槽位：个数 + sha256 + 前若干 token）。"""
    import hashlib

    out: dict[str, Any] = {}
    for name, seq in sorted(tokens.items()):
        payload = ",".join(str(x) for x in seq).encode("utf-8")
        out[name] = {"n": len(seq), "sha256": hashlib.sha256(payload).hexdigest(),
                     "head": list(seq[:8])}
    return out


def numeric_completeness_verdict(
    *, expected_slots: Sequence[str], kv: dict,
    tokens: dict, logits_keys: Sequence[str],
    n_steps: int, n_layers: int | None,
    n_kv_heads: int | None, head_dim: int | None, dtype: str | None,
    label: str = "pkv",
) -> tuple[bool, list[str]]:
    """**采集完整性**判据：确认"该采到的都采到了"，再谈"两臂是否相同"。

    ★★ 为什么必须**先**判完整性（2026-10-03 加）
    --------------------------------------------
    "两臂相同"只有在**两侧都真的采到东西**时才有意义。缺了这层，
    一条"根本没采到 KV"的运行会以 `{} == {}` 的形式**判成通过**——
    本仓已经把这类**恒定值假通过**在 `_compare_kv` / `_compare_logits` /
    token 三处各修过一次。但那种修法是"每一处 `==` 都加一个非空守卫"，
    **是防守**；这一条把它变成**正面的验收**：

      1. **预期槽位**：快照的槽位集合必须**恰好**等于场景声明的那些（少一个 =
         该请求的 KV 没采到；多一个 = 采到了不该在的东西）；
      2. **模型层数**：每层的层数必须等于模型的真实层数（**用配置核对，不是非零即可**）
         —— 只判"非空"时，"采到 1 层"与"采到 28 层"一样通过；
      3. **张量形状/dtype**：K/V 形状必须是 `(1, n_kv_heads, L, head_dim)`，
         且 dtype 与遥测声明一致（精度变了最容易被忽略，也最先毁掉可比性）；
      4. **token 采集点**：每个预期槽位都要有 token（不能只有一个槽位有）；
      5. **logits 采集点集合**：每个**步**都至少有一个采集点、每个**槽位**都至少
         被覆盖一次（比"个数 > 0"强得多：个数够也可能是全挤在同一两步上）。

    返回 `(ok, notes)`；`notes` 是**人话**，指出具体缺哪一格。
    """
    notes: list[str] = []
    exp = set(expected_slots)

    # ---- ① 预期槽位 ----
    got = set(kv)
    if not got:
        notes.append(f"{label}：KV 快照**一个槽位都没有**（预期 {sorted(exp)}）")
    else:
        missing = sorted(exp - got)
        extra = sorted(got - exp)
        if missing:
            notes.append(f"{label}：缺槽位 {missing}（KV 没采到）")
        if extra:
            notes.append(f"{label}：多出非预期槽位 {extra}")

    # ---- ② 模型层数 ----
    for name, layers in kv.items():
        if n_layers is not None and len(layers) != int(n_layers):
            notes.append(f"{label}/{name}：层数 {len(layers)} ≠ 模型层数 {n_layers}")
        if not layers:
            notes.append(f"{label}/{name}：**层列表为空**（层数 0）")

    # ---- ③ 形状 / dtype ----
    want_shape = None
    if n_kv_heads and head_dim:
        want_shape = (1, int(n_kv_heads), None, int(head_dim))
    for name, layers in kv.items():
        for li, pair in enumerate(layers):
            try:
                k, v = pair
                kshape, vshape = tuple(k.shape), tuple(v.shape)
                kdt, vdt = str(k.dtype).replace("torch.", ""), str(v.dtype).replace("torch.", "")
            except Exception as exc:                               # noqa: BLE001
                notes.append(f"{label}/{name} L{li}：读不出形状/dtype（{exc}）")
                continue
            if kshape != vshape:
                notes.append(f"{label}/{name} L{li}：K/V 形状不同 {kshape} vs {vshape}")
            if len(kshape) != 4:
                notes.append(f"{label}/{name} L{li}：K 不是 4 维 {kshape}")
            elif want_shape is not None:
                ok = (kshape[0] == want_shape[0] and kshape[1] == want_shape[1]
                      and kshape[3] == want_shape[3])
                if not ok:
                    notes.append(f"{label}/{name} L{li}：K 形状 {kshape} "
                                 f"与预期 (1,{n_kv_heads},L,{head_dim}) 不符")
            if dtype and (kdt != dtype or vdt != dtype):
                notes.append(f"{label}/{name} L{li}：dtype {kdt}/{vdt} ≠ 遥测声明 {dtype}")

    # ---- ④ token 采集点 ----
    for name in sorted(exp):
        if not tokens.get(name):
            notes.append(f"{label}/{name}：**没有 token 记录**")

    # ---- ⑤ logits 采集点集合（按步 × 槽位覆盖，不只看个数） ----
    seen_steps: set[str] = set()
    seen_slots: set[str] = set()
    malformed: list[str] = []
    for key in logits_keys:
        step, _, slot = str(key).partition(":")
        if not slot:
            malformed.append(str(key))
            continue
        seen_steps.add(step)
        seen_slots.add(slot)
    if malformed:
        notes.append(f"{label}：{len(malformed)} 个 logits 采集点键**格式不对**：{malformed[:3]}")
    if not logits_keys:
        notes.append(f"{label}：**没有任何 logits 采集点**")
    else:
        want_steps = {str(i) for i in range(int(n_steps))}
        miss_steps = sorted(want_steps - seen_steps, key=lambda x: int(x))
        if miss_steps:
            notes.append(f"{label}：{len(miss_steps)} 个**步**没有 logits 采集点：{miss_steps[:5]}")
        miss_slots = sorted(exp - seen_slots)
        if miss_slots:
            notes.append(f"{label}：{len(miss_slots)} 个槽位**从未**出现在 logits 采集点："
                         f"{miss_slots}")
    return (not notes), notes


def tokens_accounting(*, measured_base: Any, measured_pkv: Any, observed: int,
                      total_base: Any, total_pkv: Any,
                      basis: Any) -> tuple[bool, str]:
    """`tokens_out` 三方对账 —— **必须用同一个口径（正式段）**。

    ★ 2026-10-03 修：口径混比
    --------------------------
    原实现把 driver 的 `tokens_out`（**总量，含预热**）与脚本**实观察**的 token 数
    （**只有正式段**）比。启用预热后两者**本来就不该相等**，实测 EXECOPT5 一个臂：

        tokens_out = 11,609        # 含预热
        tokens_out_measured = 7,865   # 正式段
        ⇒ 预热段 = 3,744

    也就是说那条判据会在"预热门开着"的每一次运行上报"记账不符"——
    **方向性错误**：账其实是对的，错的是拿两个口径相比。
    这类错最坏的地方在于它**看起来像被测代码的问题**，会把人引去查 `tokens_out` 的
    累加器，而真正该改的是这条判据自己。

    ⇒ 现在：
      * 一律用 **`measured_window.tokens_out_measured`**（正式段）与实观察比；
      * 拿不到正式段口径 ⇒ **判失败并说明**（不许退回总量"凑一个相等"）；
      * 有预热时**额外**核一致性：`总量 == 正式段 + 预热段`，且 `总量 > 正式段`。
    """
    if measured_base is None or measured_pkv is None:
        return False, ("拿不到**正式段**口径的 `tokens_out_measured`"
                       "⇒ 无法做同口径对账（不得退回含预热的 `tokens_out` 去凑相等）")
    m0, m1 = int(measured_base), int(measured_pkv)
    ok = (m0 == m1 == int(observed))
    detail = (f"正式段口径：baseline={m0} pkv={m1} 实观察={observed}"
              f"（总量含预热：baseline={total_base} pkv={total_pkv}；basis={basis}）")

    warm = None
    if total_base is not None and total_pkv is not None:
        warm = (int(total_base) - m0, int(total_pkv) - m1)
    if basis == "total_minus_warmup":
        if warm is None:
            ok = False
            detail += "；⚠️ basis 声称已扣预热，但**拿不到总量** ⇒ 无法核一致性"
        else:
            if warm[0] < 0 or warm[1] < 0:
                ok = False
                detail += f"；⚠️ 预热段为负（{warm}）⇒ 总量 < 正式段，口径自相矛盾"
            elif warm[0] != warm[1]:
                # ⚠️ 要判的是**预热段**是否一致（`warm`），不是正式段（`m0`/`m1`）——
                #    第一版写成 `m0 != m1`，于是"两臂预热段不同"这个反例判不出来。
                ok = False
                detail += (f"；⚠️ 两臂预热段不同（baseline {warm[0]} / pkv {warm[1]}）"
                           "⇒ 跨臂比较口径不一致")
            else:
                detail += f"；预热段 = {warm[0]}（两臂一致）"
    return ok, detail


def _compare_logits(base: dict, opt: dict) -> tuple[bool, list[str]]:
    """逐步逐行对照 logits 摘要。`sha256` 逐位判据。"""
    if set(base) != set(opt):
        only_b = sorted(set(base) - set(opt))
        only_o = sorted(set(opt) - set(base))
        return False, [f"采集点集合不同：缺 {only_b[:3]} / 多 {only_o[:3]}"]
    bad = [k for k in sorted(base) if base[k] != opt[k]]
    if not bad:
        return True, []
    k = bad[0]
    return False, [
        f"{len(bad)}/{len(base)} 个采集点摘要不同；首个 {k}："
        f"base(argmax={base[k].get('argmax')}, sha={str(base[k].get('sha256'))[:12]}) vs "
        f"pkv(argmax={opt[k].get('argmax')}, sha={str(opt[k].get('sha256'))[:12]})"
    ]


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #

def evaluate(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """把**证据**判成逐条结论。纯函数：便于用合成证据做正反例。

    ⚠️ 这里**不**做 IO、不装载模型 ⇒ 反例测试可以直接喂构造好的 evidence。
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    eo = evidence["telemetry"]
    st_base, st_pkv = evidence["stats_base"], evidence["stats_pkv"]

    # ---- 前置：这条路径真的被走到过 ----
    check("快路径被走过", eo["n_fastpath_steps"] > 0,
          f"n_fastpath_steps={eo['n_fastpath_steps']}")
    check("混长回退发生",
          eo["n_fallback_steps"] > 0 and "unequal_kv_len" in (eo["fallback_reasons"] or {}),
          f"n_fallback_steps={eo['n_fallback_steps']} reasons={eo['fallback_reasons']}")
    check("★ 重入发生（本项关键）", eo["n_resyncs"] > 0,
          f"n_resyncs={eo['n_resyncs']}")

    # ---- ★ 采集完整性（**先判它，再判两臂是否相同**） ----
    #
    # 顺序很重要：`{} == {}` 与"全都采到且相同"在"两臂一致"这条判据下**同分**。
    # 完整性判据把"该采的都采到了"变成**正面验收**（预期槽位 / 模型层数 /
    # 形状与 dtype / token 与 logits 的采集点覆盖），而不是靠每处 `==` 自己加守卫。
    tlm = evidence.get("telemetry") or {}
    for arm_key, kv_key, tok_key, lg_key, layers_key in (
        ("baseline", "kv_base_reentry", "tokens_base", "logits_base", "n_layers_base"),
        ("pkv", "kv_pkv_reentry", "tokens_pkv", "logits_pkv", "n_layers"),
    ):
        comp_ok, comp_notes = numeric_completeness_verdict(
            expected_slots=evidence.get("expected_kv_slots") or [],
            kv=evidence.get(kv_key) or {},
            tokens=evidence.get(tok_key) or {},
            logits_keys=list((evidence.get(lg_key) or {})),
            n_steps=evidence.get("n_steps") or 0,
            n_layers=evidence.get(layers_key),
            n_kv_heads=tlm.get("n_kv_heads"), head_dim=tlm.get("head_dim"),
            dtype=tlm.get("dtype"), label=f"{arm_key}@重入",
        )
        check(f"★ 数值采集完整性（{arm_key}）：预期槽位/层数/形状/dtype/采集点",
              comp_ok,
              "全部核对通过" if comp_ok else "；".join(comp_notes[:5]))

    # ---- 数值：token 级 ----
    # ★ 同"共同缺失"守卫：两侧都是空 dict 时 `==` 也成立 ⇒ 必须显式要求非空。
    n_tok_base = sum(len(v) for v in evidence["tokens_base"].values())
    n_tok_pkv = sum(len(v) for v in evidence["tokens_pkv"].values())
    tok_ok = (evidence["tokens_base"] == evidence["tokens_pkv"]
              and n_tok_base > 0 and n_tok_pkv > 0)
    if tok_ok:
        tok_detail = f"两臂 token 全序列相同（{n_tok_base} 个）"
    elif n_tok_base == 0 and n_tok_pkv == 0:
        tok_detail = "两臂 token **都是空的** ⇒ 对照无效；**缺失不等于相同**"
    else:
        bad = [k for k in evidence["tokens_base"]
               if evidence["tokens_base"][k] != evidence["tokens_pkv"].get(k)]
        tok_detail = f"token 不同：槽位 {bad}"
    check("token 逐一相同", tok_ok, tok_detail)

    # ---- 数值：logits 摘要（比 token 更早暴露） ----
    lg_ok, lg_notes = _compare_logits(evidence["logits_base"], evidence["logits_pkv"])
    n_pts = len(evidence["logits_base"])
    # ★ 采集点为 0 ⇒ 开关没生效 ⇒ 这条判据**什么也没验**，必须硬失败。
    #   否则「logits 逐点相同」会在两个空 dict 上恒真（经典的"恒定值假通过"）。
    check("logits 摘要逐点相同", lg_ok and n_pts > 0,
          f"{n_pts} 个采集点全部相同" if (lg_ok and n_pts > 0)
          else ("采集点为 0：`capture_logits_digest` 未生效 ⇒ 这条判据什么也没验"
                if n_pts == 0 else "；".join(lg_notes[:4])))

    # ---- 数值：逐层 KV（**重入那一刻** + 结束时刻） ----
    kv_re_ok, kv_re_notes = _compare_kv(evidence["kv_base_reentry"],
                                        evidence["kv_pkv_reentry"])
    check("★ 重入时刻逐层 KV 逐位相同", kv_re_ok,
          "全部层相同" if kv_re_ok else "；".join(kv_re_notes[:6]))
    kv_fin_ok, kv_fin_notes = _compare_kv(evidence["kv_base_final"],
                                          evidence["kv_pkv_final"])
    check("结束时刻逐层 KV 逐位相同", kv_fin_ok,
          "全部层相同" if kv_fin_ok else "；".join(kv_fin_notes[:6]))

    # ---- 口径：正式窗口 vs 含预热（**先读**，记账要用它） ----
    mw = st_pkv.get("measured_window") or {}
    mw_base = st_base.get("measured_window") or {}
    check("measured_window 可读且与 total 分开（两臂都要）",
          "pkv_fastpath_measured" in mw and "tokens_out_measured" in mw
          and "tokens_out_measured" in mw_base,
          f"pkv keys={sorted(mw)[:8] if mw else '缺失'}；"
          f"baseline keys={sorted(mw_base)[:8] if mw_base else '缺失'}")

    # ---- 记账：tokens_out **同口径**对账 ----
    # ⚠️ 只比两臂相等是不够的——两臂**同样漏记**时照样相等（那正是曾经发生过的：
    #    快路径漏计 tokens_out，实测 17 个只记 9 个）。所以要"两臂 + 实观察"三方一致。
    # ⚠️★ 但**三方必须同口径**：driver 的 `tokens_out` 含预热，而实观察只数正式段
    #   ⇒ 启用预热时两者本来就不该相等（EXECOPT5 实测 11,609 vs 7,865）。
    #   判据一律用 `tokens_out_measured`，见 `tokens_accounting`。
    observed = sum(len(v) for v in evidence["tokens_pkv"].values())
    acct_ok, acct_detail = tokens_accounting(
        measured_base=mw_base.get("tokens_out_measured"),
        measured_pkv=mw.get("tokens_out_measured"),
        observed=observed,
        total_base=st_base.get("tokens_out"),
        total_pkv=st_pkv.get("tokens_out"),
        basis=mw.get("basis"),
    )
    check("tokens_out 对账（两臂 + 实观察，正式段口径）", acct_ok, acct_detail)

    # ---- 预热口径：**没预热就不许宣称预热扣除已验证** ----
    warmup_s = float(evidence.get("warmup_s", 0.0))
    basis = (mw or {}).get("basis")
    if warmup_s <= 0:
        verified = False
        detail = ("未预热（--warmup-s=0）⇒ `basis` 应为 `no_warmup`，"
                  f"实测 basis={basis}。**预热扣除未被验证**，"
                  "不得据此声称已扣除预热。")
        # 没预热时 basis 必须是 no_warmup：若它写着 total_minus_warmup，
        # 说明我们对口径的理解与驱动不一致 ⇒ 必须失败。
        check("预热口径自洽（无预热 ⇒ basis=no_warmup）", basis == "no_warmup", detail)
    else:
        nf, nfm = st_pkv.get("n_forwards"), (mw or {}).get("n_forwards_measured")
        verified = bool(basis == "total_minus_warmup"
                        and nf is not None and nfm is not None and nfm < nf)
        check("预热扣除**真的发生**（basis=total_minus_warmup 且 measured<total）",
              verified,
              f"warmup_s={warmup_s} basis={basis} n_forwards={nf} measured={nfm}")
    evidence["warmup_deduction_verified"] = verified

    return checks


# --------------------------------------------------------------------------- #
# 身份 / 输出路径
# --------------------------------------------------------------------------- #

def collect_identity(root: Path) -> dict[str, Any]:
    """记录**源码**身份。戳不在本机（只投到远端）时如实标注，不猜。"""
    ident: dict[str, Any] = {
        "tree_digest": None,
        "tree_digest_basis": "unavailable",
        "stamp": None,
    }
    try:
        from ascend_opt.provenance import compute_tree_digest, read_source_stamp
    except Exception as exc:  # pragma: no cover
        ident["error"] = f"{type(exc).__name__}: {exc}"
        return ident
    try:
        ident["tree_digest"] = compute_tree_digest(root)
        ident["tree_digest_basis"] = "computed_on_host"
    except Exception as exc:  # pragma: no cover
        ident["error"] = f"compute_tree_digest 失败：{type(exc).__name__}: {exc}"
    try:
        stamp = read_source_stamp(root)
        if stamp.get("validated") is True:
            ident["stamp"] = stamp
    except Exception:  # pragma: no cover - 没有戳是正常情况
        ident["stamp"] = None
    return ident


def default_out_path(root: Path, tree_digest: str | None) -> Path:
    """默认输出：**天然唯一**（UTC 时间戳 + 源码摘要前 12 位）。"""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dig = (tree_digest or "nodigest")[:12]
    return root / "results" / "raw" / "_s04_verify" / f"{ts}-{dig}.json"


def guard_out_path(p: Path) -> str | None:
    """**拒绝覆盖**：目标已存在就返回错误原因。"""
    if p.exists():
        return (f"输出路径已存在，拒绝覆盖：{p}\n"
                "  每次运行必须用唯一路径（避免把旧结果当成本次结果；"
                "也避免一次失败把上一次的 PASS 留下来）。")
    return None


def collect_environment(driver_desc: dict[str, Any] | None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "env_passthrough": {
            k: os.environ.get(k)
            for k in ("MODEL_NAME", "MODEL_REVISION", "HF_HOME", "HF_ENDPOINT",
                      "LD_PRELOAD")
            if os.environ.get(k)
        },
    }
    try:
        import torch
        env["torch"] = getattr(torch, "__version__", None)
        try:
            import torch_musa  # noqa: F401
            env["torch_musa"] = getattr(torch_musa, "__version__", "unknown")
        except Exception:
            env["torch_musa"] = None
    except Exception as exc:
        env["torch"] = f"unavailable: {type(exc).__name__}"
    if driver_desc:
        env["driver"] = driver_desc
    return env


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _parse_args(argv: Sequence[str] | None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="基线配置（S04 **未开** 那份）：脚本自己派生两条臂")
    ap.add_argument("--prompt-len", type=int, default=9)
    ap.add_argument("--prompt-len-long", type=int, default=17,
                    help="用于制造混长的更长 prompt")
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--steady-steps", type=int, default=3, help="重入前稳定快路径步数")
    ap.add_argument("--reenter-steps", type=int, default=4,
                    help="★ 重入后继续走的步数（判据只在这里生效）")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="预热时长（秒）。**默认 0 = 不预热**，此时脚本会如实"
                         "标注「预热扣除未被验证」，不得据此声称已验证")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan-only", action="store_true",
                    help="只做离线可验的部分（读配置/派生两臂/采集身份/校验单变量性/"
                         "定输出路径），**不装载模型**。CLI 离线集成测试走这条。")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # ---------- 1) 读配置（项目 load_config ⇒ 带校验） ----------
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"[s04-verify] ✗ 配置不存在：{cfg_path}", file=sys.stderr)
        return 2
    try:
        base_cfg, pkv_cfg = build_arm_configs(
            cfg_path, capacity=args.capacity, capture_logits_digest=True)
    except Exception as exc:
        print(f"[s04-verify] ✗ 配置加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # ---------- 2) 单变量性：两臂 runtime 的差集必须正好是 PKV 那两个键 ----------
    delta = arm_runtime_delta(base_cfg, pkv_cfg)
    delta_keys = tuple(sorted(delta))
    single_var_ok = delta_keys == tuple(sorted(_PKV_RUNTIME_KEYS))
    if not single_var_ok:
        print(f"[s04-verify] ✗ 单变量性不成立：两臂 runtime 差集={delta_keys}，"
              f"期望={tuple(sorted(_PKV_RUNTIME_KEYS))}", file=sys.stderr)
        return 2

    # ---------- 3) 身份 + 输出路径 ----------
    identity = collect_identity(_ROOT)
    out_path = Path(args.out) if args.out else default_out_path(
        _ROOT, identity.get("tree_digest"))
    why = guard_out_path(out_path)
    if why:
        print(f"[s04-verify] ✗ {why}", file=sys.stderr)
        return 3

    plan = {
        "mode": "plan_only" if args.plan_only else "run",
        "config": str(cfg_path),
        "arm_runtime_delta": {k: [str(v[0]), str(v[1])] for k, v in delta.items()},
        "single_variable_ok": single_var_ok,
        "capacity": args.capacity,
        "warmup_s": args.warmup_s,
        "scenario": {
            "plen": args.prompt_len, "plen_long": args.prompt_len_long,
            "gen": args.gen_len, "n_steady": args.steady_steps,
            "n_reenter": args.reenter_steps,
        },
        "out": str(out_path),
        "source_identity": identity,
        "effective_configs": {
            "baseline": base_cfg.describe(),
            "pkv": pkv_cfg.describe(),
        },
    }
    print(f"[s04-verify] 配置={cfg_path}")
    print(f"[s04-verify] 两臂 runtime 差集={delta_keys}（单变量性 OK）")
    print(f"[s04-verify] 输出={out_path}")

    if args.plan_only:
        # 离线路径也**真的写出**计划产物，供 CLI 集成测试核对落盘行为。
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({**plan, "verdict": "PLAN_ONLY"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        print("[s04-verify] --plan-only：未装载模型，已写出计划产物")
        return 0

    # ---------- 4) 真跑两臂 ----------
    scenario = dict(plen=args.prompt_len, gen=args.gen_len,
                    n_steady=args.steady_steps, n_reenter=args.reenter_steps,
                    plen_long=args.prompt_len_long)
    print(f"[s04-verify] 场景={scenario} capacity={args.capacity} "
          f"warmup_s={args.warmup_s}")

    base = _run_arm(base_cfg, warmup_s=args.warmup_s, **scenario)
    pkv = _run_arm(pkv_cfg, warmup_s=args.warmup_s, **scenario)

    eo = pkv["stats"]["execution_optimizations"]["persistent_kv_slots"]
    print(f"[s04-verify] telemetry: fastpath={eo['n_fastpath_steps']} "
          f"fallback={eo['n_fallback_steps']} resyncs={eo['n_resyncs']} "
          f"reasons={eo['fallback_reasons']}")
    n_lg = len(pkv["logits"])
    print(f"[s04-verify] logits 采集点={n_lg}（0 说明 capture_logits_digest 没生效）")

    evidence: dict[str, Any] = {
        "telemetry": eo,
        "stats_base": base["stats"], "stats_pkv": pkv["stats"],
        "tokens_base": base["tokens"], "tokens_pkv": pkv["tokens"],
        "logits_base": base["logits"], "logits_pkv": pkv["logits"],
        "kv_base_reentry": base["kv_at_reentry"], "kv_pkv_reentry": pkv["kv_at_reentry"],
        "kv_base_final": base["kv_final"], "kv_pkv_final": pkv["kv_final"],
        "warmup_s": args.warmup_s,
        # ★ 完整性判据的输入（2026-10-03）：预期槽位、模型层数、采集点全集。
        #   没有这些，"两臂相同"只能证明"两边一样地缺"。
        "expected_kv_slots": list(KV_SNAPSHOT_SLOTS),
        "expected_token_slots": list(SCENARIO_SLOTS),
        "n_layers": pkv.get("n_layers"),
        "n_layers_base": base.get("n_layers"),
        "n_steps": (pkv["stats"].get("measured_window") or {}).get(
            "n_logical_steps_measured"),
    }
    checks = evaluate(evidence)

    for c in checks:
        print(f"  {'✓' if c['pass'] else '✗'} {c['name']}: {c['detail']}")

    all_ok = all(c["pass"] for c in checks)
    verdict = "PASS" if all_ok else "FAIL"
    print(f"\n[s04-verify] === {verdict} ===")

    payload = {
        **plan,
        "mode": "run",
        "verdict": verdict,
        "telemetry": eo,
        "measured_window": pkv["stats"].get("measured_window"),
        "warmup_deduction_verified": bool(evidence.get("warmup_deduction_verified")),
        "environment": collect_environment(pkv["driver"]),
        "checks": checks,
        # ★ **轻量数值摘要**（2026-10-03 加）：把 token / logits / KV 压成可复算、
        #   可日后直接对照的摘要落盘。原来产物只写"相同/不同"这个**结论**，
        #   想复核任何一处都得**重跑真机**——而机时是本项目最贵的资源。
        #   这里只存 sha256 与标量，不存张量，故体积与层数同阶、与序列长度无关。
        "numeric_digests": {
            "tokens": {"baseline": _tokens_digest(base["tokens"]),
                       "pkv": _tokens_digest(pkv["tokens"])},
            "logits": {"baseline": dict(base["logits"]), "pkv": dict(pkv["logits"])},
            "kv_at_reentry": {"baseline": _kv_digest(base["kv_at_reentry"]),
                              "pkv": _kv_digest(pkv["kv_at_reentry"])},
            "kv_final": {"baseline": _kv_digest(base["kv_final"]),
                         "pkv": _kv_digest(pkv["kv_final"])},
        },
        "preregistered_criteria": [
            "n_fastpath_steps > 0",
            "n_fallback_steps > 0 且 unequal_kv_len 在 reasons 里",
            "n_resyncs > 0",
            "token 全序列相同",
            "logits 摘要逐点相同（sha256 逐位）",
            "重入时刻逐层 KV 逐位相同",
            "结束时刻逐层 KV 逐位相同",
            "tokens_out 两臂相等且等于实观察 token 数",
            "measured_window 可读且与 total 分开",
            "预热口径自洽（无预热 ⇒ basis=no_warmup）",
        ],
        "does_not_assert": [
            "性能（pack_ms 降幅）—— 属性能复核，判据与预算另行冻结",
            "过载 / SLO —— 本脚本不涉及",
            "非重入的尾部回退路径 —— 由单测覆盖",
            ("预热扣除 —— 本次**未预热**，故**未验证**（见 warmup_deduction_verified）"
             if args.warmup_s <= 0 else
             "预热扣除 —— 本次预热了，验证结果见 warmup_deduction_verified"),
        ],
        # ★ 产物**只存摘要，不存 KV 本体**（2026-10-03）：`numeric_digests` 里每层
        #   只有 `sha256`/`shape`/`dtype`/`sum`/`min`/`max`，体积与**层数**同阶、
        #   与序列长度**无关**。这样既能让别人独立复核（重算 sha 比一比），
        #   又不会让产物随 `max_seq_len` 膨胀到搬不动（本仓有过 2.6 MB `meta.json`
        #   回拉挂起的先例）。
        "stores": {
            "digests_only": True,
            "what": "token/logits/KV 的轻量摘要（sha256 + 形状 + dtype + 标量）",
            "not_stored": "KV 张量本体",
            "why": "独立复核不必重跑真机；同时避免产物随序列长度膨胀",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"[s04-verify] 结果已写入 {out_path}（verdict={verdict}）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
