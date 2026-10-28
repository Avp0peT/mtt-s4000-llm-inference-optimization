#!/usr/bin/env python3
"""PHASE 1 第②步：真实 FP16 路径的数值等价性验证（在远程 S4000 上运行）。

用法（远端）::

    source /root/musa-envs/qwen310/activate_musa.sh
    python validate_fp16_path.py --model Qwen/Qwen2.5-0.5B-Instruct --out <json路径>

设计（与 VALIDATION_PLAN_PHASE1.md 对齐）：

- **参考实现** = 原生 Transformers，逐 token 贪心执行，每步只喂 1 个 token、
  用 `past_key_values` 累积 KV。它不依赖本项目任何代码，是独立的真值来源。
- **被测实现** = `benchmark/real_backend.TransformerDriver`（本项目新写的后端）。
- 两者喂**同一份** prompt id（都走 `SyntheticPromptSource.prompt_ids`），
  保证比对的是"实现差异"而不是"输入差异"。

四个场景：
  V1 单请求        —— 1 个请求，逐 token 贪心 gen=N
  V2 多步 KV decode —— 走驱动的 forward 路径，验证 KV 拼接/position 递推
  V3 变长 batch    —— 三段：① 变长批的逐行 prefill 分布；② 分组前向的逐行一致性；
                      ③ **不同历史长度的合批 decode**（这才是真正覆盖 padding 的一段）
  V4 chunked prefill —— 长 prompt 分块，prefill 末位分布须与不分块相同
  V4b 分块后的多步 KV decode —— **相同历史**（teacher forcing）下**逐个分岔步**验收

**"部分检查通过"不等于"完整路径验收通过"**（三轮修正过的 over-claim）：

1. ``V4`` 原先只做到"分块 prefill 之后的首 token 分布"，随后立刻 ``retire``。
   产物里因此出现"参考生成 16 个 token、分块驱动只生成 1 个 token"而判据
   仍然 passed —— 它从未验证分块建起来的 KV 能否支撑后续 decode。
   现由 **V4b** 补齐（多步 decode + 逐步分布比对 + 步数硬判据）。
2. ``V3``/``V4`` 里"分布等价但 token 分岔"的三条分支原先**全部**只写 ``notes``，
   而 ``passed = not failures``，于是**取证失败**（不知道是不是近平局）
   与 **gap 明显不小**（与"分布等价"自相矛盾）也能拿到 ``passed=true``。
   现在按三态分流：近平局记 ``notes``，另两种进 ``needs_review`` 并**阻断通过**。
3. **2026-09-24 第三次修正（两处漏检，都是"看起来通过"）**：

   a. ``V4b`` 原先只验收**首个**分岔步。首步若是近平局（可解释），
      后面某步 gap 很大（与"分布等价"直接矛盾）就被**完全掩盖**。
      而 teacher forcing 下每个位置的输入都来自参考 ⇒ **每一步都是独立验收点**，
      没有任何"前面错了所以后面不可比"的借口。现逐个验收
      （``all_token_divergences``）。

      ⚠️ **这条只适用于"共享参考历史"的比较（V4b / ``mixed_history_batched_decode``）。**
      ``V3`` 的两条序列是**贪心自喂**的，第 k 步分岔后第 k+1 步起历史就不同了，
      后面的分岔是同一因果链的**下游**，不是独立验收点——那里仍只验首个分岔步。
      首跑实测（7B, v3_0）：第 4 步近平局（gap=0.0000），第 6~15 步的
      "分岔"gap 0.24~6.51；若逐步验收会凭空造出 10 条"需复核"。
      **"首步近平局掩盖后续错误"这个顾虑，在 V3 上由 teacher forcing 的
      V4b/history 两类判据承担，不由贪心序列承担。**

   b. ``V3`` 的"补位检查"**从未触发过补位，也从未合批**：它给每行传
      ``n_new_tokens=r.prompt_len``，而各行 prompt 长度互不相同 ⇒ 驱动按
      ``n_new_tokens`` 拆组 ⇒ **每组只有 1 行**、``pad = past_max - past_len`` 恒为 0。
      产物里 ``padding_preserves_rows: true`` 因此被读成"补位已验证"——
      **一个不会失败的判据**。而补位只在**同组内 ``past_len`` 参差**时发生，
      那正是"合批 decode 且各行步数不同"的形态（真实连续批处理里先到的行先走了几步）。
      现由 ``batched_mixed_history_decode`` 覆盖：先逐行推进**不同**步数
      （历史长度因此不同），再合批 decode 并逐行比分布；
      **若各行 KV 长度相同、或这些行没有在同一次 forward 里一起算，直接判失败**
      —— 否则又会退化成"什么都没测却通过"。

4. **2026-09-24 第四次修正（"恒假判据"，与"恒真判据"镜像）**：
   ``V4`` 的槽位检查在 ``retire`` **之前**读 ``live_slots()``，于是每次读到的都是 1
   （请求还没释放，本来就该是 1）⇒ 判据"收尾应为 0"**恒不成立**。
   恒假判据同样不携带信息，而且看起来像"真的抓到了 KV 泄漏"。
   本脚本改完后的**第一次真机跑**（0.5B/7B 各一次）就撞上了：5 个 V4 切法
   全部报"仍有 1 个常驻 KV 槽"，而两模型其余判据都正常。
   现改为 **retire 之后**读（必须 0），并把 **retire 之前**的值一并记入产物
   当作**反向对照**（单请求收尾应为 1）：若它一直是 0，说明计数器本身不可信，
   "释放后为 0"就是在什么都没测的情况下通过的 ⇒ 进 ``needs_review``。

另测资源生命周期：max_live_slots == concurrency、live_slots_at_end == 0、
n_retired == n_requests（这部分由 RealBackend 的 kv_lifecycle 统计给出，
本脚本在 V2/V3/V4 里复用驱动直接检查）。

**阈值是预声明的**（见 VALIDATION_PLAN_PHASE1.md §1），本脚本只判定、不调参。
判据不通过时退出码非零，并在产物里列出每个失败项。

关于 V1 的 fp32 误差判据——**它的能力边界必须说清**（实测校准，2026-09-22）：

- 观测值 ``max_abs_diff = 0.0`` 是**预期**结果，不是"判据没生效"：
  参考实现与被测驱动喂的是同一份权重、同一个 batch=1 的 prefill，
  设备上算的就是同一件事，逐位相同才对。
- 该判据的**动态范围已实测**：同输入重复 = 0.0；把 prompt 第一个 token
  改一位 = 7.379。所以 1e-3 的门槛落在有效区间内，判据对输入变化是敏感的。
- 但它**测不出**"驱动被旁路"这类失效：参考与驱动共用同一个模型实例
  （驱动拿到的是 ``_LogitsRecorder(model32)``），若驱动内部根本没走
  自己的 KV/padding 逻辑而直接转调，本判据一样是 0.0。
  真正能证伪旁路的是 V2/V3/V4 的 token id 逐位比对（它们必须经过
  驱动的 forward、KV 拼接与 padding 才能得到正确结果），以及
  ``driver_first_forward_shapes`` 里可核对的调用形状。
  因此**不得**仅凭 V1 的 0.0 宣称"数值路径已验证"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt import Request  # noqa: E402
from real_backend import (  # noqa: E402
    ForwardItem,
    SyntheticPromptSource,
    TransformerDriver,
)

# --------------------------------------------------------------------------- #
# 预声明阈值（与 VALIDATION_PLAN_PHASE1.md §1 逐条对应；此处是唯一执行处）
# --------------------------------------------------------------------------- #

FP32_MAX_ABS_DIFF = 1e-3
FP32_MEAN_ABS_DIFF = 1e-4
# fp16 逐 token / 变长 batch: 要求 token id 序列**逐位相同**（无容差）

# fp16 下"近平局"的判定线：分岔步的 top1-top2 gap **小于**此值，才认为
# fp16 归约顺序的舍入足以翻转 argmax，因而与实现无关。
# 实测（7B, v3_0）：分岔步 gap = 0.0000，属近平局。
#
# 为什么必须提成具名常量：V3 与 V4 两处都要用它判"这次分岔能不能被解释"，
# 而判据是**安全边界**——两处各写一个字面量，改一处忘一处就会让某一侧
# 悄悄放宽（或收紧）而没有任何提示。原先它内联在 ``_first_divergence_detail``。
NEAR_TIE_GAP = 0.05

# 分岔取证的三种结局。**"取证失败"与"gap 很大"都必须阻断通过**：
#   ① near_tie        —— 分岔步本身就是近平局，fp16 舍入可解释 → 记 note
#   ② forensics_failed—— 我们**不知道**该步是不是近平局。"不知道"不能当"没问题"
#   ③ not_near_tie    —— gap 明显不小，与"分布等价"直接矛盾 → 需复核
# ② 之所以最危险：它的产物看起来像"跑了但没结论"，而 passed 却仍是 true。
DIVERGENCE_NEAR_TIE = "near_tie"
DIVERGENCE_FORENSICS_FAILED = "forensics_failed"
DIVERGENCE_NOT_NEAR_TIE = "not_near_tie"


# --------------------------------------------------------------------------- #
# 参考实现：原生 Transformers 逐 token 贪心
# --------------------------------------------------------------------------- #


def reference_greedy_trace(model, prompt_ids, gen_len, device):
    """逐 token 贪心生成，**逐步**返回 ``(token, 该步末位分布)``。

    这是参考实现唯一的执行体：``reference_greedy`` 只取其中的 token。
    刻意写得"笨"：每步只喂 1 个 token，用 past_key_values 累积。
    这样它与被测实现的差异只可能来自被测实现，
    而非参考实现自己的批处理技巧。

    为什么要逐步记录分布：多步 decode 的正确性判据是"**每一步**在**相同
    历史**下的分布是否等价"。只拿最终 token 序列比会漏掉"序列恰好相同、
    中间分布已经错了"的情况；只在最后一步比则漏掉整段 decode。
    """
    import torch

    tokens: list[int] = []
    logits: list = []
    past = None
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(gen_len):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[0, -1, :].detach().float().cpu()
            logits.append(lg)
            nxt = int(lg.argmax().item())
            tokens.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tokens, logits


def reference_greedy(model, tokenizer, prompt_ids, gen_len, device):
    """逐 token 贪心生成，返回新产出的 token id 列表（见 trace 版）。

    ``tokenizer`` 保留在签名里只为不改动既有调用点；本函数不再需要它
    （合成 prompt 已是 token id，回归"分词"这一步没有必要）。
    """
    tokens, _ = reference_greedy_trace(model, prompt_ids, gen_len, device)
    return tokens


def reference_forced_trace(model, prompt_ids, forced_tokens, device):
    """参考实现按**指定** token 序列逐 token 前向，返回每步末位分布。

    与 ``reference_greedy_trace`` 的唯一区别是**不采样**：喂什么就是什么。
    需要它是因为"合批 decode 的逐行归属"必须唯一：按内容（喂进去的 token）
    归属行时，若两行的 token 相同就无法判定。**由调用方自己挑 token**
    才能保证各行互不相同——采样出来的 token 控制不了。

    ``logits[k]`` 对应"prompt + forced_tokens[0..k-1]"这段历史下的末位分布，
    因此它的 argmax 本应是 ``forced_tokens[k]``。返回 ``(logits, None)`` 形状与
    ``reference_greedy_trace`` 对齐（第二项是 feed 进去的 token 序列，这里就是入参）。
    """
    import torch

    tokens: list[int] = list(forced_tokens)
    logits: list = []
    past = None
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for k in range(len(tokens)):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1, :].detach().float().cpu())
            cur = torch.tensor([[int(tokens[k])]], dtype=torch.long, device=device)
    return logits, tokens


def reference_first_forward_logits(model, prompt_ids, device, dtype):
    """参考实现的首次前向 logits（fp32 用于误差比对）。"""
    import torch

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.logits.detach().to(torch.float32).cpu()


class _LogitsRecorder:
    """薄包装：记录最近一次 forward 收到的调用参数与返回的 logits。

    为什么需要它：``TransformerDriver.forward`` **刻意只返回 token id**，
    不把中间张量漏出去（那是"驱动只管执行"的设计）。但本步要验证的恰恰是
    "驱动喂进去的 logits 与参考实现是否数值一致"，所以需要在**不改动生产代码**
    的前提下拿到那份 logits。用包装器记录是唯一不侵入的做法。

    注意：它必须是透明的——``__getattr__`` 转发一切，只拦截 ``__call__``。
    若改成继承/子类化，模型内部的 ``self.xxx`` 会走到包装器上而语义漂移。
    """

    def __init__(self, inner):
        self._inner = inner
        self.last_logits = None
        self.last_kwargs: dict = {}

    def __call__(self, *args, **kwargs):
        import torch

        out = self._inner(*args, **kwargs)
        logits = getattr(out, "logits", None)
        if logits is not None:
            self.last_logits = logits.detach().to("cpu", torch.float32)
        self.last_kwargs = {k: _describe_value(v) for k, v in kwargs.items()}
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _describe_value(v):
    """把 forward 的实参压成可 JSON 序列化的**摘要**。

    这里踩过一次坑：原实现写的是 ``tuple(v.shape) if hasattr(v, "shape") else v``，
    而 ``past_key_values`` 是 ``DynamicCache`` 对象——它没有 ``.shape``，
    于是对象本身被原样塞进结果字典，到 ``json.dump`` 时抛
    ``TypeError: Object of type DynamicCache is not JSON serializable``。

    校验结果必须在**任何情况下**都能落盘：写不出来的产物等于没有产物，
    而且这个失败发生在所有测量都跑完之后，最容易被误当成"校验没跑"。
    所以未知类型一律降级为类型名摘要，绝不原样传递对象。

    另一处刻意为之：这里**不 import torch**。踩过——在某台机器上 ``import torch``
    命中了一个同名命名空间包（无 ``__init__.py`` 的目录），于是
    ``torch.Tensor`` 直接 AttributeError，本该"兜底"的函数自己先崩了。
    兜底逻辑不能依赖重依赖：用 ``shape``/``dtype`` 鸭子判定即可。
    """
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        # 张量（torch / numpy 皆适用），只记形状与类型，不搬数据
        try:
            shape = list(v.shape)
        except Exception:
            shape = None
        return {"tensor": shape, "dtype": str(getattr(v, "dtype", None))}
    # DynamicCache / tuple-of-tuples / dict 等一律只留类型与层数摘要
    summary: dict = {"type": type(v).__name__}
    layers = getattr(v, "layers", None)
    if layers is not None:
        try:
            summary["n_layers"] = len(layers)
        except Exception:
            pass
    if isinstance(v, (tuple, list)):
        summary["len"] = len(v)
    try:
        summary["seq_length"] = int(v.get_seq_length())
    except Exception:
        pass
    return summary


def _calibrate_sensitivity(ref_logits, model, prompt_ids, device):
    """校准 fp32 判据的动态范围：同输入重复 vs 改一位 token。

    为什么必须做：观测到 ``max_abs_diff == 0.0`` 时有两种可能——
    (a) 实现确实算出同一件事；(b) 判据本身不敏感（比如比较了同一个张量）。
    只报 0.0 而不给出动态范围，读产物的人无法区分这两者。
    这里给出一个**正对照**：改一位 token 后误差必须明显大于阈值，
    否则该判据无效，本次验证不得据此声称数值等价。

    刻意不把 torch 当参数传：本函数内部要用 torch 的 dtype 构造输入，
    外部传进来的那个名字不一定就是真 torch（曾命中同名命名空间包）。
    统一在这里 import，行为只有一处。
    """
    import torch

    same = reference_first_forward_logits(model, prompt_ids, device, torch.float32)
    same_d = float((ref_logits - same).abs().max().item())

    changed = list(prompt_ids)
    # 改第一个 token：位置 0 会影响所有后续位置，信号最强
    changed[0] = (changed[0] + 1) % 1000
    pert = reference_first_forward_logits(model, changed, device, torch.float32)
    pert_d = float((ref_logits - pert).abs().max().item())

    return {
        "repeat_max_abs_diff": same_d,
        "one_token_changed_max_abs_diff": pert_d,
        "threshold": FP32_MAX_ABS_DIFF,
        "sensitive": pert_d > FP32_MAX_ABS_DIFF,
    }


def _jsonable(obj):
    """把任意嵌套结构压成可 JSON 序列化的形态（用于序列化兜底）。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return {"type": type(obj).__name__}


def _single_forward_last_logits(model, rows_ids, device):
    """单次前向（可多行同长）取最后一位分布，按行返回列表。

    直接调原生模型，不经驱动 —— 这是"参考侧"的取分布方式。
    """
    import torch

    if not rows_ids:
        return []
    widths = {len(r) for r in rows_ids}
    if len(widths) != 1:
        raise ValueError("参考侧要求同长（本函数不做补位）")
    ids = torch.tensor(rows_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    lg = out.logits.detach().float().cpu()
    return [lg[i, -1, :] for i in range(lg.shape[0])]


class _PerRowCapture:
    """按"本次 forward 的行"归位 logits，供变长批的逐行比对。

    ⚠️ 为什么不能只用 ``_LogitsGrabber.last``：驱动按 ``n_new_tokens`` 分组
    （``_shape_groups``，从大到小），变长批会被拆成多次 forward，
    ``last`` 只留最后一次 —— 于是"批里第 i 行"的分布根本取不到，
    拿去比对就会得到量级完全错误的差异（实测假失败 max|Δ|=12.6）。

    归属方法：驱动对每个 shape 组调用一次，组内各行长度相同；
    记录该次 forward 的 ``input_ids`` 内容与输出。查询时按
    (单次 token 数, 首 token) 找 —— 变长批里 (n, 首token) 唯一即可定位到组，
    再用组内**行序**取对应行。若出现歧义（同长度同首 token 的行），
    视为无法归属，判据不放行（宁可不判定，也不给错判）。
    """

    def __init__(self, inner):
        self._inner = inner
        #: {(n_new, first_id): [每行的末位分布]}
        self.groups: dict[tuple[int, int], list] = {}
        self.ambiguous: set[tuple[int, int]] = set()
        #: 每次 forward 的**行数**与单次 token 数。为什么必须记下来：
        #: "变长批的补位"只有在**同一次 forward 里有多行且 past_len 参差**时才会发生。
        #: 各行 ``n_new`` 互不相同时，驱动按 ``n_new`` 拆组，每次只有 1 行——
        #: 于是 `pad` 恒为 0，检查会**在从未补位的情况下通过**。
        #: 把每次调用的行数落盘，读产物的人才能看出"到底合批了没有"。
        self.call_rows: list[int] = []
        self.call_n_new: list[int] = []

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        ids = kwargs.get("input_ids")
        lg = getattr(out, "logits", None)
        if ids is not None and lg is not None:
            n_new = int(ids.shape[1])
            self.call_rows.append(int(ids.shape[0]))
            self.call_n_new.append(n_new)
            lgc = lg.detach().float().cpu()
            # 逐行记录：组内所有行共享 n_new，用每行自己的首 token 作为键；
            # 首 token 相同则视为歧义（不放行，避免张冠李戴）。
            for i in range(ids.shape[0]):
                key = (n_new, int(ids[i, 0].item()))
                if key in self.groups:
                    self.ambiguous.add(key)
                else:
                    self.groups[key] = [lgc[i, -1, :]]
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def by_row(self, n_new: int, first_id: int, row_index: int = 0):
        """取某组第 row_index 行的末位分布；歧义或缺失返回 None。"""
        key = (n_new, first_id)
        if key in self.ambiguous or key not in self.groups:
            return None
        rows = self.groups[key]
        return rows[row_index] if row_index < len(rows) else None


class _StepRowCapture:
    """逐**次** forward 记录"每行末位分布"，供多步**合批 decode** 的逐行比对。

    与 ``_PerRowCapture`` 的两点差别，都是多步场景逼出来的：

    1. ``_PerRowCapture`` 是**单次** forward 用的：同一个 key 第二次出现就被判歧义。
       多步 decode 里"同一个 token 在不同步再次被喂进来"很常见，
       照那个口径会把正常情况判成歧义。
    2. 多步必须能回答"**第 s 步**那一行的分布是多少"，所以每次调用单独成一份记录。

    行归属仍按内容（``(单次 token 数, 首 token)``），不按行序——驱动可能重排，
    按内容归属才自证；同一份记录里 key 重复即该次不可判定（不放行）。
    """

    def __init__(self, inner):
        self._inner = inner
        #: 每次 forward 一项：{(n_new, first_id): 末位分布}
        self.calls: list[dict] = []
        #: 与 ``calls`` 对齐：该次里出现重复 key 的集合
        self.duplicated: list[set] = []

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        ids = kwargs.get("input_ids")
        lg = getattr(out, "logits", None)
        rec: dict = {}
        dup: set = set()
        if ids is not None and lg is not None:
            n_new = int(ids.shape[1])
            lgc = lg.detach().float().cpu()
            for i in range(ids.shape[0]):
                key = (n_new, int(ids[i, 0].item()))
                if key in rec:
                    dup.add(key)
                else:
                    rec[key] = lgc[i, -1, :]
        self.calls.append(rec)
        self.duplicated.append(dup)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def call_containing(self, keys, start: int = 0) -> int | None:
        """找**第一次**同时含有全部 ``keys`` 的 forward（从 ``start`` 起）。

        找不到 ⇒ 这些行**没有在同一次 forward 里一起算**，也就是"合批"没有发生。
        调用方必须把它当失败：那正是"补位检查被拆成多个单请求"的长相。
        """
        for ci in range(start, len(self.calls)):
            if all(k in self.calls[ci] and k not in self.duplicated[ci] for k in keys):
                return ci
        return None


def _single_forward_last_logits_one(model, ids, device):
    """单行单次前向的末位分布；失败返回 None。"""
    got = _single_forward_last_logits(model, [ids], device)
    return got[0] if got else None


def _first_divergence_detail(
    model, prompt_ids, driver_seq, tokens_ref, device, index: int | None = None
):
    """若 token 序列分岔，报告**某个**分岔步的 top1/top2 差距（判定是否近平局）。

    ``index=None``（默认）取**首个**分岔步；显式传 ``index`` 时可**逐个**验收。
    逐个验收是必须的：只看首个分岔时，首步若是近平局（可解释），
    后面某步 gap 很大（与"分布等价"矛盾）就会被完全掩盖
    （见 ``all_token_divergences``）。

    为什么必须给这个数：``分布等价但 token 不同`` 有两种含义——
    (a) 分岔步本身是近平局（gap≈0），fp16 舍入就能翻转 argmax，与实现无关；
    (b) 分岔步 gap 很大，那分布等价就说不通了，需要复核。
    只报"token 不同"而不给 gap，读产物的人无法区分这两者。
    实测（7B, v3_0）：分岔步 gap = 0.0000，属 (a)。

    返回值绝不静默为 None：调用方是在"已知分岔"的分支里调用的，
    所以"取不到"本身就是异常，必须显形为 ``{"error": ...}``，
    否则会把"没测出来"显示成"没有问题"。
    """
    import torch

    if driver_seq == tokens_ref:
        # 调用方说分岔、这里却一致：说明两侧比对口径不同，显形而不是返回 None
        return {"error": "调用方判定分岔，但比对时两侧序列相同（口径不一致）"}

    n = min(len(driver_seq), len(tokens_ref))
    if index is None:
        idx = next((i for i, (a, b) in enumerate(zip(driver_seq, tokens_ref)) if a != b), n)
    else:
        idx = int(index)
        if idx >= n or driver_seq[idx] == tokens_ref[idx]:
            # 指定的步**并不是**分岔步 ⇒ 调用方的清单与事实不符，显形为取证失败，
            # 不能返回一个"看起来正常"的 gap 把口径不一致藏起来。
            return {
                "error": f"指定的第 {idx} 步并非分岔步（调用方清单与序列不一致）",
                "index": idx,
            }
    try:
        # 用参考侧复现到分岔步，读该步的 top1/top2
        past = None
        cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(idx):
                out = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1, :].argmax().item())
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        lg = out.logits[0, -1, :].float()
        v, i = torch.topk(lg, 2)
    except Exception as exc:  # 取证失败不能伪装成"无近平局"
        return {"error": f"{type(exc).__name__}: {exc}", "index": idx}

    return {
        "index": idx,
        "ref_token": tokens_ref[idx] if idx < len(tokens_ref) else None,
        "driver_token": driver_seq[idx] if idx < len(driver_seq) else None,
        "top1": int(i[0]),
        "top2": int(i[1]),
        "top1_top2_gap": float(v[0] - v[1]),
        "is_near_tie": float(v[0] - v[1]) < NEAR_TIE_GAP,
    }


def _top1_top2_gap(lg) -> float:
    """单个位置的分布上 top1 与 top2 的差距。"""
    import torch

    v = torch.topk(lg.float(), 2)[0]
    return float(v[0] - v[1])


def gap_detail_from_logits(lg, idx: int) -> dict:
    """从**已记录的**参考分布里取出第 ``idx`` 步的 top1/top2 详情。

    与 ``_first_divergence_detail`` 的区别：后者要**重跑**参考到分岔步
    （因为它只有 token 序列）；多步 decode 已经把每步的分布记下来了，
    再跑一遍纯属浪费，而且重跑路径与记录路径不一致时会给出矛盾的结论。
    """
    import torch

    v, i = torch.topk(lg.float(), 2)
    gap = float(v[0] - v[1])
    return {
        "index": idx,
        "top1": int(i[0]),
        "top2": int(i[1]),
        "top1_top2_gap": gap,
        "is_near_tie": gap < NEAR_TIE_GAP,
    }


def first_token_divergence(driver_tokens, ref_tokens) -> dict:
    """逐步比 token 序列，返回首个分岔步。

    返回 ``{"length_match": bool, "first_divergence": int|None,
    "n_driver": int, "n_ref": int}``。

    **长度不一致时 ``first_divergence`` 一律为 ``None`` 且 ``length_match=False``**
    —— 第一版直接在长度不等时返回 ``None``，那会和"完全一致"撞成同一个值：
    "驱动只跑了 1 步、参考跑了 16 步"于是显示为"没有分岔"。判据里"说不清"
    与"没问题"必须是两个值，这里用 ``length_match`` 把两者分开，
    由调用方决定怎么处置（本脚本按失败处置）。
    """
    n_d, n_r = len(driver_tokens), len(ref_tokens)
    if n_d != n_r:
        return {
            "length_match": False,
            "first_divergence": None,
            "n_driver": n_d,
            "n_ref": n_r,
        }
    for i, (a, b) in enumerate(zip(driver_tokens, ref_tokens)):
        if a != b:
            return {
                "length_match": True,
                "first_divergence": i,
                "n_driver": n_d,
                "n_ref": n_r,
            }
    return {
        "length_match": True,
        "first_divergence": None,
        "n_driver": n_d,
        "n_ref": n_r,
    }


def all_token_divergences(driver_tokens, ref_tokens) -> list[int]:
    """**全部**分岔步的下标（不是只有第一个）。

    为什么必须逐个验收：只看首个分岔时，**首步若是近平局（可解释），
    后面某步 gap 很大（与"分布等价"直接矛盾）就会被完全掩盖**——
    而 teacher forcing 保证了每个位置的输入都来自参考，
    因此**每一步都是彼此独立的验收点**，没有任何"前面错了所以后面不可比"的理由。

    实测（7B, V4b 首轮）：分岔发生在第 0 步（gap=0.0000），
    若此后还有非近平局的分岔，旧判据会把它读成"已解释"。

    长度不等时只比公共前缀——长度不等本身由调用方按失败处置（见
    ``first_token_divergence`` 的说明），这里不重复表态。
    """
    n = min(len(driver_tokens), len(ref_tokens))
    return [i for i in range(n) if driver_tokens[i] != ref_tokens[i]]


def judged_divergence_steps(
    driver_tokens, ref_tokens, *, shared_history: bool
) -> list[int]:
    """**可以逐个判态**的分岔步清单。

    能不能"逐个验收"取决于**比较口径**，不取决于序列本身：

    * ``shared_history=True``（teacher forcing：每一步都喂参考 token）——
      两侧在**每个位置**的历史都相同 ⇒ 每一步都是**独立验收点**，全部返回。
    * ``shared_history=False``（贪心自喂）——第 k 步一分岔，第 k+1 步起两侧
      喂进去的历史就不同了，后面的"分岔"是同一条因果链的**下游**，
      不是独立证据 ⇒ **只返回首个**。

    为什么抽成函数而不是内联在 ``main`` 里：内联时这条规则只能靠读注释理解，
    而"该逐个还是该只取首个"恰恰是**已经判错一次**的地方——把 V4b 的口径
    （共享历史）套到 V3（贪心）上，真机首跑凭空造出 10 条"需复核"。
    变成纯函数后，两个方向都能被直接断言。

    长度不等时 ``all_token_divergences`` 只比公共前缀、可能返回空表；
    **空表不等于"没有分岔"**，由调用方按失败处置（见 ``first_token_divergence``）。
    """
    steps = all_token_divergences(driver_tokens, ref_tokens)
    return steps if shared_history else steps[:1]


def compare_step_logits(driver_steps, ref_steps) -> dict:
    """逐步比分布：返回每步的 max|Δ| 与总体最差步。

    为什么逐步比而不是只比最后一步：只比末步会漏掉"中间某步已经错了、
    末尾恰好对上"；只比 token 序列则会漏掉"序列相同但分布已经错了"。
    形状对齐的比对必须**显式检查步数**——两串长度不同时 `zip` 静默截断，
    "驱动少跑了 15 步"会变成"逐位一致"。
    """
    if len(driver_steps) != len(ref_steps):
        return {
            "error": (
                f"步数不一致：驱动 {len(driver_steps)} 步 vs 参考 "
                f"{len(ref_steps)} 步（不得截断比较）"
            )
        }
    per_step: list[dict] = []
    for i, (a, b) in enumerate(zip(driver_steps, ref_steps)):
        if a is None or b is None:
            return {
                "error": (
                    f"第 {i} 步分布缺失（驱动 {'缺' if a is None else '有'} / "
                    f"参考 {'缺' if b is None else '有'}）"
                )
            }
        per_step.append(_compare_logits(a, b))
    worst_i = max(range(len(per_step)), key=lambda j: per_step[j]["max_abs_diff"])
    return {
        "n_steps": len(per_step),
        "per_step": per_step,
        "max_abs_diff": per_step[worst_i]["max_abs_diff"],
        "worst_step": worst_i,
    }


def classify_token_divergence(where: str, det: dict | None) -> tuple[str, str]:
    """把"分布等价但 token 分岔"分成三态，返回 ``(verdict, message)``。

    **这是本次修掉的核心缺陷**：原先三种情况（近平局 / 取证失败 / gap 不小）
    全都只写进 ``notes``，而 ``passed = not failures``，于是
    "取证失败"与"gap 不小"也能拿到 ``passed=true``——一个**永远不会失败**的判据。
    它比崩掉更危险：产物看起来完整、退出码为 0，读的人会据此宣称验收通过。

    三态的处置（``main`` 里落实）：
      - ``near_tie``           → note。fp16 舍入即可翻转 argmax，与实现无关。
      - ``forensics_failed``   → needs_review，**阻断通过**。我们**不知道**该步
        是不是近平局；"不知道"不得等同于"没问题"。
      - ``not_near_tie``       → needs_review，**阻断通过**。gap 明显不小却
        "分布等价"，说明比对口径或实现有问题，必须人看。

    ``det`` 为 ``None`` / 含 ``error`` / 缺 gap 字段，一律按取证失败处理——
    取证失败有四种长相，每一种都必须落到同一态，漏掉任何一种都会重新打开缺口。
    """
    if not det:
        return (
            DIVERGENCE_FORENSICS_FAILED,
            f"{where}: 分布等价但 token 分岔，**取证结果为空**（既非近平局也"
            f"无错误信息）——无法判断分岔是否可解释，需复核",
        )
    if "error" in det:
        return (
            DIVERGENCE_FORENSICS_FAILED,
            f"{where}: 分布等价但 token 分岔，分岔步**取证失败**"
            f"（{det['error']}）——不知道该步是否为近平局，需复核",
        )
    gap = det.get("top1_top2_gap")
    if gap is None:
        return (
            DIVERGENCE_FORENSICS_FAILED,
            f"{where}: 分布等价但 token 分岔，取证结果里**没有 gap 字段**"
            f"（keys={sorted(det)}）——无法判断是否近平局，需复核",
        )
    idx = det.get("index")
    if gap < NEAR_TIE_GAP:
        return (
            DIVERGENCE_NEAR_TIE,
            f"{where}: 分布等价，token 序列在第 {idx} 步分岔；该步是近平局"
            f"（top1={det.get('top1')} top2={det.get('top2')} gap={gap:.4f} "
            f"< {NEAR_TIE_GAP}），fp16 舍入即可翻转 argmax，**不是实现错误**",
        )
    return (
        DIVERGENCE_NOT_NEAR_TIE,
        f"{where}: 分布等价但 token 在第 {idx} 步分岔，该步 gap={gap:.4f} "
        f">= {NEAR_TIE_GAP}（不够小）——与「分布等价」矛盾，需复核",
    )


#: 会**阻断通过**的裁决。``near_tie`` 刻意不在其中：它是唯一有解释的分岔。
BLOCKING_VERDICTS = frozenset({DIVERGENCE_FORENSICS_FAILED, DIVERGENCE_NOT_NEAR_TIE})


def blocks_acceptance(verdict: str) -> bool:
    """该裁决是否阻断验收通过。

    抽成函数是为了**可被测试**：原先"分岔怎么办"的逻辑散在 main 的
    分支里，只能靠读代码判断，于是"三种情况全进 notes"这个缺陷
    没有任何测试能抓住。现在判据与裁决都是纯函数，测试可直接断言。
    """
    return verdict in BLOCKING_VERDICTS


def resource_lifecycle_verdict(
    *,
    driver_live_at_end: dict[str, int | None],
    v4_live_after_retire: dict[str, int],
    v4_live_before_retire: dict[str, int],
) -> tuple[list[str], list[str]]:
    """把"KV 槽位是否及时释放"的三份记录判成 ``(failures, needs_review)``。

    判定规则只有两条，但**读的时机**是关键：

    * ``*_live_at_end``（``retire`` **之后**读）必须是 0；
    * ``v4_live_before_retire``（``retire`` **之前**读）单请求收尾时应恰好是 1,
      它不是正确性判据，而是**反向对照**：若它一直是 0，说明计数器恒 0，
      那么上面那些 0 就是"什么都没测"得来的 ⇒ 进 ``needs_review``。

    为什么抽成纯函数：这条判据第一版在 ``retire`` **之前**读 ``live_slots()``，
    于是**恒报失败**（真机首跑 5 个 V4 切法全中）。恒假判据与恒真判据一样
    不携带信息，而它看起来像"真的抓到了 KV 泄漏"。纯函数才能用合成记录直接
    证明三件事：该失败时会失败、该通过时会通过、计数器不可信时会显形。

    ``None`` = 该驱动没有槽位计数能力（不判），**不是**"测出来是 0"。
    """
    failures: list[str] = []
    needs_review: list[str] = []
    for name, live in driver_live_at_end.items():
        if live is None:
            continue
        if live != 0:
            failures.append(f"{name} 收尾仍有 {live} 个常驻 KV 槽（应为 0）")
    for name, live in v4_live_after_retire.items():
        if live != 0:
            failures.append(f"{name} 收尾仍有 {live} 个常驻 KV 槽（应为 0）")
    for name, live in v4_live_before_retire.items():
        if live != 1:
            needs_review.append(
                f"{name} 在 retire **之前**的常驻槽位是 {live}（单请求应为 1）——"
                "槽位计数器可能不可信，「释放后为 0」这条判据因此不成立，需复核"
            )
    return failures, needs_review


def overall_passed(failures, needs_review) -> bool:
    """总通过条件：**failures 与 needs_review 都必须为空**。

    这一行是本次修复的落点。原先是 ``not failures``，于是
    "分岔取证失败""多步 decode 只跑了 1 步"这类**必须人看**的条目
    也能拿到 ``passed=true``——一个永远不会失败的判据。
    """
    return not failures and not needs_review


class _LogitsGrabber:
    """记录最近一次 forward 的 logits（用于逐位置分布比对）。

    与 ``_LogitsRecorder`` 的区别：只关心 logits，不做 kwargs 摘要，供
    "分块/变长批的分布是否等价"这类判据使用。
    """

    def __init__(self, inner):
        self._inner = inner
        self.last = None

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        lg = getattr(out, "logits", None)
        if lg is not None:
            self.last = lg.detach().float().cpu()
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _compare_logits(a, b) -> dict:
    """两个同形状分布的差异摘要。"""
    d = (a - b).abs()
    return {
        "max_abs_diff": float(d.max().item()),
        "mean_abs_diff": float(d.mean().item()),
    }


def decode_with_teacher_forcing(drv, grab, req, ref_tokens, gen_len):
    """在**已 prefill 完**的驱动上跑 ``gen_len - 1`` 步 KV decode。

    每一步的输入是 ``ref_tokens[t-1]``（参考的第 t-1 个 token），而不是驱动
    自己上一步采样出的 token。这是"相同历史下比较分布"的前提：
    ``_Slot.take_next`` 在 prompt 算完之后喂的是 ``slot.last_token``，若让驱动
    自喂，某一步因近平局分岔之后两侧历史就不同了，逐位置比 logits 比的是
    **两段不同的历史**——看起来很严格，其实不可比。

    返回 ``(own_tokens, step_logits)``：
      - ``own_tokens`` 是这些步里驱动**自己**产出的 token（用于和参考比）；
      - ``step_logits[k]`` 是第 k 步的末位分布。

    **绝不静默降级**：槽位不存在（prefill 完就被释放）、prompt 没喂完
    （teacher forcing 会把参考 token 当 prompt 段用）、参考序列不够长
    （下一次索引会 IndexError 在别处炸）——三种都直接抛错，由调用方
    显形记为失败。旧实现走的正是"槽位已释放"这条路，而它只跑 1 步也通过。
    """
    if gen_len < 1:
        raise ValueError(f"gen_len 必须 >= 1，收到 {gen_len}")
    if len(ref_tokens) < gen_len:
        raise ValueError(
            f"参考序列只有 {len(ref_tokens)} 个 token，不足 {gen_len} 步；"
            "无法构造相同历史"
        )
    own: list[int] = []
    steps: list = []
    for t in range(1, gen_len):
        slot = drv._slots.get(req.req_id)
        if slot is None:
            raise RuntimeError(
                f"decode 第 {t} 步：序列 {req.req_id} 已不在驱动里"
                "（prefill 之后被提前释放），多步 KV decode 无法进行"
            )
        if slot.past_len < len(slot.prompt):
            raise RuntimeError(
                f"decode 第 {t} 步：past_len={slot.past_len} < prompt_len="
                f"{len(slot.prompt)}，prompt 尚未喂完，teacher forcing 口径不成立"
            )
        slot.last_token = ref_tokens[t - 1]
        out = drv.forward(
            [ForwardItem(req=req, n_new_tokens=1, need_token=True)]
        )
        own += [s.token_id for s in out.slots if s.token_id is not None]
        steps.append(grab.last[0, -1, :].clone() if grab.last is not None else None)
    return own, steps


def batched_mixed_history_decode(
    drv, cap, reqs, forced, extra_steps, n_steps, ref_logits
) -> dict:
    """把**已有不同历史长度**的多行放进同一次 forward 做 decode，逐行取分布并比对。

    为什么必须单独有这一项：原 V3 的补位检查只覆盖"**prompt 长度**不同"，
    而且它把逐行分布检查拆成了**多个单请求**（每行单独建驱动）——
    **"不同历史长度的合批 decode"从未被测过**。真实连续批处理里同时存在的行，
    不仅 prompt 长度不同，**已生成 token 数也不同**（先到的行先走了几步）。
    只测前者测不到"KV 长度不同"这一类；拆成单请求则根本没测合批。

    做法：
      1. 逐行**单独**推进 ``extra_steps[i]`` 步（按 ``forced`` 喂 token），
         ⇒ 各行 KV 长度从此各不相同；
      2. 之后 ``n_steps`` 步，**所有行同一次** ``forward(n_new_tokens=1)``，
         并在该次记录里按内容逐行取末位分布；
      3. 与参考在**同一绝对位置**的分布比。

    为什么让调用方给 ``forced``：逐行归属按"喂进去的 token"做，两行 token 相同
    就无法判定。采样 token 控制不了互不相同，**自己挑**才能保证。
    ``forced[rid]`` 至少要 ``extra_steps + n_steps + 1`` 个（参考需要下标多一格）。

    任何"取不到 / 没合批 / 槽位提前释放"都进 ``errors`` 并**由调用方判失败**：
    本函数**不返回一个看起来正常的空表**——那正是"跑了但什么也没测"的长相。
    """
    res: dict = {"rows": {}, "errors": [], "batch_calls": [], "histories": {}, "rows_per_call": []}

    for i, r in enumerate(reqs):
        rid = r.req_id
        need = int(extra_steps[i]) + n_steps + 1
        if len(forced.get(rid, [])) < need:
            res["errors"].append(
                f"{rid}: 强制 token 只有 {len(forced.get(rid, []))} 个，至少需 {need}"
            )
        if len(ref_logits.get(rid, [])) < need:
            res["errors"].append(
                f"{rid}: 参考分布只有 {len(ref_logits.get(rid, []))} 个，至少需 {need}"
            )
    if res["errors"]:
        return res

    # 1) 逐行推进到**各不相同**的历史长度（teacher forcing，单行 forward）
    for i, r in enumerate(reqs):
        rid = r.req_id
        slot = drv._slots.get(rid)
        if slot is None:
            res["errors"].append(f"{rid}: 不在驱动里（槽位已释放），合批 decode 无法进行")
            continue
        if slot.past_len < len(slot.prompt):
            res["errors"].append(
                f"{rid}: past_len={slot.past_len} < prompt_len={len(slot.prompt)}，"
                "prompt 尚未喂完，比对口径不成立"
            )
            continue
        for k in range(int(extra_steps[i])):
            slot.last_token = forced[rid][k]
            drv.forward([ForwardItem(req=r, n_new_tokens=1, need_token=True)])
    if res["errors"]:
        return res
    for i, r in enumerate(reqs):
        slot = drv._slots[r.req_id]
        res["histories"][r.req_id] = {
            "prompt_len": int(len(slot.prompt)),
            "extra_steps": int(extra_steps[i]),
            "kv_len_at_first_batch_step": int(slot.past_len),
        }

    # **本检查成立的前提**：同一次 forward 里各行的 `past_len` 必须不同。
    # 驱动的补位（``_forward_group`` 里的 ``pad = past_max - s.past_len``）
    # 只在"同组内 past_len 参差"时才发生；各行历史一样长时 `pad` 恒为 0，
    # 于是本检查会**在什么都没补位的情况下通过**。
    kv_lens = [h["kv_len_at_first_batch_step"] for h in res["histories"].values()]
    if len(set(kv_lens)) != len(kv_lens):
        res["errors"].append(
            f"各行 KV 长度相同（{kv_lens}）⇒ 合批时不会发生补位，本检查不成立"
            "（必须让各行处于不同的 decode 位置）"
        )
        return res

    # 2) 合批 decode：每一步都要求"所有行在同一次 forward 里"
    search_from = len(cap.calls)
    for s in range(n_steps):
        keys = []
        for i, r in enumerate(reqs):
            rid = r.req_id
            tok = int(forced[rid][int(extra_steps[i]) + s])
            drv._slots[rid].last_token = tok
            keys.append((1, tok))
        drv.forward(
            [ForwardItem(req=r, n_new_tokens=1, need_token=True) for r in reqs]
        )
        ci = cap.call_containing(keys, start=search_from)
        if ci is None:
            res["errors"].append(
                f"第 {s} 步：{len(reqs)} 行**没有在同一次 forward 里一起算**"
                f"（keys={keys}）——合批没有发生，本检查不成立"
            )
            continue
        search_from = ci + 1
        res["batch_calls"].append(ci)
        res.setdefault("rows_per_call", []).append(len(keys))
        for i, r in enumerate(reqs):
            rid = r.req_id
            pos = int(extra_steps[i]) + s
            key = (1, int(forced[rid][pos]))
            got = cap.calls[ci].get(key)
            # 捕获到的那一步分布 = 历史 "prompt + forced[0..pos]" ⇒ 参考下标 pos+1
            ref = ref_logits[rid][pos + 1]
            d = _compare_logits(got, ref) if got is not None else {"error": "该行分布缺失"}
            res["rows"].setdefault(rid, []).append({"step": s, **d})
    return res


# fp16 下"不同归约顺序"造成的 logits 扰动上界。实测（7B, 40-token prompt）：
# 分块 [16,16,8] 与整段的 max|Δlogits| = 0.0312，mean = 0.0052。
# 取 0.25 作为"数值等价"的判定线：它比实测扰动宽约 8 倍（容得下更大 batch
# 累积的舍入差），又远小于"实现错误"该有的量级（实测改一位输入达 21.7）。
FP16_LOGIT_TOL = 0.25


def run_driver_greedy(drv, reqs, prompt_source, gen_len):
    """用被测驱动跑贪心生成，返回 {req_id: [token ids]}。

    直接用 driver.forward，不经过 RealBackend 的调度器：
    本步验证的是**驱动的数值正确性**，调度语义是第③步的对象。

    ⚠️ 返回的是 token 序列，而 token 序列**不是**可靠的判据：合成 prompt 的
    续写常落在近平局上，fp16 归约顺序的差异即可翻转 argmax（实测 7B：
    某些步的 top1-top2 gap 恰为 0.0000）。正确性判据是分布等价，
    见 ``FP16_LOGIT_TOL`` 与 ``_compare_logits``。
    """
    out: dict[str, list[int]] = {r.req_id: [] for r in reqs}

    # prefill：完整喂入 prompt，必须产出首 token
    outcome = drv.forward(
        [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=True) for r in reqs]
    )
    for slot in outcome.slots:
        if slot.token_id is not None:
            out[slot.req_id].append(slot.token_id)

    # decode
    for _ in range(gen_len - 1):
        outcome = drv.forward(
            [ForwardItem(req=r, n_new_tokens=1, need_token=True) for r in reqs]
        )
        for slot in outcome.slots:
            if slot.token_id is not None:
                out[slot.req_id].append(slot.token_id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="FP16 路径数值等价性验证")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--out", default="validate_fp16_result.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result: dict = {
        "model": args.model,
        "device": args.device,
        "gen_len": args.gen_len,
        "thresholds": {
            "fp32_max_abs_diff": FP32_MAX_ABS_DIFF,
            "fp32_mean_abs_diff": FP32_MEAN_ABS_DIFF,
            "fp16_logit_tol": FP16_LOGIT_TOL,
            "near_tie_gap": NEAR_TIE_GAP,
            "token_ids": "exact",
        },
        "hostname": os.uname().nodename,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checks": {},
        "failures": [],
        # 「不通过」与「通过但有需要说明的现象」分开记：
        # token 分岔若发生在近平局上，分布其实等价——把它塞进 failures
        # 会把"实现正确"记成"实现错误"；直接丢掉又会让读产物的人
        # 看到 token 序列不同却找不到解释。所以单开一类。
        "notes": [],
        # 第三类：**必须人看，且阻断通过**。典型是"token 分岔但分岔步取证失败"——
        # 我们不知道该步是不是近平局，"不知道"不得等同于"没问题"。
        # 它曾经被写进 notes 而 passed 仍为 true：一个永远不会失败的判据。
        "needs_review": [],
    }

    print(f"[1/6] 载入模型 {args.model} 到 {args.device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 参考实现用 fp32 做首前向误差比对（与探针口径一致）
    model32 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(args.device)
    model32.eval()

    vocab = int(model32.config.vocab_size)
    # 排除特殊 token：命中 EOS 会让 gen_len 悄悄变短
    excluded = []
    for name in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"):
        v = getattr(tokenizer, name, None)
        if isinstance(v, int):
            excluded.append(v)

    src = SyntheticPromptSource(vocab_size=vocab, seed=1234, excluded_ids=excluded)

    # ---- V1: 单请求 + fp32 首前向误差 ----
    print("[2/6] V1 单请求（参考 vs 驱动）…")
    v1_req = Request(req_id="v1", prompt_len=24, gen_len=args.gen_len)
    v1_ids = src.prompt_ids(v1_req)

    # 参考实现：原生 Transformers，直接前向（不带驱动的批处理/KV 管理）
    ref_logits = reference_first_forward_logits(model32, v1_ids, args.device, torch.float32)
    ref_tokens_v1 = reference_greedy(model32, tokenizer, v1_ids, args.gen_len, args.device)

    # 被测实现：驱动。用包装器拿到它喂进去的那份 logits。
    recorder = _LogitsRecorder(model32)
    driver_fp32 = TransformerDriver(
        model_name=args.model, device=args.device, precision="fp32", model=recorder,
        tokenizer=tokenizer, prompt_source=src,
    )
    driver_fp32.load()
    d1 = driver_fp32.forward([ForwardItem(req=v1_req, n_new_tokens=24, need_token=True)])
    drv_logits = recorder.last_logits
    drv_tokens_v1 = [s.token_id for s in d1.slots if s.token_id is not None]
    for _ in range(args.gen_len - 1):
        d = driver_fp32.forward([ForwardItem(req=v1_req, n_new_tokens=1, need_token=True)])
        drv_tokens_v1.extend(s.token_id for s in d.slots if s.token_id is not None)

    v1_ok = ref_tokens_v1 == drv_tokens_v1
    result["checks"]["V1_single_request"] = {
        "reference_tokens": ref_tokens_v1,
        "driver_tokens": drv_tokens_v1,
        "token_ids_exact_match": v1_ok,
        "driver_first_forward_shapes": recorder.last_kwargs,
    }
    if not v1_ok:
        result["failures"].append(
            f"V1 token id 序列不一致：ref={ref_tokens_v1} drv={drv_tokens_v1}"
        )
    driver_fp32.retire(["v1"])

    # fp32 首前向 logits 误差：drv_logits 是驱动最后一次前向的 logits，
    # 而最后一次是 decode 步（seq 长度不同），不能直接与"首前向"比。
    # 因此这里单独再做一次"只 prefill"的驱动前向，取它的 logits。
    #
    # 同时要就地校准判据的**动态范围**：同输入重复应得 0，输入改一位应远大于阈值。
    # 不做这一步，0.0 这个观测值无法区分"计算正确"与"判据不敏感"。
    _calib = _calibrate_sensitivity(ref_logits, model32, v1_ids, args.device)
    result["checks"]["V1_threshold_calibration"] = _calib
    if not _calib.get("sensitive", False):
        result["failures"].append(
            "V1 fp32 判据对输入改动不敏感（改一位 token 后误差仍 < 阈值）"
            f"：此判据不能作为数值等价的证据。calib={_calib}"
        )

    if drv_logits is not None:
        recorder.last_logits = None
        driver_fp32b = TransformerDriver(
            model_name=args.model, device=args.device, precision="fp32", model=recorder,
            tokenizer=tokenizer, prompt_source=src,
        )
        driver_fp32b.load()
        driver_fp32b.forward(
            [ForwardItem(req=v1_req, n_new_tokens=24, need_token=True)]
        )
        prefill_logits = recorder.last_logits
        driver_fp32b.retire(["v1"])

        if prefill_logits is None:
            result["failures"].append("V1 无法取到驱动的 prefill logits（不能当作通过）")
        else:
            # 两者都取"prompt 最后一个 token 位置"的输出分布
            drv_last = prefill_logits[0, prefill_logits.shape[1] - 1, :]
            ref_last = ref_logits[0, ref_logits.shape[1] - 1, :]
            diff = (drv_last - ref_last).abs()
            max_d = float(diff.max().item())
            mean_d = float(diff.mean().item())
            result["checks"]["V1_fp32_logits"] = {
                "max_abs_diff": max_d,
                "mean_abs_diff": mean_d,
                "argmax_ref": int(ref_last.argmax().item()),
                "argmax_drv": int(drv_last.argmax().item()),
                "driver_logits_shape": list(prefill_logits.shape),
                "reference_logits_shape": list(ref_logits.shape),
                "limitation": (
                    "参考与驱动共用同一模型实例，故 max_abs_diff==0 是预期值；"
                    "该判据不能证伪'驱动被旁路'，旁路由 V2/V3/V4 的 token id 逐位比对证伪。"
                ),
            }
            if max_d > FP32_MAX_ABS_DIFF:
                result["failures"].append(
                    f"V1 fp32 max_abs_diff {max_d:.3e} > {FP32_MAX_ABS_DIFF:.0e}"
                )
            if mean_d > FP32_MEAN_ABS_DIFF:
                result["failures"].append(
                    f"V1 fp32 mean_abs_diff {mean_d:.3e} > {FP32_MEAN_ABS_DIFF:.0e}"
                )
            if result["checks"]["V1_fp32_logits"]["argmax_ref"] != result["checks"][
                "V1_fp32_logits"
            ]["argmax_drv"]:
                result["failures"].append("V1 fp32 argmax 不一致（硬门禁）")
    else:
        result["failures"].append(
            "V1 拿不到驱动的 logits —— 无法验证数值误差（不能当作通过）"
        )

    # ---- V2/V3/V4 用 fp16 驱动（这是本步的目标精度）----
    print("[3/6] 载入 fp16 模型（与驱动共用权重实例）…")
    model16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model16.eval()

    # ---- V2: 多步 KV decode ----
    print("[4/6] V2 多步 KV decode …")
    v2_req = Request(req_id="v2", prompt_len=32, gen_len=args.gen_len)
    v2_ids = src.prompt_ids(v2_req)
    ref_v2 = reference_greedy(model16, tokenizer, v2_ids, args.gen_len, args.device)

    drv2 = TransformerDriver(
        model_name=args.model, device=args.device, precision="fp16",
        model=model16, tokenizer=tokenizer, prompt_source=src,
    )
    drv2.load()
    drv_v2 = run_driver_greedy(drv2, [v2_req], src, args.gen_len)["v2"]
    v2_ok = ref_v2 == drv_v2
    result["checks"]["V2_multistep_decode"] = {
        "reference_tokens": ref_v2,
        "driver_tokens": drv_v2,
        "token_ids_exact_match": v2_ok,
    }
    if not v2_ok:
        result["failures"].append(f"V2 token 序列不一致：ref={ref_v2} drv={drv_v2}")
    drv2.retire(["v2"])

    # ---- V3: 变长 batch ----
    print("[5/6] V3 变长 batch（prompt 长度故意不等）…")
    lengths = [16, 24, 33, 48]
    v3_reqs = [
        Request(req_id=f"v3_{i}", prompt_len=n, gen_len=args.gen_len)
        for i, n in enumerate(lengths)
    ]
    # 参考：逐个单跑（各自独立，等价于串行）
    ref_v3: dict[str, list[int]] = {}
    for r in v3_reqs:
        ids = src.prompt_ids(r)
        ref_v3[r.req_id] = reference_greedy(
            model16, tokenizer, ids, args.gen_len, args.device
        )

    drv3 = TransformerDriver(
        model_name=args.model, device=args.device, precision="fp16",
        model=model16, tokenizer=tokenizer, prompt_source=src,
    )
    drv3.load()
    drv_v3 = run_driver_greedy(drv3, v3_reqs, src, args.gen_len)
    v3_per_row = {rid: (ref_v3[rid] == drv_v3[rid]) for rid in ref_v3}
    drv3.retire([r.req_id for r in v3_reqs])

    # token 逐位相同只是**次要**判据：合成 prompt 的续写常常落在近平局上，
    # fp16 的归约顺序差异就足以改变 argmax（实测 7B 上 max|Δlogits|≈0.03，
    # 而某些步的 top1-top2 差距本身就是 0.0000）。
    # 真正的主判据是**分布在 fp16 舍入范围内等价**：逐行取"该行自己的
    # prefill 末位分布"，与单跑参考比，并记录该处的 top1-top2 差距。
    #
    # ⚠️ 踩过的坑：想"一次性拿批里某行的分布"是做不到的。驱动按
    # ``n_new_tokens`` 分组（``_shape_groups``，从大到小），变长批会被拆成
    # **多次** forward；``_LogitsGrabber.last`` 只保留最后一次，于是拿到的
    # 是**别的行**的分布 —— 实测据此报出 max|Δ|=12.6 的假失败，实际是
    # 把 len=48 的参考与 len=16 那一次前向的输出比。
    # 正确做法：逐行单独建驱动（单行 ⇒ 只有一个组 ⇒ last 就是该行），
    # 再另外用**等长的批**验证 padding 不破坏结果（见下面的 lockstep 检查）。
    v3_logits: dict[str, dict] = {}
    v3_rows_ok = True
    for r in v3_reqs:
        ids = src.prompt_ids(r)

        # 参考：单跑整段 prefill 的最后一位分布
        ref_lg = _single_forward_last_logits_one(model16, ids, args.device)
        if ref_lg is None:
            v3_rows_ok = False
            v3_logits[r.req_id] = {"error": "参考分布取不到"}
            continue

        # 驱动：该行单独一条（无补位），与参考同形可比
        g = _LogitsGrabber(model16)
        d_one = TransformerDriver(
            model_name=args.model, device=args.device, precision="fp16",
            model=g, tokenizer=tokenizer, prompt_source=src,
        )
        d_one.load()
        d_one.forward(
            [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=False)]
        )
        drv_lg = g.last[0, -1, :] if g.last is not None else None
        d_one.retire([r.req_id])

        if drv_lg is None:
            v3_rows_ok = False
            v3_logits[r.req_id] = {"error": "拿不到分布（不能当作通过）"}
            continue

        lg_d = _compare_logits(drv_lg, ref_lg)
        drv_t1 = int(drv_lg.argmax().item())
        ref_t1 = int(ref_lg.argmax().item())
        v = torch.topk(ref_lg, 2)[0]
        gap = float(v[0] - v[1])

        row = {
            **lg_d,
            "argmax_ref": ref_t1,
            "argmax_drv": drv_t1,
            "ref_top1_top2_gap": gap,
            "token_ids_exact_match": v3_per_row[r.req_id],
            "within_fp16_tol": lg_d["max_abs_diff"] <= FP16_LOGIT_TOL,
        }
        v3_logits[r.req_id] = row
        if not row["within_fp16_tol"]:
            v3_rows_ok = False

    # --- 分组前向的逐行一致性（**不是**补位检查，见下）---
    #
    # ⚠️ 这一段原名"补位检查"、并自称"唯一能真正验证 padding 的检查"——
    # **它从未触发过补位，也从未合批**：给每行传的是 ``n_new_tokens=r.prompt_len``，
    # 而 4 行的 prompt 长度（16/24/33/48）互不相同 ⇒ ``_shape_groups`` 按
    # ``n_new_tokens`` 分组后**每组只有 1 行** ⇒ 实际发出 **4 次 batch=1 的 forward**，
    # ``pad = past_max - s.past_len`` 恒为 0。产物里 ``padding_preserves_rows: true``
    # 于是被读成"补位已验证"，而事实是"补位从未发生"——**一个不会失败的判据**。
    # 它真正能测的只有"各行各自的 prefill 分布与单跑一致"（有用，但不是补位）。
    # 补位由下面的「合批 decode」一段覆盖：那里各行的 ``past_len`` 才真的参差。
    pad_rows: dict[str, dict] = {}
    pad_ok = True
    cap = _PerRowCapture(model16)
    dp = TransformerDriver(
        model_name=args.model, device=args.device, precision="fp16",
        model=cap, tokenizer=tokenizer, prompt_source=src,
    )
    dp.load()
    dp.forward(
        [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=False)
         for r in v3_reqs]
    )
    dp.retire([r.req_id for r in v3_reqs])

    for r in v3_reqs:
        ids = src.prompt_ids(r)
        ref_lg = _single_forward_last_logits_one(model16, ids, args.device)
        # 该行在其 shape 组里的分布：按 (本次喂入 token 数, 首 token) 定位组。
        # 本段每组只有 1 行（见上面的说明），所以这实际上就是"单行前向"的结果。
        got = cap.by_row(r.prompt_len, ids[0], row_index=0)
        if got is None:
            pad_ok = False
            pad_rows[r.req_id] = {
                "error": "未能唯一归属该行的分布（判据无效，不放行）",
                "keys_seen": sorted(cap.groups.keys()),
            }
            continue
        lg_d = _compare_logits(got, ref_lg) if ref_lg is not None else None
        if lg_d is None:
            pad_ok = False
            pad_rows[r.req_id] = {"error": "参考分布取不到"}
            continue
        row_ok = lg_d["max_abs_diff"] <= FP16_LOGIT_TOL
        pad_rows[r.req_id] = {**lg_d, "within_fp16_tol": row_ok}
        if not row_ok:
            pad_ok = False

    # --- 合批 decode 专项：**不同历史长度**的行放进同一次 forward ---
    #
    # 为什么必须补这一段：上面两段各缺一半——
    #   · 逐行分布检查是**单请求**做的（每行单独建驱动）⇒ 没测合批；
    #   · "分组 prefill"虽是同一次调用，但各行 n_new 不同 ⇒ 每组 1 行 ⇒ 没合批、没补位。
    # 而补位只发生在"同组内 past_len 参差"，那正是**合批 decode 且各行步数不同**
    # 的形态（真实连续批处理里先到的行先走了几步）。这一段把两件事一起构造出来。
    extra = [0, 1, 2, 3]
    n_steps = 3
    # 强制喂入的 token **由我们挑**：逐行归属按"喂进去的 token"做，
    # 采样出来的 token 无法保证各行互不相同（相同就无法归属、判据要么误判要么弃权）。
    forced: dict[str, list[int]] = {}
    ref_forced_logits: dict[str, list] = {}
    for i, r in enumerate(v3_reqs):
        k = extra[i] + n_steps + 1
        forced[r.req_id] = [1000 + 1000 * i + j for j in range(k)]
        lgs_f, _toks_f = reference_forced_trace(
            model16, src.prompt_ids(r), forced[r.req_id], args.device
        )
        ref_forced_logits[r.req_id] = lgs_f

    cap2 = _StepRowCapture(model16)
    dm = TransformerDriver(
        model_name=args.model, device=args.device, precision="fp16",
        model=cap2, tokenizer=tokenizer, prompt_source=src,
    )
    dm.load()
    dm.forward(
        [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=True)
         for r in v3_reqs]
    )
    mixed = batched_mixed_history_decode(
        dm, cap2, v3_reqs, forced, extra, n_steps, ref_forced_logits
    )
    dm.retire([r.req_id for r in v3_reqs])

    mixed_max: float | None = None
    mixed_bad_rows: list[str] = []
    for _rid, _rows in (mixed.get("rows") or {}).items():
        for _row in _rows:
            if "error" in _row:
                mixed_bad_rows.append(f"{_rid} 第 {_row.get('step')} 步：{_row['error']}")
                continue
            mixed_max = (
                _row["max_abs_diff"] if mixed_max is None
                else max(mixed_max, _row["max_abs_diff"])
            )
            if _row["max_abs_diff"] > FP16_LOGIT_TOL:
                mixed_bad_rows.append(
                    f"{_rid} 第 {_row['step']} 步 max|Δ|={_row['max_abs_diff']:.4f}"
                    f" > {FP16_LOGIT_TOL}"
                )

    result["checks"]["V3_variable_length_batch"] = {
        "prompt_lengths": lengths,
        "per_row_match": v3_per_row,
        "logits_equivalence_per_row": v3_logits,
        # 名字如实反映它测的东西：各行 n_new 不同 ⇒ 驱动按 n_new 拆组。
        # `grouped_prefill_forward_call_rows` 会显示每次前向只有 1 行（= 没有合批）。
        "grouped_prefill_per_row": pad_rows,
        "grouped_prefill_forward_call_rows": list(cap.call_rows),
        "grouped_prefill_forward_call_n_new": list(cap.call_n_new),
        "fp16_logit_tol": FP16_LOGIT_TOL,
        "reference_tokens": ref_v3,
        "driver_tokens": drv_v3,
        "token_ids_exact_match": all(v3_per_row.values()),
        "logits_equivalent": v3_rows_ok,
        "grouped_prefill_preserves_rows": pad_ok,
        # 真正覆盖"合批 + 补位"的一段
        "mixed_history_batched_decode": mixed,
        "mixed_history_max_abs_diff": mixed_max,
        "mixed_history_within_fp16_tol": (
            not mixed.get("errors") and bool(mixed.get("rows")) and not mixed_bad_rows
        ),
    }
    # 「合批 decode」必须**能失败**：
    for _e in mixed.get("errors") or []:
        result["failures"].append(f"V3 合批 decode：{_e}")
    if not (mixed.get("rows") or {}):
        result["failures"].append(
            "V3 合批 decode：一行都没比到（不能当作通过）"
        )
    for _b in mixed_bad_rows:
        result["failures"].append(f"V3 合批 decode 行分布不等价：{_b}")

    # 把"分组 prefill 段其实没合批"显式写进 notes：产物里那两个 true
    # 若仍被读成"变长批已验证"，缺口就会再次被藏住。
    if cap.call_rows and max(cap.call_rows) == 1:
        result["notes"].append(
            f"V3 分组 prefill：{len(cap.call_rows)} 次 forward 每次只有 1 行"
            "（各行 n_new 互不相同 ⇒ 驱动按 n_new 拆组，`pad` 恒为 0）"
            "——**该段不产生补位/合批**；变长合批 decode 与补位由"
            "「mixed_history_batched_decode」一段覆盖"
        )
    # 主判据（分布等价）不通过 → failures。token 分岔**分流**：
    # 近平局记 note；取证失败 / gap 不小 → needs_review，**同样阻断通过**。
    # 原先三种情况一律进 notes，等于这三个分支永远不会让验收失败。
    for rid, row in v3_logits.items():
        if "error" in row:
            result["failures"].append(f"V3 行 {rid}：{row['error']}")
        elif not row["within_fp16_tol"]:
            result["failures"].append(
                f"V3 行 {rid} 分布超出 fp16 容差：max|Δ|={row['max_abs_diff']:.4f} "
                f"> {FP16_LOGIT_TOL}"
            )
        elif not row["token_ids_exact_match"]:
            # 分岔就给证据：分岔步的 top1/top2 差距。gap≈0 ⇒ 近平局，与实现无关。
            #
            # ⚠️ 必须按 rid 反查目标请求，不能用循环残留变量：本块之前的
            # `for r in v3_reqs` 结束时 r 停在**最后一个**请求上，用它会把
            # 别的行的分岔情况报成本行的（踩过：v3_0 分岔却拿到 v3_3 的对比，
            # 于是 det=None、结论变成"需复核"，把一个已知的近平局说成待查）。
            #
            # ⚠️⚠️ **这里只验收首个分岔步，是唯一正确的口径，不是偷懒。**
            # V3 的两条序列都是**贪心自喂**的（见 ``run_driver_greedy``）：
            # 第 k 步一旦分岔，第 k+1 步起两侧喂进去的历史就**不同**了，
            # 后面的"分岔"是这条因果链的下游，**不是彼此独立的验收点**。
            # 真机实测（7B, v3_0）：第 4 步是近平局（gap=0.0000），
            # 而第 6~15 步的"分岔"gap 高达 0.24~6.51 —— 逐步验收会把这
            # 一次已解释的平局放大成 10 条"需复核"（首跑实测正是如此）。
            # 「每一步都是独立验收点」**只在共享参考历史（teacher forcing）时成立**，
            # 那条判据由 V4b 与「mixed_history_batched_decode」承担。
            # 全部分岔步仍记进产物供人核对，但不逐个判态。
            _req = next(x for x in v3_reqs if x.req_id == rid)
            div_steps = all_token_divergences(drv_v3[rid], ref_v3[rid])
            row["divergence_steps"] = div_steps
            # 贪心自喂 ⇒ **只**首个分岔步是独立验收点（见 judged_divergence_steps）。
            judged = judged_divergence_steps(
                drv_v3[rid], ref_v3[rid], shared_history=False
            )
            if not judged:
                # 公共前缀完全一致却进了"分岔"分支 ⇒ 只可能是长度不等
                # （``token_ids_exact_match`` 比的是整条序列）。
                # 长度不等时"第几步分岔"没有定义，**不能**随便取一步的 gap 蒙混过去。
                row["first_divergence"] = None
                row["divergence_verdict"] = DIVERGENCE_FORENSICS_FAILED
                row["any_blocking_divergence"] = True
                result["needs_review"].append(
                    f"V3 行 {rid}: token 数不一致（驱动 {len(drv_v3[rid])} / "
                    f"参考 {len(ref_v3[rid])}）但公共前缀一致 —— 长度不等时"
                    "无「分岔步」可言，需复核"
                )
                continue
            # 用**记录下来的**参考分布取证（不再逐步重跑参考），并顺带核对
            # "重跑参考 == 先前记录"：V3 的参考序列原本只存了 token
            # （``reference_greedy``），这里跑一次带日志的版本，既拿到分布，
            # 也验证参考实现自身可复现。两次不一致 ⇒ 取证口径不成立。
            _ref_toks, _ref_lgs = reference_greedy_trace(
                model16, src.prompt_ids(_req), args.gen_len, args.device
            )
            i0 = judged[0]
            if _ref_toks != ref_v3[rid]:
                det: dict = {
                    "error": "重跑的参考序列与先前记录的不一致"
                    "（参考实现自身不可复现，取证口径不成立）",
                    "index": i0,
                }
            else:
                lg0 = _ref_lgs[i0] if i0 < len(_ref_lgs) else None
                det = (
                    gap_detail_from_logits(lg0, i0)
                    if lg0 is not None
                    else {"error": f"参考第 {i0} 步分布缺失", "index": i0}
                )
            row["first_divergence"] = det
            # 口径核对：再走一次**重跑参考到分岔步**那条路径，与上面用记录分布
            # 得到的 gap 比。两条路径给出不同结论时必须显形——那说明其中一条
            # 有问题，而不是"取哪个都行"。这正是 ``gap_detail_from_logits``
            # 文档里"保留一次重跑做口径核对"那条要求的落点。
            _det_rerun = _first_divergence_detail(
                model16, src.prompt_ids(_req), drv_v3[rid], ref_v3[rid],
                args.device, index=i0,
            )
            row["first_divergence_rerun"] = _det_rerun
            _g_record = det.get("top1_top2_gap")
            _g_rerun = _det_rerun.get("top1_top2_gap")
            # 只在"记录路径确实取到了 gap"时比：记录路径自己就失败的话，
            # 上面已经把根因报出去了，这里再报一条只是同一件事的复述。
            if _g_record is not None and (
                _g_rerun is None or abs(float(_g_record) - float(_g_rerun)) > 1e-6
            ):
                result["needs_review"].append(
                    f"V3 行 {rid}: 两条取证路径给出不同的分岔步 gap"
                    f"（记录分布 {_g_record} vs 重跑参考 {_g_rerun}）"
                    "——取证口径不成立，需复核"
                )
            verdict, msg = classify_token_divergence(f"V3 行 {rid}", det)
            row["divergence_verdict"] = verdict
            row["any_blocking_divergence"] = blocks_acceptance(verdict)
            if verdict == DIVERGENCE_NEAR_TIE:
                result["notes"].append(
                    f"（max|Δ|={row['max_abs_diff']:.4f}≈0）{msg}"
                )
            else:
                result["needs_review"].append(msg)
    for rid, row in pad_rows.items():
        if "error" in row:
            result["failures"].append(f"V3 补位检查 行 {rid}：{row['error']}")
        elif not row["within_fp16_tol"]:
            result["failures"].append(
                f"V3 补位检查 行 {rid} 分布超出 fp16 容差：max|Δ|={row['max_abs_diff']:.4f}"
            )

    # ---- V4: chunked prefill ----
    print("[6/6] V4 chunked prefill（分块 vs 不分块）…")
    v4_req = Request(req_id="v4", prompt_len=40, gen_len=args.gen_len)
    v4_ids = src.prompt_ids(v4_req)
    # 参考侧同时给出**每步**的分布：多步 decode 的判据要用它们。
    ref_v4, ref_logits_v4 = reference_greedy_trace(
        model16, v4_ids, args.gen_len, args.device
    )

    # V4 驱动的槽位数，**两个时刻各记一次**：
    #   * ``_after_run``   —— 全部 forward 做完、``retire`` 之前。单请求收尾时
    #     它有且只有 1 个（请求还没被释放，本来就该在）；
    #   * ``_after_retire``—— 显式释放之后，必须是 0。
    # 为什么要留前一个：它是这条判据的**反向对照**。若计数器坏了恒返回 0，
    # "释放后为 0"会**在什么都没测的情况下通过**——记录一次"应该是 1"就能识破。
    v4_live_at_end: dict[str, int] = {}
    v4_live_before_retire: dict[str, int] = {}

    def _v4_decode(chunks: list[int] | None, gen_len: int):
        """跑一次 V4：先 prefill（``chunks=None`` 表示整段），再走 ``gen_len`` 步 decode。

        ``gen_len=1`` = 只取 prefill 之后的首 token（旧 V4 的全部内容）。
        ``gen_len=N`` = 继续 N-1 步 **KV decode**——这正是旧 V4 缺的那一段：
        它 prefill 完立刻 ``retire``，于是"分块建起来的 KV 能否支撑后续 decode"
        从未被验证过；产物里表现为"参考生成 16 个 token、分块驱动只生成 1 个"，
        而判据仍能 passed。

        decode 每步**强制注入参考 token**（teacher forcing）：``_Slot.take_next``
        在 prompt 算完之后喂的是驱动**自己采样出**的 token，一旦某步因近平局与
        参考分岔，此后两侧历史不同，逐位置比 logits 就变成"比两段不同历史"。
        注入参考 token 才能保证判据是"相同历史下比较"。

        返回 ``(自己产出的 token 列表, 每步末位分布列表)``；分布列表长度
        == 实际完成的步数 + 1（含 prefill 那一步）。
        """
        g = _LogitsGrabber(model16)
        d = TransformerDriver(
            model_name=args.model, device=args.device, precision="fp16",
            model=g, tokenizer=tokenizer, prompt_source=src,
        )
        d.load()
        own: list[int] = []
        steps: list = []

        if chunks is None:
            o = d.forward(
                [ForwardItem(req=v4_req, n_new_tokens=len(v4_ids),
                             need_token=True)]
            )
            own += [s.token_id for s in o.slots if s.token_id is not None]
        else:
            for ci, n in enumerate(chunks):
                last = ci == len(chunks) - 1
                o = d.forward(
                    [ForwardItem(req=v4_req, n_new_tokens=n, need_token=last)]
                )
                own += [s.token_id for s in o.slots if s.token_id is not None]
        steps.append(g.last[0, -1, :].clone() if g.last is not None else None)

        # decode 段抽成模块级函数（见 decode_with_teacher_forcing）：
        # 它是 V4b 的核心机制，必须能被单独测试——写在闭包里就只能靠真机跑一次
        # 才知道"参考 token 到底喂进去了没有"。
        dec_own, dec_steps = decode_with_teacher_forcing(
            d, g, v4_req, ref_v4, gen_len
        )
        own += dec_own
        steps += dec_steps

        key = "whole" if chunks is None else (
            str(chunks) if len(chunks) <= 5 else f"[1]x{len(chunks)}"
        )
        # ⚠️ **顺序不能反。** 原先是先读 ``live_slots()`` 再 ``retire``，于是
        # 读到的永远是 1（请求还没释放）⇒ 判据"收尾应为 0"**恒不成立**。
        # 这是"恒真判据"的镜像缺陷——**恒假判据**同样不携带信息，而且看起来
        # 像"真的抓到了 KV 泄漏"。实测：本脚本改完后的**第一次真机跑**
        # （2026-09-24，0.5B 与 7B 各一次）5 个 V4 切法全部报"仍有 1 个常驻 KV 槽"，
        # 而两次 run 的其它判据都正常 —— 当时没有别的解释，只能是这条自身写错。
        if hasattr(d, "live_slots"):
            v4_live_before_retire[f"v4_{key}"] = d.live_slots()
        d.retire(["v4"])
        if hasattr(d, "live_slots"):
            v4_live_at_end[f"v4_{key}"] = d.live_slots()
        return own, steps

    def _v4_run(chunks: list[int] | None):
        """只做 prefill 的 V4（旧口径）：返回 (首 token 列表, 首分布)。"""
        own, steps = _v4_decode(chunks, gen_len=1)
        return own, (steps[0] if steps else None)

    chunks = [16, 16, 8]
    whole_tokens, whole_lg = _v4_run(None)
    chunk_tokens, chunk_lg = _v4_run(chunks)

    if whole_lg is None or chunk_lg is None:
        result["checks"]["V4_chunked_prefill"] = {
            "chunk_sizes": chunks,
            "error": "拿不到 prefill 末位分布（不能当作通过）",
        }
        result["failures"].append("V4 拿不到分布 —— 无法验证分块正确性")
    else:
        lg_d = _compare_logits(chunk_lg, whole_lg)
        # 参考自身的 top1/top2 gap，用于判断 token 分岔是否只是近平局。
        # 直接用已记录的参考首分布，不再单独跑一次前向（两次前向给出不同
        # 结论的情形一定意味着其中一条路径有问题）。
        ref_lg = ref_logits_v4[0]
        gap = _top1_top2_gap(ref_lg)

        # 参考与"整段"之间也应等价 —— 若不等价，说明连基线都不可信，
        # 此时"分块 vs 整段"的一致就不能当作分块正确的证据。
        ref_vs_whole = (
            _compare_logits(whole_lg, ref_lg) if ref_lg is not None else None
        )

        within = lg_d["max_abs_diff"] <= FP16_LOGIT_TOL
        result["checks"]["V4_chunked_prefill"] = {
            "chunk_sizes": chunks,
            "scored": "prefill 末位分布（首 token）——**仅覆盖 prefill**，"
                      "多步 decode 见 V4b",
            "whole_vs_chunked_logits": lg_d,
            "reference_vs_whole_logits": ref_vs_whole,
            "fp16_logit_tol": FP16_LOGIT_TOL,
            "within_fp16_tol": within,
            "whole_first_token": whole_tokens[0] if whole_tokens else None,
            "chunked_first_token": chunk_tokens[0] if chunk_tokens else None,
            "reference_first_token": ref_v4[0] if ref_v4 else None,
            "argmax_whole": int(whole_lg.argmax().item()),
            "argmax_chunked": int(chunk_lg.argmax().item()),
            "ref_top1_top2_gap": gap,
            "chunked_vs_whole_tokens_match": whole_tokens == chunk_tokens,
        }
        if not within:
            result["failures"].append(
                f"V4 分块与整段分布不等价：max|Δ|={lg_d['max_abs_diff']:.4f} "
                f"> {FP16_LOGIT_TOL}"
            )
        if ref_vs_whole is not None and ref_vs_whole["max_abs_diff"] > FP16_LOGIT_TOL:
            result["failures"].append(
                f"V4 整段与参考分布不等价：max|Δ|={ref_vs_whole['max_abs_diff']:.4f} "
                f"> {FP16_LOGIT_TOL}（基线不可信，分块结论无从谈起）"
            )
        elif whole_tokens != chunk_tokens:
            # 首 token 分岔：用**同一步**的参考 gap 判三态，而不是拿别处的 gap 糊过去。
            verdict, msg = classify_token_divergence(
                "V4 首 token", gap_detail_from_logits(ref_lg, 0)
            )
            if verdict == DIVERGENCE_NEAR_TIE:
                result["notes"].append(f"{msg}；分块实现本身正确（分布等价）")
            else:
                result["needs_review"].append(msg)

    # ---- V4b: 分块 prefill 之后继续**多步 KV decode** ----
    #
    # 上面 V4 只验证到"prefill 之后的首 token 分布"。旧实现 prefill 完立刻
    # retire，于是"分块建起来的 KV 能否支撑后续 decode"从未被验证过——
    # 7B 的产物里参考生成 16 个 token、分块驱动只生成 1 个，判据仍然 passed。
    # 这一段补齐缺口：在**相同历史**下逐步比分布（teacher forcing 见 _v4_decode）。
    print("[6/6] V4b 分块后多步 KV decode（teacher forcing，逐步比分布）…")
    try:
        dec_whole_own, dec_whole_steps = _v4_decode(None, gen_len=args.gen_len)
        dec_chunk_own, dec_chunk_steps = _v4_decode(chunks, gen_len=args.gen_len)
        v4b_error = None
    except Exception as exc:  # noqa: BLE001
        # 槽位被提前释放 / prompt 没喂完都会走到这里。必须显形为失败：
        # 这正是旧实现"只跑 1 步也通过"的那条路。
        dec_whole_own = dec_whole_steps = None
        dec_chunk_own = dec_chunk_steps = None
        v4b_error = f"{type(exc).__name__}: {exc}"

    if v4b_error is not None:
        result["checks"]["V4b_chunked_multi_step_decode"] = {"error": v4b_error}
        result["failures"].append(f"V4b 多步 decode 无法完成：{v4b_error}")
    else:
        vs_ref = compare_step_logits(dec_chunk_steps, ref_logits_v4)
        vs_whole = compare_step_logits(dec_chunk_steps, dec_whole_steps)
        div = first_token_divergence(dec_chunk_own, ref_v4)
        result["checks"]["V4b_chunked_multi_step_decode"] = {
            "scored": "分块 prefill → 多步 KV decode，每步用参考 token 作为输入"
                      "（相同历史），逐步比末位分布",
            "gen_len": args.gen_len,
            "n_steps_chunked": len(dec_chunk_steps),
            "n_steps_whole": len(dec_whole_steps),
            "chunked_vs_reference_logits": vs_ref,
            "chunked_vs_whole_logits": vs_whole,
            "fp16_logit_tol": FP16_LOGIT_TOL,
            "n_tokens_chunked": len(dec_chunk_own),
            "n_tokens_whole": len(dec_whole_own),
            "n_tokens_reference": len(ref_v4),
            "reference_tokens": ref_v4,
            "driver_tokens_chunked": dec_chunk_own,
            "driver_tokens_whole": dec_whole_own,
            "length_match": div["length_match"],
            "first_divergence_step": div["first_divergence"],
            "whole_vs_chunked_tokens_match": dec_whole_own == dec_chunk_own,
        }
        # ① 步数／token 数：少一步都不行。"驱动只跑了 1 步"正是旧缺陷的长相。
        if not div["length_match"]:
            result["failures"].append(
                f"V4b 分块驱动产出 {div['n_driver']} 个 token，参考产出 "
                f"{div['n_ref']} 个（长度不等，不得截断比较）"
            )
        if len(dec_chunk_steps) != args.gen_len:
            result["failures"].append(
                f"V4b 分块驱动只完成 {len(dec_chunk_steps)} 步 decode，"
                f"应为 {args.gen_len} —— 多步 KV decode 未被完整验证"
            )
        # ② 逐步分布等价：这是分块 + KV decode 正确性的主判据。
        for label, cmp_res in (("与参考", vs_ref), ("与整段", vs_whole)):
            if "error" in cmp_res:
                result["failures"].append(f"V4b 分布比对{label}失败：{cmp_res['error']}")
            elif cmp_res["max_abs_diff"] > FP16_LOGIT_TOL:
                result["failures"].append(
                    f"V4b 分块 decode 逐步分布{label}不等价："
                    f"max|Δ|={cmp_res['max_abs_diff']:.4f} > {FP16_LOGIT_TOL}"
                    f"（第 {cmp_res['worst_step']} 步最差）"
                )
        # ③ token 分岔：**逐个分岔步**三态分流，非近平局一律阻断通过。
        #
        # 为什么必须逐个：只看首个分岔时，首步若是近平局（可解释），
        # 后面某步 gap 很大（与"分布等价"直接矛盾）会被**完全掩盖**——
        # 而 teacher forcing 下每个位置的输入都来自参考，
        # **每一步都是彼此独立的验收点**，没有"前面错了所以后面不可比"的借口。
        # （这条口径**只**适用于共享历史；V3 的贪心序列只验首个，
        #   见 judged_divergence_steps 的说明。）
        div_steps = judged_divergence_steps(
            dec_chunk_own, ref_v4, shared_history=True
        )
        div_report: list[dict] = []
        for i in div_steps:
            ref_lg_i = ref_logits_v4[i] if i < len(ref_logits_v4) else None
            det = (
                gap_detail_from_logits(ref_lg_i, i)
                if ref_lg_i is not None
                else {"error": f"参考第 {i} 步分布缺失", "index": i}
            )
            verdict, msg = classify_token_divergence(
                f"V4b 多步 decode（第 {i} 步）", det
            )
            div_report.append({"step": i, "verdict": verdict, "message": msg})
            if verdict == DIVERGENCE_NEAR_TIE:
                result["notes"].append(f"{msg}；分块 decode 实现本身正确（分布等价）")
            else:
                result["needs_review"].append(msg)
        _v4b_block = result["checks"]["V4b_chunked_multi_step_decode"]
        _v4b_block["divergence_steps"] = div_steps
        _v4b_block["n_divergence_steps"] = len(div_steps)
        _v4b_block["divergence_verdicts"] = div_report
        _v4b_block["any_blocking_divergence"] = any(
            blocks_acceptance(d["verdict"]) for d in div_report
        )

    # 分块的自洽性还要求多种切法都一致 —— 只测一种切法可能恰好避开问题
    alt_splits = [[20, 20], [39, 1], [1] * len(v4_ids)]
    alt_ok: dict[str, bool] = {}
    for sp in alt_splits:
        t, lg = _v4_run(sp)
        key = str(sp) if len(sp) <= 5 else f"[1]x{len(sp)}"
        # 与整段比分布（这才是正确性；token 可能因近平局分岔）
        alt_ok[key] = (
            lg is not None
            and _compare_logits(lg, whole_lg)["max_abs_diff"] <= FP16_LOGIT_TOL
        ) if whole_lg is not None else False
    result["checks"]["V4_alt_splits_equivalent"] = alt_ok
    for k, ok in alt_ok.items():
        if not ok:
            result["failures"].append(f"V4 切法 {k} 与整段分布不等价")

    # ---- 资源生命周期 ----
    #
    # V4 的驱动在 ``_v4_run`` 内部创建并已 retire，这里不能再引用它；
    # 改为直接断言"每次 _v4_run 结束后槽位回到 0"，并把结果并入本块。
    result["checks"]["resource_lifecycle"] = {
        "drv2_live_at_end": drv2.live_slots() if hasattr(drv2, "live_slots") else None,
        "drv3_live_at_end": drv3.live_slots() if hasattr(drv3, "live_slots") else None,
        "v4_drivers_live_at_end": v4_live_at_end,
        # 反向对照：单请求全部 forward 做完、retire 之前应当恰好是 1。
        # 它**只用于证伪计数器**（恒 0 的计数器会让"释放后为 0"白通过），
        # 不构成正确性判据，所以不参与 passed。
        "v4_drivers_live_before_retire": v4_live_before_retire,
    }
    _rl_failures, _rl_review = resource_lifecycle_verdict(
        driver_live_at_end={
            "drv2": drv2.live_slots() if hasattr(drv2, "live_slots") else None,
            "drv3": drv3.live_slots() if hasattr(drv3, "live_slots") else None,
        },
        v4_live_after_retire=v4_live_at_end,
        v4_live_before_retire=v4_live_before_retire,
    )
    result["failures"].extend(_rl_failures)
    result["needs_review"].extend(_rl_review)

    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # ---- 源码溯源：产物必须能回答"这是哪一版判据跑出来的" ----
    #
    # 为什么必须放进产物：这套判据在 2026-09-24 一天之内改过四版
    # （收紧两处漏检 → 又修掉两处判据自身的缺陷），而**不带溯源戳的 JSON 与
    # 任何一版都对不上**——四份产物的 `passed` 字段可以完全一样、判据内容不同。
    # 部署戳的摘要作用域包含 ``scripts/`` 与 ``src/ascend_opt``，所以戳有效
    # 就等于"这份产物出自那棵树"，**包括本脚本自己的版本**。
    # 取不到时 ``validated=False`` 且 ``git_commit=None``：**不填一个像真的值**
    # （"没记录"与"记错了"是两回事，后者更危险）。
    try:
        from ascend_opt.provenance import read_source_stamp

        result["source_provenance"] = read_source_stamp(_ROOT)
    except Exception as exc:  # noqa: BLE001 —— 溯源失败不能吞掉已经跑完的测量
        result["source_provenance"] = {
            "git_commit": None,
            "validated": None,
            "reason": f"溯源不可用：{type(exc).__name__}: {exc}",
        }
    # ``needs_review`` 与 ``failures`` **同样阻断通过**（判据本体见 overall_passed，
    # 它同时被测试直接断言）。
    result["passed"] = overall_passed(result["failures"], result["needs_review"])

    # 先序列化、再落盘。
    #
    # 顺序很重要：原实现直接在文件句柄上 json.dump，一旦遇到不可序列化的对象，
    # 异常会从 dump 里抛出，文件被留在"写了一半"的状态，而**已经跑完的全部测量
    # 一并丢失**。踩过一次：DynamicCache 漏进 result，六步测量白跑。
    # 现在即使序列化失败，也要把失败显形地写进产物，测量结果不丢。
    try:
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        write_error = None
    except TypeError as exc:
        # 兜底：把不可序列化的部分剥掉，保留可以留的证据，并把失败记录在案
        payload = json.dumps(
            {
                "passed": False,
                "failures": result["failures"]
                + [f"结果无法序列化，已降级落盘：{type(exc).__name__}: {exc}"],
                # 降级产物里 needs_review 也要留下：只留 failures 会让
                # "必须人看"的证据在降级时消失，而 passed 本来就是 False。
                "needs_review": list(result["needs_review"]),
                "checks": _jsonable(result["checks"]),
                "note": "序列化降级产物——不可序列化的字段已替换为类型摘要，"
                        "数值判据请以 failures / needs_review 为准。",
            },
            ensure_ascii=False,
            indent=2,
        )
        write_error = exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n"：证据文件的字节即证据，跨平台必须一致
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.write("\n")

    if write_error is not None:
        print(f"⚠️  产物经过降级序列化：{write_error}", file=sys.stderr)

    print("\n" + "=" * 60)
    if result["passed"]:
        print("全部判据通过 ✅")
    else:
        print(f"有 {len(result['failures'])} 项不通过 ❌")
        for f in result["failures"]:
            print(f"  - {f}")
    if result["needs_review"]:
        # 与 failures 分开印：它们是"必须人看"的条目，不是"已确认实现错误"。
        # 但两者**都**让退出码非零——这是本次修复的关键。
        print(f"\n需复核（阻断通过，共 {len(result['needs_review'])} 条）：")
        for n in result["needs_review"]:
            print(f"  ! {n}")
    if result["notes"]:
        print(f"\n说明（不构成失败，共 {len(result['notes'])} 条）：")
        for n in result["notes"]:
            print(f"  · {n}")
    print(f"\n产物：{out_path}")
    print("=" * 60)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
