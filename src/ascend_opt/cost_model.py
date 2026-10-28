"""成本模型：**加载、指纹比对、范围检查、预测**（第 12 步的模型层，纯逻辑）。

三条纪律（都来自第 11 步的实测，不是设计口味）
----------------------------------------------
1. **指纹不符即拒**（验收第 5 条）。模型文件里存的是**标定当时**的硬件/软件/模型/精度；
   运行时再取一份 `observed` 逐项比对。**不可测的值算"没确认"，默认也拒**
   —— "没确认"与"确认相符"是两件事，把前者当后者就是"静默使用"。
   要放宽必须显式传 `strict=False`，且放宽了什么会被记进 `describe()`。
2. **越界即回退**（预注册 §六）。标定范围是逐 `batch` 的 `prefill_tok` / `kv_len` 实测边界；
   越界返回 `None`，由调用方回退 —— **不得外推**（第 11 步实测：外推到 `kv=2048` 时
   低估 p95 达 23.11%，且 92% 的情况是低估）。
3. **安全余量从"低估尾部"推**（预注册 §四）。低估 ⇒ 控制器以为来得及 ⇒ SLO 违约；
   高估只是保守。余量取留出形状上的低估 p95，随预测一起乘上去。

   ⚠️ **2026-09-26 更正（这条的"量级"不可信）**：`1.77%` 那个数是 **1-in-4 划分下的留出误差**，
   而换更严的划分会崩到 **68.78%（1-in-3）／95.25%（1-in-2）** ⇒ 真因是 **`kv` 维不可辨识**
   （每段 3 参数、而网格里 `kv` 只有 3 个取值）。⇒ **"余量 = 低估 p95" 这条纪律仍然成立，
   但用 `model_v1.json` 里的 `safety_margin` 当"已覆盖低估尾部"是站不住的。**
   要恢复可信度需要**可辨识的网格**（v4：kv 加密到 5 值）**+ 更严的划分**（1-in-3）。

⚠️ `n_forwards` **不是**范围的一部分，而是**乘子**：实测它 1→4 的**单前向成本逐位相同**
（加和性成立），所以越界的判断只看 `(batch, prefill_tok, kv_len)`。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: 指纹里必须逐项比对的叶子路径（与 `fingerprint` 的两层结构一致）。
FINGERPRINT_LEAVES: tuple[tuple[str, str], ...] = (
    ("hardware", "device_model"),
    ("hardware", "driver_version"),
    ("software", "torch"),
    ("software", "torch_musa"),
    ("software", "transformers"),
    ("software", "python"),
    ("model", "id"),
    ("model", "revision"),
)
#: 单独处理的标量指项（不在两层结构里）。
FINGERPRINT_SCALARS: tuple[str, ...] = ("precision",)


@dataclass(frozen=True)
class PredictedCost:
    """一次预测。**带上"用了哪个模型的哪一段、加了多少余量"**，便于事后追责。"""

    per_forward_ms: float
    total_ms: float
    margin: float
    batch: int
    prefill_tok: int
    kv_len: int
    n_forwards: int
    # Exact execution tables measure host once over the entire logical step.
    host_ms: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "per_forward_ms": round(self.per_forward_ms, 4),
            "total_ms": round(self.total_ms, 4),
            "margin": self.margin,
            "shape": {"batch": self.batch, "prefill_tok": self.prefill_tok,
                      "kv_len": self.kv_len, "n_forwards": self.n_forwards},
        }


class CostModelError(ValueError):
    """模型加载/比对失败。**继承 ValueError**：配置类错误就该在加载期炸掉。"""


@dataclass
class CostModel:
    payload: dict[str, Any]
    #: 放宽过什么（空 = 严格模式、且全部叶子都确认相符）。
    relaxations: list[str] = field(default_factory=list)

    def predict_execution(self, shape: Mapping[str, Any], *,
                          execution_signature: Mapping[str, Any],
                          apply_margin: bool = True) -> tuple[PredictedCost | None, str, str]:
        """Exact whole-step prediction. Legacy scalar tables are diagnostics only.

        No sum of independently measured groups is assumed. A new table must
        include the measured total forward and host cost for this full vector.
        """
        from .execution_shape import execution_shape_key
        items = shape["items"]
        if any(i["kv_len"] is None for i in items):
            return None, "unknown_kv", "缺少逐请求 KV 长度，不能用 0 或最大值代替"
        if not items:
            return None, "idle", "空闲执行形状"
        table = self.payload.get("execution_shape_table")
        if table is None:
            return None, "execution_contract_unvalidated", "旧标定与当前执行语义未对齐，需锚点复核/整步标定"
        if self.payload.get("execution_signature") != dict(execution_signature):
            return None, "execution_signature_mismatch", "标定与当前模型/执行开关不一致"
        if self.payload.get("execution_timing") != "whole_step_fwd_plus_host_v1":
            return None, "timing_contract_mismatch", "缺少整步前向与 host 计时契约"
        hit = table.get(execution_shape_key(shape))
        if hit is None:
            return None, "missing_execution_key", "真实整步执行形状未标定（不可插值/外推）"
        values = [hit.get("fwd_ms"), hit.get("host_ms")]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in values) or values[0] <= 0 or values[1] < 0):
            return None, "invalid_execution_record", "整步标定成本缺失或非法"
        margin = float(self.payload.get("safety_margin") or 0.0) if apply_margin else 0.0
        if not math.isfinite(margin) or margin < 0:
            return None, "invalid_execution_record", "误差余量非法"
        total = float(values[0]) * (1.0 + margin)
        n = int(shape["n_forwards"])
        return PredictedCost(total / n, total, margin, len(items),
                             sum(i["n_new_tokens"] for i in items if i["role"] == "prefill"),
                             max(i["kv_len"] for i in items), n,
                             host_ms=float(values[1])), "exact_execution_key", "精确整步形状命中"

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, path: str | Path, observed: Mapping[str, Any], *,
             strict: bool = True, expect_protocol: str | None = None) -> "CostModel":
        """加载并比对指纹。**不满足即抛** `CostModelError`（不返回半可用的模型）。

        * `strict=True`（默认）：**不符或未测**都拒。
        * `strict=False`：只有**真实不符**才拒；未测的记进 `relaxations`（显式放宽）。
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        relaxations: list[str] = []
        problems: list[str] = []
        unmeasured: list[str] = []

        if "execution_shape_table" in payload:
            if payload.get("data_label") != "MEASURED" or (payload.get("validation") or {}).get("model_holds") is not True:
                problems.append("整步执行模型必须来自 MEASURED 且通过独立验证")
            code = payload.get("execution_code_digest")
            if not code or code != observed.get("execution_code_digest"):
                problems.append("执行源码摘要缺失/不符，旧执行路径成本不可直接复用")

        if expect_protocol is not None:
            got = payload.get("protocol_version")
            if got != expect_protocol:
                problems.append(f"protocol_version={got!r} != 期望 {expect_protocol!r}")

        exp = payload.get("fingerprint") or {}
        for group, leaf in FINGERPRINT_LEAVES:
            want = (exp.get(group) or {}).get(leaf)
            got = (observed.get(group) or {}).get(leaf) if isinstance(
                observed.get(group), Mapping) else None
            if want is None:
                problems.append(f"{group}.{leaf}: 模型里没有期望值 ⇒ 无法比对")
            elif got is None:
                unmeasured.append(f"{group}.{leaf}（期望 {want!r}，运行时不可测）")
            elif got != want:
                problems.append(f"{group}.{leaf}: 实测 {got!r} != 标定 {want!r}")
        for leaf in FINGERPRINT_SCALARS:
            want = exp.get(leaf)
            got = observed.get(leaf)
            if want is None:
                problems.append(f"{leaf}: 模型里没有期望值 ⇒ 无法比对")
            elif got is None:
                unmeasured.append(f"{leaf}（期望 {want!r}，运行时不可测）")
            elif got != want:
                problems.append(f"{leaf}: 实测 {got!r} != 标定 {want!r}")

        if strict:
            problems.extend(f"{u} ⇒ 未确认" for u in unmeasured)
        if problems:
            raise CostModelError(
                "成本模型指纹比对未通过（失配不能静默使用）：\n  - " + "\n  - ".join(problems))
        if unmeasured:
            relaxations.append("显式放宽了未测项：" + "、".join(unmeasured))

        for key in ("coefficients", "calibrated_range", "safety_margin"):
            if key not in payload:
                raise CostModelError(f"模型缺少 {key}")
        return cls(payload=payload, relaxations=relaxations)

    @classmethod
    def load_offline(cls, path: str | Path) -> "CostModel":
        """**离线**加载：跳过指纹校验，但把这件事**记进 `relaxations`**。

        为什么要有这个入口，而不是让调用方造一个"声明指纹"去比对：
        离线（模拟器）根本没有真实运行环境可观测，**造一个指纹去比对就是假装校验过了**
        —— 那种"看起来更严格"的做法比明说"这里没校验"更危险。
        ⇒ 明说：`relaxations` 里带一条记录，且会随 `describe()` 进产物。

        ⚠️ 正式比较（第 14 步）必须走 `load()` 的严格路径。
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("coefficients", "calibrated_range", "safety_margin"):
            if key not in payload:
                raise CostModelError(f"模型缺少 {key}")
        return cls(payload=payload, relaxations=[
            "离线路径：**未做指纹校验**（离线没有真实运行环境可观测）⇒ "
            "这一层防护只在带实测指纹的路径上生效"])

    # ------------------------------------------------------------------ 范围
    def in_range(self, *, batch: int, prefill_tok: int, kv_len: int) -> bool:
        """是否落在**实测**标定范围内。训练集里没出现过的组合一律算越界。"""
        rng = (self.payload.get("calibrated_range") or {}).get(str(int(batch)))
        if not rng:
            return False
        for axis, val in (("prefill_tok", int(prefill_tok)), ("kv_len", int(kv_len))):
            lo, hi = (rng.get(axis) or [None, None])
            if lo is None or hi is None:
                return False
            if not (lo <= val <= hi):
                return False
        return True

    # ------------------------------------------------------------ 候选形状
    def lookup_prefill_candidates(self, *, batch: int, kv_len: int,
                                  cap: int) -> tuple[int, ...] | None:
        """`lookup` 模式下，本 `(batch, kv_len)` 桶**实际有键**的 `prefill_tok`（`0 < c ≤ cap`，升序）。

        ★ 为什么需要它（2026-09-29 实测的反例）
        ----------------------------------------
        `lookup` 是**稀疏**的：只有实测采过的形状才有键。缺键时 :meth:`predict`
        返回 `None`，那是"**未知**"，**不是**"不可行"。而在 `c` 上做二分需要
        "`T(n, c)` 随 `c` 单调 + 可行域连续"两个前提 —— 稀疏表上**两个都不成立**。

        实测（`model_v7`，`b1` / `kv` 桶 1024）：可预测的只有
        `c ∈ {128, 256, 512, 1024}`，`c ∈ {192, 2048, 4096}` **均缺键**。
        于是二分从 `mid=2048`（缺键）起一路 `hi = mid - 1`
        ⇒ 即便 `c=1024` 明明可行（预算 400 ms > 316.9 ms），也收敛到 **`chunk = 0`（decode-only）**。
        ⇒ 这不是"保守"，而是**把一个最接近方法基线人为削弱**（会把 `slo_weave`
        的 decode-only 计数抬高、动作幅度压小），正是"不通过削弱基线制造普遍收益"要禁止的。

        返回 `None` ⇒ **非 `lookup` 模式**（调用方沿用论文的二分形式，那里
        `T` 由 `segmented` 线性分段给出、单调且连续）。
        """
        if str(self.payload.get("default_mode") or "segmented") != "lookup":
            return None
        tbl = self.payload.get("lookup_table") or {}
        if not tbl:
            return None
        bk = self.payload.get("lookup_bucket") or {}
        kv_step = int(bk.get("kv_step") or 0)
        _kv = (int(kv_len) // kv_step) * kv_step if kv_step else int(kv_len)
        prefix = f"b{int(batch)}_pf"
        suffix = f"_kv{_kv}"
        out: set[int] = set()
        for key in tbl:
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            mid = key[len(prefix):len(key) - len(suffix)]
            if mid.isdigit() and 0 < int(mid) <= int(cap):
                out.add(int(mid))
        return tuple(sorted(out))

    # ------------------------------------------------------------------ 预测
    def predict(self, *, batch: int, prefill_tok: int, kv_len: int,
                n_forwards: int = 1, apply_margin: bool = True,
                mode: str | None = None) -> PredictedCost | None:
        """预测一步的成本。**越界/未命中返回 `None`**（调用方必须回退，不得外推）。

        ``mode``（第 14 步预注册分支）
        --------------------------------
        * ``segmented``（默认）：按 ``batch`` 分段、段内对 ``(prefill_tok, kv_len)`` 线性插值。
          覆盖率高，但**插值误差不可忽略** —— 秩保证划分下实测低估 p95 = **6.12%**。
        * ``lookup``：**只用实测点**（键 = 精确的 `(batch, prefill_tok, kv_len)`），
          取值取该键实测的**最大值**（保守：绝不低估）；**未命中就回退**。
          无插值误差，代价是**覆盖率** ⇒ 覆盖率必须与误差**并排报**。
        """
        m = str(mode or self.payload.get("default_mode") or "segmented")
        if m == "lookup":
            tbl = self.payload.get("lookup_table") or {}
            # **分桶**（把连续取值量化到桶）——纯精确表的覆盖率实测为 0.00%
            # （真实 kv 是 1568，不匹配任何标定点）。桶宽与取舍见
            # `scripts/sweep_lookup_buckets.py` 的曲线：kv 桶 256 ⇒ 覆盖 53.6%、
            # 保守高估 p95 仅 0.13%；桶 1024 ⇒ 覆盖 100% 但代价 p95 9.76%。
            bk = self.payload.get("lookup_bucket") or {}
            pf_step = int(bk.get("pf_step") or 0)
            kv_step = int(bk.get("kv_step") or 0)
            _pf = (int(prefill_tok) // pf_step) * pf_step if pf_step else int(prefill_tok)
            _kv = (int(kv_len) // kv_step) * kv_step if kv_step else int(kv_len)
            hit = tbl.get(f"b{int(batch)}_pf{_pf}_kv{_kv}")
            if not hit:
                # 未命中 ⇒ 回退。**不插值、不外推** —— 这正是与 segmented 的区别。
                return None
            per = float(hit.get("max") or 0.0)
            if per <= 0:
                return None
            margin = (float(self.payload.get("safety_margin") or 0.0)
                      if apply_margin else 0.0)
            per_m = per * (1.0 + margin)
            return PredictedCost(per_forward_ms=per_m, total_ms=per_m * int(n_forwards),
                                 margin=margin, batch=int(batch),
                                 prefill_tok=int(prefill_tok), kv_len=int(kv_len),
                                 n_forwards=int(n_forwards))
        if m != "segmented":
            raise ValueError(f"未知的预测方式 mode={m!r}（只支持 segmented / lookup）")
        if not self.in_range(batch=batch, prefill_tok=prefill_tok, kv_len=kv_len):
            return None
        coef = (self.payload.get("coefficients") or {}).get(str(int(batch)))
        if not coef or len(coef) != 3:
            return None
        c0, c1, c2 = coef
        per = c0 + c1 * float(prefill_tok) + c2 * float(kv_len)
        if per <= 0:
            # 负成本说明这一段的系数在本点外推出去了 ⇒ 判定为不可用而不是返回一个负数
            return None
        margin = float(self.payload.get("safety_margin") or 0.0) if apply_margin else 0.0
        per_m = per * (1.0 + margin)
        return PredictedCost(per_forward_ms=per_m, total_ms=per_m * int(n_forwards),
                             margin=margin, batch=int(batch), prefill_tok=int(prefill_tok),
                             kv_len=int(kv_len), n_forwards=int(n_forwards))

    # ------------------------------------------------------------------ 描述
    def describe(self) -> dict[str, Any]:
        p = self.payload
        return {
            "model_version": p.get("model_version"),
            "protocol_version": p.get("protocol_version"),
            "grid_sha256": p.get("grid_sha256"),
            "n_records": p.get("n_records"),
            "fit_on_sessions": p.get("fit_on_sessions"),
            "safety_margin": p.get("safety_margin"),
            "relaxations": list(self.relaxations),
        }
