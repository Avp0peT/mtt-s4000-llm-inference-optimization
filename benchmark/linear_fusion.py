"""小批量线性投影融合（P3）：QKV 3→1、gate/up 2→1。

设计纪律（逐条对应 `REFERENCE_8_PLANS.md` §P3）
----------------------------------------------
1. **权重只在装载时拼接一次**，绝不每 token 拼（每 token 拼 = 把省下的 launch
   换成等量的显存搬运，还额外多一次分配）。
2. **两个融合各自独立开关**（`fuse_qkv` / `fuse_gate_up`），可单独消融。
3. **保留原权重可追溯**：融合权重是 `cat` 出来的**新张量**，原 `nn.Linear` 的
   `weight`/`bias` **一个字节都不动** ⇒ `state_dict` 键不变、随时可回退。
4. **不假设调用顺序**：QKV 融合靠"q 先算、k/v 复用"来避免一次算三遍；
   但复用以 `(data_ptr, shape)` + **剩余消费计数**双重校验，任一条不成立就
   **回退到原 `F.linear`**。⇒ 顺序变了只会"没省到"，**不会算错**。
5. **切分要 contiguous 的地方如实付代价**：HF 的 attention 对 q/k/v 做
   ``.view(...)``，而"从融合输出里切出来的窄片"**不是连续张量**（行 stride 仍是
   融合宽度）⇒ 必须 ``.contiguous()``，那是一次真实拷贝。
   ⚠️ 这是 QKV 融合的**固有代价**，也是"融合不一定有净收益"的机制来源，
   所以逐调用把它**记数、记字节**（`n_split_copies` / `bytes_copied`）而不是当作没有。
   （gate/up 那边不需要：SiLU 与逐元素乘法对非连续张量天然可用。）

纯逻辑：**不在模块顶层 import torch**（`benchmark/` 会被 `--dry-run` 路径导入，
那里不该要求装 torch）。torch 只在函数内部拿。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Sequence

__all__ = ["FuseManifest", "install_fusions", "uninstall_fusions", "iter_layers"]


# --------------------------------------------------------------------------- #
# 清单（落盘用）
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class _Group:
    """一层里的一组待融合投影。持有拼接后的权重与原模块的引用。"""

    name: str
    modules: list[Any]                    # 原 nn.Linear，顺序 = 拼接顺序
    weight: Any                           # cat 出来的融合权重（新张量）
    bias: Any                             # 无 bias 时为 None
    split: tuple[int, ...]                # 各分量的输出维
    shapes: tuple[int, ...] | None        # 允许融合的 token 数；None = 全部
    need_contiguous: bool                 # 切出的片是否必须 contiguous
    #: 复用状态：`(data_ptr, shape)` + 剩余可消费分量数。任何不匹配 ⇒ 回退。
    pending_out: Any = None
    pending_key: tuple[Any, ...] | None = None
    pending_remaining: int = 0
    n_fused_calls: int = 0                # 走融合路径的次数（一组算一次）
    n_component_calls: int = 0            # 被调用的分量个数（原路径会是同值）
    n_reuse: int = 0                      # 复用命中次数
    n_fallback_calls: int = 0             # 回退到原 F.linear 的次数
    n_skipped_by_shape: int = 0           # 因形状不在集合内而回退
    n_split_copies: int = 0               # 切分时实际做的 contiguous 次数
    bytes_copied: int = 0                 # 切分拷贝的字节数（累计）

    def enabled_for(self, x: Any) -> bool:
        if self.shapes is None:
            return True
        return int(_token_count(x)) in self.shapes

    def clear(self) -> None:
        self.pending_out = None
        self.pending_key = None
        self.pending_remaining = 0


def _token_count(x: Any) -> int:
    """本步的"token 数" M = 除最后一维外的元素数（[B,S,H] ⇒ B*S）。"""
    n = 1
    for d in x.shape[:-1]:
        n *= int(d)
    return n


@dataclasses.dataclass
class FuseManifest:
    """融合的**全部可核验事实**：装了哪些、形状是什么、各路径走了几次。"""

    fuse_qkv: bool
    fuse_gate_up: bool
    shapes: tuple[int, ...] | None
    n_layers: int
    groups: list[_Group] = dataclasses.field(default_factory=list)
    skipped: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def installed(self) -> int:
        return len(self.groups)

    def _agg(self, name: str) -> dict[str, Any]:
        rows = [g for g in self.groups if g.name == name]
        if not rows:
            return {"groups": 0}
        g0 = rows[0]
        return {
            "groups": len(rows),
            "fused_calls": sum(g.n_fused_calls for g in rows),
            "component_calls": sum(g.n_component_calls for g in rows),
            "reuse_hits": sum(g.n_reuse for g in rows),
            "fallback_calls": sum(g.n_fallback_calls for g in rows),
            "skipped_by_shape": sum(g.n_skipped_by_shape for g in rows),
            "split_copies": sum(g.n_split_copies for g in rows),
            "bytes_copied": sum(g.bytes_copied for g in rows),
            "weight_shape": list(g0.weight.shape),
            "bias": g0.bias is not None,
            "split": list(g0.split),
            "dtype": str(g0.weight.dtype),
            "stride": list(g0.weight.stride()),
            "need_contiguous_on_split": g0.need_contiguous,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "fuse_qkv": self.fuse_qkv,
            "fuse_gate_up": self.fuse_gate_up,
            "shapes": None if self.shapes is None else list(self.shapes),
            "n_layers": self.n_layers,
            "installed_groups": self.installed,
            "skipped": list(self.skipped),
            "qkv": self._agg("qkv"),
            "gate_up": self._agg("gate_up"),
        }


# --------------------------------------------------------------------------- #
# 模型遍历
# --------------------------------------------------------------------------- #

def iter_layers(model: Any) -> list[Any]:
    """取出 decoder 层列表。支持 `model.model.layers` 与 `model.layers`。

    找不到就返回空列表 ⇒ 融合清单里 `installed_groups=0`，
    而不是抛异常——**"这个模型没有可融合的投影"是一种结果，不是错误**。
    """
    inner = getattr(model, "inner", model)          # 测试用的记录器/包装器
    for attr in ("model", None):
        obj = getattr(inner, attr, inner) if attr else inner
        layers = getattr(obj, "layers", None)
        if layers is not None:
            return list(layers)
    return []


def _find_proj(layer: Any, names: Sequence[str]) -> list[Any] | None:
    """按顺序在一层里找投影模块。任一缺失 ⇒ `None`（不部分融合）。"""
    for owner_attr in ("self_attn", "attention", "mlp"):
        owner = getattr(layer, owner_attr, None)
        if owner is not None and all(hasattr(owner, n) for n in names):
            return [getattr(owner, n) for n in names]
    return None


# --------------------------------------------------------------------------- #
# 安装 / 卸载
# --------------------------------------------------------------------------- #

def _concat(torch: Any, modules: Sequence[Any]) -> tuple[Any, Any, str]:
    """沿**输出维**拼接权重与 bias。返回 `(weight, bias, 说明)`。

    bias 必须**同进同出**：一部分有、一部分没有时不能融合——拼出来的 bias
    长度对不上，而"给缺的那部分补零"是**静默改变语义**（原路径没有 bias，
    补零等于加了一个原来不存在的加法）。那种情况如实跳过。
    """
    weights = [m.weight for m in modules]
    has_bias = [m.bias is not None for m in modules]
    if any(has_bias) and not all(has_bias):
        return None, None, "bias 不同进同出（部分投影有 bias、部分没有）⇒ 不融合"
    weight = torch.cat(weights, dim=0)
    bias = torch.cat([m.bias for m in modules], dim=0) if all(has_bias) else None
    return weight, bias, ""


def install_fusions(
    model: Any,
    *,
    fuse_qkv: bool = False,
    fuse_gate_up: bool = False,
    shapes: Iterable[int] | None = None,
) -> FuseManifest:
    """在 `model` 上安装融合。返回 :class:`FuseManifest`（含全部可核验事实）。

    ⚠️ **就地修改模型的 `forward`**（不改权重、不改 `state_dict`）。
    要撤销请用 :func:`uninstall_fusions`。
    """
    shapes_t = None if shapes is None else tuple(int(s) for s in shapes)
    layers = iter_layers(model)
    manifest = FuseManifest(fuse_qkv=bool(fuse_qkv), fuse_gate_up=bool(fuse_gate_up),
                            shapes=shapes_t, n_layers=len(layers))
    if not layers:
        manifest.skipped.append({"reason": "找不到 decoder 层列表（model.model.layers）"})
        return manifest

    torch = _torch()
    import torch.nn.functional as F  # noqa: PLC0415

    specs: list[tuple[str, tuple[str, ...], bool]] = []
    if fuse_qkv:
        specs.append(("qkv", ("q_proj", "k_proj", "v_proj"), True))
    if fuse_gate_up:
        specs.append(("gate_up", ("gate_proj", "up_proj"), False))

    for name, proj_names, need_contig in specs:
        for idx, layer in enumerate(layers):
            mods = _find_proj(layer, proj_names)
            if mods is None:
                manifest.skipped.append(
                    {"fusion": name, "layer": idx,
                     "reason": f"层里找不到 {list(proj_names)}"})
                continue
            weight, bias, why = _concat(torch, mods)
            if weight is None:
                manifest.skipped.append({"fusion": name, "layer": idx, "reason": why})
                continue
            split = tuple(int(m.weight.shape[0]) for m in mods)
            group = _Group(name=name, modules=list(mods), weight=weight, bias=bias,
                           split=split, shapes=shapes_t, need_contiguous=need_contig)
            _patch_group(F, group)
            manifest.groups.append(group)
    return manifest


def _torch() -> Any:
    import torch  # noqa: PLC0415
    return torch


def _patch_group(F: Any, g: _Group) -> None:
    """把一组投影的 `forward` 换成"融合一次 + 切片复用"。

    实例属性赋值 ⇒ 调用时**不绑定 self**（`nn.Module.__call__` 取 `self.forward`
    得到的就是这个普通函数），因此闭包签名直接是 `(x)`。
    """
    # 保存**绑定**的原 forward：回退路径必须走原样实现。
    # 取 `m.forward` 拿到的就是绑定方法（此刻还没被覆盖），可直接调用；
    # 卸载时把它删掉即可恢复类上的实现。
    for m in g.modules:
        m._fusion_orig_forward = m.forward

    def _make_first(index: int):
        take = g.split[index]

        def _first(x: Any) -> Any:
            if not g.enabled_for(x):
                g.n_skipped_by_shape += 1
                g.clear()
                return _orig(g.modules[index], x)
            y = F.linear(x, g.weight, g.bias)
            g.n_fused_calls += 1
            g.n_component_calls += 1
            g.pending_out = y
            g.pending_key = (x.data_ptr(), tuple(x.shape))
            g.pending_remaining = len(g.split) - 1
            out = y[..., :take]
            if g.need_contiguous:
                # `.view()` 需要连续张量；窄片不是连续的（行 stride 仍是融合宽度）。
                # 这一次拷贝**如实记字节**，它是融合的固有代价。
                g.n_split_copies += 1
                g.bytes_copied += int(out.numel()) * int(out.element_size())
                out = out.contiguous()
            return out

        return _first

    def _make_rest(index: int, expected_remaining: int):
        lo = sum(g.split[:index])
        hi = lo + g.split[index]

        def _rest(x: Any) -> Any:
            key = (x.data_ptr(), tuple(x.shape))
            if (g.pending_out is not None and g.pending_remaining == expected_remaining
                    and g.pending_key == key):
                y = g.pending_out
                g.pending_remaining -= 1
                g.n_reuse += 1
                g.n_component_calls += 1
                if g.pending_remaining == 0:
                    g.clear()
                out = y[..., lo:hi]
                if g.need_contiguous:
                    g.n_split_copies += 1
                    g.bytes_copied += int(out.numel()) * int(out.element_size())
                    out = out.contiguous()
                return out
            # 顺序/输入对不上 ⇒ **回退**（只损失优化，绝不改变数值）
            g.n_fallback_calls += 1
            g.n_component_calls += 1
            g.clear()
            return _orig(g.modules[index], x)

        return _rest

    def _orig(module: Any, x: Any) -> Any:
        """原路径。**权重与 bias 从未被改动** ⇒ 这就是安装前的语义。"""
        saved = module.__dict__.get("_fusion_orig_forward")
        if saved is not None:
            return saved(x)
        return F.linear(x, module.weight, module.bias)

    for i, m in enumerate(g.modules):
        m.forward = _make_first(i) if i == 0 else _make_rest(i, len(g.split) - i)


def uninstall_fusions(manifest: FuseManifest) -> int:
    """撤销安装，返回被恢复的模块数。权重从未被改动 ⇒ `state_dict` 前后一致。"""
    n = 0
    for g in manifest.groups:
        for m in g.modules:
            saved = m.__dict__.pop("_fusion_orig_forward", None)
            if saved is not None:
                # 删掉实例属性 ⇒ 重新走类上的（原）forward。
                m.__dict__.pop("forward", None)
                n += 1
        g.clear()
    manifest.groups = []
    return n
