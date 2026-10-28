"""CPU 微型模型的测试夹具（S01/S02/S03 共用）。

为什么抽出来：三个会话的用例都要"一个 CPU 上能跑的确定性微型 Qwen2 + 一个把
真机驱动跑在 CPU 上的子类"。各写一份的后果不是重复，而是**三份夹具迟早不一致**
（本项目已在别处吃过"两条路径各写一份"的亏）。

⚠️ `CpuDriver.hardware_backed = False`：它**没有在硬件上执行**，
因此它的任何产物都不得被当成性能证据（`RealBackend` 会拒绝它）。
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

TORCH_MISSING_REASON = ""
try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    TORCH_MISSING_REASON = f"{type(exc).__name__}: {exc}"

try:  # pragma: no cover
    from transformers import Qwen2Config, Qwen2ForCausalLM
except Exception as exc:  # pragma: no cover
    Qwen2Config = None  # type: ignore[assignment]
    Qwen2ForCausalLM = None  # type: ignore[assignment]
    if not TORCH_MISSING_REASON:
        TORCH_MISSING_REASON = f"{type(exc).__name__}: {exc}"


class RecordingModel(nn.Module if torch is not None else object):
    """把内层模型每次前向的 logits 记下来（只存在于测试侧）。

    ⚠️ 不为了让生产代码"可测"而在驱动里留调试口 —— 调试口一旦存在，
    就有人会在正式运行里打开它。
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner
        self.logits: list[Any] = []
        self.kwargs: list[dict[str, Any]] = []
        self.calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.inner(*args, **kwargs)
        self.logits.append(out.logits.detach().clone())
        self.kwargs.append(dict(kwargs))
        self.calls += 1
        return out


def make_model(seed: int = 0, vocab: int = 64, layers: int = 2) -> Any:
    """确定性微型 Qwen2（随机权重即可 —— 这里验的是**计算语义等价**）。"""
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def make_driver(model: Any, **flags: Any) -> Any:
    """构造一个跑在 CPU 上的真机驱动（`hardware_backed=False`）。

    时钟用**递增计数器**：`pack_ms` 是"两次读钟之差"，固定时钟下它恒为 0，
    于是"这一步发生了补位搬运"这类断言会永远失败，而失败原因看起来像
    "补位没做"。
    """
    from real_backend import SyntheticPromptSource, TransformerDriver

    class _CpuDriver(TransformerDriver):
        hardware_backed = False

        def _device_available(self, torch_mod: Any) -> bool:
            return True

    tick = itertools.count()
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 64))
    return _CpuDriver(
        model_name="local-tiny",
        device="cpu",
        precision="fp32",
        max_seq_len=512,
        model=model,
        prompt_source=SyntheticPromptSource(vocab_size=vocab),
        clock=lambda: next(tick) * 1e-6,
        **flags,
    )


def req(name: str, prompt_len: int, gen_len: int = 4) -> Any:
    from ascend_opt import Request
    return Request(req_id=name, prompt_len=prompt_len, gen_len=gen_len)


# --------------------------------------------------------------------------- #
# "本机必测"纪律：缺输入要么红、要么**显式**标为外部依赖
# --------------------------------------------------------------------------- #

class ExternalArtifactMissing(unittest.SkipTest):
    """**外部依赖缺失**：产物由远端跑出、尚未回拉 ⇒ 无法核验（**允许** skip）。

    用作 `skipTest` 的**理由类型**：它把"因为远端产物没回拉"与"因为本机文件缺失"
    分开。只有**前者**才是合法的 skip 理由。
    """


def require_local_file(path: Path, *, what: str) -> None:
    """本机**必测**输入不存在 ⇒ **直接失败**（用 `assert` 而非 `skipTest`）。

    ★★ 为什么不能 `skipTest`（用户 2026-10-03 点名）：
      `skipTest` 让"**没跑**"在测试报告里显示成 **skip**，而 skip 在多数 CI
      摘要里与"通过"难以区分（`pytest -q` 的 `s` 与 `.` 只差一个字符）。
      本仓已吃过这个亏：`-r5` 不存在时 6 条端到端反例**静默跳过**，
      而当时报告看起来是"全绿"。
      ⇒ 纪律：**本机就在仓库里、且应当存在的输入**缺失 ⇒ 是**环境坏**，
        不是"跳过"。必须红。

    ⚠️ 与 `ExternalArtifactMissing` 的分工：远端产物（`results/raw/**`）**可能**
      尚未回拉，那是**外部依赖**，允许显式 skip；而**配置 / 脚本 / 已提交的
      离线产物**都在仓库里，缺了就是仓库坏。
    """
    if not Path(path).is_file():
        raise AssertionError(
            f"本机必测输入缺失：{what} = {path}\n"
            f"  ⇒ 该路径**应当在仓库内**（不是外部依赖）⇒ 以**失败**终止，"
            f"**不**转成 skip（`skipTest` 会让『没跑』看起来像『通过』）。")

