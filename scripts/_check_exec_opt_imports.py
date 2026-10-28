"""远端自检：本轮执行优化新增的模块能否在**真实 MUSA 环境**下导入。

存在理由：`exec_opt_batch.py preflight` 只核到环境与 revision，不核"我推上去的代码
本身有没有语法/依赖问题"。这个文件由 `push_files.sh` 推上去、远端直接 `python3` 跑，
**不经过任何引号嵌套**（手工多层引号踩过两次 ⇒ 一律走文件）。

用法（远端）：cd <root> && python3 scripts/_check_exec_opt_imports.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# 被测模块：入口链上的三个（run_exec_opt_matrix 会 import 它们）
MODULES = [
    "benchmark.exec_opt_plan",
    "benchmark.linear_fusion",
    "benchmark.real_backend",
]


def main() -> int:
    # ⚠️ 两条路径**都要**：
    #    ① 仓库根 ⇒ `import benchmark.xxx` 成立；
    #    ② `benchmark/` 本身 ⇒ `real_backend.py` 里以**裸名**导入的 `arm_factory` 成立
    #       （只加仓库根会 ModuleNotFoundError: arm_factory；
    #        只加 benchmark 会 ModuleNotFoundError: benchmark。两个都踩过。）
    #    照抄 `run_exec_opt_matrix.py` 的既有约定，不另立一套。
    root = Path(__file__).resolve().parent.parent
    for p in (str(root), str(root / "benchmark"), str(root / "src"),
              str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    bad = 0
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            print(f"IMPORT_OK {name} <- {getattr(m, '__file__', '?')}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"IMPORT_FAIL {name} {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # 关键符号自检：开关真的在**承载它们的那一层**上（不只是"文件能 import"）。
    # ⚠️ 判据来自现读：#783 `class TransformerDriver` 的 `__init__`（781 行起）
    #    持有这 6 个开关；`RealBackend` 在 1830 行，**不含**这些形参。
    #    最初写成查 `RealBackend.__init__` ⇒ 恒 False（假失败）。
    try:
        from benchmark import real_backend as rb
        sig_ok = all(
            k in rb.TransformerDriver.__init__.__code__.co_varnames
            for k in ("logits_last_only", "sample_needed_only",
                      "retire_empty_cache", "fuse_qkv", "fuse_gate_up",
                      "linear_fuse_shapes")
        )
        print(f"SWITCHES_PRESENT {sig_ok} (on TransformerDriver.__init__)")
        bad += 0 if sig_ok else 1

        # ★ 否定判据：换一个**不该有**这些形参的类，必须为 False。
        #   没有这一条，"恒真"与"真的找到了"无法区分（恒定值假通过）。
        neg = any(
            k in rb.RealBackend.__init__.__code__.co_varnames
            for k in ("logits_last_only", "sample_needed_only")
        )
        print(f"NEGATIVE_CONTROL RealBackend_has_switches={neg} (期望 False)")
        if neg:
            print("NEGATIVE_CONTROL_UNEXPECTED ⇒ 判据定位错了（开关不止在一处？）")
            bad += 1
    except Exception as exc:  # noqa: BLE001
        print(f"SWITCHES_CHECK_FAIL {type(exc).__name__}: {exc}")
        bad += 1

    print(f"---- bad={bad} ----")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
