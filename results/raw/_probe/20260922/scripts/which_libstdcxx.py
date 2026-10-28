"""打印进程**实际载入**的 libstdc++，用来确证选库机制。

用法：
    python which_libstdcxx.py torch        # 只 import torch
    python which_libstdcxx.py tok_torch    # 先 import tokenizers 再 import torch

不停留在"哪个更早"，直接读 /proc/self/maps 看加载器最终选了哪一个。
两个库同名（soname 都是 libstdc++.so.6），所以**先载入的那个会被复用**——
这正是"导入顺序能改变成败"的机制。
"""
import sys
import traceback

mode = sys.argv[1] if len(sys.argv) > 1 else "torch"


def show(tag: str) -> None:
    found = []
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "libstdc++" in line:
                path = line.rstrip().split()[-1]
                if path not in found:
                    found.append(path)
    print(f"  [{tag}] libstdc++ -> {found or '(未载入)'}")


print(f"mode = {mode}")
try:
    if mode == "tok_torch":
        import tokenizers  # noqa: F401

        show("after tokenizers")
    import torch  # noqa: F401

    show("after torch")
    print("  RESULT ok, torch", torch.__version__)
except Exception as exc:  # noqa: BLE001
    show("at failure")
    print(f"  RESULT FAIL {type(exc).__name__}: {str(exc)[:120]}")
    traceback.print_exc()
    sys.exit(1)
