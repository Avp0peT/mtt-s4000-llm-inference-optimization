"""对候选版本做**安装前**核对：Requires-Python 与关键依赖约束。

为什么不在本地凭印象定版本：本项目已经吃过"看起来对、实际错"的亏。
这一步把"能否在 py3.10 + torch 2.2.0 上装"从印象变成可核对的事实，
并且**在装之前**暴露会顶掉 torch / numpy 的依赖。
"""
import json
import re
import sys
import urllib.request

CANDIDATES = [
    ("transformers", "4.46.3"),
    ("transformers", "4.44.2"),
    ("tokenizers", "0.20.3"),
    ("safetensors", "0.4.5"),
    ("huggingface-hub", "0.26.2"),
    ("accelerate", "1.1.1"),
    ("regex", "2024.11.6"),
]

WATCH = ("python", "torch", "numpy", "tokenizers", "safetensors", "huggingface-hub")


def fetch(pkg: str, ver: str) -> dict:
    url = f"https://pypi.org/pypi/{pkg}/{ver}/json"
    with urllib.request.urlopen(url, timeout=25) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    print(f"本地目标：python 3.10.8, torch 2.2.0, numpy 1.26.4\n")
    for pkg, ver in CANDIDATES:
        try:
            info = fetch(pkg, ver)["info"]
        except Exception as exc:  # noqa: BLE001
            print(f"{pkg}=={ver}: 取元数据失败 {type(exc).__name__} {exc}")
            continue
        print(f"--- {pkg}=={ver} ---")
        print(f"    Requires-Python: {info.get('requires_python')}")
        reqs = info.get("requires_dist") or []
        interesting = [
            r for r in reqs
            if any(w in r.lower() for w in WATCH) and "extra ==" not in r.lower()
        ]
        if interesting:
            for r in sorted(interesting):
                print(f"    dep: {r}")
        else:
            print("    dep: (无 python/torch/numpy/相关约束)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
