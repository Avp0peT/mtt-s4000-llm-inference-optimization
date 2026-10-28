"""列出与模型推理相关的已装包版本（含哪些来自 base）。

关键要看清的是 **numpy**：torch 2.2.0 是按 numpy 1.x 构建的，若独立环境里装了
numpy 2.x，它会遮蔽 base 的 numpy 并把 torch 弄坏——而症状是 import torch 时报
"compiled with NumPy 1.x cannot run in NumPy 2.x"，看起来却像 torch 坏了。
"""
import importlib
import sys

WATCH = [
    "numpy",
    "packaging",
    "filelock",
    "tqdm",
    "regex",
    "requests",
    "yaml",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "accelerate",
    "transformers",
    "sentencepiece",
    "einops",
    "PIL",
    "protobuf",
    "google.protobuf",
]

print("executable:", sys.executable)
print(f"{'package':<18} {'version':<16} origin")
print("-" * 62)
for name in WATCH:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<18} {'-- (未安装)':<16} {type(exc).__name__}")
        continue
    version = getattr(mod, "__version__", "?")
    origin = getattr(mod, "__file__", "?") or "?"
    tag = "base" if "miniconda3" in origin else ("venv" if "qwen310" in origin else "other")
    print(f"{name:<18} {str(version):<16} {tag}: {origin}")
