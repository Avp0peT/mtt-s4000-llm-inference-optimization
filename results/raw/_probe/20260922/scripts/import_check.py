"""逐个导入并打印完整回溯——ImportError 的原因不能靠猜。"""
import traceback

ORDER = [
    "numpy",
    "regex",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "torch",
    "accelerate",
    "transformers",
]

for name in ORDER:
    try:
        mod = __import__(name)
        print(f"OK   {name:<18} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name:<18} {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("-" * 60)
