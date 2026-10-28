"""验证"导入顺序是否影响 torch 能否加载"。

现象：同一个解释器，`import torch` 先跑就成功；先 import numpy / tokenizers 再 import
torch 就报
    libstdc++.so.6: version GLIBCXX_3.4.30 not found (required by libtorch_python.so)
典型成因：前面的库先把一个**较旧的 libstdc++** 载进进程，之后 dlopen libtorch_python.so
就按已载入的那份解析符号。这类问题必须用受控实验定位，不能靠猜。
"""
import subprocess
import sys

CASES = [
    ("torch 单独", "import torch; print('ok', torch.__version__)"),
    ("numpy -> torch", "import numpy, torch; print('ok')"),
    ("tokenizers -> torch", "import tokenizers, torch; print('ok')"),
    ("safetensors -> torch", "import safetensors, torch; print('ok')"),
    ("huggingface_hub -> torch", "import huggingface_hub, torch; print('ok')"),
    ("torch -> numpy", "import torch, numpy; print('ok')"),
]

print("解释器:", sys.executable)
print()
for label, code in CASES:
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    status = "OK " if proc.returncode == 0 else "FAIL"
    detail = ""
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        detail = lines[-1][:150] if lines else "(无 stderr)"
    print(f"  {status} {label:<24} {detail}")
