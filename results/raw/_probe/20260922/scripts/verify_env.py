"""打印当前解释器的 torch / torch_musa 归属与 MUSA 可用性。

用途：在**独立环境**里确认 torch 没有被 venv 内新装的版本遮蔽——
`--system-site-packages` 的 venv 一旦装上 torch 依赖，很容易悄悄把 base 的
torch_musa 版本顶掉，而那个组合才是能驱动 MTT 卡的组合。
"""
import os
import sys

print("executable :", sys.executable)
print("python     :", sys.version.split()[0])
print("prefix     :", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("in_venv    :", sys.prefix != sys.base_prefix)

import torch  # noqa: E402

print("torch      :", torch.__version__)
print("torch file :", torch.__file__)
print("cuda field :", getattr(torch.version, "cuda", None))

try:
    import torch_musa  # noqa: E402

    print("torch_musa :", getattr(torch_musa, "__version__", "?"))
    print("musa file  :", torch_musa.__file__)
except Exception as exc:  # noqa: BLE001
    print("torch_musa : IMPORT FAIL", type(exc).__name__, exc)

try:
    print("musa avail :", torch.musa.is_available())
    print("musa count :", torch.musa.device_count())
    print("musa name  :", torch.musa.get_device_name(0))
except Exception as exc:  # noqa: BLE001
    print("musa query : FAIL", type(exc).__name__, exc)

try:
    free, total = torch.musa.mem_get_info()
    print(f"musa mem   : free {free / 2**20:.0f} MiB / total {total / 2**20:.0f} MiB")
except Exception as exc:  # noqa: BLE001
    print("musa mem   : FAIL", type(exc).__name__, exc)

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "<unset>"))
