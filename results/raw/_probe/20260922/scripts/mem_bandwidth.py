"""设备内存带宽实测（MTT S4000 / MUSA）。

为什么值得单独测：本轮实测到 0.5B 与 7B 的每步耗时只差 22%（47.3ms → 57.6ms），
而参数量差 15.4 倍。从这两点可以拟合出"固定开销 + 访存"两段模型，但那是**推算**。
推算要先知道这块卡的真实带宽才有意义——否则"第 15.4 倍的参数只多花 10ms"到底
说明"带宽很高"还是"根本没到带宽上限"，无法区分。

做法：大张量 copy_（读 + 写各一遍），取多次平均。这是**上界口径**（纯连续搬运，
无注意力/无激活），所以真实算子的有效带宽只会更低，不会更高。
"""
from __future__ import annotations

import json
import time

import torch
import torch_musa  # noqa: F401  必须显式导入：torch.musa 是 torch_musa 注册进来的属性，
#                   不导入它 `torch.musa` 根本不存在（AttributeError: module 'torch' has no
#                   attribute 'musa'）。别指望"别的库顺手导入过"——那依赖导入顺序，
#                   顺序一变就崩，而且报错信息指向 torch 而不是指向缺失的导入。


def bench_readwrite(n_bytes: int, reps: int = 20) -> dict:
    """读 n_bytes + 写 n_bytes，返回实测 GB/s。"""
    n_el = n_bytes // 2                       # fp16
    a = torch.empty(n_el, dtype=torch.float16, device="musa").normal_()
    b = torch.empty_like(a)
    torch.musa.synchronize()

    for _ in range(3):                        # 预热：首次 kernel 与内存池开销不计入
        b.copy_(a)
    torch.musa.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        b.copy_(a)
    torch.musa.synchronize()
    dt = time.perf_counter() - t0

    moved = 2 * n_bytes                       # 读一遍 + 写一遍
    out = {
        "tensor_mib": round(n_el * 2 / 2**20, 1),
        "reps": reps,
        "per_copy_ms": round(dt / reps * 1000, 4),
        "effective_gbps": round(moved / (dt / reps) / 1e9, 1),
    }
    del a, b
    torch.musa.empty_cache()
    return out


def main() -> int:
    res: dict = {"device": torch.musa.get_device_name(0)}
    free, total = torch.musa.mem_get_info()
    res["mem_total_mib"] = round(total / 2**20, 1)

    for mib in (256, 2048):
        try:
            res[f"copy_{mib}mib"] = bench_readwrite(mib * 2**20)
        except Exception as exc:  # noqa: BLE001 - 失败要留痕，不能跳过当没事
            res[f"copy_{mib}mib"] = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
