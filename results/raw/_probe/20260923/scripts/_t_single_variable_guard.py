"""正向对照：证明 gen_phase1_step3_configs 的单变量自检**真的会失败**。

一个不会失败的守卫等于没有守卫。这里故意在 adaptive 臂上多改一个字段
（``chunk.max_num_batched_tokens``），使 ``static_chunk -> adaptive`` 的差异
从 1 处变成 2 处。自检必须以非零退出码报出来。

跑法：python scripts/_t_single_variable_guard.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_phase1_step3_configs as G  # noqa: E402


def run_check() -> int:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        return G.main(["--check"])


def main() -> int:
    failures: list[str] = []

    # ---- 1) 未注入时应当通过 ----
    G.ARMS["adaptive"] = {
        "chunk.enabled": True,
        "controller.enabled": True,
    }
    baseline_rc = run_check()
    if baseline_rc != 0:
        failures.append(f"未注入时自检应当通过，实际退出码 {baseline_rc}")
    print(f"[1] 未注入        -> rc={baseline_rc}（期望 0）")

    # ---- 2) 注入第二个变量后必须失败 ----
    G.ARMS["adaptive"] = {
        "chunk.enabled": True,
        "controller.enabled": True,
        # 多余的第二个变量：与 static_chunk 之间多了一处差异
        "chunk.max_num_batched_tokens": 4096,
    }
    injected_rc = run_check()
    if injected_rc == 0:
        failures.append("注入第二个变量后自检仍然通过 —— 守卫不会失败，等于没有守卫")
    print(f"[2] 注入第二变量  -> rc={injected_rc}（期望非 0）")

    # ---- 3) 注入并发轴内的第二处差异，也必须失败 ----
    G.ARMS["adaptive"] = {
        "chunk.enabled": True,
        "controller.enabled": True,
    }
    original_common = dict(G._COMMON)
    # 让 num_requests 随并发变化 —— 于是同一臂的不同并发点之间差了并发**和**
    # 请求数两处，正是"两个变量一起动"的那种错误。
    real_build = G.build

    def patched_build() -> dict:
        out = real_build()
        for c in G.CONCURRENCIES:
            out[f"adaptive_c{c}"]["workload"]["num_requests"] = c * 2
        return out

    G.build = patched_build  # type: ignore[assignment]
    axis_rc = run_check()
    G.build = real_build  # type: ignore[assignment]
    G._COMMON = original_common
    if axis_rc == 0:
        failures.append("并发轴内多一处差异时自检仍然通过 —— 守卫漏判")
    print(f"[3] 轴内多变量    -> rc={axis_rc}（期望非 0）")

    print()
    if failures:
        for f in failures:
            print(f"对照失败：{f}", file=sys.stderr)
        return 1
    print("正向对照通过：守卫在两种注入下都报错，在未注入时通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
