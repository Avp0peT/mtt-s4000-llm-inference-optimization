#!/usr/bin/env python3
"""PHASE 1 第③步矩阵配置的预检（可在本机或远程跑）。

为什么需要单独一步
------------------
矩阵一旦开跑要花掉几十分钟远程时间。跑完才发现"某个配置加载不了"、
"两个臂其实差了三个字段"、"负载不是我以为的那条"，代价就全白付了。
本脚本在**开跑之前**把这些都查一遍，并且把查到的内容**打印出来**——
预检的输出本身就是"我打算跑什么"的记录。

查四件事
--------
1. 12 个配置都能被 ``load_config`` 加载（未知字段、类型错误在这里暴露）。
2. 每个配置的负载能构造出请求，且请求数与并发符合预期。
3. 臂之间的差异**仍然**只有设计的那一处（防止生成之后被人手工改坏）。
4. 打印将要被测量的实际条件（模型、精度、chunk、controller、repeats、
   平台声明），供人核对。

用法::

    python scripts/check_step3_configs.py                 # 默认 configs/mthreads/phase1_step3
    python scripts/check_step3_configs.py --dir <目录>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "benchmark"))
sys.path.insert(0, str(_ROOT / "src"))

from run_benchmark import load_config  # noqa: E402
from workloads import get_workload  # noqa: E402
from ascend_opt import build_requests  # noqa: E402

DEFAULT_DIR = _ROOT / "configs" / "mthreads" / "phase1_step3"

#: 臂之间预期且**仅**预期的差异（点分路径）。
ARM_EDGES: dict[tuple[str, str], str] = {
    ("static", "static_chunk"): "chunk.enabled",
    ("static_chunk", "adaptive"): "controller.enabled",
}
CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8)
ARMS: tuple[str, ...] = ("static", "static_chunk", "adaptive")


def _flatten(prefix: str, obj: object, out: dict[str, object]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
        return
    out[prefix] = obj


def _check_model_id(model: object, hf_home: str | None) -> list[str]:
    """校验 ``runtime.model`` 是一个**能命中缓存**的 repo id。

    为什么值得单独查：HF 按 **repo id**（``<org>/<name>``）拼缓存目录名
    （``models--<org>--<name>``）。写成 ``Qwen2.5-7B-Instruct``（缺 ``Qwen/``）
    时，本地那份 ``models--Qwen--Qwen2.5-7B-Instruct`` **完全匹配不上**，
    于是明明已缓存的模型会去走网络，报出来的却是
    ``RepositoryNotFoundError: 404``——看起来像"模型没下载"，
    实际是"名字少了一段"。这个错只会在真机上、且只在加载时才暴露。

    本机没有 ``HF_HOME`` 时只做格式检查（缺 ``/`` 一律拦下），
    在远程跑时会额外核对缓存目录是否真的存在。
    """
    probs: list[str] = []
    if not model:
        return ["runtime.model 未配置"]
    name = str(model)
    if "/" not in name:
        probs.append(
            f"runtime.model={name!r} 缺少 `<org>/` 前缀；"
            f"HF 按 repo id 找缓存，写成裸模型名会命中不了本地缓存并去走网络"
        )
        return probs  # 名字不合法就不必再查缓存了
    if hf_home:
        cached = Path(hf_home) / "hub" / ("models--" + name.replace("/", "--"))
        if not cached.is_dir():
            probs.append(f"HF 缓存里没有 {cached}（模型未下载，或 model 写错了）")
    return probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="预检第③步矩阵配置")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="配置目录")
    args = ap.parse_args(argv)
    cfg_dir = Path(args.dir)

    problems: list[str] = []
    loaded: dict[str, object] = {}

    # ---- 1 & 2：能加载、能构造请求 ----
    for arm in ARMS:
        for c in CONCURRENCIES:
            path = cfg_dir / f"{arm}_c{c}.yaml"
            if not path.exists():
                problems.append(f"缺少配置：{path}")
                continue
            try:
                cfg = load_config(path)
            except Exception as exc:  # 加载失败必须显形，不能跳过
                problems.append(f"{path.name} 加载失败：{type(exc).__name__}: {exc}")
                continue
            loaded[f"{arm}_c{c}"] = cfg
            try:
                spec = get_workload(cfg.workload_name, **cfg.workload_overrides)
                reqs = build_requests(spec)
            except Exception as exc:
                problems.append(f"{path.name} 负载构造失败：{type(exc).__name__}: {exc}")
                continue
            if spec.concurrency != c:
                problems.append(
                    f"{path.name} 并发不符：配置名说 c={c}，负载是 {spec.concurrency}"
                )
            if not reqs:
                problems.append(f"{path.name} 构造出 0 个请求")
            # cache 必须关：真实后端显式拒绝 cache.enabled=true
            if cfg.cache.enabled:
                problems.append(f"{path.name} cache.enabled=true —— 真实后端不支持前缀缓存")
            if cfg.run.repeats < 3:
                problems.append(f"{path.name} repeats={cfg.run.repeats} < 3，不能出结论")

    if not loaded:
        for p in problems:
            print(f"预检失败：{p}", file=sys.stderr)
        return 1

    # ---- 1b：模型 id 必须是能命中缓存的 repo id ----
    # HF_HOME 只在远程被 activate 脚本设置；本机没设时只做格式检查。
    import os

    hf_home = os.environ.get("HF_HOME")
    models = {str(c.runtime.get("model")) for c in loaded.values()}
    for m in sorted(models):
        for p in _check_model_id(m, hf_home):
            problems.append(p)

    # ---- 3：臂间仍只差设计的那一处 ----
    flat: dict[str, dict[str, object]] = {}
    for name, cfg in loaded.items():
        d: dict[str, object] = {}
        # 只看**会被策略层消费或影响测量**的字段；run.notes 是标签，天然不同。
        #
        # 这些是 dataclass 而非 dict：必须先经 describe() 摊平。直接把它们当叶子
        # 会得到一个对象，两个不同的对象永远不等于彼此 —— 于是每个臂都"差一处"，
        # 守卫退化成恒真的报警器（本脚本第一次跑就踩了这个，见输出里的
        # "实际还差 ['chunk']"）。describe() 是项目契约里 dataclass 的标准出口。
        for attr in ("chunk", "controller", "cache", "scheduler"):
            obj = getattr(cfg, attr, None)
            if obj is None:
                continue
            if hasattr(obj, "describe"):
                obj = obj.describe()
            elif isinstance(obj, dict):
                pass
            else:
                obj = {"_repr": repr(obj)}
            _flatten(attr, obj, d)
        flat[name] = d

    for (a, b), expect in ARM_EDGES.items():
        ka, kb = f"{a}_c1", f"{b}_c1"
        if ka not in flat or kb not in flat:
            continue
        diffs = sorted(
            k for k in set(flat[ka]) | set(flat[kb]) if flat[ka].get(k) != flat[kb].get(k)
        )
        unexpected = [d for d in diffs if d != expect]
        if unexpected:
            problems.append(f"{a} vs {b}：预期只差 {expect}，实际还差 {unexpected}")

    # ---- 4：打印将要被测量的条件 ----
    print("=" * 78)
    print("即将测量的条件（每一格）")
    print("=" * 78)
    print(f"{'格':<22}{'chunk':<7}{'ctl':<5}{'chunk_sz':<10}{'repeats':<9}{'模型':<26}{'并发':<6}{'请求数'}")
    for arm in ARMS:
        for c in CONCURRENCIES:
            name = f"{arm}_c{c}"
            if name not in loaded:
                continue
            cfg = loaded[name]
            spec = get_workload(cfg.workload_name, **cfg.workload_overrides)
            print(
                f"{name:<22}"
                f"{str(cfg.chunk.enabled):<7}"
                f"{str(cfg.controller.enabled):<5}"
                f"{cfg.chunk.chunk_size:<10}"
                f"{cfg.run.repeats:<9}"
                f"{str(cfg.runtime.get('model')):<26}"
                f"{spec.concurrency:<6}"
                f"{spec.num_requests}"
            )
    print()
    first = loaded["static_c1"]
    spec = get_workload(first.workload_name, **first.workload_overrides)
    print(f"负载：{spec.name}")
    print(f"  prompt_len = {spec.prompt_len_min}（固定单点）"
          f"  gen_len = {spec.gen_len_min}（固定单点）")
    print(f"  arrival = {spec.arrival}   concurrency = {spec.concurrency}"
          f"   num_requests = {spec.num_requests}")
    print(f"  shared_prefix_len = {spec.shared_prefix_len}  prefix_groups = {spec.prefix_groups}")
    print(f"平台声明：execution_environment = {first.runtime.get('execution_environment')}"
          f"   hardware_platform = {first.runtime.get('hardware_platform')}")
    print(f"精度：{first.runtime.get('precision')}   warmup_s = {first.run.warmup_s}"
          f"   control_interval_steps = {first.run.control_interval_steps}")

    print()
    if problems:
        for p in problems:
            print(f"预检失败：{p}", file=sys.stderr)
        return 1
    print(f"预检通过：{len(loaded)} 个配置可加载，臂间差异符合设计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
