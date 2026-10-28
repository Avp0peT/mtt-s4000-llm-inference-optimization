#!/usr/bin/env python3
"""生成 **S6 服务收益档** 的三臂配置集（`configs/mthreads/phase1_s6/`）。

依据
----
`EXPERIMENTS.md` §S6「服务收益档 · 预注册（先冻结规则，再看数据）」。
本节写于 S6 产生任何数据**之前**；本脚本是它的机械实现。

负载 = §7d 第 5 节机械选出的 **`concurrency=2` / `max_num_seqs=4`**
（底稿 `configs/mthreads/phase1_d1_scan/fixed_cost_scan_c2_m4.yaml`）。

唯一变量（三臂只差两个开关，**不得再有第三处差异**）
-----------------------------------------------------
| 臂 | `controller.enabled` | `controller.kind` | 走哪条路径 |
| --- | --- | --- | --- |
| `static`     | `false` | （任意，不读） | run 路径**整块跳过**控制器 |
| `threshold`  | `true`  | `threshold`    | `AdaptiveController.observe()` → 规则 A/B/C/D |
| `cost`       | `true`  | `cost`         | `observe()` **且** `apply_arm(kind="cost")` |

⚠️ **`static` 臂必须写 `controller.enabled: false`**（不是 `kind: static`：
`ControllerConfig.__post_init__` 只接受 `threshold`/`cost`/`slo_weave`，
写 `static` 会在加载期抛 `ValueError` —— 这是刻意的，防止"以为在跑静态臂"）。

⚠️ **`threshold`/`cost` 臂必须 `controller.enabled: true`**
（`kind != "threshold"` 且 `enabled == False` 会在加载期抛错，
因为那会让 run 路径整块跳过控制器、该臂根本不被执行、轨迹为空）。

⚠️ 三臂的可比性由「**同格三臂只差这两个键**」保证。本生成器在 `verify()` 里
**逐文件断言**：拿 `static` 当基准，另两臂除这两处外**逐字节相同**。

用法::

    python scripts/gen_phase1_s6_configs.py            # 写盘
    python scripts/gen_phase1_s6_configs.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 底稿 = §7d 机械选出的 `c2_m4`（低并发扫描批的胜出格）。
REFERENCE = _ROOT / "configs" / "mthreads" / "phase1_d1_scan" / "fixed_cost_scan_c2_m4.yaml"
OUT_DIR = _ROOT / "configs" / "mthreads" / "phase1_s6"

#: 三臂定义（§S6 第 2 节）。**只差这两个开关**。
#: 元组常量而不是散在代码里：`tests/` 与本脚本的校验共用同一处定义。
#:
#: ⚠️ **`static` 臂的 `kind` 必须是 `threshold`，不能留底稿的 `cost`**：
#: `ControllerConfig.__post_init__` 规定 `kind != "threshold"` 时 `enabled` 必须为 True
#: ⇒ `kind: cost` + `enabled: false` 会在**加载期**抛 ValueError。
#: 这是**实测踩到的**：文本级校验放过了它（只数 `enabled: true` 的出现次数），
#: 真加载器一构造就现形。教训：**只做文本校验不够，必须过一遍真加载器**。
#: `kind: threshold` + `enabled: false` 是合法的（规则只对非 threshold 生效），
#: 且与"静态臂不读 kind"的语义一致。
ARMS: tuple[tuple[str, bool, str, str], ...] = (
    # (文件主干, controller.enabled, controller.kind, notes 里的臂标签)
    ("static", False, "threshold", "static（控制器整块关闭）"),
    ("threshold", True, "threshold", "threshold（阈值自适应）"),
    ("cost", True, "cost", "cost（成本模型自适应）"),
)

#: 底稿里必须**逐字**出现的锚点。任何一处不是恰好 1 次 ⇒ 底稿结构变了，
#: 本脚本的替换是错的（静默替换错位置的后果是"三臂其实只跑了一条臂"）。
_ANCHOR_CTRL_ENABLED = "  enabled: true\n"
_ANCHOR_CTRL_KIND = "  kind: cost\n"
_ANCHOR_NOTES_TAIL = "（两处已同步）"


def cell_name(arm: str) -> str:
    """文件名主干。臂名进名字 ⇒ 产物与配置可机械对应。"""
    return f"service_benefit_c2_m4_{arm}"


def _count_ctrl_enabled(text: str) -> int:
    """数 `controller.enabled` 那一个 `  enabled: true`。

    底稿里另有 `chunk.enabled: true`，两者缩进相同（都是两空格）——
    所以不能按 `  enabled: true` 全数，必须**限定在 controller 段内**。
    """
    lines = text.splitlines(keepends=True)
    in_ctrl = False
    n = 0
    for ln in lines:
        if ln.startswith("controller:"):
            in_ctrl = True
            continue
        if in_ctrl and ln and not ln.startswith((" ", "#")):
            in_ctrl = False      # 到下一个顶层键
        if in_ctrl and ln == _ANCHOR_CTRL_ENABLED:
            n += 1
    return n


def _replace_in_ctrl(text: str, old: str, new: str) -> str:
    """在 controller 段内替换**第一处** `old` ⇒ `new`（不碰其它段）。"""
    lines = text.splitlines(keepends=True)
    in_ctrl = False
    out: list[str] = []
    done = False
    for ln in lines:
        if ln.startswith("controller:"):
            in_ctrl = True
            out.append(ln)
            continue
        if in_ctrl and ln and not ln.startswith((" ", "#")):
            in_ctrl = False
        if in_ctrl and not done and ln == old:
            out.append(new)
            done = True
            continue
        out.append(ln)
    if not done:
        raise SystemExit(f"✗ controller 段内找不到锚点 {old!r}")
    return "".join(out)


def build() -> dict[str, str]:
    if not REFERENCE.exists():
        raise SystemExit(f"✗ 找不到底稿 {REFERENCE}")
    base = REFERENCE.read_text(encoding="utf-8")

    # ---- 底稿形状校验（形状变了就停下，不要带病替换）----------------------
    if base.count(_ANCHOR_CTRL_KIND) != 1:
        raise SystemExit(
            f"✗ 底稿里 controller.kind 锚点 {_ANCHOR_CTRL_KIND!r} 出现 "
            f"{base.count(_ANCHOR_CTRL_KIND)} 次（期望恰好 1 次）—— 底稿结构变了？")
    if _count_ctrl_enabled(base) != 1:
        raise SystemExit(
            f"✗ 底稿 controller 段内 `enabled: true` 出现 "
            f"{_count_ctrl_enabled(base)} 次（期望恰好 1 次）—— 底稿结构变了？")
    if base.count(_ANCHOR_NOTES_TAIL) != 1:
        raise SystemExit(
            f"✗ 底稿 notes 尾 {_ANCHOR_NOTES_TAIL!r} 出现 "
            f"{base.count(_ANCHOR_NOTES_TAIL)} 次（期望恰好 1 次）—— 底稿结构变了？")

    out: dict[str, str] = {}
    for arm, ctrl_enabled, kind, label in ARMS:
        t = base
        # ① controller.enabled
        t = _replace_in_ctrl(
            t, _ANCHOR_CTRL_ENABLED,
            f"  enabled: {'true' if ctrl_enabled else 'false'}\n")
        # ② controller.kind —— **总是**写（不"留底稿值"）：
        #    底稿是 cost，而 `static` 臂必须写 threshold（见 ARMS 的注释）。
        #    "留底稿值"会让 static 臂带上 cost ⇒ 加载期抛错。
        t = _replace_in_ctrl(t, _ANCHOR_CTRL_KIND, f"  kind: {kind}\n")
        # ③ notes 里写明臂 —— 让 `meta.json` 自带臂标签，事后可从产物重建
        t = t.replace(
            _ANCHOR_NOTES_TAIL,
            f"（两处已同步）｜**S6 服务收益档 §S6**：臂={label}"
            f"（controller.enabled={ctrl_enabled} / kind={kind}）")
        out[f"{cell_name(arm)}.yaml"] = t
    return out


def verify(files: dict[str, str] | None = None) -> list[str]:
    """逐文件断言三臂只差那两处开关。返回问题列表（空 = 通过）。

    ⚠️ **两道校验缺一不可**：
    ① 文本级（三臂逐字节差异只允许出现在开关行）—— 抓"引入了第二个变量"；
    ② **真加载器级**（`ControllerConfig` 实构造）—— 抓"开关组合非法"。
    实测踩到：文本级只数 `enabled: true` 出现次数，放过了
    `kind: cost` + `enabled: false`（加载期抛错）⇒ 只做 ① 是不够的。
    """
    files = files if files is not None else build()
    problems: list[str] = []

    # 1) 三份都在
    for arm, _, _, _ in ARMS:
        fn = f"{cell_name(arm)}.yaml"
        if fn not in files:
            problems.append(f"缺文件 {fn}")

    # 2) 三臂逐字节差异**只允许**出现在那两行
    base_arm = ARMS[0][0]
    base_lines = files[f"{cell_name(base_arm)}.yaml"].splitlines()
    for arm, _, _, _ in ARMS[1:]:
        other = files[f"{cell_name(arm)}.yaml"].splitlines()
        if len(other) != len(base_lines):
            problems.append(f"{arm}: 行数与 {base_arm} 不同"
                            f"（{len(other)} vs {len(base_lines)}）")
            continue
        for i, (a, b) in enumerate(zip(base_lines, other), start=1):
            if a == b:
                continue
            # 允许不同：controller.enabled / controller.kind / notes
            allowed = ("  enabled: " in a or "  kind: " in a or "S6 服务收益档" in a)
            if not allowed:
                problems.append(f"{arm}: 第 {i} 行不应有差异：{a!r} vs {b!r}")

    # 3) 语义校验：解析各臂的 (enabled, kind)，与 ARMS 声明一致
    for arm, ctrl_enabled, kind, _ in ARMS:
        text = files[f"{cell_name(arm)}.yaml"]
        n_on = _count_ctrl_enabled(text)
        expect_on = 1 if ctrl_enabled else 0
        if n_on != expect_on:
            problems.append(
                f"{arm}: controller 段内 enabled: true 出现 {n_on} 次（期望 {expect_on}）")
        if f"  kind: {kind}\n" not in text:
            problems.append(f"{arm}: 找不到 `  kind: {kind}`")

    # 4) 三臂之间 kind/enabled 组合互不相同（防止复制粘贴造出两条相同的臂）
    sigs = [(en, kind) for _, en, kind, _ in ARMS]
    if len(set(sigs)) != len(sigs):
        problems.append(f"三臂的 (enabled, kind) 组合有重复：{sigs}")

    # 5) ★ 真加载器校验（唯一权威）：非法组合会在这里抛错。
    #    必须 `import` 真类，不能自己重写规则 —— 重写一份就必然与实现分叉。
    try:
        import yaml  # type: ignore
        sys.path.insert(0, str(_ROOT / "src"))
        from ascend_opt.controller import ControllerConfig  # type: ignore
        for arm, _en, _kind, _ in ARMS:
            doc = yaml.safe_load(files[f"{cell_name(arm)}.yaml"])
            c = dict(doc["controller"])
            try:
                ControllerConfig(**c)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{arm}: 真加载器拒绝该 controller 组合 —— "
                                f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"真加载器校验未能执行（{type(e).__name__}: {e}）"
                        "—— 不得当作通过")

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只校验、不写盘")
    args = ap.parse_args(argv)

    files = build()
    problems = verify(files)

    print(f"[s6] 底稿 = {REFERENCE.relative_to(_ROOT)}")
    print(f"[s6] 三臂 = {[a for a, *_ in ARMS]}")
    print(f"[s6] 目标目录 = {OUT_DIR.relative_to(_ROOT)}")
    print(f"[s6] 校验问题 = {len(problems)} 处")
    for p in problems:
        print(f"  ✗ {p}")
    if problems:
        return 1

    if args.check:
        print("[s6] ✓ 校验通过（--check，未写盘）")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn, text in sorted(files.items()):
        # ⚠️ 必须显式 newline="\n"：Windows 上 Path.write_text 会把 \n 翻成 \r\n，
        #    而底稿是 LF ⇒ 三臂会以"只差开关"的名义带上 CRLF 差异。
        with open(OUT_DIR / fn, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(f"[s6] ✓ 已写 {len(files)} 个文件到 {OUT_DIR.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
