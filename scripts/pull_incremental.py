#!/usr/bin/env python3
"""**增量取回**远端产物：只取本地还没有的 run 目录，并与远端核对数量。

为什么需要它（而不是每次手敲 `pull`）
------------------------------------
远端实例重启会**把系统盘回滚到快照**（`PLAYBOOK.md` §5.4 实测），
所以"等整批跑完再取"等于把整批放在一个会消失的地方。
⇒ 纪律是**边跑边取**。但手敲的循环有两个容易漏的点，本脚本把它们变成机械动作：

1. **不覆盖已有产物**：`results/raw/**` 是证据、append-only。
   已存在的本地目录**跳过**（并提示），绝不原地替换。
2. **取回后逐目录核对**：`meta.json` 在、四个文件齐、`data_label` 有值；
   任何一项不满足就**返回非零**（"取回来了"与"取回来的是完整产物"是两件事）。

⚠️ 它**不删**任何东西：远端目录一律保留（远端才是原件的所在地，本地是副本）。

用法::

    python scripts/pull_incremental.py --tag S10
    python scripts/pull_incremental.py --tag S10 --dry-run     # 只看差集，不取
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_RUN = _ROOT / "scripts" / "remote_run.py"
#: 一次性会被取回的完整产物应有的文件（缺任何一个都不算"完整产物"）。
_REQUIRED = ("meta.json", "requests.jsonl", "telemetry.jsonl", "config.snapshot.yaml")

#: 远端产物根目录。**当前平台（MTT S4000 / D1）用 `_d1` 后缀那个。**
#:
#: ⚠️ 2026-09-27 踩过：本参数默认值是迁移前的 `/root/ascend_probe/results/raw`，
#: 而 D1 迁移后一切都在 `/root/ascend_probe_d1`。旧默认值让枚举恒为空，
#: 而当时的报错文案是"**远端没有产物（或远端不可达）**"——
#: 于是"根目录写错"被读成"批次还没跑"。**先修根目录，再谈别的。**
DEFAULT_REMOTE_ROOT = "/root/ascend_probe_d1/results/raw"

#: paramiko 只装在托管 venv 里（裸解释器会 ModuleNotFoundError）。
_VENV_PY = (Path.home() / ".workbuddy" / "binaries" / "python" / "envs"
            / "default" / "Scripts" / "python.exe")

#: 远端命令失败的特征串（与 remote_run.py / check_remote_busy.py 保持一致）。
_FAIL_MARKERS = (
    "连接失败", "SSHException", "EOFError", "No existing session",
    "banner", "Authentication", "paramiko",
)


def _pick_interpreter() -> str:
    """选一个**能 import paramiko** 的解释器（见 check_remote_busy 的同名函数）。

    不能用 ``sys.executable``：本脚本常被**系统解释器**调用（带 torch 但没 paramiko），
    于是 remote_run.py 整块失败，而症状被上面那句"或远端不可达"吞掉。
    """
    for p in (str(_VENV_PY), sys.executable):
        if not p or not Path(p).exists():
            continue
        try:
            r = subprocess.run([p, "-c", "import paramiko"], capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            continue
        if r.returncode == 0:
            return p
    return sys.executable


def _looks_failed(text: str) -> bool:
    return any(m in text for m in _FAIL_MARKERS)


#: `remote_run.py pull` 会在**远端同名父目录**留下暂存包 `<arc>.tar`
#: （`do_pull` 行 715：`remote_tar = posixpath.join(parent, f"{arc}.tar")`）。
#: 因此 `ls -d <root>/<tag>-*` 会**把暂存包也列成"一个 run"**。实测已踩：
#: 一次失败的回拉在远端证据目录留下 `S14B2-...-r1.tar` 与其倍乘
#: `S14B2-...-r1.tar.tar`，它们随后被当成 run：
#:   * 进入"待取"清单 ⇒ 白白传一遍，本地多出 `_...-r1.tar.tar` 的怪名字；
#:   * **永远过不了 `_verify`**（没有 `meta.json`）⇒ 完整性核对报**假失败**；
#:   * 还会把 `--expect N` 的计数抬高，让"批次还没跑完"看起来像"跑完了但坏了"。
#: ⇒ 列举时**按后缀剔除**，并在 `remote` 之外单独记录被剔除了什么（不静默吞）。
_STAGING_SUFFIXES = (".tar", ".tar.gz", ".tgz")


def _split_staging(names: list[str]) -> tuple[list[str], list[str]]:
    """把列举结果分成 (run 目录, 暂存包)。暂存包**只报告**，不当 run。"""
    runs, staging = [], []
    for n in names:
        (staging if n.lower().endswith(_STAGING_SUFFIXES) else runs).append(n)
    return runs, staging


def _remote_dirs(remote_root: str, tag: str, py: str) -> tuple[list[str] | None, str]:
    """列出远端该 tag 的目录名。返回 ``(目录列表, 原始输出)``。

    ★ **``None`` 与 ``[]`` 必须分开**：
    * ``None`` = 探测本身失败（链路/权限/根目录不存在）⇒ "没查到"；
    * ``[]``   = 命令成功执行且**确实没有**该 tag 的产物。

    混成一个空会让"链路故障 / 根目录写错"伪装成"批次还没跑"——
    本脚本原来就是这个问题（报"远端没有产物（**或远端不可达**）"，
    两种成因写在同一句话里，等于都不告诉）。

    ★ 暂存包（`<arc>.tar` 等，见 `_STAGING_SUFFIXES`）**不算产物**：
    它们是 `pull` 自己的中间文件，落在与 run 同级的父目录里。
    这里剔除掉，并把剔除情况附在原始输出里（调用方会打印）⇒ 不静默。
    """
    cmd = [py, str(_REMOTE_RUN), "exec",
           f"ls -d {remote_root}/{tag}-* 2>/dev/null | xargs -n1 basename"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0 or _looks_failed(out):
        return None, out
    names = [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]
    runs, staging = _split_staging(names)
    if staging:
        out += ("\n[staging] 已剔除 %d 个暂存包（不是产物）：%s"
                % (len(staging), ", ".join(staging)))
    return runs, out


def _pull(remote_root: str, name: str, local_root: Path, py: str) -> int:
    cmd = [py, str(_REMOTE_RUN), "pull",
           f"{remote_root}/{name}", str(local_root / name)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def _verify(d: Path) -> list[str]:
    """核对一个取回目录是否**完整**。返回问题清单（空 = 完整）。"""
    bad: list[str] = []
    for f in _REQUIRED:
        if not (d / f).exists():
            bad.append(f"缺 {f}")
    meta = d / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if not m.get("data_label"):
                bad.append("meta.data_label 为空")
            if not ((m.get("summary") or {}).get("n_requests")):
                bad.append("meta.summary.n_requests 缺失")
        except Exception as e:                      # noqa: BLE001
            bad.append(f"meta.json 不可解析：{type(e).__name__}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="增量取回远端产物")
    ap.add_argument("--tag", required=True, help="run_id 前缀，如 S10")
    ap.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT,
                    help=f"远端产物根目录（默认当前平台的 {DEFAULT_REMOTE_ROOT}）")
    ap.add_argument("--local-root", default=str(_ROOT / "results" / "raw"))
    ap.add_argument("--python", default=None,
                    help="跑 remote_run.py 用的解释器；默认自动选**有 paramiko** 的那个")
    ap.add_argument("--dry-run", action="store_true", help="只看差集，不取")
    ap.add_argument("--expect", type=int, default=None,
                    help="本批预期产物的总数；给了才提示'还差几个'。"
                         "⚠️ **不给就不猜**——写死一个数对别的批次会报错数（本脚本第一版就是 15）。")
    args = ap.parse_args(argv)

    py = args.python or _pick_interpreter()
    local_root = Path(args.local_root)
    remote, raw = _remote_dirs(args.remote_root, args.tag, py)
    if remote is None:
        # ★ 三态：这是"没查到"，**不是**"确实没有"。绝不能读成"批次还没跑完"。
        print(f"UNKNOWN：远端枚举失败——**不得**当作'没有产物'。", file=sys.stderr)
        print(f"  远端根  ：{args.remote_root}", file=sys.stderr)
        print(f"  解释器  ：{py}", file=sys.stderr)
        print(f"  原始输出：\n{raw[:1500]}", file=sys.stderr)
        return 2
    if not remote:
        # 命令成功执行，且确实一个都没有。此时**根目录写错**是最常见成因，
        # 故把根目录一起打出来（而不是含糊说"或远端不可达"）。
        print(f"远端 {args.remote_root} 下没有 {args.tag}-* 的产物"
              f"（命令执行成功 ⇒ 确实是空的；请顺手确认远端根是否写对）")
        return 2

    # ⚠️ 本地 `have` 必须与远端 `remote` 用**同一套**过滤：
    # `results/raw/` 里同样会残留回拉中间态（`_<arc>.tar`、以及空的
    # `<arc>.tar` 目录）。用裸 `glob` 会让 `...-r1.tar` 进 `have`，
    # 于是差集比较把"暂存残留"与"真 run"混为一谈。
    have = {d.name for d in local_root.glob(f"{args.tag}-*") if d.is_dir()}
    have_runs, have_staging = _split_staging(sorted(have))
    have = set(have_runs)
    if have_staging:
        print(f"[staging] 本地 {len(have_staging)} 个暂存残留（不计入 have）："
              f"{', '.join(have_staging)}")
    # ⚠️ 远端剔除的暂存包**必须打印**：`_remote_dirs` 把提示附在 `raw` 里返回，
    # 但成功路径本来不打印 `raw` ⇒ 提示被造出来又丢掉（= 静默吞）。
    # 剔除是**对计数有影响**的动作（它决定"还剩几个"），不打印就没人发现
    # 曾经有东西被剔掉了。
    if "[staging]" in raw:
        for _line in raw.splitlines():
            if _line.startswith("[staging]"):
                print(_line)
    missing = [n for n in remote if n not in have]
    print(f"远端 {len(remote)} 个 / 本地已有 {len(have)} 个 / **待取 {len(missing)} 个**")
    if missing:
        for n in missing:
            print(f"   - {n}")
    if args.dry_run:
        return 0
    if not missing:
        print("（没有新产物；本地已是最新）")
    else:
        for n in missing:
            rc = _pull(args.remote_root, n, local_root, py)
            print(f"   {'✓' if rc == 0 else '✗'} 取回 {n}（rc={rc}）")

    # ---- 核对：取回后逐目录检查完整性 ----
    problems: list[str] = []
    checked = 0
    for n in remote:
        d = local_root / n
        if not d.is_dir():
            problems.append(f"{n}: 取回后本地仍不存在")
            continue
        bad = _verify(d)
        checked += 1
        if bad:
            problems.append(f"{n}: {'；'.join(bad)}")
    print(f"\n=== 完整性核对：{checked}/{len(remote)} 个目录 ===")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        print("  ⇒ 返回非零：'取回来了'不等于'取回来的是完整产物'")
        return 1
    print("  ✓ 全部完整（4 个文件齐、data_label 与 n_requests 有值）")
    if args.expect is not None:
        if len(remote) < args.expect:
            print(f"\n⚠️ 远端当前 {len(remote)} 个 / 预期 {args.expect} 个 ⇒ **批次还没跑完**，"
                  "隔一会儿再取一次。")
        elif len(remote) > args.expect:
            print(f"\n⚠️ 远端 {len(remote)} 个 > 预期 {args.expect} 个 ⇒ **预期值可能过时**"
                  "（别按旧数去判断批次状态）。")
    else:
        print(f"\n（远端当前 {len(remote)} 个；未给 --expect ⇒ 不对'批次是否跑完'下判断）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
