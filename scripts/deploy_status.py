"""本地源码相对**已部署版本**的状态（**离线，不连远端**）。

★ 为什么需要它（2026-10-03）
----------------------------
本项目用"部署戳"（`SOURCE_PROVENANCE.json`）把产物绑到源码版本上。但戳是打在
**远端**的，而代码是在**本机**改的 —— 于是有一个安静的危险窗口：

    改了代码 → 本机 `tree_digest` 变了 → 远端戳**还是旧的** →
    有人拿它当"当前源码的身份"用 ⇒ "这批数字来自我现在这份代码"

而它其实来自**改动之前**那份。⇒ 本脚本把这件事变成一个**可查询的状态**：
只要本机摘要 ≠ 已部署摘要，就把本地版本标成 **尚未部署**，
并且**明确写出"旧戳不代表当前源码"**。

⚠️ 本脚本只读本地文件、只写一份状态文件；**不 SSH、不启动远端命令、不碰 `results/raw`**。
⚠️ `--record` 只应在**真的做过一次部署**之后调用（部署动作本身不在这里做）——
   否则会用一个"声称已部署"的记录盖住真实状态，比不记录更坏。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: 部署状态记录（**放进交接目录**，不是 `results/raw/`；后者只读）。
RECORD_PATH = ROOT / "handoff" / "optimization_20260929" / "DEPLOYMENT.json"

SCHEMA = "deployment-status-v1"


def local_tree_digest(root: Path = ROOT) -> str:
    """本机源码树摘要（与产物里 `tree_digest` **同一套算法**）。"""
    from ascend_opt.provenance import compute_tree_digest

    return compute_tree_digest(root)


def read_record(path: Path = RECORD_PATH) -> dict[str, Any] | None:
    """读部署记录；**读不到返回 None**（不返回 `{}`：`{}` 会被读成"记录存在但没内容"）。"""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def deployment_state(*, local_digest: str | None,
                     record: dict[str, Any] | None) -> dict[str, Any]:
    """**纯函数**：本机源码相对已部署版本处于什么状态。

    三态，且**三态都不许被读成"没问题"除非真的是 IN_SYNC**：

    | state | 含义 | `redeploy_needed` |
    | --- | --- | --- |
    | `NEVER_DEPLOYED` | 没有部署记录（或记录里没有摘要） | `True` |
    | `IN_SYNC` | 本机摘要 == 已部署摘要 | `False` |
    | `LOCAL_AHEAD` | 本机摘要 ≠ 已部署摘要 ⇒ 本地版本**尚未部署** | `True` |
    | `UNKNOWN` | 本机摘要取不到 ⇒ **无法判断**（不等于同步） | `True` |

    ⚠️ `UNKNOWN` 必须与 `IN_SYNC` 分开：把"取不到"当成"同步"，
    正是本项目反复记的"缺失伪装成合格"。
    """
    dep = str((record or {}).get("deployed_tree_digest") or "").strip() or None
    at = (record or {}).get("recorded_at")

    if not local_digest:
        return {
            "schema": SCHEMA, "state": "UNKNOWN", "redeploy_needed": True,
            "local_tree_digest": None, "deployed_tree_digest": dep,
            "recorded_at": at,
            "message": ("本机源码摘要**取不到** ⇒ 无法判断本地版本是否已部署。"
                        "**不得**把这种情况当成「已同步」。"),
        }
    if dep is None:
        return {
            "schema": SCHEMA, "state": "NEVER_DEPLOYED", "redeploy_needed": True,
            "local_tree_digest": local_digest, "deployed_tree_digest": None,
            "recorded_at": at,
            "message": ("没有部署记录 ⇒ 本地版本**尚未部署**。"
                        "任何既有产物的 `tree_digest` 都**不能**代表这份源码。"),
        }
    if local_digest == dep:
        return {
            "schema": SCHEMA, "state": "IN_SYNC", "redeploy_needed": False,
            "local_tree_digest": local_digest, "deployed_tree_digest": dep,
            "recorded_at": at,
            "message": "本机源码与已部署版本**一致**（摘要相同）。",
        }
    return {
        "schema": SCHEMA, "state": "LOCAL_AHEAD", "redeploy_needed": True,
        "local_tree_digest": local_digest, "deployed_tree_digest": dep,
        "recorded_at": at,
        "message": (f"本机源码**已改动**（{local_digest[:12]}… ≠ 已部署 "
                    f"{dep[:12]}…）⇒ 本地版本**尚未部署**。"
                    "⚠️ 旧远端部署戳**不代表当前源码**：既有产物仍绑在改动前那份代码上，"
                    "**不得**用当前代码去「复核」它们的数字；要引用新代码必须重新部署并重打戳。"),
    }


def write_record(*, digest: str, note: str = "") -> dict[str, Any]:
    """记录一次**真实部署**的摘要。只应在部署动作之后调用。"""
    payload = {
        "schema": SCHEMA,
        "deployed_tree_digest": digest,
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "note": note,
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true",
                    help="把**当前**本机摘要记为「已部署」（只在真的部署过之后用）")
    ap.add_argument("--record-as", default=None, metavar="DIGEST",
                    help="把**指定的**摘要记为「已部署」。用于**事后补记**一次历史部署"
                         "（例如从产物里读到的 `tree_digest`）——"
                         "⚠️ 它必须来自**产物或戳**，不能凭记忆写。")
    ap.add_argument("--note", default="", help="写进记录的备注")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args(argv)

    try:
        digest: str | None = local_tree_digest()
    except Exception as exc:                                       # noqa: BLE001
        digest = None
        print(f"[deploy-status] ⚠️ 算本机摘要失败：{type(exc).__name__}: {exc}",
              file=sys.stderr)

    if args.record_as:
        rec = write_record(digest=str(args.record_as).strip(), note=args.note)
        print(f"[deploy-status] 已补记历史部署摘要 = {rec['deployed_tree_digest'][:16]}…")
        return 0

    if args.record:
        if not digest:
            print("[deploy-status] ✗ 算不出本机摘要 ⇒ 拒绝写记录"
                  "（写一条假的「已部署」比不写更坏）", file=sys.stderr)
            return 2
        rec = write_record(digest=digest, note=args.note)
        print(f"[deploy-status] 已记录部署摘要 = {rec['deployed_tree_digest'][:16]}…")
        return 0

    state = deployment_state(local_digest=digest, record=read_record())
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print(f"[deploy-status] state = {state['state']}")
        print(f"[deploy-status] 本机 = {str(state['local_tree_digest'])[:16]}… "
              f"已部署 = {str(state['deployed_tree_digest'])[:16]}…")
        print(f"[deploy-status] {state['message']}")
    # 退出码：需要重新部署 ⇒ 1（便于挂进前置检查）
    return 1 if state["redeploy_needed"] else 0


if __name__ == "__main__":
    sys.exit(main())
