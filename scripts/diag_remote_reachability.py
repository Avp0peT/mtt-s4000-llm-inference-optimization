#!/usr/bin/env python3
"""远端可达性诊断：把 `PLAYBOOK.md` §5.1 那套**人工步骤**做成可执行的一条命令。

为什么需要它
------------
本项目最长的一次误判，来自把**三种完全不同的状态读成同一件事**：

| 观测 | 真正的含义 | 对策 |
| --- | --- | --- |
| 读到 SSH banner | 主机在、sshd 在应答 | 可用 |
| `recv()` 返回 `b''`（**EOF**） | 主机在、**该端口没监听** | 核对端口号（重启/换镜像会改端口） |
| `socket.timeout` | 包被**静默丢弃** | 先关代理复测；再查实例状态 |

"空 banner" 这一个词会把上面三行**合并成一行**，于是"端口变了"被读成"实例关机"。
本脚本强制把三态分开输出，并且**在同一次运行里**取全部对照
（§5.1.1：判据只对同一时间窗口内的对照成立，**不得**跨会话引用旧签名）。

⚠️ 另外两条已实测的边界，脚本只在输出里提醒、不代为判断：
* **"TCP 能连"不等于实例在**：本机代理会懒应答（`create_connection` 16 ms 就"成功"）；
* **HTTP 代理的 `CONNECT` 会"秒回 200"然后上游静默** ⇒ 必须**读到目标协议的
  首个字节**才算数（本脚本读 160 B，正是为此）。

依赖：**只用标准库** ⇒ 任意解释器都能跑（不需要 paramiko）。
用法::

    python scripts/diag_remote_reachability.py             # 读 ~/.workbuddy/remote.env
    python scripts/diag_remote_reachability.py --host H --port P
    python scripts/diag_remote_reachability.py --json out.json

退出码：0 = 目标拿到 banner（可 SSH）；1 = 目标不可达/异常；2 = 输入错（读不到 host:port）。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 8.0
ENV_PATH = Path.home() / ".workbuddy" / "remote.env"

#: 条目：(标签, host, port, 期望签名说明)
def default_targets(host: str, port: int) -> list[tuple[str, str, int, str]]:
    return [
        ("目标：远端映射端口", host, port, "应回 SSH banner"),
        ("对照A：同主机 :9", host, 9, "应 EOF 或超时（必然没服务）"),
        ("对照B：同类路由 SSH", "gitee.com", 22, "应回 SSH banner"),
        ("对照C：国外 SSH", "gitlab.com", 22, "应回 SSH banner"),
        ("对照D：通用出网 HTTP", "www.baidu.com", 80, "代理 CONNECT 常假活，仅供参考"),
    ]


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _read_first_byte(sock, timeout: float) -> tuple[str, str]:
    """只做"读首个字节"这一件事，并把三态分开返回。"""
    try:
        sock.settimeout(timeout)
        data = sock.recv(160)
    except socket.timeout:
        # 连上了但对方一个字节都不发 ⇒ 隧道假活（代理 CONNECT 秒回 200 的典型形态）
        return "TIMEOUT_AFTER_CONNECT", "连上但对方不发一个字节（隧道假活）"
    except OSError as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"
    if data == b"":
        return "EOF", "对端主动关闭（主机在、该端口没服务）"
    return "BANNER", data[:120].decode("utf-8", "replace").replace("\r\n", " / ").strip()


def probe(host: str, port: int, label: str, *, timeout: float = DEFAULT_TIMEOUT,
          connect=None, note: str = "") -> dict:
    """一次探测。`connect` 可注入（测试用），默认 socket.create_connection。"""
    connect = connect or socket.create_connection
    t0 = time.perf_counter()
    row = {"label": label, "target": f"{host}:{port}", "verdict": None,
           "detail": "", "ms": None, "note": note}
    try:
        sock = connect((host, port), timeout)
    except socket.timeout:
        row["verdict"] = "TIMEOUT"
        row["detail"] = "静默超时（包被丢弃：连不上 / 映射已撤销）"
    except ConnectionRefusedError:
        row["verdict"] = "REFUSED"
        row["detail"] = "对端 RST（主机可达、端口无监听）"
    except OSError as exc:
        row["verdict"] = "ERROR"
        row["detail"] = f"{type(exc).__name__}: {exc}"
    else:
        try:
            verdict, detail = _read_first_byte(sock, timeout)
            row["verdict"], row["detail"] = verdict, detail
        finally:
            try:
                sock.close()
            except OSError:
                pass
    row["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return row


def interpret(rows: list[dict]) -> tuple[int, list[str]]:
    """把探测结果翻成"读法"与退出码。**纯函数**（可单测，不碰网络）。"""
    if not rows:
        return 2, ["✗ 没有任何探测结果 ⇒ 未判"]
    target = rows[0]
    others = {r["verdict"] for r in rows[1:]}
    lines: list[str] = []

    if target["verdict"] == "BANNER" and "SSH-" in target["detail"]:
        lines.append("★ 目标可 SSH 可达（拿到真实 banner）⇒ 隧道与凭据链路正常。")
        lines.append("   下一步：跑一次 `scripts/remote_run.py exec <命令>`，"
                     "**认证**成功才算真的通（banner ≠ 认证通过）。")
        return 0, lines

    if target["verdict"] == "EOF":
        lines.append("△ 目标端口被**主动关闭** ⇒ 主机在、但该端口没有服务。")
        lines.append("   最可能是**端口号变了**（AutoDL 重启 / 换镜像 / 迁移都会改端口）。")
        lines.append("   动作：拿控制台当前的登录指令核对端口，只改 ~/.workbuddy/remote.env（仓库外）。")
        return 1, lines

    if target["verdict"] in ("TIMEOUT", "TIMEOUT_AFTER_CONNECT", "REFUSED", "ERROR"):
        if "BANNER" in others:
            lines.append("△ 目标异常、但有对照拿到 banner ⇒ 指向**远端侧**"
                         "（实例未运行 / 端口映射撤销 / 端口号变了）。")
            lines.append("   ⚠️ 但 §5.1.2 实测过「代理开着时目标也会这样」的案件 ⇒ "
                         "**先关掉本机代理复测一次**（几秒钟成本）。")
        else:
            lines.append("⚠️ 目标不通，且**对照也不正常** ⇒ 判据此刻**没有分辨力**，"
                         "**不得**下任何关于远端的结论。")
            lines.append("   （§5.1.1：对照的签名本身会变；跨时间窗口引用旧表格 ＝ 拿昨天的尺子量今天的东西。）")
        lines.append(f"   本次对照签名集合：{sorted(others)}")
        return 1, lines

    lines.append(f"⚠️ 目标返回 {target['verdict']} ⇒ 需人工判读。")
    return 1, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="远端可达性诊断（PLAYBOOK §5.1 的可执行版）")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--json", default=None, help="把结构化结果写到这个路径")
    ap.add_argument("--no-controls", action="store_true", help="只探目标（不推荐：判据会失去分辨力）")
    args = ap.parse_args(argv)

    env = read_env()
    host = args.host or env.get("REMOTE_HOST", "")
    try:
        port = args.port or int(env.get("REMOTE_PORT", "0") or 0)
    except ValueError:
        port = 0
    if not host or not port:
        print(f"✗ 读不到 host:port（命令行未给，且 {ENV_PATH} 里没有 REMOTE_HOST/REMOTE_PORT）",
              file=sys.stderr)
        return 2

    print(f"时间窗口：{time.strftime('%Y-%m-%d %H:%M:%S')}"
          "   ⚠️ 判据只对**同一次批内**的对照成立")
    print(f"本机代理：{os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or '(未设)'}")
    try:
        print(f"DNS 解析：{host} -> {socket.gethostbyname(host)}"
              "   ⚠️ 198.18.0.0/16 是代理 fake-ip，不是真实地址")
    except OSError as exc:
        print(f"DNS 解析失败：{exc}")

    targets = default_targets(host, port)
    if args.no_controls:
        targets = targets[:1]

    rows = [probe(h, p, label, timeout=args.timeout, note=note) for label, h, p, note in targets]

    print()
    print(f"{'条目':<26}{'结果':<26}{'耗时':>9}  说明")
    print("-" * 110)
    for r in rows:
        print(f"{r['label']:<26}{r['verdict']:<26}{(str(r['ms']) + ' ms'):>9}  {r['detail'][:58]}")

    rc, lines = interpret(rows)
    print("\n=== 读法 ===")
    for ln in lines:
        print(ln)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "proxy": os.environ.get("HTTP_PROXY"),
                       "rows": rows, "rc": rc}, fh, ensure_ascii=False, indent=1)
        print(f"\n已写出：{p}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
