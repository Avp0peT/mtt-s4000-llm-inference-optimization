#!/usr/bin/env python3
"""执行优化线的**自驱动**矩阵运行器（在远端推理主机上跑）。

机制覆盖：S01 输出层冗余 / S02 allocator 生命周期 / S03 小批量线性融合 /
**S04 常驻 KV 缓冲**（臂集合由 `--arms` 显式给出，或 `--stage` 预设）。

设计目标：**跑满 + 全程无人值守 + 本机零干预**。

它替调用方做的四件事
--------------------
1. **穿插顺序**：`for pass in range(repeats): for cell in cells: run(cell, pass)`。
   背靠背重复会把一次偶发停顿整段归因成"这个配置更差"（实测同类现象：
   同配置重复两次 tpot 只差 1.23×，而 stall_p99 差 **4.35×**）。
   顺序逻辑在 `benchmark/exec_opt_plan.py`（纯函数，本机有单测）。
2. **均衡重复**：预算不够跑完当前一轮就**干净收尾**，不跑半轮
   （否则有的臂 n=5、有的 n=4，比较退化成"谁的 n 大"）。
3. **逐项原子落进度 + 30 s 心跳**：本机轮询这个文件即可区分
   「在跑」与「卡住」，不需要登录远端看进程。心跳由后台线程更新，
   因为**单次 run 可能跑 20+ 分钟**，只在 run 边界写进度会让长 run 看起来像死机。
4. **协作式停止**：`--stop-file` 一旦出现，**当前 run 跑完后**退出。
   （本机沙箱无法对远端子进程发信号，所以停止只能靠文件约定。）

判据（不是"没抛异常"）
----------------------
* **成功 = 产物判定通过**（复用 `scripts/run_step3_matrix.read_run_status`：
  失败/拒绝/未完成任一 > 0 即判不通过，`dropped_requests` 也算）。
  "跑完了但活没干完"**不算成功**，且产物**留在证据目录**不删。
* **不覆盖既有产物**：`run_once` 在目标目录已存在时抛 `RunDirConflict`（证据保护）。
* **断点续跑**：已存在的项按产物分类（ok / 不 ok / 不完整）⇒ 已 ok 的跳过，
  不 ok 的**记录但重跑会撞目录**，此时报错让人用新 `--tag`，**绝不静默覆盖**。

用法（远端）::

    python scripts/run_exec_opt_matrix.py --dry-plan
    python scripts/run_exec_opt_matrix.py --tag EXECOPT1 --fill-budget --budget-minutes 300
    python scripts/run_exec_opt_matrix.py --tag EXECOPT1 --repeats 3

退出码：0 = 全部项判定通过；1 = 有项不通过（或跑不完）；2 = 参数/环境错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "benchmark"), str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exec_opt_plan import (  # noqa: E402
    ORDER_MODES,
    order_for_pass,
    BASELINE_DEFAULT,
    CFG_DIR_DEFAULT,      # ★ 唯一真相源在 exec_opt_plan（外层编排也用同一个）
    build_plan,
    completion_flags,
    decide_stop,
    plan_summary,
    progress_payload,
    resolve_arms,
    run_dir_name,
    utc_stamp,
    write_json_atomic,
)
from run_benchmark import (  # noqa: E402
    CONFIG_SNAPSHOT_NAME,
    config_data_digest,
    load_config,
    run_once,
)
# ★ 复用产物判定（**不重写**）：`read_run_status` 是"成功 = 活干完了"的唯一实现。
from run_step3_matrix import format_run_status, read_run_status  # noqa: E402

HEARTBEAT_S = 30.0

#: 阶段预设。**把"先少量筛选、别一上来开全网格"这条纪律写进工具**，
#: 而不是靠操作者记住（`01_PROJECT_PLAN.md` §7：「先少量代表形状筛选，
#: 不能先开全网格」）。`screen` 每个机制取**一个代表性最强者**：
#:   * S01 取组合（两开关叠加，是这一支最有收益形态）；
#:   * S02 取唯一开关（没有可选的中间态）；
#:   * S03 取 **gate/up**（不是 QKV）—— 因为 QKV 的窄片必须 contiguous、
#:     每层每次 3 次拷贝（见 `linear_fusion.py`），机制上更可能被拷贝抵掉；
#:     筛选阶段先测**代价形状更干净**的那一支，避免被 QKV 的拷贝掩盖信号。
STAGE_ARMS: dict[str, list[str] | None] = {
    "screen": ["s01_both", "s02_no_empty_cache", "s03_fuse_gate_up"],
    "full": None,          # None = 配置目录下全部
}


# --------------------------------------------------------------------------- #
# 已存在产物的分类（断点续跑 / 覆盖保护）
# --------------------------------------------------------------------------- #

#: 可信的源码身份来源。`stamp_stale` = 有戳但重算摘要不符；`none` = 从来没戳。
#: 二者都**不能**证明"这份产物出自当前源码"。
_TRUSTED_GIT_SOURCES = ("git", "stamp")


def resolve_source_digest(root: Path) -> str | None:
    """求**根目录**这棵树的源码内容摘要；取不到返回 `None`（= **测不出**）。

    ★ 用部署戳的 ``tree_digest`` 而不是现算：判据要比的是"产物落盘那一刻的树"与
    "现在的树"，而戳正是那份"投放到远端、并已校验通过"的断言。现算一个摘要
    只能描述**现在**，两个都现算的摘要相等**并不能**证明产物出自它——
    因为产物可能是很久以前用另一棵树跑的，而那时没有留痕。
    """
    try:
        from ascend_opt.provenance import read_source_stamp
    except Exception:  # pragma: no cover
        return None
    stamp = read_source_stamp(root)
    if stamp.get("validated") is not True:
        return None
    digest = stamp.get("tree_digest")
    return str(digest) if digest else None


def identity_complaint(run_dir: Path, *, arm: str, cfg_file: Path,
                       model_revision: str | None,
                       source_digest: str | None) -> str | None:
    """续跑时核验**身份**（源码摘要 / 配置摘要 / 模型 revision），不只数"请求跑完没"。

    ★ 为什么必须：`read_run_status(dir)["ok"]` 只回答"活干完了"，
    完全不回答"这份产物是不是**当前这份代码 + 这份配置**跑出来的"。
    实测（EXECOPT4）：12 份产物全部 `git_commit=NOT_MEASURED`，
    而旧判据照样能把它们当"已完成"跳过 ⇒ 续跑会拿**来路不明**的产物充当证据。

    ★★ 为什么**不能只比 commit**（2026-10-02）：本机工作区常年是脏的（实测 83 项
    未提交改动），此时 `git_commit` 只标出"分支点"，**不描述被测的那份代码**——
    两份 commit 相同、代码却不同的产物，在只比 commit 的判据下**一模一样**。
    所以源码身份以**内容摘要**（`tree_digest`）为准。

    ★★★ **缺失 ⇒ 拒绝复用**（不是"跳过这一项比对"）。旧实现有三处"缺席即通过"：
      * `if src is not None and src not in _TRUSTED...` —— 产物没这个字段就放行；
      * `if snap.is_file() and cfg_file.is_file()` —— 快照或当前配置缺一个就不比；
      * `if model_revision and got and ...` —— 任一侧为空就不比。
    三处都犯同一个错：**把"测不出"读成了"没问题"**。真正的语义应当是
    "我核不了 ⇒ 我不能说这份产物可信 ⇒ 拒绝复用"。

    返回 `None` = 身份可接受；否则返回一句人话原因（调用方据此拒绝复用）。
    """
    mp = run_dir / "meta.json"
    if not mp.is_file():
        return None                      # 交给 `incomplete` 那条分支去报
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "meta.json 读不出/坏掉"

    repro = meta.get("reproducibility") or {}

    # ---------------- 1) 源码身份：摘要级 ----------------
    #
    # ⚠️ 注意 `_TRUSTED_GIT_SOURCES` 里那两个来源在**摘要**这一关是**不对等**的：
    #   * `stamp` —— 戳自带 tree_digest ⇒ 能过；
    #   * `git`   —— `detect_source_identity` **刻意不给摘要**（本机开发树的语义是
    #                "HEAD + 未提交改动"，一个纯 HEAD 摘要反而不描述它）
    #                ⇒ 走到下面 C 那条时**必然拒绝**。
    # 这是**有意的保守**，不是漏洞：真 git 树上做续跑，本来就没有可比的内容摘要，
    # 此时"核不了"必须读成"不许复用"。反过来（放行）才是那个致命的错。
    src = repro.get("git_source")
    if src is None or not str(src).strip():
        return ("源码身份**缺失**（产物没有 git_source 字段）⇒ 无法核验"
                "它是不是当前这棵树跑出来的，拒绝复用")
    if src not in _TRUSTED_GIT_SOURCES:
        return (f"源码身份不可信（git_source={src!r}，git_commit="
                f"{repro.get('git_commit')!r}）")

    if source_digest is None:
        return ("本机基线目录没有可用部署戳 ⇒ **无法**给出要比对的源码摘要，"
                "拒绝复用（改跑：先让部署戳通过）")

    got_digest = repro.get("tree_digest")
    if got_digest is None or not str(got_digest).strip():
        return ("产物**没有** tree_digest（旧产物或戳没带摘要）⇒ 无法做摘要级"
                "身份比对，拒绝复用。注意 `git_commit` 相同**不足**以证明是同一棵树："
                "脏树上它只是分支点。换新 --tag 重跑")
    if str(got_digest) != str(source_digest):
        return (f"源码摘要不同（产物 {str(got_digest)[:12]}… ≠ 当前 "
                f"{str(source_digest)[:12]}…）")

    # ---------------- 2) 配置身份：**参数内容的规范摘要** ----------------
    why = config_identity_complaint(run_dir, cfg_file)
    if why is not None:
        return why

    # ---------------- 3) 模型身份 ----------------
    if not model_revision:
        return "当前配置**没有**给出 model_revision ⇒ 无法核验模型身份，拒绝复用"
    got = repro.get("model_revision")
    if not got:
        return ("产物**没有** model_revision ⇒ 无法核验模型身份，拒绝复用")
    if got != model_revision:
        return f"模型 revision 不同（产物 {got!r} ≠ 当前 {model_revision!r}）"
    return None


def _load_yaml_mapping(path: Path) -> dict[str, Any] | str | None:
    """把 YAML 读成映射。

    返回值刻意用**三种形态**区分"读到了""读不出""不是映射"：

    * `dict`   —— 成功；
    * `str`    —— 读不出（内容是错误摘要，供报错直接用）；
    * `None`   —— 读出来了，但根节点不是映射。

    ⚠️ 不合并成 `None`：把"坏文件"与"结构不对"混成一个值，报错就只能说一句
    "配置有问题"，而这两者的修法完全不同（换文件 vs 改结构）。
    """
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:                                        # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return data if isinstance(data, dict) else None


def config_identity_complaint(run_dir: Path, cfg_file: Path) -> str | None:
    """核验**配置身份**：产物里的快照是否与当前配置是**同一组参数**。

    ★★ 约定（写读统一，2026-10-03 修）
    -----------------------------------
    身份 = **参数内容的规范摘要**（`config_data_digest`），**不是**源文件字节。

    原先的实现是 `snapshot.read_bytes() == cfg_file.read_bytes()`，而写端落的快照
    是 `yaml.safe_dump(cfg.raw, ...)` —— 一次**重新序列化**（注释、空行、键序、
    引号风格全被规范化）。本仓每份配置都带大段注释 ⇒ 两边**永远不可能相等**
    ⇒ 配置身份**恒判不一致** ⇒ **续跑永远不复用**。症状是"每次都重跑"，
    而报错说"配置已变"（指向错方向）。

    ⇒ 现在：解析快照 → 取规范摘要；与当前配置的规范摘要比。
      * 只有注释/排版变了 ⇒ 摘要相同 ⇒ **允许复用**（这才是"真实写端生成的
        相同配置快照必须允许复用"）；
      * 任一参数变了 ⇒ 摘要不同 ⇒ 拒绝；
      * 快照缺失/读不出/不是映射 ⇒ "测不出" ⇒ **拒绝**（不得读成"没问题"）；
      * 若产物另外记了 `reproducibility.config_digest`，它必须与快照自洽，
        否则拒绝（两份记录打架时不许挑对自己有利的那份）。

    返回 `None` = 可接受；否则一句人话原因。
    """
    snap = run_dir / CONFIG_SNAPSHOT_NAME
    if not snap.is_file():
        return (f"产物里**没有** {CONFIG_SNAPSHOT_NAME} ⇒ 配置摘要**缺失**，"
                "无法核验配置身份，拒绝复用")
    if not cfg_file.is_file():
        return f"当前配置 {cfg_file} **不存在** ⇒ 无法核验配置身份，拒绝复用"

    # ⚠️ **两侧用同一个解析器**（`yaml.safe_load`），不要一侧用 `load_config`。
    #    理由：① 判据只关心**参数内容**，不需要跑 schema 校验；
    #    ② `load_config` 是**严格**的（未知字段直接报错），拿它读"当前配置"会把
    #       "这份配置不合法"误报成"配置身份核验失败"，指向错方向；
    #    ③ 一旦两侧用的不是同一个解析路径，任何一侧的规范化都会静默破坏相等性——
    #       这正是本项要修的那类缺陷（写端重排 vs 读端比字节）。
    #    真正的"这份配置能不能跑"由批次启动时的 `load_config` 负责，不在这里。
    snap_data = _load_yaml_mapping(snap)
    if isinstance(snap_data, str):          # 返回字符串 = 出错了（见该函数约定）
        return (f"产物里的 {CONFIG_SNAPSHOT_NAME} **读不出**（{snap_data}）"
                "⇒ 配置摘要**缺失**，拒绝复用")
    if snap_data is None:
        return (f"产物里的 {CONFIG_SNAPSHOT_NAME} 不是映射"
                "⇒ 配置摘要**缺失**，拒绝复用")
    cur_data = _load_yaml_mapping(cfg_file)
    if isinstance(cur_data, str):
        return (f"当前配置 {cfg_file.name} **读不出**（{cur_data}）"
                "⇒ 无法核验配置身份，拒绝复用")
    if cur_data is None:
        return (f"当前配置 {cfg_file.name} 不是映射 ⇒ 无法核验配置身份，拒绝复用")

    got, want = config_data_digest(snap_data), config_data_digest(cur_data)
    if got != want:
        # 给出**具体差在哪**，避免又出现"说配置变了、却查不出差异"。
        # ⚠️ 保留「配置已变」这个**稳定子串**：它是运维口径与既有测试都在用的标记，
        #    换成别的说法会让"看得懂的人"和"看得懂的测试"同时失效。
        diff = _config_diff(snap_data, cur_data)
        return (f"配置已变（**参数**不同）：快照 {got[:12]}… ≠ 当前 {want[:12]}…；"
                f"差异：{diff}")

    # 若 meta 里另记了摘要，必须与快照自洽（两份记录不许打架）。
    mp = run_dir / "meta.json"
    recorded: str | None = None
    if mp.is_file():
        try:
            recorded = (json.loads(mp.read_text(encoding="utf-8"))
                        .get("reproducibility") or {}).get("config_digest")
        except Exception:                                           # noqa: BLE001
            recorded = None
    if recorded is not None and str(recorded) != got:
        return (f"产物自相矛盾：meta 记的 config_digest={str(recorded)[:12]}… "
                f"与快照算出的 {got[:12]}… 不一致 ⇒ 拒绝复用")
    return None


def _config_diff(a: dict[str, Any], b: dict[str, Any]) -> str:
    """给人看的、**有界**的参数差异列表（最多列 5 条路径）。"""
    flat_a, flat_b = _flatten(a), _flatten(b)
    keys = sorted(set(flat_a) | set(flat_b))
    lines = [
        f"{k}: {flat_a.get(k, '<缺失>')!r} -> {flat_b.get(k, '<缺失>')!r}"
        for k in keys if flat_a.get(k) != flat_b.get(k)
    ]
    if not lines:
        return "（参数完全一致——那说明差异只在排版上，属实现缺陷）"
    more = "" if len(lines) <= 5 else f"（共 {len(lines)} 处，仅列前 5）"
    return "；".join(lines[:5]) + more


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def classify_existing(run_dir: Path) -> str:
    """`absent` / `ok` / `bad` / `incomplete`。

    ⚠️ `incomplete` 与 `bad` 必须分开：前者是"没落盘/落了一半"（可能是有意的
    中断），后者是"跑完了但活没干完"。两者都**不许当成已完成、也不许静默覆盖**，
    但成因不同，报错文案要能区分 —— 否则人会去修错的东西。
    """
    if not run_dir.exists():
        return "absent"
    status = read_run_status(run_dir)
    if status.get("ok"):
        return "ok"
    label = status.get("label")
    if label is None and not (run_dir / "meta.json").is_file():
        return "incomplete"
    return "bad"


# --------------------------------------------------------------------------- #
# 心跳
# --------------------------------------------------------------------------- #

class Heartbeat:
    """后台线程定期刷新进度文件的 `heartbeat_epoch`。

    为什么需要：进度只在 run 边界写。单次 run 可能 20+ 分钟，期间进度文件
    一动不动 ⇒ 本机侧无法区分「正在跑一个长 run」与「进程卡死」，
    于是要么白等、要么误判重启（而重启会撞目录或重复烧机时）。

    ★ **必须传 provider（每次重新计算 payload），不能传一份快照 dict。**
    踩过的形状（2026-09-30，EXECOPT4 实测）：心跳持有一份**构造时**的
    `snapshot(None)` ⇒ 每 30 s 把 `n_done=0` / `current=None` / `elapsed_s=<构造时>`
    **写回**进度文件，覆盖掉 run 边界刚写的真实值。后果：
      ① 本机侧轮询看到的 `current` 与 `n_done` **时真时假**（取决于查询落在
         30 s 窗口的哪一段）——`cat` 看到 `baseline#r0`，同一时刻解析却得到 `None`；
      ② `decide_wait_exit` 的**冗余退出通道** `n_done>=n_total` 因此**不可靠**
         （summary 通道仍正常 ⇒ 本批未被拖住，但那是运气，不是设计）。
    这与「等待侧必须枚举全部退出通道」是**同一族**问题：枚举了通道，却让其中
    一条的内容被别的写入者周期性破坏。
    """

    def __init__(self, provider, path: Path,
                 interval: float = HEARTBEAT_S) -> None:
        #: `provider()` 必须返回**当前**的完整 payload（含 `heartbeat_epoch`）。
        self._provider = provider
        self.path = path
        #: 可注入，便于单测（默认 30 s；测试传 0.02 级别，不必真等半分钟）。
        self._interval = float(interval)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                # 每次重新算 ⇒ 心跳写的内容与 run 边界写的**同源**，不会互相覆盖。
                write_json_atomic(self.path, self._provider())
            except Exception:                                    # noqa: BLE001
                # 心跳写失败**不能**拖垮批次：它只是给本机看的辅助信息。
                pass

    def start(self) -> None:
        self._t = threading.Thread(target=self._loop, daemon=True,
                                   name="exec-opt-heartbeat")
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _cell_run_id(tag: str, cfg: Any) -> str:
    """与既有批次命名一致：`<tag>-<workload>-<cfg stem>`。"""
    return f"{tag}-{cfg.workload_name}-{cfg.stem}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=False, default=None,
                    help="批次标签（进入 run_id；不覆盖既有产物）")
    ap.add_argument("--cfg-dir", default=CFG_DIR_DEFAULT)
    ap.add_argument("--baseline", default=BASELINE_DEFAULT,
                    help="基线配置（作为对照臂，固定在穿插表的第一位）")
    ap.add_argument("--stage", choices=sorted(STAGE_ARMS), default="screen",
                    help="screen = 每机制取一个代表臂（默认，先筛选）；full = 全部臂")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="只跑这些臂（显式给出时**覆盖** --stage）")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--order", choices=list(ORDER_MODES), default="block",
                    help="轮内格序：block=每轮同序（历史行为）；"
                         "abba=奇数轮反转 ⇒ 抵消「轮内位置」这一系统性因素。"
                         "⚠️ 计划/执行/续跑共用同一张顺序表（同一函数生成）。")
    ap.add_argument("--repeats", type=int, default=None,
                    help="固定轮数（与 --fill-budget 二选一）")
    ap.add_argument("--fill-budget", action="store_true",
                    help="跑满：整轮整轮地跑，直到预算不够再跑完一轮")
    ap.add_argument("--min-repeats", type=int, default=3,
                    help="--fill-budget 下的下限（<3 次不能作结论）")
    ap.add_argument("--max-repeats", type=int, default=50,
                    help="--fill-budget 下的安全上限（防跑飞）")
    ap.add_argument("--budget-minutes", type=float, default=None,
                    help="时间预算；留给回拉与核验的余量由调用方预留")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--summary-file", default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--keep-going", action="store_true",
                    help="默认首个不通过即停；给了它就跑完剩下的")
    ap.add_argument("--dry-plan", action="store_true", help="只打印计划，不跑")
    ap.add_argument("--dry-run", action="store_true",
                    help="★ 走**离线模拟**后端跑整条编排（不碰真机）。"
                         "产物会被标成 dry-run/TEST_DATA，不得当性能证据；"
                         "它的用途是**验证编排本身**（轮数、完成度、续跑、传参）。")
    ap.add_argument("--notes", default="", help="写进每个 run 的 meta notes")
    args = ap.parse_args(argv)

    tag = args.tag or f"EXECOPT-{utc_stamp()}"
    out_dir = Path(args.out).resolve()
    progress_file = Path(args.progress_file or f"/root/exec_opt_progress_{tag}.json")
    summary_file = Path(args.summary_file or f"/root/exec_opt_summary_{tag}.json")
    stop_file = Path(args.stop_file or f"/root/exec_opt_STOP_{tag}")

    cfg_dir = Path(args.cfg_dir)
    if not cfg_dir.is_absolute():
        cfg_dir = _ROOT / cfg_dir
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = _ROOT / baseline_path

    # ---- 解析臂与配置 --------------------------------------------------------
    # 显式 --arms 覆盖 --stage；否则用阶段预设。
    include = args.arms if args.arms else STAGE_ARMS[args.stage]
    try:
        arms = resolve_arms(cfg_dir, include=include, exclude=args.exclude or ())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[exec-opt] ✗ {exc}", file=sys.stderr)
        return 2
    if not baseline_path.is_file():
        print(f"[exec-opt] ✗ 基线配置不存在：{baseline_path}", file=sys.stderr)
        return 2

    cells = ["baseline"] + [a for a in arms]
    configs: dict[str, Any] = {"baseline": load_config(baseline_path)}
    for a in arms:
        configs[a] = load_config(cfg_dir / f"{a}.yaml")

    # 每个臂**实际**用的配置路径。★ 必须与上面 `configs` 的装配逐条对应：
    # "baseline" 臂用的是 `--baseline`，**不是** `cfg_dir/baseline.yaml`
    # （那个文件通常根本不存在）。旧实现把续跑判据的 `cfg_file` 写成
    # `cfg_dir / f"{name}.yaml"`，于是 baseline 臂恒指向一个不存在的路径 ⇒
    # 判定分支里那个 `cfg_file.is_file()` 恒假 ⇒ **配置身份这一项从来不比**。
    cfg_path_of: dict[str, Path] = {"baseline": baseline_path}
    for a in arms:
        cfg_path_of[a] = cfg_dir / f"{a}.yaml"

    # 源码身份（摘要级）在**批次开始时求一次**：整批同源，逐个 item 重复求值是浪费。
    # ⚠️ `source_digest` 为 `None` 表示**测不出**（本机没有可用部署戳），
    #    这时身份判据对**任何**既有产物都必须**拒绝复用**——不是"跳过比对"。
    source_digest = resolve_source_digest(_ROOT)
    if source_digest is None:
        print("[exec-opt] ⚠️ 本机源码摘要**测不出**（部署戳不可用）⇒ 既有产物一律"
              "不复用；本次只能跑新 --tag 的空目录。")

    # 记录每个臂的**执行优化开关**，让"这一批到底开了什么"不必去翻配置。
    # ⚠️ 这张清单必须**随机制增长**（踩过的坑的镜像：字段清单不覆盖新形状 ⇒
    #    键在产物里恒为缺失，而"缺失"与"没开"在报告里看不出区别）。加机制时
    #    在这一行补键，**不要**另起一处真相源。
    _SWITCH_KEYS = (
        "logits_last_only", "sample_needed_only", "retire_empty_cache",
        "fuse_qkv", "fuse_gate_up", "linear_fuse_shapes",
        # ---- S04 / P2b-c ----
        "persistent_kv_slots", "persistent_kv_capacity",
    )
    switches = {name: {k: cfg.runtime.get(k) for k in _SWITCH_KEYS
                       if k in cfg.runtime} for name, cfg in configs.items()}

    run_ids = {name: _cell_run_id(tag, cfg) for name, cfg in configs.items()}

    # ---- 轮数：固定 or 填预算 ------------------------------------------------
    # 默认行为是 **填预算**（跑满）—— 无人值守时"给一个时间预算就不管了"
    # 比"猜一个轮数"更贴合"跑满且不超时"。显式给 --repeats 则改为固定轮数。
    if args.repeats is not None and args.fill_budget:
        print("[exec-opt] ✗ --repeats 与 --fill-budget 只能给一个", file=sys.stderr)
        return 2
    fill_budget = bool(args.fill_budget) or args.repeats is None
    repeats = int(args.repeats) if args.repeats else int(args.min_repeats)
    if repeats < 1:
        print("[exec-opt] ✗ 轮数必须 >= 1", file=sys.stderr)
        return 2

    budget_s = None if args.budget_minutes is None else float(args.budget_minutes) * 60.0

    if args.dry_plan:
        print(json.dumps({
            "tag": tag, "cells": cells, "repeats": repeats,
            "fill_budget": fill_budget,
            "budget_minutes": args.budget_minutes,
            "switches": switches,
            "run_ids": run_ids,
            "plan": plan_summary(cells, repeats, order=args.order)["order"],
            "out_dir": str(out_dir),
        }, ensure_ascii=False, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 恢复：把已完成项先标出来 -------------------------------------------
    def item_dir(name: str, pass_index: int) -> Path:
        return out_dir / run_dir_name(run_ids[name], pass_index)

    # ---- 主循环（可多轮；--fill-budget 时动态决定跑几轮）--------------------
    started = time.time()
    done_keys: list[str] = []
    failures: list[dict[str, Any]] = []
    durations: list[float] = []
    skipped_existing: list[str] = []
    total_planned = len(cells) * repeats

    def snapshot(current: str | None, stop=None, extra=None) -> dict[str, Any]:
        plan = build_plan(cells, repeats, order=args.order)
        return progress_payload(tag=tag, plan=plan, done_keys=done_keys,
                                current=current, started_epoch=started,
                                now_epoch=time.time(), stopped=stop, extra=extra)

    #: 心跳与 run 边界**共用**这一个可变状态 ⇒ 心跳不再需要冻结一份快照。
    #: ⚠️ 只有 `current` 需要放进去：`done_keys` 是 list 引用、`started` 是闭包变量，
    #:    两者本来就会随进度变化；`current` 是**值**参数，冻结了就永远是旧值。
    hb_state: dict[str, str | None] = {"current": None}
    hb = Heartbeat(lambda: snapshot(hb_state["current"]), progress_file)
    hb.start()
    print(f"[exec-opt] tag={tag} 臂={cells} 轮数={repeats} 合计={total_planned} 项")
    print(f"[exec-opt] 进度文件={progress_file}")
    print(f"[exec-opt] STOP 文件={stop_file}（出现即在本 run 结束后干净退出）")

    rc = 0
    try:
        pass_index = 0
        while pass_index < (args.max_repeats if fill_budget else repeats):
            # ★ 顺序**只从这里来**（与计划、续跑同一个函数）。
            #   旧实现自己拼了一份 `for name in cells` —— 与 `build_plan` 并列，
            #   于是"计划写 ABBA、实际跑 ABAB"两边都不报错。
            plan_this_pass = [(name, pass_index) for name in
                              order_for_pass(cells, pass_index, order=args.order)]
            # 本轮里"尚未开始"的项数（不含正在跑的）——均衡重复的判据用它
            remaining_this_pass = len(plan_this_pass)

            for name, p in plan_this_pass:
                key = f"{name}#r{p}"
                rdir = item_dir(name, p)
                state = classify_existing(rdir)
                if state == "ok":
                    # ★ **"跑完了"不等于"是当前这份代码/配置跑出来的"**。
                    #   续跑必须核身份，否则会拿 `git_commit=NOT_MEASURED`
                    #   之类来路不明的产物充当证据（EXECOPT4 的 12 份全是那种）。
                    why = identity_complaint(
                        rdir, arm=name, cfg_file=cfg_path_of[name],
                        model_revision=(configs[name].runtime or {}).get("model_revision"),
                        source_digest=source_digest,
                    )
                    if why is None:
                        remaining_this_pass -= 1
                        if key not in done_keys:
                            done_keys.append(key)
                        skipped_existing.append(key)
                        continue
                    failures.append({
                        "key": key, "run_dir": str(rdir), "state": "identity_mismatch",
                        "reason": why,
                    })
                    print(f"[exec-opt] ✗ {key} 产物已判定完成，但**身份不符**：{why}\n"
                          f"           ⇒ 不复用、不覆盖、不重跑。换新 --tag，"
                          f"或先让部署戳通过再跑。")
                    rc = 1
                    if not args.keep_going:
                        raise KeyboardInterrupt
                    remaining_this_pass -= 1
                    continue
                if state in ("bad", "incomplete") and rdir.exists():
                    # 绝不覆盖证据；也不重复烧机时。报错让人用新 --tag。
                    failures.append({
                        "key": key, "run_dir": str(rdir), "state": state,
                        "reason": format_run_status(read_run_status(rdir)),
                    })
                    print(f"[exec-opt] ✗ {key} 已有产物且判定不通过（{state}）"
                          f"⇒ 不覆盖、不重跑。重跑请用新的 --tag。")
                    rc = 1
                    if not args.keep_going:
                        raise KeyboardInterrupt
                    remaining_this_pass -= 1
                    continue

                stop = decide_stop(
                    elapsed_s=time.time() - started, budget_s=budget_s,
                    items_done=len(done_keys), items_total=total_planned,
                    avg_item_s=(sum(durations) / len(durations)) if durations else None,
                    remaining_items_this_pass=remaining_this_pass,
                    stop_file_exists=stop_file.exists(),
                )
                if stop.stop:
                    print(f"[exec-opt] ⏹ {stop.reason}")
                    raise KeyboardInterrupt

                print(f"[exec-opt] ▶ {key} -> {rdir.name}", flush=True)
                # ★ **开跑前**就把 `current` 落到进度文件：单次 run 可能 20+ 分钟，
                #   若等 run 结束才写，本机侧整个 run 期间看到的都是**上一项**
                #   （或初始化值）。心跳会用同一个 provider 持续刷新。
                hb_state["current"] = key
                write_json_atomic(progress_file, snapshot(key))
                t0 = time.time()
                ok = True
                try:
                    run_once(configs[name], dry_run=bool(args.dry_run), out_dir=out_dir,
                             repeat_index=p, run_id=run_ids[name],
                             extra_meta={"exec_opt_batch": tag, "exec_opt_cell": name,
                                         "exec_opt_pass": p, "notes": args.notes},
                             verbose=True)
                except Exception as exc:                          # noqa: BLE001
                    ok = False
                    failures.append({"key": key, "run_dir": str(rdir),
                                     "state": "exception",
                                     "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"[exec-opt] ✗ {key} 抛异常：{type(exc).__name__}: {exc}")
                dt = time.time() - t0
                durations.append(dt)
                status = read_run_status(rdir)
                # ★ 成功判据是**产物**，不是"没抛异常"
                if ok and status.get("ok"):
                    done_keys.append(key)
                    print(f"[exec-opt] ✓ {key} {dt / 60:.1f} min")
                else:
                    failures.append({"key": key, "run_dir": str(rdir),
                                     "state": "not_ok",
                                     "reason": format_run_status(status)})
                    rc = 1
                    print(f"[exec-opt] ✗ {key} {dt / 60:.1f} min —— "
                          f"{format_run_status(status)}")
                    if not args.keep_going:
                        raise KeyboardInterrupt
                remaining_this_pass -= 1
                hb_state["current"] = key
                write_json_atomic(progress_file, snapshot(key))

            pass_index += 1
            if not fill_budget:
                # ★★★ 固定轮数模式：**不能 break**（2026-10-03 修）。
                #
                # "跑几轮"的唯一权威是循环条件 `pass_index < repeats`。原先在这里
                # `break`，于是 `--repeats 5` **只跑 1 轮** —— 实测（2 格 × 5 轮）：
                #
                #     run_once 调用 **2** 次（应为 10）、n_done=2、n_items=10、
                #     completed_passes=1、balanced=**True**、failures=**[]**、rc=**0**
                #
                # ⇒ 少跑 8 项，却报"完整成功"。`balanced` 判据看不出来它：
                #   2 % 2 == 0 本来就"均衡"。真正的判据是**对着计划总数核完成度**。
                continue
            # 填预算模式：一轮跑完后再决定要不要下一轮
            stop = decide_stop(
                elapsed_s=time.time() - started, budget_s=budget_s,
                items_done=len(done_keys), items_total=total_planned,
                avg_item_s=(sum(durations) / len(durations)) if durations else None,
                remaining_items_this_pass=len(cells),      # 下一轮的整轮
                stop_file_exists=stop_file.exists(),
            )
            if stop.stop:
                print(f"[exec-opt] ⏹ 不再开新一轮：{stop.reason}")
                break
            if len(done_keys) >= args.max_repeats * len(cells):
                print(f"[exec-opt] ⏹ 达到 --max-repeats={args.max_repeats}")
                break
            # ⚠️ **不要复用 `total_planned`**：它是"计划总数"（= len(cells) * repeats），
            #    上一轮起 `decide_stop(items_total=total_planned)` 要用它做"是否全部完成"
            #    的判据。曾在这里写成 `total_planned = len(cells) * (pass_index + 1)`
            #    （本意只是**报告**当前累计项数）⇒ r0 结束后把它从 12 改成了 4
            #    ⇒ r1 结束后 `decide_stop(items_done=8, items_total=4)` 判成
            #    **"全部项已完成"** ⇒ 3 轮的批次只跑了 2 轮就"干净收尾"，
            #    而 summary 里 `failures=[]`、日志也不报错（**静默少跑一轮**，
            #    实测 EXECOPT2：n_done=8/12、completed_passes=2）。
            print(f"[exec-opt] ↻ 开始第 {pass_index + 1} 轮（已完成 {len(done_keys)} 项，"
                  f"已用 {(time.time() - started) / 60:.1f} min）")
    except KeyboardInterrupt:
        pass
    finally:
        hb.stop()

    # ---- 收尾 ---------------------------------------------------------------
    completed_passes = (len(done_keys) // len(cells)) if cells else 0
    # ★★★ 完成度判据（2026-10-03 加）：判据本体在纯逻辑层 `completion_flags`
    #   （见它的 docstring：`balanced` 在任何整数轮上都为真，拦不住"少跑一整批"）。
    flags = completion_flags(n_done=len(done_keys), n_cells=len(cells),
                             repeats=repeats, fill_budget=bool(fill_budget))
    n_done_total = len(done_keys)

    summary = {
        "schema": "exec-opt-summary-v1",
        "tag": tag,
        "cells": cells,
        "run_ids": run_ids,
        "switches": switches,
        "n_items": len(build_plan(cells, repeats)),
        "n_done": n_done_total,
        "n_skipped_existing": len(skipped_existing),
        "completed_passes": completed_passes,
        # ★ 完成度四件套（判据在 `completion_flags`）：
        #   `n_planned_fixed` 为 None = 填预算模式（目标开区间）；
        #   `under_run` 是**唯一的"少跑"判据**，rc 由它决定。
        **flags,
        "dry_run": bool(args.dry_run),
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
        "avg_item_s": round(sum(durations) / len(durations), 1) if durations else None,
        "out_dir": str(out_dir),
        # ⚠️ 这个键**曾经**写成 `budget_s is not None` —— 那说的是「**有没有配**预算」，
        #    而不是「**是不是被预算停下的**」。实测本批：`budget_limited=false`
        #    （10/10 跑满）却同时 `stopped_by_budget=true`，两个字段看起来互相矛盾
        #    ⇒ 读汇总的人会以为这一批被截断了。
        #    ⇒ 改为**如实**取「预算真的限制了轮数/项数」那个量（= `budget_limited`）。
        "stopped_by_budget": bool(flags.get("budget_limited")),
        #: 另存一条：本批**是否配置了**预算（与「是否被它停下」是两件事）。
        "budget_configured": budget_s is not None,
        "notes": args.notes,
    }
    hb_state["current"] = None
    write_json_atomic(progress_file, snapshot(None))
    write_json_atomic(summary_file, summary)
    print(f"[exec-opt] 汇总={summary_file}")
    print(json.dumps({k: summary[k] for k in
                      ("tag", "n_done", "n_items", "n_planned_fixed", "complete",
                       "under_run", "completed_passes", "balanced",
                       "n_skipped_existing", "elapsed_s", "failures")},
                     ensure_ascii=False, indent=2))
    if failures:
        rc = 1
    if flags["under_run"]:
        print(f"[exec-opt] ✗ **少跑**：计划 {flags['n_planned_fixed']} 项，"
              f"实际完成 {n_done_total} 项"
              f"（completed_passes={completed_passes}/{repeats}）"
              "⇒ 不得当作完整成功；补齐或改小 --repeats 后重跑。")
        rc = 1
    if not flags["balanced"]:
        print("[exec-opt] ⚠️ 各臂完成轮数**不均衡** ⇒ 比较会退化成'谁的 n 大'；"
              "要么补跑到均衡，要么按最小 n 截断（不得直接混算）")
        rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
