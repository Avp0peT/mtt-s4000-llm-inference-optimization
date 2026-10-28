#!/usr/bin/env python3
"""远程主机执行/传文件的薄封装（paramiko）。

为什么需要它：本项目的权威推理环境在远程 S4000 主机上，所有硬件相关操作
（模型加载、推理、基准、显存测量、profiling）都必须在**那台机器**上执行。
本机只做编排。

凭证安全约定：
- 凭证从 ``~/.workbuddy/remote.env`` 读取（**在仓库之外**，绝不入库）；
- 凭证不经命令行传递（命令行会出现在进程列表里），只走本进程内存；
- 任何输出都不打印密码。

用法::

    python scripts/remote_run.py exec  "nvidia-smi; ls /root"
    python scripts/remote_run.py put   <本地路径> <远端绝对路径>
    python scripts/remote_run.py get   <远端绝对路径> <本地路径>
    python scripts/remote_run.py push  <远端根目录> <本地路径>...   # 打成 tar 一次传
    python scripts/remote_run.py pull  <远端目录> <本地落点目录>    # 打包取回
    python scripts/remote_run.py stamp <远端根目录>                  # 投放并校验源码溯源戳

关于在远端跑 python——**``exec`` 不激活任何环境**:

    ``exec`` 用非交互 shell 执行，**不会** source 任何 profile/activate 脚本。
    远端 ``python``/``python3`` 因此解析到 ``/root/miniconda3``（**没有** ``transformers``），
    直接跑基准会 ``ModuleNotFoundError``。必须先激活 venv，且有两处坑：

    1. **要把激活和命令写在同一个 shell 里**，用 ``bash -c``（``sh`` 不认识 ``source``）::

           python scripts/remote_run.py exec \
               'bash -c "source /root/musa-envs/qwen310/activate_musa.sh && python3 -c \"import transformers\""'

       分开写两行是不行的——每次 ``exec`` 都是**新连接**，上一次的 ``source`` 不会留下。

    2. **起批次要走 ``launch_batch.sh --activate``**，不要在起跑器外面 ``source``：
       起跑用 ``setsid`` 在**新会话**里重新 exec，外层环境不会继承（踩过，见该脚本 ④ 段）。

为什么需要 ``push``：``put`` 每次都要新开一条 SSH 连接，13 个文件的批量上传
会在中途超时（实测发生过，传到第 11 个被 SIGTERM）。``push`` 把多个路径打进
一个内存 tar、用**一条**连接送过去再解开，既快又少一次"传了一半"的机会。
``pull`` 是它的镜像（paramiko 的 ``get`` 不支持目录）。

关于 ``stamp``：远端那棵源码树是**拷贝**过去的，不带 ``.git`` ⇒
``git rev-parse HEAD`` 必然失败 ⇒ 产物的 ``git_commit`` 恒为 ``NOT_MEASURED``
（72 份历史产物就是这样）。``stamp`` 把本机的 ``HEAD`` + 源码树摘要写成
``SOURCE_PROVENANCE.json`` 投到远端源码树根，让远端读到的版本可追溯。
**投放之后立刻在远端重算摘要校验**——校验不过就不算成功（见 ``do_stamp``）。

``push`` 默认**在推送源码后自动投放戳**（``--no-stamp`` 可关）：推送了源码却
没投放戳，等于让后续所有 run 悄悄退回 ``NOT_MEASURED``，而这种退化不会有报错。

**落点语义（push/pull 一致）**：``push`` 的相对路径按整条相对路径保留，
``pull`` 的远端目录的**内容**落到本地落点目录本身——不是落点里面再套一层。

退出码：远端命令的退出码；本地错误（连不上、凭证缺失）返回 2；
无法生成可信戳返回 4，投放后校验不通过返回 6。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# 凭证载入：只从仓库外的文件 + 环境变量读
# --------------------------------------------------------------------------- #

_CRED_FILE = Path.home() / ".workbuddy" / "remote.env"


def _load_credentials() -> dict[str, str]:
    """载入连接参数。环境变量优先，其次读仓库外的凭证文件。

    缺凭证时**报错退出**，不猜、不用空密码试探——静默降级会让人以为连上了。
    """
    env: dict[str, str] = {}
    if _CRED_FILE.exists():
        for raw in _CRED_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()

    # 环境变量覆盖文件值（便于临时切换主机而不改文件）
    for key in ("REMOTE_HOST", "REMOTE_PORT", "REMOTE_USER", "REMOTE_PASSWORD"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    missing = [k for k in ("REMOTE_HOST", "REMOTE_USER", "REMOTE_PASSWORD") if not env.get(k)]
    if missing:
        print(
            f"缺少连接参数 {missing}；请在 {_CRED_FILE} 里填写，"
            f"或用同名环境变量传入。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    env.setdefault("REMOTE_PORT", "22")
    return env


def _connect(creds: dict[str, str], attempts: int = 4, backoff: float = 3.0):
    """建立 SSH 连接；**在单次调用内重试**。

    为什么要重试：实测这台远端在批次跑满时，约 1/3 概率在**握手阶段**就被拒
    （``Error reading SSH protocol banner`` / ``Authentication timeout``），
    而**重试即通**（本轮实测：一次调用内第 5 次成功；换一次调用第 1 次就成功）。
    这不是"链路死了"，而是控制面限流 ⇒ 把重试放进连接层，
    让上层（回拉脚本、看护脚本）不必各自实现一遍。

    ⚠️ **只对连接建立重试**，不对已经发出的命令重试——有副作用的命令
    盲目重试会执行两次。命令层的重试由调用方按只读/只写自行决定。
    """
    import paramiko
    import time as _time

    last: Exception | None = None
    for i in range(max(1, attempts)):
        client = paramiko.SSHClient()
        # 已知主机白名单：本项目在首次连接时接受并记住主机密钥（AutoDL 实例会重建主机密钥）
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=creds["REMOTE_HOST"],
                port=int(creds["REMOTE_PORT"]),
                username=creds["REMOTE_USER"],
                password=creds["REMOTE_PASSWORD"],
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,   # 不用本机密钥试探，避免把「密钥被拒」误报成「密码错」
                allow_agent=False,
            )
            return client
        except Exception as exc:  # noqa: BLE001
            last = exc
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            if i + 1 < attempts:
                _time.sleep(backoff * (i + 1))
    raise last if last else RuntimeError("_connect 失败且无异常记录")


# --------------------------------------------------------------------------- #
# 三个动作
# --------------------------------------------------------------------------- #


def _exec_capture(client, command: str, timeout: float | None) -> tuple[int, str, str]:
    """执行远端命令并**返回** ``(退出码, stdout, stderr)``，不回显。

    存在的理由：有些命令的输出是**要解析的**（部署戳校验），而不是给人看的。
    走 ``do_exec`` 会先把输出打到终端，调用方拿不到字符串——于是只能"再看一遍
    终端输出"，而那正是"解析器对着人眼输出写"的开始。

    解码用 ``errors="replace"``：远端环境是 POSIX/C locale 时输出里可能混入
    非 UTF-8 字节，直接 decode 会抛异常，把"命令跑完了、结果有点乱码"
    变成"命令失败"。
    """
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdin.close()

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def do_exec(client, command: str, timeout: float | None) -> int:
    """在远端执行命令，实时回显 stdout/stderr，返回远端退出码。

    用 get_pty=False 以保持输出纯净（不混入终端控制字符）。
    """
    code, out, err = _exec_capture(client, command, timeout)

    if out:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    if err:
        sys.stderr.write(err)
        if not err.endswith("\n"):
            sys.stderr.write("\n")
    return code


def do_put(client, local: str, remote: str) -> int:
    local_path = Path(local)
    if not local_path.exists():
        print(f"本地文件不存在：{local_path}", file=sys.stderr)
        return 2

    # 远端路径必须用 POSIX 语义解析：本机是 Windows，Path("/root/x/y").parent
    # 会变成 "\\root\\x"（反斜杠、丢掉前导斜杠），送到 sftp 就是一个不存在的路径。
    # 这类错误在本地看不出来，只在远端报 FileNotFoundError。
    remote = _normalize_remote(remote)
    remote_dir = posixpath.dirname(remote)
    if not remote_dir:
        print(f"远端路径必须是绝对路径：{remote!r}", file=sys.stderr)
        return 2

    sftp = client.open_sftp()
    try:
        _ensure_remote_dir(sftp, remote_dir)
        sftp.put(str(local_path), remote)
        size = sftp.stat(remote).st_size
        if size != local_path.stat().st_size:
            print(
                f"put 后大小不一致：本地 {local_path.stat().st_size} != 远端 {size}",
                file=sys.stderr,
            )
            return 2
        print(f"put OK  {local_path} -> {remote}  ({size} bytes)")
    finally:
        sftp.close()
    return 0


def do_get(client, remote: str, local: str) -> int:
    remote = _normalize_remote(remote)
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote, str(local_path))
        print(f"get OK  {remote} -> {local_path}  ({local_path.stat().st_size} bytes)")
    finally:
        sftp.close()
    return 0


def _remote_exists(sftp, path: str) -> bool:
    """远端点是否存在。用 sftp.stat 而不是 shell 的 [ -e ]：

    踩过——arcname 里带 Windows 反斜杠时，`[ -e 'scripts\\x.py' ]` 在 Linux 上
    检查的是一个**含字面反斜杠的文件名**，恒假；于是推送实际成功、
    卫兵却报"落点缺失"。把判断从 shell 挪到 sftp，分隔符与引号都不再参与。
    """
    try:
        sftp.stat(path)
        return True
    except OSError:
        return False


def do_push(
    client,
    remote_root: str,
    locals_: list[str],
    *,
    stamp: bool = True,
    repo_root: str | Path | None = None,
    allow_dirty: bool = False,
    verify_python: str = "python3",
) -> int:
    """把若干本地路径打进一个 tar，经**一条** SSH 连接送到远端再解开。

    相对路径的保留规则：每个**相对**路径以它自己被给出的**整条相对路径**为
    tar 内的名字，因此 ``push /root/proj scripts/ benchmark/`` 会在远端得到
    ``<proj>/scripts/...``、``<proj>/benchmark/...``；绝对路径只取 basename。
    所以 ``push /root/proj scripts/x.py`` 落到 ``/root/proj/scripts/x.py``，
    **不是** ``/root/proj/x.py``。调用方因此可以精确控制落点，不依赖 cwd。

    传输后**逐项比对文件数**：只报"tar 解开了"是不够的——解到错误的层级、
    或者被 umask/权限挡掉一部分，tar 自己不会失败（踩过：上传报成功、
    文件却写到别处）。

    推送成功后默认继续 :func:`do_stamp` 投放源码溯源戳（``stamp=False`` 关闭）。
    默认开着，是因为"推送源码但不投放戳"的代价是**静默**的：后续所有 run 的
    ``git_commit`` 退回 ``NOT_MEASURED``，没有任何报错。默认关的守卫等于没有守卫。
    """
    remote_root = _normalize_remote(remote_root)

    buf = io.BytesIO()
    n_files = 0
    skipped = 0

    def _skip(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        """排除 __pycache__/*.pyc。

        本机是 Python 3.13、远端是 3.10，把 cpython-313 的 pyc 传过去虽不会被
        加载（magic number 不符），但会让远端目录里混着两套无关的字节码，
        使"源码到底传对没有"这件事变得难以一眼判断。
        """
        nonlocal skipped
        if "__pycache__" in ti.name or ti.name.endswith(".pyc"):
            skipped += 1
            return None
        return ti

    with tarfile.open(fileobj=buf, mode="w") as tar:
        landed: list[str] = []
        for raw in locals_:
            p = Path(raw)
            if not p.exists():
                print(f"本地路径不存在：{p}", file=sys.stderr)
                return 2
            # arcname 的选取规则（踩过一次，必须写清）：
            #   相对路径 → 用**给定的路径本身**做 arcname，于是 `scripts/x.py`
            #     落到 <remote_root>/scripts/x.py。若像早先那样一律取 p.name，
            #     `scripts/x.py` 会落到 <remote_root>/x.py —— 推送打印"成功"、
            #     调用方也打印"完成"，但远端真正被执行的 `scripts/x.py`
            #     **一点没变**。真踩过：改了检查器、推送成功、跑出来的还是旧版输出，
            #     这类"报成功却没生效"不会有任何报错。
            #   绝对路径 → 只能用 basename，否则会试图还原本机的目录树。
            # arcname 必须用 **POSIX 形式**：Windows 上 str(Path("a/b")) 给的是
            # "a\\b"，反斜杠进 tar 后既不是分隔符、又会让后续存在性检查
            # 去查一个含字面反斜杠的名字（踩过，见 _remote_exists）。
            arc = p.as_posix() if not p.is_absolute() else p.name
            if arc.startswith("..") or arc.startswith("/"):
                print(f"拒绝把路径写到目标目录之外：{arc}", file=sys.stderr)
                return 2
            tar.add(str(p), arcname=arc, recursive=True, filter=_skip)
            landed.append(arc)
            if p.is_dir():
                n_files += sum(
                    1
                    for _ in p.rglob("*")
                    if _.is_file() and "__pycache__" not in _.parts
                )
            else:
                n_files += 1
    payload = buf.getvalue()

    sftp = client.open_sftp()
    try:
        _ensure_remote_dir(sftp, remote_root)
        remote_tar = posixpath.join(remote_root, "_push.tar")
        with sftp.open(remote_tar, "wb") as fh:
            fh.write(payload)
        size = sftp.stat(remote_tar).st_size
        if size != len(payload):
            print(
                f"tar 上传大小不一致：本地 {len(payload)} != 远端 {size}",
                file=sys.stderr,
            )
            return 2
        print(f"push 传输完成：{len(payload)} bytes, {n_files} 个文件 -> {remote_tar}")
        print(f"  落点（相对 {remote_root}）：{', '.join(landed)}")
        if skipped:
            print(f"（跳过 {skipped} 个 __pycache__/*.pyc 条目）")
    finally:
        sftp.close()

    # 解开并清理 tar。用 exec_command 走 shell，便于拿到解包报错。
    # 落点存在性**不在 shell 里判**（见 _remote_exists），这里只负责解开。
    # PUSH_UNPACKED 是整目录文件总数、与本次推送无关，别拿它当推送成功的证据。
    cmd = (
        f"cd {remote_root} && tar xf _push.tar && rm -f _push.tar && "
        f"echo PUSH_UNPACKED=$(find . -type f | wc -l)"
    )
    code = do_exec(client, cmd, timeout=120)
    if code != 0:
        print(f"远端解包失败（退出码 {code}）", file=sys.stderr)
        return code

    sftp = client.open_sftp()
    try:
        missing = [
            a for a in landed
            if not _remote_exists(sftp, posixpath.join(remote_root, a))
        ]
    finally:
        sftp.close()
    if missing:
        print(f"推送后这些落点不存在：{missing}", file=sys.stderr)
        return 3
    print(f"  已确认落点存在：{len(landed)} 项")

    if not stamp:
        # 显式关掉时**把代价说出来**，否则"这次先不投"会变成默认。
        print(
            "  （--no-stamp：未投放源码溯源戳 ⇒ 后续 run 的 git_commit 会是 "
            "NOT_MEASURED）"
        )
        return 0

    try:
        scope = _import_provenance().SOURCE_DIGEST_SCOPE
    except Exception as exc:
        print(f"无法加载部署戳模块，因此不能投放：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    uncovered = _pushed_covers_scope(landed, scope)
    if uncovered:
        # 不拦下来，因为"只推一个文件"是常见且正当的操作；但要说清校验会拿
        # **远端现存内容**去比，因此远端若落后于本机，校验就会失败。
        print(
            f"  注意：本次推送未覆盖摘要作用域里的 {uncovered}——"
            f"它们将按**远端现存内容**参与校验；远端若落后于本机，校验会失败"
            f"（这正是想要的效果：它在说「远端树已经不是被戳住的那棵」）。"
        )
    return do_stamp(
        client,
        remote_root,
        repo_root,
        allow_dirty=allow_dirty,
        verify_python=verify_python,
    )


# --------------------------------------------------------------------------- #
# 部署戳：投放 + 在**投放现场**校验
# --------------------------------------------------------------------------- #
#
# 为什么必须现场校验
# ------------------
# 戳有两种坏法，都不是"报错"型：
#   1. 摘要算错了（写端与读端的作用域/算法不一致）⇒ 戳从投下去那一刻就是过期的，
#      现场表现只是"git_commit 还是 NOT_MEASURED"，跟没投一模一样；
#   2. 远端树与戳不符（推了一半、远端有旧文件）⇒ 戳指向一个**错误**的 commit。
# 所以投放不是"写个文件"，而是一次**判定**：在远端用**将要读它的那份代码**
# 重算摘要，不符就返回非零。这样"戳有效"这件事在投放当时就被证明过一次，
# 而不是等到几十个 run 跑完、回头查溯源时才发现。

VERIFY_MARKER = "VERIFY_JSON "
VERIFY_PY_MARKER = "VERIFY_PY "
MANIFEST_MARKER = "MANIFEST "

#: 远端校验脚本。刻意用 heredoc 而不是"推一个校验脚本过去"——校验脚本若落在
#: `scripts/` 里就会进摘要作用域，于是"放校验器"这个动作本身会把摘要改掉。
#: 也刻意只 import、不重新实现摘要：写端读端必须是同一份实现。
_VERIFY_SCRIPT = """\
import json, sys
sys.path.insert(0, 'src')
try:
    from ascend_opt.provenance import read_source_stamp
except Exception as exc:
    print('VERIFY_IMPORT_ERROR ' + type(exc).__name__ + ': ' + str(exc))
    raise SystemExit(3)
print('VERIFY_PY ' + sys.executable + ' ' + sys.version.split()[0])
print('VERIFY_JSON ' + json.dumps(read_source_stamp('.'), sort_keys=True))
"""

_MANIFEST_SCRIPT = """\
import sys
sys.path.insert(0, 'src')
try:
    from ascend_opt.provenance import iter_scope_files
except Exception as exc:
    print('MANIFEST_IMPORT_ERROR ' + type(exc).__name__ + ': ' + str(exc))
    raise SystemExit(3)
for rel, sha in iter_scope_files('.'):
    print('MANIFEST ' + rel + ' ' + sha)
"""


def _import_provenance():
    """导入 ``src/ascend_opt/provenance.py``——部署戳的读写都走它。

    刻意**不在模块顶层导入**：这个脚本要能被单独加载做单元测试
    （见 ``tests/test_remote_run_pull.py``），顶层导入会在加载时就去摸 ``src/``。
    也刻意不在这里另写一份摘要算法：写端与读端一旦分家，戳在投放的**那一刻**
    就已经是过期的，而现场只是"git_commit 又是 NOT_MEASURED"。
    """
    root = Path(__file__).resolve().parent.parent
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from ascend_opt import provenance  # noqa: PLC0415  (刻意延迟导入)

    return provenance


def _pushed_covers_scope(landed: list[str], scope: tuple[str, ...]) -> list[str]:
    """返回摘要作用域里**没被本次推送覆盖**的条目（纯函数，便于测试）。"""
    out: list[str] = []
    for entry in scope:
        covered = False
        for arc in landed:
            a = arc.rstrip("/")
            if a == entry or entry.startswith(a + "/"):
                covered = True
                break
        if not covered:
            out.append(entry)
    return out


def parse_stamp_verification(output: str) -> dict | None:
    """从远端校验输出里取回 ``read_source_stamp`` 的结果；取不到返回 ``None``。

    **取不到必须是 None，不能是"看起来还行"的默认值。** 远端 python 起不来、
    远端源码是旧版没有 ``read_source_stamp``、命令被 shell 吃掉——这些都必须
    落成"校验未通过"，而不是"没看到问题所以没问题"。
    """
    for line in output.splitlines():
        if line.startswith(VERIFY_MARKER):
            try:
                data = json.loads(line[len(VERIFY_MARKER):])
            except Exception:
                return None
            return data if isinstance(data, dict) else None
    return None


def parse_manifest(output: str) -> dict[str, str] | None:
    """从远端清单输出里取回 ``{相对路径: sha256}``；一条都没有则返回 ``None``。"""
    rows: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith(MANIFEST_MARKER):
            continue
        rel, _, sha = line[len(MANIFEST_MARKER):].partition(" ")
        rel, sha = rel.strip(), sha.strip()
        if rel and sha:
            rows[rel] = sha
    return rows or None


def parse_verify_python(output: str) -> str:
    """取远端校验用的解释器（记录用；取不到返回空串）。"""
    for line in output.splitlines():
        if line.startswith(VERIFY_PY_MARKER):
            return line[len(VERIFY_PY_MARKER):].strip()
    return ""


def diagnose_digest_mismatch(
    client,
    remote_root: str,
    python: str,
    local_pairs: list[tuple[str, str]],
    limit: int = 20,
) -> None:
    """摘要不符时，逐文件定位到"哪些文件不同"。

    只在**失败路径**上跑：平时不必付这份开销，出事时最缺的恰恰是"哪不一样"。
    复用远端那份 ``iter_scope_files``，所以两边列的是同一套清单。
    """
    cmd = f"cd {shlex.quote(remote_root)} && {python} - <<'PY'\n{_MANIFEST_SCRIPT}PY"
    _, out, err = _exec_capture(client, cmd, timeout=180)
    remote = parse_manifest(out)
    if remote is None:
        print(
            "  （无法取得远端文件清单，不能定位差异——远端 provenance 可能是旧版，"
            "或远端 python 起不来）",
            file=sys.stderr,
        )
        if err.strip():
            print(f"  远端 stderr：{err.strip()[:800]}", file=sys.stderr)
        return

    local = dict(local_pairs)
    only_local = sorted(set(local) - set(remote))
    only_remote = sorted(set(remote) - set(local))
    differ = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])
    print(
        f"  差异定位：本机 {len(local)} 个文件 / 远端 {len(remote)} 个文件",
        file=sys.stderr,
    )
    for label, items in (
        ("仅本机有（远端缺，可能没推上去）", only_local),
        ("仅远端有（远端有改动或本地已删）", only_remote),
        ("同路径但内容不同", differ),
    ):
        if items:
            shown = items[:limit]
            more = f" …（共 {len(items)} 个）" if len(items) > limit else ""
            print(f"    {label}：{shown}{more}", file=sys.stderr)
    if not (only_local or only_remote or differ):
        # 逐文件全等却摘要不同 ⇒ 只可能是算法/作用域分家。
        # 这条分支**不该出现**；写出来是为了让它出现时不会被当成"又是一次摘要不符"。
        print(
            "    逐文件比较**完全一致**，但聚合摘要不同 ⇒ 两端摘要算法或作用域不一致"
            "（不是源码被改过）。",
            file=sys.stderr,
        )


def do_stamp(
    client,
    remote_root: str,
    repo_root: str | Path | None = None,
    *,
    allow_dirty: bool = False,
    verify_python: str = "python3",
) -> int:
    """把本机 ``HEAD`` + 源码树摘要写成部署戳投到远端源码树根，并当场校验。

    三步，任何一步不过都返回非零：
      1. 在**本机**生成戳（本机是 git 仓库，所以这里才有 commit 可记）；
      2. 投到 ``<远端根>/SOURCE_PROVENANCE.json``；
      3. 在**远端**用将要读它的那份代码调 ``read_source_stamp`` 重算摘要，
         要求 ``validated is True`` 且读回的 commit 与投出去的一致。

    第 3 步是整件事的关键：没有它，"戳投好了"只是一个未经证实的断言。

    退出码：4 = 生成不了可信戳；6 = 投放后校验不通过。
    """
    remote_root = _normalize_remote(remote_root)
    root = (
        Path(repo_root).resolve()
        if repo_root
        else Path(__file__).resolve().parent.parent
    )

    try:
        prov = _import_provenance()
    except Exception as exc:
        print(
            f"无法加载部署戳模块（{root / 'src' / 'ascend_opt' / 'provenance.py'}）："
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4

    try:
        stamp = prov.build_source_stamp(root, allow_dirty=allow_dirty)
    except prov.SourceStampError as exc:
        print(f"无法生成可信部署戳：\n{exc}", file=sys.stderr)
        return 4

    payload = json.dumps(stamp, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    remote_stamp = posixpath.join(remote_root, prov.SOURCE_STAMP_NAME)
    print(
        f"部署戳：commit={stamp['git_commit'][:12]} dirty={stamp['git_dirty']} "
        f"digest={stamp['tree_digest'][:12]}… -> {remote_stamp}"
    )

    with tempfile.TemporaryDirectory(prefix="_stamp_") as tmpdir:
        tmp = Path(tmpdir) / prov.SOURCE_STAMP_NAME
        # 显式 newline="\n"：Windows 上默认会把 \n 翻成 \r\n。
        # 戳是给人看、给 json.loads 读的，多出来的 \r 不影响解析，但会让同一份
        # 戳的字节数在两个平台上不同，而"字节不同"在这条线上就等于"来源不同"。
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        code = do_put(client, str(tmp), remote_stamp)
    if code != 0:
        return code

    cmd = (
        f"cd {shlex.quote(remote_root)} && {verify_python} - <<'PY'\n"
        f"{_VERIFY_SCRIPT}PY"
    )
    _, out, err = _exec_capture(client, cmd, timeout=180)
    if err.strip():
        print(f"远端校验 stderr：{err.strip()[:800]}", file=sys.stderr)

    seen = parse_stamp_verification(out)
    if seen is None:
        print(
            "投放后校验**未拿到结果**，因此不算通过。\n"
            f"  远端没有输出校验标记 {VERIFY_MARKER.strip()!r}。常见原因："
            f"远端 python 起不来、远端 src/ascend_opt 是旧版（没有 read_source_stamp）。\n"
            f"  ---- 远端输出 ----\n{out}",
            file=sys.stderr,
        )
        return 6

    py = parse_verify_python(out)
    if seen.get("validated") is not True:
        print(
            "投放后校验**未通过**（戳与远端源码树不一致），因此这次投放不算成功。\n"
            f"  原因：{seen.get('reason')}\n"
            f"  远端读到的：commit={seen.get('git_commit')!r} "
            f"deployed_at={seen.get('deployed_at')!r}\n"
            f"  校验解释器：{py or '(未报告)'}",
            file=sys.stderr,
        )
        diagnose_digest_mismatch(
            client, remote_root, verify_python, prov.iter_scope_files(root)
        )
        return 6

    if seen.get("git_commit") != stamp["git_commit"]:
        # 理论上不可达（摘要相符时读回的就是戳里的 commit）。写出来是因为
        # "不可达"若是真的不可达，这段代码永远不会执行；若哪天执行了，
        # 说明读端换了取值逻辑——那正是必须被人看到的事。
        print(
            f"投放后校验异常：远端读回的 commit {seen.get('git_commit')!r} 与投出的 "
            f"{stamp['git_commit']!r} 不一致（摘要却相符）。已按失败处理。",
            file=sys.stderr,
        )
        return 6

    if seen.get("git_dirty") is True:
        print(
            "  ⚠️ 戳里 git_dirty=true：本机工作区当时有未提交改动，"
            "因此 git_commit 只应被当作「这批代码的分支点」，不能当作它的版本号。"
        )
    print(
        f"投放并校验通过：远端以 {py or verify_python} 重算摘要相符，"
        f"读回 commit={seen['git_commit'][:12]}（来源 stamp）"
    )
    return 0


def _extract_archive(
    local_tar: Path, arc: str, dest_parent: Path
) -> tuple[dict[str, int], list[str]]:
    """解开 tar 到 ``dest_parent``，返回（剥掉 ``arc/`` 前缀的相对路径→大小, 布局不符项）。

    抽成独立函数是为了**可测**：这里曾经有一个会静默通过的缺陷——解包目标选错
    层级（解进落点目录本身，而不是它的父目录），于是数据实际落在
    ``<落点>/<arc>/...``，而校验按 tar 内部路径去查，结论是"每个文件都在、
    大小都对"。把布局判定抽出来，就能用合成 tar 直接证明这条判据会失败。

    布局不符时**不写任何文件**，避免在落点旁留下一堆"半对"的东西。
    """
    prefix = f"{arc}/"
    with tarfile.open(local_tar, "r:gz") as tar:
        want = [m for m in tar.getmembers() if m.isfile()]
        off = [m.name for m in want if not m.name.startswith(prefix)]
        stripped = {
            m.name[len(prefix):]: m.size
            for m in want
            if m.name.startswith(prefix)
        }
        if not off:
            tar.extractall(str(dest_parent), filter="data")
    return stripped, off


def do_pull(client, remote_dir: str, local_dir: str) -> int:
    """把远端一个目录整体取回本地（``push`` 的镜像）。

    为什么不用现成的 ``get``：paramiko 的 ``SFTPClient.get`` **不支持目录**，
    传目录会抛 ``IsADirectoryError``；而"先列文件再逐个 get"在成百上千个小文件上
    慢到不可用。这里让远端打包成**一个** tar 取回，再在本地解开。

    三项核对，缺一不可（"取回来了"和"取全了"是两件事）：
      * 取回的 tar 字节数与远端 ``stat`` 一致 —— 截断会静默给出"半个目录"；
      * 解出的每个文件**是否存在且大小与 tar 内记录一致** —— 解到错误层级不报错；
      * 远端临时 tar 用完即删 —— 别把打包产物留在证据目录旁边。
    """
    remote_dir = _normalize_remote(remote_dir).rstrip("/")
    parent = posixpath.dirname(remote_dir)
    arc = posixpath.basename(remote_dir)
    remote_tar = posixpath.join(parent, f"{arc}.tar")

    # 落点名必须与远端目录名**同名**（见 _extract_archive：解包按 tar 内成员名
    # 原样落地，成员顶层就是 arc，所以 dest_parent/arc 必须正好等于落点）。
    # 不同名时**先报错**，不要等打包取回、解开之后才由大小核对发现：
    # 那时数据其实已经写到 `<落点父目录>/<远端目录名>/` 了，报错只说
    # "这些文件缺失或大小不符"，完全看不出真正的原因是"名字不一致"。
    # 踩过：把远端 `_pull_b` 拉到本地 `results/_incoming_b`，报错信息与
    # 数据实际位置都指向别处，看起来像"拉取不完整"。
    local_root = Path(local_dir).resolve()
    if local_root.name != arc:
        print(
            f"落点目录名必须与远端目录名一致：远端 {arc!r} vs 落点 "
            f"{local_root.name!r}。\n"
            f"  pull 的落点语义是「远端目录的**内容**落到落点目录本身」，"
            f"而解包按 tar 内成员名原样落地，因此两者必须同名。\n"
            f"  改法：把落点改成 …/{arc}（或先 `mv {local_dir} "
            f"…/{arc}` 再拉），或把远端目录改名成 {local_root.name}。",
            file=sys.stderr,
        )
        return 5

    # 远端临时 tar 必须在**所有**路径上清掉，包括失败路径。
    # 踩过：路径参数带了尾随 '\r' 时 `tar czf X\r.tar ... X\r` 会**先建出空归档**
    # 再因为成员不存在而失败；而失败分支直接 return，于是 37 个体积为零、
    # 名字里带着 '\r' 的 .tar 留在了证据目录旁。它们还会被通配符统计进去，
    # 让"远端文件数"比"本地文件数"多出正好一个 run 数——看起来像拉取不完整，
    # 其实是被自己的临时文件污染的计数。
    try:
        code = do_exec(
            client,
            f"cd {parent} && tar czf '{arc}.tar' --exclude=__pycache__ "
            f"--exclude='*.pyc' '{arc}'",
            timeout=900,
        )
        if code != 0:
            print(f"远端打包失败（退出码 {code}）", file=sys.stderr)
            return code

        local_root = Path(local_dir)
        local_root.parent.mkdir(parents=True, exist_ok=True)
        local_root.mkdir(parents=True, exist_ok=True)
        # 临时 tar 放在**落点之外**：放里面的话，一旦清理失败就会混进证据目录。
        local_tar = local_root.parent / f"_{arc}.tar"

        sftp = client.open_sftp()
        try:
            expected = sftp.stat(remote_tar).st_size
            sftp.get(remote_tar, str(local_tar))
        finally:
            sftp.close()

        got = local_tar.stat().st_size
        if got != expected:
            print(
                f"取回的 tar 大小不一致：远端 {expected} != 本地 {got}"
                f"（截断会给出一个看起来正常的半个目录）",
                file=sys.stderr,
            )
            return 2

        # 落点语义：远端 remote_dir 的**内容**落到 local_dir。
        # tar 内成员的顶层名是 arc（= remote_dir 的 basename），所以解包目标是
        # local_dir 的**父目录**，解出来才正好是 local_dir/<内容>。
        # 踩过：往 local_dir 里解 → local_dir/<arc>/<内容>，多套一层；而当时的
        # 校验按 **tar 内部路径**去查，于是"每个文件都在、大小都对"全部通过——
        # 检查过了，数据却不在预期位置。布局判定见 _extract_archive。
        try:
            stripped, off = _extract_archive(local_tar, arc, local_root.parent)
        finally:
            local_tar.unlink(missing_ok=True)

        if off:
            print(
                f"tar 内成员的顶层不是 {arc!r}，落点语义不成立："
                f"{off[:5]}（共 {len(off)} 项）",
                file=sys.stderr,
            )
            return 4

        bad = [
            rel
            for rel, size in stripped.items()
            if not (local_root / rel).is_file()
            or (local_root / rel).stat().st_size != size
        ]
        if bad:
            print(f"解包后这些文件缺失或大小不符（相对 {local_root}）：{bad[:10]}",
                  file=sys.stderr)
            return 3

        print(
            f"pull OK  {remote_dir} -> {local_root}"
            f"  ({len(stripped)} 个文件, tar {got} bytes, 落点已按预期布局核对)"
        )
        return 0
    finally:
        do_exec(client, f"rm -f '{remote_tar}'", timeout=60)


def _normalize_remote(remote: str) -> str:
    """把远端路径归一化成 POSIX 绝对路径；无法判定为绝对路径时**报错退出**。

    两类本机特有的破坏必须先识破，否则会静默把文件写到错误的地方：

    1. MSYS（Git Bash）路径转换：把 ``/root/x`` 改写成
       ``<Git 安装前缀>/root/x``，于是文件落到 ``/Users/.../root/x``。
       产物里只多一行 "put OK"，看起来是一次成功的上传。
    2. 本机 ``Path`` 语义：``Path("/root/x").parent`` == ``"\\root"``。

    判据：远端路径必须是**以单个斜杠开头**的 POSIX 绝对路径。不满足就退出，
    绝不猜（猜错 = 静默写错位置；报错 = 一次可见的失败）。
    """
    original = remote

    # MSYS 路径转换的典型产物：单个盘符 + 冒号，如 C:/root/x 或 C:\root\x
    drive, sep, rest = remote.partition(":")
    if sep and len(drive) == 1 and drive.isalpha() and rest.startswith(("/", "\\")):
        remote = rest

    remote = remote.replace("\\", "/")
    # 折叠多余前导斜杠：POSIX 里 "//x" 是实现定义（可能被当成网络路径），
    # MSYS 环境下常用双斜杠来阻止路径转换，这里统一收敛成单斜杠绝对路径。
    if remote.startswith("/"):
        remote = "/" + remote.lstrip("/")

    if not remote.startswith("/"):
        print(
            f"远端路径不是绝对路径：{original!r} -> {remote!r}\n"
            f"  远端路径必须以 '/' 开头。若在 Git Bash 下传入 POSIX 路径被改写，"
            f"可加 MSYS_NO_PATHCONV=1 前缀。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # 识破 MSYS 已经改写过的形态：本机根本不存在的挂载点被人为造出来。
    # 这些前缀一旦出现，说明参数在进入本进程之前就已经被改写了。
    for bad in ("/Users/", "/c/Users/", "/mnt/c/", "/cygdrive/"):
        if original.startswith(bad) or remote.startswith(bad):
            print(
                f"远端路径像是被本机路径转换改写过：{original!r}\n"
                f"  这会把文件写到远端一个不该存在的位置，而命令仍然报成功。\n"
                f"  请在命令前加 MSYS_NO_PATHCONV=1，或用 '//root/...' 形态传入。",
                file=sys.stderr,
            )
            raise SystemExit(2)

    return posixpath.normpath(remote)


def _ensure_remote_dir(sftp, remote_dir: str) -> None:
    """在远端递归建目录（sftp 没有 mkdir -p）。"""
    if remote_dir in ("", ".", "/"):
        return
    # 逐级拼接用 posixpath，保证 /a/b 拆成 /a、/a/b 而不是 a、a/b
    parts = [p for p in remote_dir.split("/") if p]
    if not remote_dir.startswith("/"):
        print(f"远端目录必须是绝对路径：{remote_dir!r}", file=sys.stderr)
        raise SystemExit(2)
    cur = "/"
    for part in parts:
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="远程主机执行/传文件")
    sub = parser.add_subparsers(dest="action", required=True)

    p_exec = sub.add_parser("exec", help="在远端执行命令")
    p_exec.add_argument("command", help="要执行的 shell 命令")
    p_exec.add_argument("--timeout", type=float, default=None, help="秒；默认不限")

    p_put = sub.add_parser("put", help="上传文件")
    p_put.add_argument("local")
    p_put.add_argument("remote")

    p_get = sub.add_parser("get", help="下载文件")
    p_get.add_argument("remote")
    p_get.add_argument("local")

    p_push = sub.add_parser(
        "push", help="把多个本地路径打成 tar，一次连接上传并解开"
    )
    p_push.add_argument("remote_root", help="远端解包根目录（绝对路径）")
    p_push.add_argument(
        "locals", nargs="+", help="本地文件或目录，可多个"
    )
    p_push.set_defaults(stamp=True)  # 默认投放溯源戳；用 --no-stamp 关
    p_push.add_argument(
        "--no-stamp",
        dest="stamp",
        action="store_false",
        help="推送后**不**投放源码溯源戳（后续 run 的 git_commit 会是 NOT_MEASURED）",
    )
    p_push.add_argument(
        "--allow-dirty",
        action="store_true",
        help="本机工作区脏时也投放戳（戳里会记 git_dirty=true）",
    )
    p_push.add_argument(
        "--repo-root", default=None, help="本机源码树根；默认取本脚本所在的仓库"
    )
    p_push.add_argument(
        "--verify-python", default="python3", help="远端校验摘要用的解释器"
    )

    p_stamp = sub.add_parser(
        "stamp", help="投放并**当场校验**源码溯源戳（SOURCE_PROVENANCE.json）"
    )
    p_stamp.add_argument("remote_root", help="远端源码树根目录（绝对路径）")
    p_stamp.add_argument(
        "--repo-root", default=None, help="本机源码树根；默认取本脚本所在的仓库"
    )
    p_stamp.add_argument(
        "--allow-dirty",
        action="store_true",
        help="本机工作区脏时也投放（戳里会记 git_dirty=true）",
    )
    p_stamp.add_argument(
        "--verify-python", default="python3", help="远端校验摘要用的解释器"
    )

    p_pull = sub.add_parser(
        "pull", help="把远端一个目录打包取回本地并解开（push 的镜像）"
    )
    p_pull.add_argument("remote_dir", help="远端目录（绝对路径）")
    p_pull.add_argument("local_dir", help="本地落点目录")

    args = parser.parse_args(argv)
    creds = _load_credentials()

    try:
        client = _connect(creds)
    except Exception as exc:  # 连接类错误统一给出可读信息
        print(f"连接失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.action == "exec":
            return do_exec(client, args.command, args.timeout)
        if args.action == "put":
            return do_put(client, args.local, args.remote)
        if args.action == "get":
            return do_get(client, args.remote, args.local)
        if args.action == "push":
            return do_push(
                client,
                args.remote_root,
                args.locals,
                stamp=args.stamp,
                repo_root=args.repo_root,
                allow_dirty=args.allow_dirty,
                verify_python=args.verify_python,
            )
        if args.action == "stamp":
            return do_stamp(
                client,
                args.remote_root,
                args.repo_root,
                allow_dirty=args.allow_dirty,
                verify_python=args.verify_python,
            )
        if args.action == "pull":
            return do_pull(client, args.remote_dir, args.local_dir)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
