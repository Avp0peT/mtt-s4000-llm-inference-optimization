#!/usr/bin/env bash
# 环境自查 + 硬件指纹采集
# ---------------------------------------------------------------------------
# 为什么单独做成一个脚本：
#   1) 硬件指纹必须在 run **开始之前** 采集，否则测出来的可能是被 run 影响的状态
#      （频率、显存碎片、常驻进程）。
#   2) 环境信息必须落成机器可读的 JSON，才能写进 meta.json 参与后续可比性判断。
#      「我记得当时是 CANN 8.x」不是可复现性。
#
# ★ 在哪台机器上跑：
#   必须在**实际执行推理的那台主机**上运行。正文 "Remote Execution
#   Environment" 一节规定权威推理环境是远程主机（研究主体平台 MTT S4000，
#   次平台昇腾），本机不是推理环境。
#   在开发机上跑本脚本是**允许**的（用于确认"这里确实没有加速卡"），
#   但它产出的指纹不能代表远程主机——指纹描述的是采集它的那台机器。
#   产物里的 discovery.looks_like_*_host 与 hostname 就是用来区分这一点的。
#
# 用法：
#   bash scripts/inspect_env.sh [输出目录]
# 默认输出到 results/raw/_env/ ，该目录被所有 run 共享为环境指纹来源。
# ---------------------------------------------------------------------------
set -u

ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
OUT_DIR="${1:-${ROOT:-.}/results/raw/_env}"
mkdir -p "$OUT_DIR"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    PY="python"
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "错误：未找到 python3/python，无法生成 JSON 指纹。" >&2
    exit 1
fi

echo "采集环境指纹 -> $OUT_DIR"

if ! "$PY" - "$OUT_DIR" "${ROOT:-}" <<'PYEOF'
"""生成 env.json 与 env.txt。

原则：**采集失败要记录为失败**，不能用空值冒充「查到了但为空」。
每个字段要么有真实来源，要么带 note 说明为什么取不到。
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def normalize_path(raw: str) -> Path:
    """把 shell 传来的路径转成 Python 能正确解析的形式。

    背景（踩过一次）：Git Bash 会把工作目录表示为 /c/Users/...，
    交给 Windows 上的 Python 会被解析为「当前盘符根下的 c\\Users\\...」，
    即 C:\\c\\Users\\...。文件确实写成功了，但写到了完全不相干的位置，
    而且没有任何报错——这类"静默写错路径"必须显式纠正。
    """
    if os.name == "nt" and len(raw) > 3 and raw[0] == "/" and raw[2] == "/" and raw[1].isalpha():
        return Path(f"{raw[1].upper()}:/{raw[3:]}")
    return Path(raw)


out_dir = normalize_path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

root = normalize_path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None


def run(cmd, timeout=20):
    """执行命令并返回 (ok, stdout)。缺失的命令与超时都返回 ok=False。"""
    if shutil.which(cmd[0]) is None:
        return False, f"{cmd[0]} not found in PATH"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(cmd)} timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


env = {
    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": platform.node(),
    "os": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "kernel": platform.platform(),
    },
    "cpu": {
        "logical_cores": os.cpu_count(),
        "model": platform.processor(),
    },
}

# ---- 软件栈 ---------------------------------------------------------------
software = {"python": sys.version.split()[0], "executable": sys.executable}
# 两个平台的运行时插件都尝试导入，**不做"只导配置里声明的那一个"的优化**：
# 导入失败本身是有价值的信息（"这台机器上没装 torch_musa"），而按声明去挑
# 就永远发现不了"配置说是 MTT 机器、实际装的是另一套栈"。
#
# vLLM-MUSA 的 python 包名无法从本项目这一侧证实，因此**不**猜：
# 走 VLLM_MUSA_VERSION 环境变量，或由 VLLM_VERSION 的构建串体现。
# 猜一个包名会让"没采到"看起来像"导入失败"，两者的修法完全不同。
_MODULE_PROBES = (
    "torch",
    "torch_npu",
    "torch_musa",
    "vllm",
    "vllm_ascend",
    "transformers",
)
for mod in _MODULE_PROBES:
    try:
        module = __import__(mod)
        software[f"{mod}_version"] = getattr(module, "__version__", "unknown")
        if mod == "torch":
            software["torch_version_cuda_field"] = getattr(module.version, "cuda", None)
            # torch_npu 会让 torch 报告 npu 后端；不导入 torch_npu 时这里是 None，
            # 因此「torch_npu 已导入但 npu 不可用」是可区分的故障态。
            software["torch_npu_available"] = bool(
                getattr(module, "npu", None) and module.npu.is_available()
            ) if hasattr(module, "npu") else None
    except ImportError:
        software[f"{mod}_version"] = None
    except Exception as exc:  # 导入报错本身是有价值的信息（多为环境冲突）
        software[f"{mod}_version"] = f"IMPORT_ERROR: {type(exc).__name__}: {exc}"
env["software"] = software

ascend_vars = {
    k: v
    for k, v in os.environ.items()
    if k.startswith(("ASCEND", "NPU", "HCCL", "RANK", "ATB", "VLLM", "MODEL"))
}
env["env_vars"] = ascend_vars

# ---- CANN / 驱动 ----------------------------------------------------------
cann = {"paths_checked": [], "version_files": {}}
for candidate in (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/latest",
    os.environ.get("ASCEND_HOME_PATH", ""),
):
    if not candidate:
        continue
    path = Path(candidate)
    cann["paths_checked"].append(str(path))
    if not path.exists():
        continue
    cann["home"] = str(path)
    for fname in ("ascend_toolkit_install.info", "version.cfg", "ascend_toolkit_version.info"):
        fpath = path / fname
        if fpath.exists():
            cann["version_files"][fname] = fpath.read_text(
                encoding="utf-8", errors="replace"
            ).strip()[:2000]
cann["ok"], cann["probe"] = run(["npu-smi", "info", "-t", "board"])
env["cann"] = cann

# ---- 设备侧 ---------------------------------------------------------------
# 两个平台的探测**各跑一次、都要跑**。为什么不是"只探配置里声明的那个平台"：
# 探测结果是**证据**，不是配置的镜子。只探声明的平台，就永远发现不了
# "配置说在这台机器上跑、指纹却是另一台机器的"——而那正是要堵的空子。
ascend_ok, npu_info = run(["npu-smi", "info"], timeout=30)
ascend_topo_ok, ascend_topo = run(["npu-smi", "info", "-t", "topo"], timeout=30)
mtt_ok, mtt_info = run(["mthreads-gmi"], timeout=30)

env["device"] = {
    # 昇腾侧：键名保持旧样子——老读端按这两个键判断昇腾可用性，改名会让它们静默失效。
    "npu_smi_available": ascend_ok,
    "npu_smi_info": npu_info if ascend_ok else None,
    "npu_smi_error": None if ascend_ok else npu_info,
    "topology": ascend_topo if ascend_topo_ok else None,
    # MTT 侧：与昇腾侧对称，但**刻意不复用 npu_* 键名**——复用会让
    # "到底是哪个工具报的"在下游看不出来，两个平台也就无法区分了。
    "mthreads_gmi_available": mtt_ok,
    "mthreads_gmi_info": mtt_info if mtt_ok else None,
    "mthreads_gmi_error": None if mtt_ok else mtt_info,
    # 规范化后的型号/数量，在下面的解析块里填。读端优先用这两个键，
    # 不必自己判断"该看哪个平台"。
    "model": None,
    "count": None,
}

env["memory"] = {}
ok_mem, meminfo = run(["free", "-m"])
if ok_mem:
    env["memory"]["free_m"] = meminfo


# ---- 规范点名的 12 项环境事实 ----------------------------------------------
# 正文 "Remote Execution Environment" 一节要求这 12 项**在跑推理的那台机器上**
# 发现（本项目为远程 MTT S4000 主机；次平台为昇腾主机），而不是从文档里抄、
# 也不是本地假设一个。键名沿用既有的 ASCEND_* 拼写：那只是**实现细节**，
# 契约是"这 12 项"本身；改名会让老读端静默失效，故不在本轮改。
#
# 实现要点：
#   * 设备型号/数量复用 ascend_opt.provenance.parse_npu_smi，避免这里再写一份
#     易碎的 npu-smi 解析（两份解析器迟早会不一致）。
#   * 模型相关的 4 项来自远程主机的环境变量。取不到就写 None 并注明
#     「未声明」，不猜一个路径——猜出来的模型路径正是规范明确禁止的事。
#   * 每一项都记 source：没有来源的字段等于没有约束力。
DISCOVERY_KEYS = (
    "ASCEND_DEVICE_MODEL",
    "ASCEND_DEVICE_COUNT",
    "ASCEND_DRIVER_VERSION",
    "CANN_VERSION",
    "TORCH_VERSION",
    "TORCH_NPU_VERSION",
    "VLLM_VERSION",
    "VLLM_ASCEND_VERSION",
    "MODEL_PATH",
    "MODEL_NAME",
    "MODEL_REVISION",
    "MODEL_PRECISION",
)

# ---- MTT（研究主体平台）侧的软件版本 ----------------------------------------
# 上面那 12 项只覆盖了昇腾侧的驱动/SDK/框架插件。研究主体平台换成 MTT S4000 之后，
# 同样三类事实必须从 MTT 侧采到，否则"支持 MTT"就只是一句声明：
#   驱动     -> MTT_DRIVER_VERSION
#   SDK      -> MUSA_VERSION
#   框架插件 -> TORCH_MUSA_VERSION / VLLM_MUSA_VERSION
#
# 与 12 项分开列，而不是塞进上面那张表：那 12 项是规范点名的契约，
# 混进来会让"漏了几项"变得数不清楚。这里的是**补充**，缺了不会让标签降级。
MTT_DISCOVERY_KEYS = (
    "MTT_DRIVER_VERSION",
    "MUSA_VERSION",
    "TORCH_MUSA_VERSION",
    "VLLM_MUSA_VERSION",
)

#: MUSA SDK 版本文件候选。与 CANN 的处理方式一致：先看环境变量，再看文件。
#:
#: ★ ``version.json`` 不能漏。实测踩到：MUSA 3.1.0 装的是
#:   ``/usr/local/musa/version.json``，而这里原先只列了 ``version`` 与
#:   ``version.txt``——文件全部找不到，MUSA 版本就静默变成 ``None``，
#:   与"这台机器根本没装 MUSA"在产物里**完全无法区分**。
MUSA_VERSION_FILES = (
    "{MUSA_HOME}/version",
    "{MUSA_HOME}/version.json",
    "/usr/local/musa/version",
    "/usr/local/musa/version.txt",
    "/usr/local/musa/version.json",
)


def field_after(text, label):
    """从 npu-smi 的 `Key : Value` 文本里取值。取不到返回 None。"""
    m = re.search(rf"^{re.escape(label)}\s*:\s*(\S.*?)\s*$", text or "", re.M)
    return m.group(1) if m else None


def first_env(names):
    """按优先级返回第一个已设置的环境变量，附带它的名字。"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip(), f"env:{name}"
    return None, None


def _import_provenance():
    """导入包内的解析/判定工具。导入失败返回 None（降级为不可用，而不是瞎解）。

    刻意只写这一处导入：解析与判定逻辑必须**只有一份**实现。
    在这里再抄一份易碎的解析器，两份迟早不一致——而不一致的那一份
    会以"数字看起来正常"的方式出错。
    """
    if root is None:
        return None
    src = root / "src"
    if not (src / "ascend_opt").exists():
        return None
    try:
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from ascend_opt import provenance as _prov

        return _prov
    except Exception:  # noqa: BLE001, pragma: no cover
        return None


def parse_npu_info(text):
    """借包内的解析器解析 npu-smi 表。导入失败时降级为不可用（而不是瞎解）。"""
    prov = _import_provenance()
    if prov is None:
        return None
    try:
        return prov.parse_npu_smi(text)
    except Exception:  # noqa: BLE001, pragma: no cover
        return None


prov = _import_provenance()
if prov is None:
    print("  警告：无法导入 ascend_opt.provenance，设备解析将降级为不可用。", file=sys.stderr)


parsed = parse_npu_info(npu_info if ascend_ok else None)
board = cann.get("probe") if cann.get("ok") else None

# ---- 把探测结果压成"平台 + 可用性" ----------------------------------------
# 关键设计：可用性来自**探测认出了设备**，不来自"命令退出码为 0"。
# 退出码只说明工具能跑；一个能跑却认不出任何设备的探测，不能证明这台机器
# 能产出实测数据。这条比旧写法严，而且只会往**收紧**方向改（把不确定写成
# 未知），不会把一个不存在的硬件说成存在。
def _probe_result(tool_present, count, has_name=False):
    """单次探测的结论。判据**只有一份**实现，在 ``ascend_opt.provenance``。

    这里刻意不自己写判据——本项目已经吃过一次亏：解析端判"数不出卡数就是未知"，
    采集端却判"有产品名就算可用"，两边不一致。于是同一句
    ``mthreads-gmi 2.7.0 supports MTT S4000``：解析器返回 ``count=None``（未知），
    采集判定却给出 ``available=True``（可用）。两套判据迟早会给出不同答案，
    而不一致的那一份会以"数字看起来正常"的方式出错。

    导入失败时返回 ``None`` 而不是自己判：宁可"未知"（会触发重新采集），
    也不要一个可能是错的"可用"。
    """
    if prov is None:
        return None
    return prov.device_probe_result(tool_present, count, bool(has_name))


ascend_model = field_after(board, "Chip Name") or (parsed or {}).get("model") or None
ascend_count = (parsed or {}).get("count")
if ascend_count is None and board:
    # 表解析不出来时回落到数 NPU ID 行；仍数不出就是 None，不写 0。
    hits = len(re.findall(r"^NPU\s+ID\s*:", board, re.M))
    ascend_count = hits or None

mtt_parsed = (
    prov.parse_mthreads_gmi(mtt_info)
    if (prov is not None and mtt_ok)
    else {"models": [], "model": "", "count": None}
)

ascend_tool = shutil.which("npu-smi") is not None
mtt_tool = shutil.which("mthreads-gmi") is not None

env["probes"] = {
    "ascend": {
        "tool": "npu-smi",
        "tool_present": ascend_tool,
        "ok": ascend_ok,
        "device_count": ascend_count,
        "model": ascend_model,
        "error": None if ascend_ok else npu_info,
        "result": _probe_result(ascend_tool, ascend_count, bool(ascend_model)),
    },
    "mthreads": {
        "tool": "mthreads-gmi",
        "tool_present": mtt_tool,
        "ok": mtt_ok,
        "device_count": mtt_parsed["count"],
        "model": mtt_parsed["model"] or None,
        "error": None if mtt_ok else mtt_info,
        "result": _probe_result(mtt_tool, mtt_parsed["count"], bool(mtt_parsed["models"])),
    },
}

_ascend_r = env["probes"]["ascend"]["result"]
_mtt_r = env["probes"]["mthreads"]["result"]
env["available"] = (
    prov.normalize_availability([_ascend_r, _mtt_r])
    if prov is not None
    else (_ascend_r if _ascend_r is not None else _mtt_r)
)
env["available_reason"] = (
    f"npu-smi -> {_ascend_r}；mthreads-gmi -> {_mtt_r}"
    f"（合成规则：任一认出设备即可用；有认不出的则为未知；否则不可用）"
)

# 平台取**唯一**一个明确认出设备的平台。两个都认出（异构机）或都没认出 -> unknown：
# 那时没有唯一答案，按"未声明"处理比随便挑一个安全——挑错会把下游按平台筛选的
# 结果直接带偏，而且不报错。
_detected = [
    name for name, r in (("ascend", _ascend_r), ("mthreads", _mtt_r)) if r is True
]
env["platform"] = _detected[0] if len(_detected) == 1 else "unknown"

env["device"]["model"] = ascend_model or (mtt_parsed["model"] or None)
env["device"]["count"] = (
    ascend_count if ascend_count is not None else mtt_parsed["count"]
)

# 指纹格式版本。有它，读端才能区分"新格式没写某字段"与"旧格式压根没这个字段"，
# 而不是靠猜键名。
env["schema_version"] = 2

cann_version = None
cann_version_source = None
for fname, content in cann.get("version_files", {}).items():
    m = re.search(r"^\s*version\s*=\s*(\S+)", content, re.M)
    if m:
        cann_version = m.group(1)
        cann_version_source = f"cann:{fname}"
        break
if cann_version is None:
    cann_version, cann_version_source = first_env(("ASCEND_TOOLKIT_VERSION", "CANN_VERSION"))

model_path, model_path_src = first_env(
    ("MODEL_PATH", "ASCEND_MODEL_PATH", "MODEL_DIR", "MODEL_WEIGHTS")
)
model_name, model_name_src = first_env(("MODEL_NAME", "MODEL_ID", "MODEL"))
model_rev, model_rev_src = first_env(("MODEL_REVISION", "MODEL_REV"))
model_prec, model_prec_src = first_env(("MODEL_PRECISION", "MODEL_DTYPE", "TORCH_DTYPE"))

# 型号/数量已在探测块里算好（那里顺带覆盖了 MTT），这里只补来源标注：
# 没有来源的字段等于没有约束力。
device_model = ascend_model or (mtt_parsed["model"] or None)
if field_after(board, "Chip Name"):
    device_model_src = "npu-smi:Chip Name"
elif (parsed or {}).get("model"):
    device_model_src = "npu-smi:device-table"
elif mtt_parsed["model"]:
    device_model_src = "mthreads-gmi:device-table"
else:
    device_model_src = None
device_count = ascend_count if ascend_count is not None else mtt_parsed["count"]
device_count_src = (
    "npu-smi:device-table"
    if ascend_count is not None
    else ("mthreads-gmi:device-table" if mtt_parsed["count"] is not None else None)
)

values = {
    "ASCEND_DEVICE_MODEL": device_model,
    "ASCEND_DEVICE_COUNT": device_count,
    "ASCEND_DRIVER_VERSION": field_after(board, "Software Version"),
    "CANN_VERSION": cann_version,
    "TORCH_VERSION": software.get("torch_version"),
    "TORCH_NPU_VERSION": software.get("torch_npu_version"),
    "VLLM_VERSION": software.get("vllm_version"),
    "VLLM_ASCEND_VERSION": software.get("vllm_ascend_version"),
    "MODEL_PATH": model_path,
    "MODEL_NAME": model_name,
    "MODEL_REVISION": model_rev,
    "MODEL_PRECISION": model_prec,
}
sources = {
    "ASCEND_DEVICE_MODEL": device_model_src,
    "ASCEND_DEVICE_COUNT": device_count_src,
    "ASCEND_DRIVER_VERSION": "npu-smi:Software Version" if field_after(board, "Software Version") else None,
    "CANN_VERSION": cann_version_source,
    "TORCH_VERSION": "python:import torch" if software.get("torch_version") else None,
    "TORCH_NPU_VERSION": "python:import torch_npu" if software.get("torch_npu_version") else None,
    "VLLM_VERSION": "python:import vllm" if software.get("vllm_version") else None,
    "VLLM_ASCEND_VERSION": "python:import vllm_ascend" if software.get("vllm_ascend_version") else None,
    "MODEL_PATH": model_path_src,
    "MODEL_NAME": model_name_src,
    "MODEL_REVISION": model_rev_src,
    "MODEL_PRECISION": model_prec_src,
}

# ---- MTT 侧软件版本 --------------------------------------------------------
# 三个来源按可靠性排序：显式环境变量 > MUSA 版本文件 > mthreads-gmi 的打印。
# 每一步取不到就落到下一步，最后仍取不到就写 None 并注明"未采集"——
# 编一个 "unknown" 会让它看起来像一次成功的采集，在交叉核对驱动版本时误事。
mtt_soft = (
    prov.parse_mthreads_software(mtt_info)
    if (prov is not None and mtt_ok)
    else {"driver_version": None, "musa_version": None}
)

mtt_driver, mtt_driver_src = first_env(("MTT_DRIVER_VERSION", "MTHREADS_DRIVER_VERSION"))
if mtt_driver is None and mtt_soft["driver_version"]:
    mtt_driver, mtt_driver_src = mtt_soft["driver_version"], "mthreads-gmi:version-line"

musa_sdk, musa_sdk_src = first_env(("MUSA_VERSION", "MUSA_SDK_VERSION"))
# prov 为 None 时**不自己写一份解析**：解析只有一份实现（ascend_opt.provenance），
# 两份迟早不一致；导入失败时上面已经警告过，这里宁可留 None。
if musa_sdk is None and prov is not None:
    musa_home = os.environ.get("MUSA_HOME", "")
    for pattern in MUSA_VERSION_FILES:
        if "{MUSA_HOME}" in pattern:
            # MUSA_HOME 未设时 .format 会把它变成 "/version" 这种以根目录开头的
            # 假路径——产物里看起来像"查过了"，其实是查了个不存在的地方。
            if not musa_home:
                continue
            candidate = Path(pattern.format(MUSA_HOME=musa_home))
        else:
            candidate = Path(pattern)
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        version = prov.parse_musa_version_file(text)
        if version:
            musa_sdk, musa_sdk_src = version, f"file:{candidate}"
            break
if musa_sdk is None and mtt_soft["musa_version"]:
    musa_sdk, musa_sdk_src = mtt_soft["musa_version"], "mthreads-gmi:musa-version-line"

vllm_musa, vllm_musa_src = first_env(("VLLM_MUSA_VERSION",))

values.update({
    "MTT_DRIVER_VERSION": mtt_driver,
    "MUSA_VERSION": musa_sdk,
    "TORCH_MUSA_VERSION": software.get("torch_musa_version"),
    "VLLM_MUSA_VERSION": vllm_musa,
})
sources.update({
    "MTT_DRIVER_VERSION": mtt_driver_src,
    "MUSA_VERSION": musa_sdk_src,
    "TORCH_MUSA_VERSION": "python:import torch_musa" if software.get("torch_musa_version") else None,
    "VLLM_MUSA_VERSION": vllm_musa_src,
})

missing = [k for k in DISCOVERY_KEYS if values[k] in (None, "")]
mtt_missing = [k for k in MTT_DISCOVERY_KEYS if values[k] in (None, "")]

# 「这台机器像不像权威推理环境」——一个**启发式**判断，不是证明。
# 直接复用探测块的结论，不再单独写一套判据：两套判据迟早会给出不同答案，
# 而"指纹文件里两个字段互相矛盾"是最难被下游发现的那种错。
looks_like_ascend_host = env["probes"]["ascend"]["result"]

env["discovery"] = {
    "hostname": env["hostname"],
    "looks_like_ascend_host": looks_like_ascend_host,
    "looks_like_mthreads_host": env["probes"]["mthreads"]["result"],
    "detected_platform": env["platform"],
    "values": values,
    "sources": sources,
    "missing": missing,
    # MTT 侧补充项单独一块：它们缺了**不会**让标签降级，因此不能与上面那 12 项
    # 混在一个 missing 列表里——混进去会让"规范要求的 12 项齐不齐"这件事
    # 变得无法一眼判断。
    "mtt_missing": mtt_missing,
    "note": (
        "本块描述的是**采集它的这台机器**。规范要求这些事实从执行推理的远程主机"
        "发现，因此在开发机上采集会得到一串 null——那不是「没有信息」，"
        "那是在说「这台机器不是推理环境」。"
    ),
}

# ---- 落盘 -----------------------------------------------------------------
json_path = out_dir / "env.json"
# 先写临时文件再原子替换。直接写目标文件时，中途异常会留下一个**被截断的
# env.json**——它仍是一份能解析的 JSON，只是内容不全，下游会静默地用错。
tmp_path = json_path.with_name("env.json.tmp")
tmp_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp_path, json_path)

# 同时留一份人读的文本副本，便于人工核对（机器读 JSON，人读这个）。
# 头条放在最前面：人最需要先看到的是"这是哪台机器、有没有卡"，
# 而不是一屏 OS 细节。
text_parts = [
    f"collected_at: {env['collected_at']}",
    f"hostname: {env['hostname']}",
    f"schema_version: {env['schema_version']}",
    f"platform: {env['platform']}",
    f"available: {env['available']}",
    f"available_reason: {env['available_reason']}",
]
for section in ("probes", "os", "software", "cann", "device", "discovery"):
    text_parts.append(f"\n## {section}")
    text_parts.append(json.dumps(env[section], ensure_ascii=False, indent=2))
(out_dir / "env.txt").write_text("\n".join(text_parts), encoding="utf-8")

print(f"  env.json  -> {json_path}")
print(f"  env.txt   -> {out_dir / 'env.txt'}")
print(f"  platform={env['platform']}  available={env['available']}")
print(f"  {env['available_reason']}")

# 三个态的措辞必须分开——它们要做的下一步动作不同：
# 可用 -> 可以用；不可用 -> 这台机器就没有该硬件；未知 -> 去修探测或换机器重采。
if env["available"] is True:
    print(f"  检出可用设备（platform={env['platform']}），指纹可作为权威环境指纹。")
    print("  注意：指纹描述的是**采集它的这台机器**，与配置里声明的平台必须一致。")
elif env["available"] is False:
    print("  未检出可用的加速设备：该指纹只能证明「此处无硬件」，不能代表远程环境。")
    print("        在无加速卡的开发机上这是预期的；在真机上出现则必须先修环境。")
else:
    print("  无法判定硬件是否可用（探测跑了但认不出设备）。")
    print("        这不是「没有硬件」，别按缺失处理——请检查上面的 probes 段。")
if missing:
    print(f"  discovery 缺失 {len(missing)}/{len(DISCOVERY_KEYS)} 项：" + ", ".join(missing))
    print("        模型相关的 4 项需在远程主机上通过环境变量显式声明。")
PYEOF
then
    # 采集失败必须**大声失败**。原来的写法在这里继续往下走、还打印"完成。"，
    # 结果是一份没被生成的指纹被当成"已采集"——下游会读到更早的旧文件，
    # 或者读到缺失，两种情况都会让 run 的环境归属变错而没有任何报错。
    echo "错误：环境指纹采集失败（原因见上方 Python 报错）。" >&2
    echo "      未生成可信的 env.json。若该目录下已有旧文件，它是上一次的指纹，" >&2
    echo "      不代表当前环境——请修复后重跑本脚本。" >&2
    exit 1
fi

# 上一次采集的副本：保留历史指纹，便于发现「环境在中途被改过」
if [ -f "$OUT_DIR/env.json" ]; then
    cp "$OUT_DIR/env.json" "$OUT_DIR/env.$(date +%Y%m%d-%H%M%S).json" 2>/dev/null || true
fi

echo "完成。"
