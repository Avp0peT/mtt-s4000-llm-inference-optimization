#!/bin/bash
# 恢复/重建远端 MUSA 推理环境。
#
# 为什么需要它
# ------------
# 2026-09-24 晚换镜像后实测：**新镜像只带 base 栈，推理环境与模型缓存都没了**——
#   · `/root/musa-envs/qwen310` 不存在（故文档里"必须先 source activate_musa.sh"无法执行）
#   · `transformers` 未安装
#   · `/root/autodl-tmp/hf/hub` 不存在
# 而 base 栈本身与 2026-09-22 记录**完全一致**（driver 2.7.0 / MUSA 3.1.0 /
# torch 2.2.0 / torch_musa 1.3.0+81caf0a / python 3.10.8），因此**不需要也不允许**
# 重装或升级驱动与 torch——只把上层补回来。配方取自 `README.md` §9 与
# `RESEARCH_LOG.md` 2026-09-22 条目（版本号逐项照抄，不"顺手升级"）。
#
# 本脚本做什么、不做什么
# ----------------------
# **做**：建 `venv --system-site-packages`、生成 `activate_musa.sh`、装四个固定版本的
#         模型依赖、并**当场按会踩坑的导入顺序**验证。
# **不做**：不动 base 环境、不装/升级 torch 与驱动、不下载模型权重（那是
#         `results/raw/_probe/20260922/scripts/download_7b.sh` 一类的事，体量大、
#         且要按实验需要决定取哪个模型）。
#
# ⚠️ `LD_PRELOAD` 不是预防性添加（RESEARCH_LOG 2026-09-22 §2）：
#    conda 自带的 libstdc++ 是 6.0.29，**缺 `GLIBCXX_3.4.30`**；`tokenizers` 的扩展
#    `.so` 没有 RUNPATH，会继承 conda python 的 RPATH 把旧的那份先载进进程，
#    同名 soname 一旦载入即被复用 ⇒ 之后 torch 再 dlopen 只能拿到旧的 ⇒ 崩。
#    本脚本会**复核这个前提**（见第 4 步），不盲抄结论。
#
# 用法（远端）：bash /root/setup_musa_env.sh
# 幂等：重复跑只做校验，不重复装。

set -u

VENV=/root/musa-envs/qwen310
ACT="$VENV/activate_musa.sh"
HF_HOME_DIR=/root/autodl-tmp/hf
RC=0

echo "===== 1. venv（--system-site-packages，保留下层 torch / torch_musa / MUSA）====="
mkdir -p /root/musa-envs
if [ -x "$VENV/bin/python" ]; then
    echo "[skip] 已存在：$VENV"
else
    python3 -m venv --system-site-packages "$VENV" || RC=1
    echo "[ok] 建好 $VENV"
fi
echo "解释器：$("$VENV/bin/python" -VV 2>&1)"

echo
echo "===== 2. activate_musa.sh（**逐字照抄既有版本**，不重新发明）====="
# ⚠️ 下面这段是 2026-09-22 那份 `activate_musa.sh` 的**原文**（本机副本
#    `C:\tmp\remote_scripts\activate_musa.sh`，2202 B）。不要"顺手改写"它：
#    第一版我按文档描述合成了一份，结果与原文在几处有实质差异——
#    `MODELS_DIR` 没设、`LD_LIBRARY_PATH` 的拼法会多出一个尾冒号、少了 echo 行。
#    ⇒ **"按文档描述恢复"不等于"恢复成原来那份"**；有原文就照抄原文。
cat > "$ACT" <<'EOF'
# MTT 推理环境激活脚本（source 使用）
# ---------------------------------------------------------------------------
#   source /root/musa-envs/qwen310/activate_musa.sh
#
# 为什么需要 LD_PRELOAD（**实测结论，不是预防性添加**）：
#   1. conda 自带的 libstdc++ 是 6.0.29，**不含 GLIBCXX_3.4.30**；
#      而 torch 的 libtorch_python.so 需要 GLIBCXX_3.4.30。
#   2. conda 的 python 带 RPATH `$ORIGIN/../lib`（= /root/miniconda3/lib）。
#      自己没有 RUNPATH 的扩展库会继承这个 RPATH，把 conda 那份旧的先载进进程；
#      实测 `tokenizers` 就是这种（它的 .so 无 RUNPATH）。
#   3. 同名 soname 一旦载入即被复用，之后 torch 再 dlopen 只能拿到那份旧的 -> 失败。
#      而 torch 自己的 .so 带 RUNPATH（$ORIGIN:/usr/local/musa/lib），
#      这些目录里没有 libstdc++，于是它会落到系统那份（6.0.30，正确）。
#   所以现象是**依赖导入顺序**：先 import torch 正常，先 import tokenizers 就崩。
#   把系统那份预先载入，就与顺序无关了。
#
#   受控实验（/proc/self/maps 实测实际载入的是哪一个）：
#     torch 单独               -> /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  OK
#     tokenizers -> torch      -> /root/miniconda3/lib/libstdc++.so.6.0.29       FAIL
#     两者 + LD_PRELOAD 系统版 -> /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  OK
# ---------------------------------------------------------------------------

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# huggingface.co 在本机不可达（curl 000），hf-mirror.com 可达（200）。
export HF_ENDPOINT=https://hf-mirror.com

# 权重放数据盘（/root 在 30G overlay 上，/root/autodl-tmp 是独立 50G 卷）。
export HF_HOME=/root/autodl-tmp/hf
export MODELS_DIR=/root/autodl-tmp/models

# ---- 模型溯源（2026-09-27 补）-----------------------------------------------
# `scripts/inspect_env.sh` 的 discovery 会读这四个（见 `MODEL_*` 前缀表），
# 缺了 `available` 判据不受影响、但**模型溯源会缺**。
# 取值**实测自远端**，不是按配置文件抄：
#   REVISION 与 `configs/mthreads/phase1_s6/*.yaml` 的 `runtime.model_revision`
#   逐字比对过（两者都是 a09a3545…）——**声明与实际必须一致**，
#   否则溯源声明本身就成了一个错误的自我认知。
# ⚠️ `MODEL_PATH` 指向 **HF hub 缓存根**（不是 snapshot 目录）：
#    HF 的 `from_pretrained(repo_id)` 路径是从 `HF_HOME/hub` 派的，
#    写 snapshot 子目录会与运行期实际用的路径**不是同一个**。
export MODEL_PATH=/root/autodl-tmp/hf/hub/models--Qwen--Qwen2.5-7B-Instruct
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
export MODEL_PRECISION=fp16

export PATH=/root/musa-envs/qwen310/bin:$PATH
export MUSA_HOME=/usr/local/musa

echo "[activate_musa] venv=/root/musa-envs/qwen310"
echo "[activate_musa] HF_ENDPOINT=$HF_ENDPOINT  HF_HOME=$HF_HOME"
echo "[activate_musa] LD_PRELOAD=$LD_PRELOAD"
echo "[activate_musa] MODEL_NAME=$MODEL_NAME  MODEL_REVISION=$MODEL_REVISION"
EOF
echo "[ok] 写入 $ACT"
mkdir -p "$HF_HOME_DIR" /root/autodl-tmp/models

echo
echo "===== 3. 模型依赖（版本照抄 RESEARCH_LOG 2026-09-22，不升级）====="
"$VENV/bin/pip" install -q --upgrade pip > /tmp/pip_bootstrap.log 2>&1
tail -2 /tmp/pip_bootstrap.log
# ⚠️ 不用 `pip install ... | tail`：管道的退出码是 tail 的（恒 0），
#    那样 `if !` 永远为假、回退分支**永远不会执行**——静默失效。
if "$VENV/bin/pip" install -q transformers==4.46.3 tokenizers==0.20.3 \
        accelerate==1.1.1 safetensors==0.4.5 > /tmp/pip_main.log 2>&1; then
    echo "[ok] 默认源安装成功"
else
    echo "[warn] 默认源失败，改用清华镜像重试。默认源日志尾部："
    tail -5 /tmp/pip_main.log
    if "$VENV/bin/pip" install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
            transformers==4.46.3 tokenizers==0.20.3 accelerate==1.1.1 safetensors==0.4.5 \
            > /tmp/pip_mirror.log 2>&1; then
        echo "[ok] 镜像源安装成功"
    else
        echo "[FAIL] 两个源都失败。镜像源日志尾部："
        tail -5 /tmp/pip_mirror.log
        RC=1
    fi
fi
"$VENV/bin/pip" list 2>/dev/null | grep -iE "^(torch|torch_musa|transformers|tokenizers|accelerate|safetensors|numpy) " || true

echo
echo "===== 4. 复核 LD_PRELOAD 的前提（不盲抄结论）====="
CNT=$(strings /root/miniconda3/lib/libstdc++.so.6 2>/dev/null | grep -c GLIBCXX_3.4.30)
echo "conda libstdc++ 含 GLIBCXX_3.4.30 的条目数 = $CNT （0 ⇒ 仍需要 LD_PRELOAD）"
if [ "$CNT" != "0" ]; then
    echo "[note] conda 那份已经有 3.4.30 ⇒ LD_PRELOAD 可能已非必要（**不要据此直接删**，需前后对照）"
fi

echo
echo "===== 5. 验证（必须按 tokenizers → torch 的顺序，否则测不出那个坑）====="
# shellcheck disable=SC1090
source "$ACT"
python - <<'PY'
import sys, os
print("python      :", sys.version.split()[0], "@", sys.executable)
import tokenizers                     # 先载入 tokenizers（故意）
print("tokenizers  :", tokenizers.__version__)
import torch                          # 再载入 torch —— 这一步在有坑时会崩
print("torch       :", torch.__version__)
import torch_musa                     # 必须先 import 才注册出 torch.musa
print("torch_musa  :", torch_musa.__version__)
import transformers
print("transformers:", transformers.__version__)
print("torch.musa  :", hasattr(torch, "musa"))
if hasattr(torch, "musa"):
    print("musa 可用   :", torch.musa.is_available())
    print("设备数      :", torch.musa.device_count())
    x = torch.randn(64, 64, device="musa", dtype=torch.float16)
    y = (x @ x).sum().item()
    print("设备上做了一次 64x64 fp16 矩阵乘，sum =", round(y, 4))
print("LD_PRELOAD  :", os.environ.get("LD_PRELOAD", "<unset>"))
maps = open("/proc/self/maps").read()
for line in maps.splitlines():
    if "libstdc++" in line:
        print("实际载入    :", line.split()[-1])
        break
PY
VRC=$?
echo "[verify] python 退出码=$VRC"
[ "$VRC" != "0" ] && RC=1

echo
echo "===== 6. 复核模型溯源四项**指向真实存在的东西**（声明 ≠ 事实）====="
# 为什么单独一条：这四个值会被写进 env.json 的 discovery，成为"本次用的哪个模型"
# 的凭据。若路径打错、或 revision 与权重不符，**不会有任何报错**
# —— 只会在很久以后被人当成事实引用。⇒ 当场核。
echo "MODEL_PATH     = ${MODEL_PATH:-<unset>}"
if [ -d "${MODEL_PATH:-/nonexistent}" ]; then
    echo "  [ok] 目录存在"
    SNAP="$MODEL_PATH/snapshots/${MODEL_REVISION:-<unset>}"
    if [ -d "$SNAP" ]; then
        echo "  [ok] revision 快照存在：$SNAP"
        N_SHARD=$(ls "$SNAP"/model-*.safetensors 2>/dev/null | wc -l)
        echo "  [ok] 权重分片数 = $N_SHARD"
        [ "$N_SHARD" -ge 1 ] || { echo "  [FAIL] 没有权重分片"; RC=1; }
    else
        echo "  [FAIL] 找不到 revision 快照 $SNAP（MODEL_REVISION 打错了？）"
        RC=1
    fi
else
    echo "  [FAIL] MODEL_PATH 指向的目录不存在"
    RC=1
fi
if [ -n "${MODEL_REVISION:-}" ] && [ "${MODEL_REVISION}" != "a09a35458c702b33eeacc393d103063234e8bc28" ]; then
    echo "  [warn] MODEL_REVISION 与配置里的 runtime.model_revision 不一致 —— 声明与配置必须同步改"
fi

echo
echo "===== 完成（RC=$RC）====="
exit "$RC"
