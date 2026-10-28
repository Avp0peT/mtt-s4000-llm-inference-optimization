"""读远端 `exec_opt_progress_<TAG>.json` 并打印摘要（避免远端引号嵌套）。

用法：python scripts/_read_progress.py <TAG> [--remote-root /root/ascend_probe_d1]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_VENV = Path.home() / ".workbuddy" / "binaries" / "python" / "envs" / "default" / "Scripts" / "python.exe"
REMOTE_RUN = REPO / "scripts" / "remote_run.py"


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "EXECOPT2"
    root = "/root/ascend_probe_d1"
    if "--remote-root" in sys.argv:
        root = sys.argv[sys.argv.index("--remote-root") + 1]

    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    prog = f"{root}/exec_opt_progress_{tag}.json"
    log = f"{root}/logs/{tag}.log"

    for label, cmd in [
        ("进度", f"cat {prog}"),
        ("完成/失败行", f"grep -aE '✓|✗|▶' {log} | tail -20"),
        ("产物", f"ls {root}/results/raw/ | grep {tag}"),
    ]:
        print(f"===== {label} =====")
        p = subprocess.run([str(PY_VENV), str(REMOTE_RUN), "exec", cmd],
                           capture_output=True, timeout=180, env=env)
        text = p.stdout.decode("utf-8", errors="replace")
        if label == "进度":
            try:
                d = json.loads(text)
                print(json.dumps({k: d[k] for k in
                                  ("n_total", "n_done", "done_keys", "current",
                                   "elapsed_s") if k in d},
                                 ensure_ascii=False, indent=2))
                continue
            except Exception:
                pass
        print(text.strip()[-2000:] or "(空)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
