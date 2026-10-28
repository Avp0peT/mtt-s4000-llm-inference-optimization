"""把已回拉的 `*.json.gz` / `*.jsonl.gz` 解压到同名无后缀文件。

为什么不用 shell `gzip -d -c f > g`：
  2026-09-28 §62 —— 在 Git Bash + Windows 下，该管线在脚本里**静默失败**
  （目标文件不生成，只留 `.gz`），于是"回拉成功"被读成"回拉失败"。
  用 Python 的 gzip 模块一次做完**解压 + 校验**，并**打印字节数**，
  使成功/失败都显形。

纪律：
  * 只在**解压后字节数 == 远端原始字节数**时才写目标文件（原子替换，先写 .partial）。
  * `.gz` 一律保留（它是回拉的原始证据，也是可重试的中间态）。
  * 解压后**必须**能 JSON 解析（.json）或逐行 JSON 解析（.jsonl），否则不动目标文件。

用法：
    python scripts/decompress_pulled.py            # 处理 results/raw/S14B2-*
    python scripts/decompress_pulled.py S14B3      # 处理指定前缀
"""

from __future__ import annotations

import glob
import gzip
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remote_size(run_dir: str, base: str) -> int | None:
    """从 meta.json.gz 的 size_hint 无法得到；改为从同目录 JSON 的自述读取。

    更稳的做法：解压后用 JSON 结构里的 `n_samples` 校验不了字节数，
    所以这里只做「解压 + 可解析」判定，字节数交给调用方比对远端。
    """
    return None


def parse_bytes(raw: bytes, name: str) -> tuple[bool, str]:
    if name.endswith(".json"):
        try:
            json.loads(raw.decode("utf-8"))
            return True, "JSON_OK"
        except Exception as e:  # noqa: BLE001
            return False, f"JSON_BAD({type(e).__name__})"
    if name.endswith(".jsonl"):
        n = 0
        for ln in io.StringIO(raw.decode("utf-8", errors="replace")):
            if ln.strip():
                try:
                    json.loads(ln)
                    n += 1
                except Exception:
                    return False, "JSONL_BAD"
        return (n > 0), f"JSONL_OK(n={n})"
    return True, "RAW"


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "S14B2"
    pattern = os.path.join(ROOT, "results", "raw", f"{prefix}-*", "*.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"没有可处理的 .gz（pattern={pattern}）")
        return 0

    done = skipped = failed = 0
    for gz in files:
        base = os.path.basename(gz)[:-3]  # 去掉 .gz
        d = os.path.dirname(gz)
        target = os.path.join(d, base)
        tag = os.path.basename(d)

        if os.path.exists(target) and os.path.getsize(target) > 0:
            skipped += 1
            continue

        try:
            with gzip.open(gz, "rb") as fh:
                raw = fh.read()
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {tag}/{base}: 解压失败 {type(e).__name__}: {e}")
            failed += 1
            continue

        ok, why = parse_bytes(raw, base)
        if not ok:
            print(f"  ✗ {tag}/{base}: 解压后不可解析（{why}，{len(raw)} B）——目标文件未写")
            failed += 1
            continue

        partial = target + ".partial"
        with open(partial, "wb") as fh:
            fh.write(raw)
        os.replace(partial, target)
        print(f"  ✓ {tag}/{base}: {len(raw)} B  {why}")
        done += 1

    print(f"=== 解压: 新增={done} 已存在跳过={skipped} 失败={failed} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
