"""覆盖率闸门的**模型来源**必须与实跑同源（`--cfg`），且两个来源不得同时给。

为什么要有这一组用例
--------------------
实测踩到（2026-09-26）：v6 链的覆盖率闸门**显式**传 `--model model_v6.json`，
而紧接着的批 `run_step3_matrix.py --cfg-dir phase1_step14` 读的是**配置**、
配置里 `cost_model_path` 指向 **`model_v5.json`** ⇒ "**验 A 跑 B**"。
两者**指纹相同**（同机器/精度/模型）⇒ 加载校验全过 ⇒ **不报错**、产物完全正常，
但那一批**没有**检验它声称要检验的东西。

⇒ 结构性修法：让闸门**从配置读**模型（`--cfg`），于是"闸门验的"与"实跑用的"
在结构上就是**同一个指代**；同时把"两个来源都给"变成**错误**（那正是温床）。
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_model_coverage as C  # noqa: E402

#: 本地一定存在的模型（用于构造"配置指向真实文件"的情形）
_LOCAL_MODEL = "configs/cost_calib/model_v4.json"


def _cfg(tmp: Path, *, model: str | None = _LOCAL_MODEL, name: str = "cfg.yaml") -> Path:
    """从一个真实配置派生（只改 model 那一行），避免手写整份 yaml。"""
    src = (_ROOT / "configs" / "mthreads" / "phase1_step14b" / "fixed_cost.yaml").read_text(
        encoding="utf-8")
    if model is None:
        # 变成"不加载模型"的臂：删掉 cost_model_path 行
        src = "\n".join(l for l in src.splitlines()
                        if "cost_model_path:" not in l) + "\n"
    else:
        src = src.replace("cost_model_path: configs/cost_calib/model_v6.json",
                          f"cost_model_path: {model}")
    src = src.replace("cost_model_fingerprint_source: observed",
                      "cost_model_fingerprint_source: declared")
    p = tmp / name
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def _run(argv: list[str]) -> tuple[int, str, str]:
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = C.main(argv)
    return rc, so.getvalue(), se.getvalue()


class TestModelSourceResolution(unittest.TestCase):
    def test_cfg_resolves_the_model_from_the_config(self):
        """★ 从配置读 —— 打印出来源，让人一眼看出闸门验的是哪个模型。"""
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td))
            rc, out, _ = _run(["--tag", "NOSUCHTAG", "--cfg", str(cfg)])
        self.assertIn("模型来源 = **配置**", out, "没有把来源打印出来 ⇒ 事后无法核对")
        self.assertIn(_LOCAL_MODEL, out, "来源里应含配置里的那个相对路径")
        # 该 tag 没有产物 ⇒ 应在解析之后以 2 退出（证明解析真的走过了）
        self.assertEqual(rc, 2)

    def test_both_sources_is_an_error(self):
        """★ 两个来源同时给 = '验 A 跑 B' 的温床 ⇒ 必须报错，不许静默取其一。"""
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td))
            rc, _, err = _run(["--tag", "NOSUCHTAG", "--cfg", str(cfg),
                               "--model", _LOCAL_MODEL])
        self.assertEqual(rc, 2, "两个来源同时给却接受了 ⇒ 守卫失效")
        self.assertIn("验 A 跑 B", err)

    def test_cfg_without_model_path_is_an_error(self):
        """配置指向的臂**不加载模型**（如 threshold）⇒ 报错，不许回落到默认模型。"""
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td), model=None, name="nomodel.yaml")
            rc, _, err = _run(["--tag", "NOSUCHTAG", "--cfg", str(cfg)])
        self.assertEqual(rc, 2)
        self.assertIn("cost_model_path 为空", err)

    def test_cfg_pointing_at_missing_file_is_an_error(self):
        """配置指向**不存在**的模型 ⇒ 报错（否则会静默用别的模型或全部回退）。"""
        with tempfile.TemporaryDirectory() as td:
            cfg = _cfg(Path(td), model="configs/cost_calib/model_does_not_exist.json")
            rc, _, err = _run(["--tag", "NOSUCHTAG", "--cfg", str(cfg)])
        self.assertEqual(rc, 2)
        self.assertIn("模型不存在", err)

    def test_neither_source_is_an_error_not_a_silent_default(self):
        """既没给 `--model` 也没给 `--cfg` ⇒ **报错**，不许有静默默认。

        ★ 这条是本次用例抓出来的真缺陷：原先默认 `model_v3.json`，而 v3 **被判否、
        故意没有导出**（见 `results/raw/_costcalib/README.md`）⇒ 不传参数时闸门会抛
        `FileNotFoundError`，看起来像"工具坏了"而不是"你没给模型"。
        教训：**一个指向"被否掉的版本"的默认值，等于静默指错** —— 而任何静默默认
        都是"验 A 跑 B"的温床。
        """
        rc, out, err = _run(["--tag", "NOSUCHTAG"])
        self.assertEqual(rc, 2, "没有模型来源却没报错 ⇒ 又回到了静默默认")
        self.assertIn("必须显式指定模型来源", err)
        self.assertIn("--cfg", err)
        self.assertIn("model_v3.json", err, "报错要说明为什么取消默认值")

    def test_missing_model_file_is_a_clean_error_not_a_traceback(self):
        """闸门崩栈 ≠ 闸门判决：崩栈会被读成"工具坏了"从而**绕过**判定。"""
        rc, _, err = _run(["--tag", "NOSUCHTAG", "--model",
                           "configs/cost_calib/no_such_model.json"])
        self.assertEqual(rc, 2)
        self.assertIn("模型加载失败", err)


if __name__ == "__main__":
    unittest.main()
