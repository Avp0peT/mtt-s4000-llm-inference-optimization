"""`scripts/plot_step16_decouple.py` 的回归用例（第 16 步 P27 图）。

守的是**图能由脚本重建**这条硬要求（`paper/sections/04_failure_analysis.md`
的 P27 占位块原文："图必须能**由脚本重建**并标注 run_id"）。
所以这里不测"画得好不好看"，只测**输入契约**：

1. 输入缺一个 ⇒ **必须报错**，不得静默画出"空图"（空图看起来像"数据全 0"）；
2. `cells` 为空 ⇒ **必须报错**，不得当作"没有数据"（缺 ≠ 0）；
3. `--check` 不写任何文件；
4. 全量跑一次 ⇒ PNG/SVG/`.source.txt` 都产出且非空，且 `.source.txt` 里含
   三个输入的 sha256（"图来自哪一份文件"可核对）；
5. `kv_step=0` 的 `coverage=0.0` 被当成**实测值**（不是缺测 ⇒ 仍进图）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "plot_step16_decouple.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_step16_decouple", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _min_decouple(mod, modulus=2):
    """最小但形状正确的 decouple 产物（2×2）。"""
    cells = {}
    for grid in mod.GRID_ORDER:
        for split in mod.SPLIT_ORDER:
            cells[f"{grid}|{split}"] = {
                "split": split, "modulus": modulus,
                "train_n": 10, "holdout_n": 12,
                "train_min_rank": 1 if split == "hash" else 3,
                "holdout_min_rank": 2,
                "underest_p95": 0.9 if split == "hash" else 0.1,
                "abs_rel_p90": 0.8, "eval_json": "x.json",
                "model_holds": False, "verdict": "v",
            }
    return {"holdout_modulus": modulus, "cells": cells}


def _min_lookup(mod):
    cells = []
    for pf in (0, 128):
        for kv in (0, 1024):
            # ★ kv=0 时 coverage 实测为 0.0（不是缺测），必须仍被画出来
            cells.append({"pf_step": pf, "kv_step": kv, "n_buckets": 10,
                          "coverage": 0.0 if kv == 0 else 1.0,
                          "over_p50": 0.0, "over_p95": 0.001 if kv == 0 else 0.1,
                          "over_max": 0.0, "loo_n": 124, "loo_skipped": 0})
    return {"calib": "c.json", "shapes_tag": "T", "n_records": 1,
            "n_shapes": 4, "cells": cells}


@pytest.fixture()
def tmp_inputs(tmp_path, monkeypatch):
    """把模块的 PROCESSED 指向临时目录，造齐最小输入。"""
    mod = _load_module()
    proc = tmp_path / "processed"
    _write(proc / "decouple_v1_v3.json", _min_decouple(mod))
    _write(proc / "decouple_v3_v4.json", _min_decouple(mod, modulus=2))
    _write(proc / "lookup_bucket_sweep.json", _min_lookup(mod))
    return mod, proc, tmp_path


def test_load_decouple_rejects_missing_file(tmp_inputs):
    mod, proc, _ = tmp_inputs
    (proc / "decouple_v3_v4.json").unlink()
    with pytest.raises(FileNotFoundError):
        mod.load_decouple(proc)


def test_load_lookup_rejects_empty_cells(tmp_inputs):
    """空 `cells` ⇒ 报错（不得画空图；空图会被读成"数据全 0"）。"""
    mod, proc, _ = tmp_inputs
    _write(proc / "lookup_bucket_sweep.json",
           {"calib": "c.json", "cells": []})
    with pytest.raises(ValueError):
        mod.load_lookup(proc)


def test_load_decouple_rejects_empty_cells(tmp_inputs):
    mod, proc, _ = tmp_inputs
    _write(proc / "decouple_v1_v3.json", {"holdout_modulus": 2, "cells": {}})
    with pytest.raises(ValueError):
        mod.load_decouple(proc)


def test_parse_decouple_requires_pipe(tmp_inputs):
    mod, _, _ = tmp_inputs
    with pytest.raises(ValueError):
        mod.parse_decouple({"没有竖线": {}})


def test_check_writes_nothing(tmp_inputs):
    """`--check` 只校验，**不得**写任何文件。"""
    mod, proc, tmp_path = tmp_inputs
    out = tmp_path / "figs"
    rc = mod.main(["--processed", str(proc), "--out", str(out), "--check"])
    assert rc == 0
    assert not out.exists() or not any(out.iterdir())


def test_full_run_produces_all_artifacts(tmp_inputs):
    """全量跑 ⇒ 4 个图 + `.source.txt`，且 `.source.txt` 记 sha256。"""
    mod, proc, tmp_path = tmp_inputs
    out = tmp_path / "figs"
    rc = mod.main(["--processed", str(proc), "--out", str(out)])
    assert rc == 0
    names = {p.name for p in out.iterdir()}
    for want in ("fig_p27a_decouple.png", "fig_p27a_decouple.svg",
                 "fig_p27b_lookup_sweep.png", "fig_p27b_lookup_sweep.svg",
                 "fig_p27.source.txt"):
        assert want in names, f"缺产物 {want}"
    for p in out.iterdir():
        assert p.stat().st_size > 0, f"{p} 为空"

    src = (out / "fig_p27.source.txt").read_text(encoding="utf-8")
    for f in ("decouple_v1_v3.json", "decouple_v3_v4.json",
              "lookup_bucket_sweep.json"):
        digest = hashlib.sha256((proc / f).read_bytes()).hexdigest()
        assert digest in src, f".source.txt 未记录 {f} 的 sha256"
    # 重建命令（可复制再跑）
    assert "python scripts/plot_step16_decouple.py" in src


def test_figure_text_has_no_missing_glyphs(tmp_inputs):
    """★ 图内文字**不得**缺字。

    实测踩到：Microsoft YaHei 不含 `✅`/`⚠️`/`⇒` ⇒ matplotlib 报
    `Glyph 9888 (WARNING SIGN) missing from font`，落到图上是**方框**。

    ⚠️ 注意本用例**必须**在"旁路 `safe_plot_text`"的前提下才有效：
    因为脚本把图内文字都过了一遍 `safe_plot_text()`，坏符号到不了 matplotlib，
    `-W error::UserWarning` **永远不会**触发 ⇒ 那个断言是**空的**（踩过）。
    所以这里同时做两件事：
      1. 断言 `safe_plot_text` 确实覆盖了脚本所声明的全部坏符号；
      2. 断言渲染路径上**没有**对坏符号的裸调用（即图内文字都过了清洗）。
    """
    mod = _load_module()
    text = SCRIPT.read_text(encoding="utf-8")

    # 1) `_GLYPH_MAP` 的键必须覆盖脚本自己在图内文段里用到的坏符号
    bad = set(mod._GLYPH_MAP)
    # 图内文段（fig A/B 的 text/title/label 参数）里出现的坏符号清单在这断言
    for ch in ("✅", "⚠️", "⇒"):
        assert ch in bad, f"_GLYPH_MAP 未覆盖 {ch!r}"

    # 2) 渲染路径：所有 fig.text / set_title / set_xlabel / set_ylabel 调用
    #    **不得**直接传含坏符号的字面量（必须经 safe_plot_text 或已用安全替代）
    #
    # ⚠️ **必须先剥注释再扫**（踩过）：第一版直接在原文上扫，命中了
    #    `# 探针需要的最小秩（3 参数 ⇒ 满秩 3）` 这条**注释**，报假失败。
    #    与"用字符串探测能力"那个坑同族 —— 注释里的字不是代码。
    import re
    code_lines = []
    for line in text.splitlines():
        stripped = re.sub(r"#.*$", "", line)
        if stripped.strip():
            code_lines.append(stripped)
    code = "\n".join(code_lines)

    unsafe_calls = []
    for m in re.finditer(
        r"(fig\.text\(|set_title\(|set_xlabel\(|set_ylabel\(|ax\d?\.text\(|suptitle\(|\.legend\()",
        code,
    ):
        # 取该调用之后到本行末（+续行）的片段
        nxt = code[m.end():m.end() + 300]
        if any(ch in nxt for ch in ("✅", "⚠", "⇒", "→")) and "safe_plot_text" not in nxt:
            unsafe_calls.append(nxt[:80].replace("\n", " "))
    assert not unsafe_calls, (
        "图内文字存在未经清洗的坏符号（会渲染成方框）：\n" + "\n".join(unsafe_calls))


def test_render_path_is_actually_sanitized(tmp_inputs, monkeypatch):
    """★ 反向验证：把清洗**旁路**掉，坏符号必须真的报缺字。

    这是本组用例的"自校准"：如果旁路后仍然不报错，说明这条守卫是**空的**
    （坏符号到不了 matplotlib，守卫就在测空气）。
    """
    mod = _load_module()
    # 旁路清洗 ⇒ 图内文字原样送到 matplotlib
    monkeypatch.setattr(mod, "safe_plot_text", lambda s: s)
    # 同时在图注里塞一个真的坏符号（用 monkeypatch 不方便改字面量，改用
    # 直接调用绘图函数的方式：给 fig_b 传一个含坏符号的标题）
    proc = tmp_inputs[1]
    out = tmp_inputs[2] / "figs_bypass"
    out.mkdir(parents=True, exist_ok=True)

    import warnings
    plt, _ = mod.setup_matplotlib()
    decouple = mod.load_decouple(proc)
    lookup = mod.load_lookup(proc)

    # 造一个含坏符号的标题，强制走渲染（绕过 safe_plot_text）
    orig_suptitle = plt.Figure.suptitle

    def bad_suptitle(self, t, *a, **kw):
        return orig_suptitle(self, t + " ⚠️✅⇒", *a, **kw)

    monkeypatch.setattr(plt.Figure, "suptitle", bad_suptitle, raising=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # 缺字警告 ⇒ 变错误
        raised = False
        try:
            mod.fig_a(plt, decouple, out)
        except Exception as e:                # noqa: BLE001
            raised = True
            assert "Glyph" in str(e) or "missing" in str(e), (
                f"抛的不是缺字错误，而是：{type(e).__name__}: {e}")
        assert raised, ("旁路清洗后仍无缺字警告 ⇒ 该守卫是**空的**"
                        "（坏符号根本没到渲染层），本用例失去意义")


def test_svg_has_no_crlf(tmp_inputs):
    """★ 生成的 `.svg` 不得含 CRLF。

    踩过（本项目老坑）：Windows 上写出的文本常是 CRLF，`git add` 会报
    "CRLF will be replaced by LF"，而**内容对、字节错**的改动**没有报错**。
    所以这里对**产物字节**直接断言 `\\r` 计数为 0。
    """
    mod, proc, tmp_path = tmp_inputs
    out = tmp_path / "figs"
    assert mod.main(["--processed", str(proc), "--out", str(out)]) == 0
    for svg in out.glob("*.svg"):
        raw = svg.read_bytes()
        assert b"\r" not in raw, f"{svg.name} 含 CR（{raw.count(b'\\r')} 个）⇒ 应归一为 LF"


def test_missing_input_is_an_error_not_an_empty_figure(tmp_inputs):
    """缺输入 ⇒ 报错（**不得**画空图，空图会被读成"数据全 0"）。"""
    mod, proc, tmp_path = tmp_inputs
    (proc / "lookup_bucket_sweep.json").unlink()
    out = tmp_path / "figs_missing"
    assert mod.main(["--processed", str(proc), "--out", str(out)]) == 2
    assert not out.exists() or not any(out.iterdir())


def test_safe_plot_text_replaces_unsafe_glyphs():
    """`safe_plot_text` 必须把非字体符号换成可渲染替代。"""
    mod = _load_module()
    for bad in ("✅", "⚠️", "⇒", "→", "✗", "✓"):
        out = mod.safe_plot_text(f"a{bad}b")
        assert bad not in out, f"{bad} 未替换"
        assert "a" in out and "b" in out        # 周围文字保留
    # 纯中文/ASCII 不受影响
    assert mod.safe_plot_text("网格 v1_v3 ｜ coverage=0.536") == "网格 v1_v3 ｜ coverage=0.536"


def test_kv_zero_coverage_is_a_measured_value(tmp_inputs):
    """★ `kv_step=0` 的 `coverage=0.0` 必须被当作**实测值**保留。

    这是本图的一个陷阱：coverage=0 同时 over_p95 最低，会被误读成"最好"。
    所以它**不能**被当成缺测而跳过——必须在图上显形。
    """
    mod, proc, _ = tmp_inputs
    look = mod.load_lookup(proc)
    zeros = [c for c in look["doc"]["cells"] if c["kv_step"] == 0]
    assert zeros, "测试输入应含 kv_step=0"
    for c in zeros:
        assert c["coverage"] == 0.0
        assert c["coverage"] is not None      # ← 实测 0，不是缺测
        assert c["over_p95"] < 0.01            # 低误差正是陷阱来源


def test_source_docstring_declares_the_pf_inert_reading():
    """读法必须写在脚本里（"照抄结论"，不靠读图人自己推）。"""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "pf_step" in text and "惰性" in text
    assert "coverage=0" in text or "coverage = 0" in text
