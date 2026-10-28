"""`phase1_step14b/`（派生配置目录）的**三条硬约束**。

背景（实测踩到的闸门/实跑脱节）
--------------------------------
v6 链的覆盖率闸门**显式**传 `--model configs/cost_calib/model_v6.json`，但紧接着的批
`run_step3_matrix.py --cfg-dir configs/mthreads/phase1_step14` 读的是**配置**，
而配置里 `cost_model_path` 指向 **`model_v5.json`** ⇒ **"用 v6 过闸、用 v5 开跑"**。
两者指纹相同（同机器同精度）⇒ **不会报错**、产物看起来完全正常。
⇒ `phase1_step14b/` 就是让**配置**指向 v6 的那份副本（语义修复只有在配置指对时才被检验）。

三条约束（缺一条，派生目录就会要么漂成"第二份真相"、要么静默指错模型）
--------------------------------------------------------------------
1. **只差一行**：除 `controller.cost_model_path` 外，逐字节等于 `phase1_step14/` 的对应文件
   （`static`/`threshold` 本来就不加载模型 ⇒ 应完全相同）。少了这条，派生副本会慢慢漂开，
   而那正是本项目最避免的"第二处真相"。
2. **同 cell 内用模型的两臂必须指同一个模型**（否则四臂里混了两个模型，四臂对比无从归因）。
3. **被引用的模型文件必须与"是否已就绪"的状态一致**：见 `PENDING_MODELS`。
   这条是**双向**的 —— 已就绪却仍挂在 `PENDING_MODELS` 里也要失败，否则名单会烂在那里、
   守卫慢慢变成摆设。
"""

from __future__ import annotations

import glob
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

STEP14 = _ROOT / "configs" / "mthreads" / "phase1_step14"
STEP14B = _ROOT / "configs" / "mthreads" / "phase1_step14b"
MT = _ROOT / "configs" / "mthreads"

#: 派生目录与源目录之间**唯一允许**的差异
_SRC_LINE = "cost_model_path: configs/cost_calib/model_v5.json"
_DST_LINE = "cost_model_path: configs/cost_calib/model_v6.json"

#: **尚未产出**的模型：路径 → 由哪一步/哪条链负责产出。
#: 双向断言：在名单里 ⇒ 必须**还不存在**；不在名单里 ⇒ 必须**已存在**。
#: 模型落地后从这里删掉，用例会立刻开始验它的存在性（名单不会烂掉）。
PENDING_MODELS: dict[str, str] = {
    # ⚠️ 首跑时真触发了这条守卫：本地**没有** `model_v5.json` —— 它由正在跑的 v5 链在
    # **远端**产出，取回之前本地不存在。这正是"配置指向不存在的模型"的真实形态
    # （跑起来不报错、只是每次都回退）。
    # 2026-09-26：v5 链以 `S14V5_DONE` 结束、`model_v5.json` 已取回 ⇒ 按双向守卫移除本条
    # （移除前该守卫实测失败，确认它确实在看着这条）。
    # 2026-09-27（自动化离线收尾）：v6 链的 `model_v6.json` 已自 `/root/ascend_probe` 取回
    # ⇒ 同 v5 的处置：移除本条。移除前该守卫同样是**红**的（报"已存在，应从 PENDING_MODELS
    # 移除"）⇒ 确认它真在看这条，而不是在删一条早已失效的摆设。
    # ⚠️ **本条守卫只查"文件在不在"，不查"模型可不可用"** —— 而这次二者正好背离：
    # `model_v6.json` 虽已就位，但它拟合的那批 v6 测点里 **mixed(batch>1) 84/84 全部报错**
    # （`TypeError: int + list`，即 P0；v6 树戳 `158f83a0` 早于修复 `da4a71a`）⇒ 其查表**不含**
    # 任何 `pf>0,batch>1` 键，自身 `validation.verdict.model_holds=false`（低估 p95 5.02% > 5%）。
    # 语义核验 `results/processed/verify_v5_v6.json` 结论为 **undetermined**（退出码 3：
    # C1 ✓、C3 ✓、C2 未判）。故"文件已在"**不得**被读成"语义修复已生效"。
    # 2026-09-26（自动化离线诊断追加）：`9f09478`（D1 候选下限配置集）让
    # `configs/mthreads/phase1_d1/*.yaml`（4 份）指向 `model_v7.json`，
    # 但**漏登记**本条 ⇒ 该守卫在 HEAD 上实测**红**。
    # `model_v7` 是 D1 的**计划产物**（`EXPERIMENTS.md` §D1 把"`model_v7.json` 已产出"
    # 列为前置条件）⇒ 属"被引用但未产出"，正是本名单的用途，按机制补登记。
    # ⚠️ v7 产出并取回后**必须**移除本条（双向守卫会把"已产出却仍挂着"判失败）。
    # ⚠️ 2026-09-27 02:59：v7 产出并**已取回**（本机 md5
    # `851252452a338fd63bcf411e854c93a4` ≡ 远端，12 077 B，自 `/root/ascend_probe_d1`）
    # ⇒ 按双向守卫移除本条。**移除前该守卫实测为红**（报"已存在，应从 PENDING_MODELS 移除"，
    # 并列出 4 份引用它的 `configs/mthreads/phase1_d1/*.yaml`）⇒ 确认它真在看这条，
    # 而不是在删一条早已失效的摆设。
    # ⚠️ 与 v6 同一条警告继续适用：**本条只查"文件在不在"，不查"模型可不可用"** ——
    # v7 是否真的修好了 `real_mixed`（C2 处理效应能否判出），须对 v7 重跑
    # `scripts/verify_calib_semantics.py` 三问来定，**不得**由"文件已在"推得。
    # 当前名单为空：所有被 `phase1_step14*` / `phase1_d1` 引用的模型都已产出并取回。
}

#: 非 yaml 的成员（说明文档），不参与"只差一行"的比对
_NON_YAML = {"README.md", "FREEZE.md"}


class TestDerivedOnlyOneLineDiffers(unittest.TestCase):
    def test_dirs_exist(self):
        self.assertTrue(STEP14.is_dir(), f"缺少 {STEP14}")
        self.assertTrue(STEP14B.is_dir(), f"缺少 {STEP14B}")

    def test_same_yaml_members(self):
        a = {p.name for p in STEP14.glob("*.yaml")}
        b = {p.name for p in STEP14B.glob("*.yaml")}
        self.assertEqual(a, b, f"两侧 yaml 成员不一致：仅左有 {a - b}｜仅右有 {b - a}")

    def test_every_file_is_source_with_one_substitution(self):
        """★ 核心约束：逐字节等于"源文件 + 一处替换"。

        用**通用换行**读取（`read_text`）故对 CRLF/LF 不敏感 —— 这是有意的：
        否则一个换行符差异会伪装成"内容改了"（踩过：diff 报'整个文件都不同'）。
        """
        bad: list[str] = []
        for f in sorted(STEP14.glob("*.yaml")):
            src = f.read_text(encoding="utf-8")
            got = (STEP14B / f.name).read_text(encoding="utf-8")
            want = src.replace(_SRC_LINE, _DST_LINE)
            if got != want:
                extra = [l for l in got.splitlines() if l not in want.splitlines()]
                missing = [l for l in want.splitlines() if l not in got.splitlines()]
                bad.append(f"{f.name}: 多出 {extra[:3]}｜缺失 {missing[:3]}")
        self.assertEqual(bad, [], "派生目录里出现了'只差一行'以外的改动：\n  " + "\n  ".join(bad))

    def test_no_freeze_doc_copied(self):
        """**不得**复制 `FREEZE.md`：冻结方案的权威源只有 `phase1_step14/FREEZE.md` 一处。"""
        self.assertFalse((STEP14B / "FREEZE.md").exists(),
                         "派生目录复制了 FREEZE.md ⇒ 出现第二处真相来源")


class TestSingleVariableAcrossArms(unittest.TestCase):
    def test_model_arms_share_one_model_and_static_arms_load_none(self):
        for cell in ("fixed", "ramp", "lenmix"):
            with self.subTest(cell=cell):
                cost = load_config(STEP14B / f"{cell}_cost.yaml")
                sw = load_config(STEP14B / f"{cell}_slo_weave.yaml")
                self.assertEqual(cost.controller.cost_model_path,
                                 sw.controller.cost_model_path,
                                 f"{cell}: cost 与 slo_weave 指了不同的模型 ⇒ 四臂混了模型")
                for arm in ("static", "threshold"):
                    c = load_config(STEP14B / f"{cell}_{arm}.yaml")
                    self.assertIsNone(c.controller.cost_model_path,
                                      f"{cell}_{arm} 不加载模型，不应有 cost_model_path")

    def test_derived_dir_points_at_the_new_model(self):
        for cell in ("fixed", "ramp", "lenmix"):
            c = load_config(STEP14B / f"{cell}_cost.yaml")
            self.assertTrue(str(c.controller.cost_model_path).endswith("model_v6.json"),
                            f"{cell}: 派生目录应指向 model_v6.json，实际 "
                            f"{c.controller.cost_model_path}")


class TestReferencedModelsExistOrArePending(unittest.TestCase):
    def _referenced(self) -> dict[str, list[str]]:
        """`configs/mthreads/**` 下每个配置引用的模型路径 → 引用它的配置列表。"""
        out: dict[str, list[str]] = {}
        for f in sorted(glob.glob(str(MT / "**" / "*.yaml"), recursive=True)):
            cfg = load_config(f)
            p = cfg.controller.cost_model_path
            if p:
                out.setdefault(str(p), []).append(
                    str(Path(f).relative_to(_ROOT)).replace("\\", "/"))
        return out

    def test_every_referenced_model_is_present_or_explicitly_pending(self):
        refs = self._referenced()
        self.assertTrue(refs, "没扫到任何引用模型的配置 ⇒ 本用例没在测东西（路径/字段名变了？）")
        missing_but_not_pending, pending_but_present = [], []
        for rel, users in refs.items():
            exists = (_ROOT / rel).exists()
            if not exists and rel not in PENDING_MODELS:
                missing_but_not_pending.append(f"{rel} ← {users}")
            if exists and rel in PENDING_MODELS:
                pending_but_present.append(f"{rel}（已存在，应从 PENDING_MODELS 移除）← {users}")
        self.assertEqual(missing_but_not_pending, [],
                         "以下配置指向**不存在**的模型（跑起来会静默回退）：\n  "
                         + "\n  ".join(missing_but_not_pending))
        self.assertEqual(pending_but_present, [],
                         "PENDING_MODELS 里有已产出的模型 ⇒ 名单烂了，守卫正在变摆设：\n  "
                         + "\n  ".join(pending_but_present))

    def test_pending_entries_name_a_reason(self):
        for k, why in PENDING_MODELS.items():
            self.assertTrue(why.strip(), f"{k} 在 PENDING_MODELS 里但没写理由")


if __name__ == "__main__":
    unittest.main()
