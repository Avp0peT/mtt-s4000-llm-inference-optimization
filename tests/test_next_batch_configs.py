"""机制小批的**配置准备**（`configs/mthreads/next_batch/`）与双预算开关的接线。

★ 为什么需要这个文件（2026-10-03，任务 E）
------------------------------------------
1. **双预算开关此前在配置层不存在**：`ControllerConfig` 没有 `prefill_deadline_ms`，
   而 `arm_factory` 也从不传它 ⇒ 机制只在代码与单测里可达、**任何批次都开不了它**。
   ⇒ 本文件锁住"开关能从 YAML 一路走到 `CostPolicyConfig`"。
2. **两条 cost 臂必须只差一个键**（否则是两变量对比）。这条要**程序化**核对，
   不能靠目测 diff。
3. **静态基线不许重复造**：`static_nochunk` 与 `threshold` 直接用第 14 步的冷冻配置；
   只有真正缺的两份才派生。

⚠️ 本文件**不加载成本模型**（它的指纹校验会在非标定机上**正确地**拒绝加载）。
   所以走 `dataclass` 与 `load_config` 两条不依赖模型的路径。
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("src", "benchmark", "scripts"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.controller import ControllerConfig  # noqa: E402
from run_benchmark import load_config  # noqa: E402

FROZEN = _ROOT / "configs" / "mthreads" / "phase1_step14"
NEXT = _ROOT / "configs" / "mthreads" / "next_batch"


class TestDualBudgetSwitchIsWired(unittest.TestCase):
    """★ `ControllerConfig.prefill_deadline_ms` 必须能从配置读到、并落进产物。"""

    def test_dataclass_field_exists_and_defaults_to_none(self):
        """默认 `None` ⇒ 不启用双预算 ⇒ 既有批次行为不变。"""
        cfg = ControllerConfig()
        self.assertIsNone(cfg.prefill_deadline_ms)

    def test_dataclass_field_is_recorded_in_describe(self):
        """必须落盘：否则"这个臂开没开双预算"事后无从核对。"""
        d = ControllerConfig(prefill_deadline_ms=2000.0).describe()
        self.assertIn("prefill_deadline_ms", d,
                      "describe() 必须带这个键（它是两条 cost 臂的实验条件差异）")
        self.assertEqual(d["prefill_deadline_ms"], 2000.0)

    def test_default_describe_records_none_not_missing(self):
        self.assertIsNone(ControllerConfig().describe()["prefill_deadline_ms"])

    def test_yaml_can_turn_it_on(self):
        c = load_config(NEXT / "cost_dual.yaml")
        self.assertEqual(c.controller.prefill_deadline_ms, 2000.0)

    def test_yaml_single_budget_arm_keeps_it_off(self):
        """`fixed_cost.yaml`（单预算）⇒ 该字段缺省 ⇒ `None`（不是 0、不是 120）。"""
        c = load_config(FROZEN / "fixed_cost.yaml")
        self.assertIsNone(c.controller.prefill_deadline_ms)

    def test_arm_factory_passes_it_into_the_policy_config(self):
        """★ 接线端到端：`arm_factory` 构造 `CostPolicyConfig` 时必须传它。

        ⚠️ 用**源码级**核对而不是真的 `build_arm`：后者会触发成本模型的指纹校验，
        在非标定机上会**正确地**报错（实测：本机 torch 2.10.0 vs 标定 2.2.0）。
        这条断言的作用是**防"死字段"**：字段写进了产物、却谁也没读。
        """
        import inspect

        import arm_factory

        src = inspect.getsource(arm_factory)
        self.assertIn("prefill_deadline_ms", src,
                      "arm_factory 必须把 prefill_deadline_ms 传给 CostPolicyConfig —— "
                      "否则 ControllerConfig 里那个字段只是个没人读的死字段")


class TestSingleVariableBetweenCostArms(unittest.TestCase):
    """★ `cost_single` 与 `cost_dual` **只许差 `prefill_deadline_ms` 一项**。"""

    @staticmethod
    def _raw(path: Path) -> dict:
        c = load_config(path)
        # 用 `raw` 的深拷贝，剔掉生成器加的头部注释（注释不在 raw 里）
        return dataclasses.asdict(c) if dataclasses.is_dataclass(c) else dict(c.raw)

    def _diff_keys(self, a: dict, b: dict, prefix: str = "") -> set[str]:
        out: set[str] = set()
        for k in set(a) | set(b):
            ka, kb = f"{prefix}{k}", f"{prefix}{k}"
            va, vb = a.get(k), b.get(k)
            if isinstance(va, dict) and isinstance(vb, dict):
                out |= self._diff_keys(va, vb, f"{ka}.")
            elif va != vb:
                out.add(ka)
        return out

    def test_only_the_dual_budget_key_differs(self):
        single = load_config(FROZEN / "fixed_cost.yaml").raw
        dual = load_config(NEXT / "cost_dual.yaml").raw
        diff = self._diff_keys(single, dual)
        self.assertEqual(diff, {"controller.prefill_deadline_ms"},
                         f"两条 cost 臂只许差这一项，实际差 {sorted(diff)}")

    def test_both_are_the_same_kind(self):
        self.assertEqual(load_config(FROZEN / "fixed_cost.yaml").controller.kind, "cost")
        self.assertEqual(load_config(NEXT / "cost_dual.yaml").controller.kind, "cost")


class TestStaticBaselines(unittest.TestCase):
    """★ 静态基线：**只派生真正缺的那一份**，其余复用冷冻配置。"""

    def test_static_small_enables_chunking_at_the_same_starting_point(self):
        """`static_small` = 开分块 + `chunk_size == 自适应臂的 chunk_size_init`。

        ⚠️ 取同一个值才能问出"控制器的收益"，而不是"换了个 chunk 大小"。
        """
        small = load_config(NEXT / "static_small.yaml")
        cost = load_config(FROZEN / "fixed_cost.yaml")
        self.assertTrue(small.raw["chunk"]["enabled"])
        self.assertEqual(small.raw["chunk"]["chunk_size"],
                         cost.controller.chunk_size_init,
                         "静态基线的 chunk 必须与自适应臂同起点")

    def test_static_small_has_controller_off(self):
        """静态 = **不控制**：控制器必须关，否则它就不是静态基线。"""
        small = load_config(NEXT / "static_small.yaml")
        self.assertFalse((small.raw.get("controller") or {}).get("enabled", False),
                         "静态臂不得开启控制器")

    def test_static_nochunk_is_reused_not_duplicated(self):
        """★ `static_nochunk` **不另建文件** —— 它就是冷冻的 `fixed_static.yaml`。

        重复造一份 = 第二处真相：改了一处、另一处不会跟着变，而产物看不出区别。
        """
        frozen = FROZEN / "fixed_static.yaml"
        self.assertTrue(frozen.is_file(), "冷冻的 static 臂必须还在")
        c = load_config(frozen)
        self.assertFalse(c.raw["chunk"]["enabled"], "static_nochunk = 不分块")
        self.assertFalse((c.raw.get("controller") or {}).get("enabled", False),
                         "static_nochunk = 不控制")
        self.assertFalse((NEXT / "static_nochunk.yaml").exists(),
                         "不许复制一份 static_nochunk（用 fixed_static.yaml）")

    def test_threshold_arm_is_reused_not_duplicated(self):
        self.assertTrue((FROZEN / "fixed_threshold.yaml").is_file())
        self.assertFalse((NEXT / "threshold.yaml").exists(),
                         "不许复制一份 threshold（用 fixed_threshold.yaml）")

    def test_every_next_batch_config_is_derived_not_handwritten(self):
        """派生文件必须能被生成器**逐字节复现**（有人手改过就会不一致）。"""
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "gen_next_batch_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(_ROOT))
        self.assertEqual(proc.returncode, 0,
                         f"派生配置与生成器不一致：\n{proc.stdout}\n{proc.stderr}")

    def test_generated_configs_are_lf_only(self):
        for p in sorted(NEXT.glob("*.yaml")):
            self.assertEqual(p.read_bytes().count(b"\r"), 0, f"{p.name} 含 CR")


class TestDryRunIsResolutelyOffline(unittest.TestCase):
    """★ 离线 dry-plan：**不初始化 GPU、不连远端**（用**否定对照**证明）。

    为什么用否定对照：光看"跑完了"说明不了它没偷偷连网或没碰设备 ——
    只有**把网卡掉**再跑一遍还成功，才排除了这条路径。
    """

    def test_dry_run_completes_with_sockets_disabled(self):
        import socket
        import tempfile

        from run_benchmark import run_once

        cfg_path = NEXT / "static_small.yaml"
        real_connect = socket.socket.connect
        real_create = socket.create_connection

        def _boom(*a, **k):                                        # noqa: ANN002
            raise AssertionError("dry-run 路径尝试建立网络连接 ⇒ 它不是离线的")

        socket.socket.connect = _boom          # type: ignore[method-assign]
        socket.create_connection = _boom       # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                run_dir = run_once(load_config(cfg_path), dry_run=True,
                                   out_dir=Path(td), verbose=False)
                self.assertTrue((run_dir / "meta.json").is_file())
                import json

                meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        finally:
            socket.socket.connect = real_connect      # type: ignore[method-assign]
            socket.create_connection = real_create    # type: ignore[assignment]

        repro = meta.get("reproducibility") or {}
        self.assertEqual(repro.get("backend"), "simulate",
                         "dry-run 必须走**模拟**后端")
        self.assertEqual(meta.get("data_label"), "TEST_DATA",
                         "dry-run 产物必须标 TEST_DATA（不得进结论）")

    def test_cost_arms_cannot_dry_run_off_the_calibrated_host(self):
        """★ **如实记录**：cost 臂的 dry-plan **在本机跑不了**。

        成本模型加载时会做**指纹比对**，本机的 torch/transformers/python 与标定环境
        不同 ⇒ 它**正确地**拒绝加载。⇒ cost 臂的 dry-plan 必须在**标定环境**（远端）做。

        ⚠️ 这条不是"环境坏了"，而是"本机不是标定环境"——把两者混为一谈，
        会让人去修一个不该修的东西。
        """
        from run_benchmark import load_config as _lc

        c = _lc(NEXT / "cost_dual.yaml")
        self.assertEqual(c.controller.kind, "cost")
        # 单预算臂同理（同一份模型）——两条 cost 臂都受这条限制
        from run_benchmark import load_config as _lc2

        self.assertEqual(_lc2(FROZEN / "fixed_cost.yaml").controller.kind, "cost")
        # 而静态基线**不涉及**成本模型 ⇒ 本机可 dry-run（见上面的用例）
        self.assertNotEqual(load_config(NEXT / "static_small.yaml").controller.kind, "cost")


class TestS01RecheckPairIsSingleVariable(unittest.TestCase):
    """★ S01 复核那一对（`s4000_fp16` vs `s01_both`）的单变量性。

    ⚠️ **实测差异是 3 个键，不是 2 个**：

        runtime.logits_last_only      ← S01 开关（要测的那个）
        runtime.sample_needed_only    ← S01 开关（要测的那个）
        runtime.model_revision        ← ★ `main` → 钉死的 commit

    第三个**不是**第二个变量，而是**同一份权重的写法差异**（生成器把 `main`
    钉死成实测解析到的 commit）。但这只在"`main` 今天仍解析到同一个 commit"时成立
    —— 否则它就是**真的**换了权重，两臂不再单变量。
    ⇒ 所以把这件事**钉成测试**：钉死的值必须等于项目记录的 `PINNED_REVISION`，
    且开跑前必须在远端确认 `main` 仍指向它。
    """

    BASE = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"
    ARM = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both.yaml"
    S01_KEYS = {"runtime.logits_last_only", "runtime.sample_needed_only"}
    PIN_KEY = "runtime.model_revision"

    @staticmethod
    def _flat(o, p: str = "") -> dict:
        out: dict = {}
        if isinstance(o, dict):
            for k, v in o.items():
                out.update(TestS01RecheckPairIsSingleVariable._flat(
                    v, f"{p}.{k}" if p else k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                out.update(TestS01RecheckPairIsSingleVariable._flat(v, f"{p}[{i}]"))
        else:
            out[p] = o
        return out

    def _diff(self) -> set[str]:
        a = self._flat(load_config(self.BASE).raw)
        b = self._flat(load_config(self.ARM).raw)
        return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

    def test_diff_is_exactly_the_two_switches_plus_the_revision_pin(self):
        diff = self._diff()
        self.assertEqual(diff, self.S01_KEYS | {self.PIN_KEY},
                         f"差异集变了：{sorted(diff)} ⇒ 单变量前提要重核")

    def test_revision_pin_equals_the_project_recorded_commit(self):
        pinned = load_config(self.ARM).raw["runtime"]["model_revision"]
        self.assertEqual(pinned, "a09a35458c702b33eeacc393d103063234e8bc28",
                         "钉死的 revision 必须等于项目记录的 PINNED_REVISION")
        self.assertEqual(load_config(self.BASE).raw["runtime"]["model_revision"],
                         "main", "源配置走的是 `main`（引用）")

    # ⚠️ **"`main` 今天是否仍指向钉死的 commit" 这条前置不写成测试**：
    #    它需要远端的 `git ls-remote`，离线侧**无法**验证。写一个恒真的断言来"代表它"
    #    比不写更坏 —— 那正是本仓反复记的"假通过"（一个不会失败的检查）。
    #    ⇒ 它作为**开跑前置**写在 `NEXT_BATCHES_PREREG.md` §7，由批次前的 preflight 执行。


class TestStaticSmallDefinitionIsPinned(unittest.TestCase):
    """★★ `static_small` 的 `chunk_size` **在两处文档里定义相反** —— 本类把实况钉住。

    实况（2026-10-03 现读）：
      * `NEXT_SESSION_TASKS.md` §3 说 = 自适应臂的 **`chunk_size_min`**（本 cell = 512）；
      * `NEXT_BATCHES_PREREG.md` §3 + 生成器说 = **`chunk_size_init`**（= 1024）；
      * **生成器实际产出的是 1024**。

    ⇒ 这不是"哪份文档错了"的判断题，而是**两个不同的候选基线**（同起点 vs 同下界）。
    本类的作用：一旦有人改了生成器的取值或改了本 cell 的 `init/min`，
    这里就必须**重核**（并同步 §9.1 的裁定），不会静默分叉。
    """

    def test_next_batch_static_small_uses_chunk_size_init(self):
        small = load_config(NEXT / "static_small.yaml")
        cost = load_config(FROZEN / "fixed_cost.yaml")
        self.assertEqual(small.raw["chunk"]["chunk_size"],
                         cost.controller.chunk_size_init,
                         "生成器当前取的是 chunk_size_init（同起点基线）")

    def test_frozen_cell_min_is_512_not_128(self):
        """★ 本 cell 的 `chunk_size_min` 是 **512**；`128` 属于**另一个** cell。

        把两者混为一谈会派生出**错的**基线 —— 这正是定义冲突的根源。
        """
        cost = load_config(FROZEN / "fixed_cost.yaml")
        self.assertEqual(cost.controller.chunk_size_min, 512,
                         "本 cell 的下界变了 ⇒ 必须重核 NEXT_BATCHES_PREREG.md §9.1")
        self.assertNotEqual(cost.controller.chunk_size_min, 128,
                            "128 是 phase1_d1/fixed_cost_min128.yaml 那**另一个** cell 的下界")

    def test_the_two_candidate_values_differ(self):
        """两个候选（同起点 / 同下界）在当前 cell 下**确实不同** ⇒ 必须二选一并写明。"""
        cost = load_config(FROZEN / "fixed_cost.yaml")
        self.assertNotEqual(cost.controller.chunk_size_init,
                            cost.controller.chunk_size_min,
                            "若两者相等，这个定义冲突就自然消失了（届时删掉本类）")


class TestCostModelVersionConsistency(unittest.TestCase):
    """★★ v5 / v6 / v7 **三个版本并存**：把"哪个配置引用哪个模型"钉成**会失败的**检查。

    实况（2026-10-03 现读，权威来源 = `configs/mthreads/**/*.yaml` 的
    `controller.cost_model_path`）：

      * `phase1_step14/*`（本小批 `cost_single` 的来源）→ **v5**（91 键）
      * `next_batch/cost_dual.yaml`（新派生，继承自上面）→ **v5**
      * `phase1_step15/*`、`phase1_step14b/*` → **v6**   ← ★ 我最初 grep 漏掉的那批
      * `phase1_d1/*`、`phase1_d1_scan/*`、`phase1_s6/*` → **v7**（107 键）

    ⚠️ 这条"三版本并存"是**先写断言、跑出来才发现的**：我最初以为只有 v5/v7 两版，
    断言写成 `{v5, v7}` 后 **测试立刻变红**（实际集合含 v6）。⇒ 保留这个断言的价值
    正在于此：它让"版本集合变了"**无法被静默忽略**。

    ⚠️ 本类**不**断言"必须都是 v7"（那是 §9.2 的**裁定**，要改配置才能成立）；
    它只保证：① 每个引用的模型文件**存在**；② 版本差异**可见**（可枚举、可计数）。
    ⇒ 若有人悄悄把某个配置改到另一个版本，`test_model_versions_in_use_are_enumerable`
    的集合会变 ⇒ 测试红 ⇒ 必须显式更新 §9.2。
    """

    def _model_paths(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted((_ROOT / "configs" / "mthreads").rglob("*.yaml")):
            try:
                c = load_config(p)
            except Exception:
                continue
            ctrl = (c.raw.get("controller") or {})
            mp = ctrl.get("cost_model_path")
            if mp:
                out[str(p.relative_to(_ROOT)).replace("\\", "/")] = str(mp)
        return out

    def test_every_referenced_model_file_exists(self):
        missing = []
        for cfg, mp in self._model_paths().items():
            if not (_ROOT / mp).is_file():
                missing.append(f"{cfg} → {mp}")
        self.assertEqual(missing, [], f"引用了不存在的成本模型：{missing}")

    def test_the_two_cost_arms_share_one_model_version(self):
        """本小批的两条 cost 臂必须**引用同一个**模型版本（否则不只是双预算一个变量）。"""
        paths = self._model_paths()
        a = paths.get("configs/mthreads/next_batch/cost_dual.yaml")
        b = paths.get("configs/mthreads/phase1_step14/fixed_cost.yaml")
        self.assertIsNotNone(a, "cost_dual.yaml 必须引用成本模型")
        self.assertIsNotNone(b, "fixed_cost.yaml 必须引用成本模型")
        self.assertEqual(a, b,
                         f"两条 cost 臂引用了不同模型：{a} vs {b} ⇒ 多了一个变量")

    def test_model_versions_in_use_are_enumerable(self):
        """★ 把"当前在用哪些版本"变成**可枚举的集合** —— 集合一变就红。

        ★ **现状集合 = {v5, v6, v7}（三个）**，不是两版。这是本类跑红之后**实测出来的**：
        我先前只按 `phase1_step14` vs `phase1_d1` 两处对读，漏掉了 `phase1_step15/*`
        与 `phase1_step14b/*` 这两个整目录都指向 **v6**。
        ⇒ 这正是"集合型断言"的价值：它把**我没有看见的东西**变成阻塞项。

        统一之后集合会变成 {v7}，届时**必须**同步更新断言并在 §9.2 记一笔。
        """
        used = {Path(mp).name for mp in self._model_paths().values()}
        self.assertTrue(used, "没有任何配置引用成本模型？检查 _model_paths 的解析路径")
        # 现状：v5 / v6 / v7 三版并存（不一致，待 §9.2 统一）。集合变了就必须显式更新。
        self.assertEqual(
            used, {"model_v5.json", "model_v6.json", "model_v7.json"},
            f"在用模型版本集合变了：{sorted(used)}；"
            "若已按 NEXT_BATCHES_PREREG.md §9.2 统一，请同步更新本断言")


if __name__ == "__main__":
    unittest.main()
