"""PHASE 1 第③步矩阵的**可归因性**守卫。

这张矩阵的结论完全依赖一件事：**每个臂只比相邻臂多一个开关**。一旦多改一处
（例如顺手把 ``chunk_size`` 也调了），配置依然合法、run 依然成功、产物依然齐全，
只是整张矩阵的归因作废了——而没有任何运行时检查会发现它。

所以这里把"单变量"打成断言，并且**连守卫本身一起测**：一个不会失败的守卫
等于没有守卫，因此每个检查都配一个反向对照（注入缺陷 → 必须报警）。

同时也钉住那个曾经真实发生过的缺陷：``runtime.model`` 写成裸模型名
（``Qwen2.5-7B-Instruct`` 而不是 ``Qwen/Qwen2.5-7B-Instruct``）时，HF 匹配不上
本地缓存目录 ``models--Qwen--Qwen2.5-7B-Instruct``，会去走网络并报 404 ——
现象像"模型没下载"，实际是"名字少了一段"。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CFG_DIR = _ROOT / "configs" / "mthreads" / "phase1_step3"
#: 修正版（第③步重跑）。两份都在测：旧的那份是复现路径，新的那份是要跑的东西。
_CFG_DIR_B = _ROOT / "configs" / "mthreads" / "phase1_step3b"
#: 第二份修正版（P3 前段）：只把 ``run.slo.tpot_ms`` 从 50 改成 100。
_CFG_DIR_C = _ROOT / "configs" / "mthreads" / "phase1_step3c"
#: 第三份修正版（P3 后段）：只把控制门限从 ``110/0.1`` 改成 ``120/0.08``。
#: 它是**当前要跑的那一版**（唯一没有已知缺陷的 profile）。
_CFG_DIR_D = _ROOT / "configs" / "mthreads" / "phase1_step3d"
_ALL_CFG_DIRS = (_CFG_DIR, _CFG_DIR_B, _CFG_DIR_C, _CFG_DIR_D)
#: 已执行、必须逐字可复现的 profile（p1s3 → P1S3-*，p1s3b → P1S3B-*/P2-*）。
_EXECUTED_PROFILES = ("p1s3", "p1s3b")

#: 脚本名以数字/字母开头的可执行脚本，用 importlib 按路径加载，
#: 避免把 scripts/ 变成一个包（项目惯例是 scripts/ 下放可执行脚本）。
def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    modname = f"_step3_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：脚本用了 ``from __future__ import annotations``，
    # 于是 @dataclass 的字段注解是**字符串**，dataclasses 解析时要拿
    # ``sys.modules[cls.__module__]`` —— 不注册就会以
    # ``AttributeError: 'NoneType' object has no attribute '__dict__'`` 整块失败。
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGeneratedConfigsExistAndLoad(unittest.TestCase):
    """12 个配置必须存在、可加载，且都指向完整 repo id 的模型。"""

    def test_all_expected_configs_exist(self):
        gen = _load_script("gen_phase1_step3_configs")
        expected = {
            f"{arm}_c{c}"
            for arm in gen.ARMS
            for c in gen.CONCURRENCIES
        }
        for cfg_dir in _ALL_CFG_DIRS:
            with self.subTest(dir=cfg_dir.name):
                found = {p.stem for p in cfg_dir.glob("*.yaml")}
                self.assertTrue(
                    expected <= found,
                    f"{cfg_dir.name} 缺少配置：{sorted(expected - found)}。"
                    f"跑 scripts/gen_phase1_step3_configs.py --profile <...> 重新生成。",
                )

    def test_all_configs_load(self):
        from run_benchmark import load_config

        for cfg_dir in _ALL_CFG_DIRS:
            for path in sorted(cfg_dir.glob("*.yaml")):
                with self.subTest(config=f"{cfg_dir.name}/{path.name}"):
                    cfg = load_config(path)
                    self.assertFalse(cfg.cache.enabled,
                                     "真实后端拒绝前缀缓存，矩阵里不该出现")
                    self.assertGreaterEqual(cfg.run.repeats, 3)

    def test_model_id_is_full_repo_id(self):
        """裸模型名会命中不了 HF 缓存并去走网络 —— 这个缺陷真实发生过。"""
        from run_benchmark import load_config

        for cfg_dir in _ALL_CFG_DIRS:
            for path in sorted(cfg_dir.glob("*.yaml")):
                cfg = load_config(path)
                model = str(cfg.runtime.get("model"))
                with self.subTest(config=f"{cfg_dir.name}/{path.name}"):
                    self.assertIn(
                        "/", model,
                        f"{model!r} 缺少 `<org>/` 前缀；HF 按 repo id 找缓存，"
                        f"写成裸名会绕过已下载的权重去走网络",
                    )


class TestSingleVariableProperty(unittest.TestCase):
    """臂之间只差一个开关；并发轴内只差并发。"""

    def test_generator_reports_single_variable(self):
        gen = _load_script("gen_phase1_step3_configs")
        self.assertEqual(gen.main(["--check"]), 0,
                         "生成器的单变量自检未通过")

    def test_arm_edges_differ_by_exactly_one_switch(self):
        gen = _load_script("gen_phase1_step3_configs")
        edges = {
            ("static", "static_chunk"): "chunk.enabled",
            ("static_chunk", "adaptive"): "controller.enabled",
        }
        # 两份 profile 都要成立：p1s3 是已执行的那一版（复现用），
        # p1s3b 是修正版。只测默认 profile 会漏掉"复现路径被改坏"。
        for pname, profile in gen.PROFILES.items():
            configs = gen.build(profile)
            for (a, b), expect in edges.items():
                with self.subTest(profile=pname, edge=f"{a}->{b}"):
                    diffs = gen._assert_single_variable(
                        configs, f"{a}_c1", f"{b}_c1"
                    )
                    unexpected = [d for d in diffs if d not in (expect, "run.notes")]
                    self.assertEqual(
                        unexpected, [],
                        f"{a} 与 {b} 除了 {expect} 之外还差了 {unexpected}；"
                        f"多变量会让整张矩阵无法归因",
                    )

    def test_concurrency_axis_differs_only_by_concurrency(self):
        gen = _load_script("gen_phase1_step3_configs")
        for pname, profile in gen.PROFILES.items():
            configs = gen.build(profile)
            for arm in gen.ARMS:
                cs = gen.CONCURRENCIES
                for a, b in zip(cs, cs[1:]):
                    with self.subTest(profile=pname, arm=arm, pair=f"{a}->{b}"):
                        diffs = gen._assert_single_variable(
                            configs, f"{arm}_c{a}", f"{arm}_c{b}"
                        )
                        unexpected = [
                            d for d in diffs
                            if d not in ("workload.concurrency", "run.notes")
                        ]
                        self.assertEqual(unexpected, [])


class TestGuardDetectsTheDefectItClaimsTo(unittest.TestCase):
    """反向对照：把缺陷注进去，守卫**必须**报警。

    没有这一段，上面那些断言只能证明"现在通过"，不能证明"以后改坏了会被发现"。
    """

    def test_guard_fires_when_second_variable_injected(self):
        gen = _load_script("gen_phase1_step3_configs")
        original = dict(gen.ARMS["adaptive"])
        try:
            gen.ARMS["adaptive"] = {
                "chunk.enabled": True,
                "controller.enabled": True,
                "chunk.max_num_batched_tokens": 4096,  # 第二个变量
            }
            self.assertNotEqual(
                gen.main(["--check"]), 0,
                "注入了第二个变量，单变量自检却仍然通过 —— 守卫不起作用",
            )
        finally:
            gen.ARMS["adaptive"] = original

    def test_model_id_check_fires_on_bare_name(self):
        mod = _load_script("check_step3_configs")
        self.assertTrue(
            mod._check_model_id("Qwen2.5-7B-Instruct", None),
            "裸模型名应当被拦下",
        )
        self.assertEqual(mod._check_model_id("Qwen/Qwen2.5-7B-Instruct", None), [])
        self.assertTrue(mod._check_model_id(None, None), "空模型名应当被拦下")


class TestControllerMovableGuard(unittest.TestCase):
    """被扫描的维度必须**真的可动**。

    第③步第一次执行的 ``adaptive`` 臂就是栽在这里：``chunk_size_init == chunk_size_min``
    使 chunk 维度死锁，36/36 次运行里 ``chunk_req`` 恒为 512/512、``n_recovered=0``。
    产物齐全、无报错、标签全绿，只有结论是错的。

    这个守卫第一版写错过：它读 ``Profile`` 的字段，而 ``p1s3`` 的
    ``chunk_size_min`` 是 ``None``（表示"沿用基线"），于是每一条检查都被
    ``if ... is not None`` 跳过 —— 守卫对**恰恰有缺陷的那份**恒真。
    所以下面既测"能放行"，也测"能拦下"。
    """

    def test_corrected_profile_has_room_in_both_directions(self):
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3b"])["adaptive_c1"]
        probs = gen._assert_controller_movable(cfg)
        self.assertEqual(probs, [], f"修正版不该有可动性问题：{probs}")
        ctl = cfg["controller"]
        self.assertLess(ctl["chunk_size_min"], ctl["chunk_size_init"],
                        "初值必须严格大于下界，否则既不能收缩也不能恢复")
        self.assertGreater(ctl["chunk_size_max"], ctl["chunk_size_init"],
                           "初值必须严格小于上界，否则不能放大（单向棘轮）")

    def test_guard_fires_on_the_executed_flawed_profile(self):
        """反向对照：守卫必须在**已执行的那份**上报错，否则它是个恒真报警器。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3"])["adaptive_c1"]
        probs = gen._assert_controller_movable(cfg)
        self.assertTrue(probs, "p1s3 的 chunk 维度不可动，守卫却没有报错")
        joined = " ".join(probs)
        self.assertIn("不可收缩", joined)
        self.assertIn("不切分", joined)

    def test_guard_uses_effective_values_not_profile_placeholders(self):
        """守卫必须读**最终产物**的生效值，不能读 ``Profile`` 里可能是 None 的字段。

        这条断言的价值在于它钉住了一个真实发生过的写法错误。若守卫改回读 profile，
        ``p1s3`` 的 ``chunk_size_min`` 是 ``None``，检查全部被跳过 —— 这个测试会失败。
        """
        gen = _load_script("gen_phase1_step3_configs")
        profile = gen.PROFILES["p1s3"]
        self.assertIsNone(profile.chunk_size_min,
                          "前提：p1s3 没显式覆盖 chunk_size_min（沿用基线）")
        cfg = gen.build(profile)["adaptive_c1"]
        self.assertIsNotNone(cfg["controller"]["chunk_size_min"],
                             "生成的配置里它有基线值；守卫就该读这个")
        self.assertTrue(gen._assert_controller_movable(cfg))

    def test_dead_zone_above_prompt_len_is_rejected(self):
        """上界超过 prompt 全长 ⇒ "grow 了但不切分"，必须被拦下。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3b"])["adaptive_c1"]
        cfg["controller"]["chunk_size_max"] = 4096  # 超过 phase1_anchor 的 2048
        probs = gen._assert_controller_movable(cfg)
        self.assertTrue(any("不切分" in p for p in probs),
                        f"超出 prompt 全长的上界未被拦下：{probs}")

    def test_generating_flawed_profile_requires_explicit_acknowledgement(self):
        """有已知缺陷的 profile 默认拒绝生成；显式确认后才放行。"""
        gen = _load_script("gen_phase1_step3_configs")
        self.assertNotEqual(gen.main(["--profile", "p1s3", "--check"]), 0,
                            "带已知缺陷的 profile 不该静默通过")
        self.assertEqual(
            gen.main(["--profile", "p1s3", "--allow-known-defects", "--check"]), 0,
            "显式确认后应放行（复现已执行证据需要这条路径）",
        )


class TestSloReachableGuard(unittest.TestCase):
    """SLO 判据必须**可达**，否则达标率三列不携带信息（P3）。

    失效形状：``run.slo`` 不参与任何调度决策 ⇒ 一个不可达的阈值**不会**让 run 失败、
    不会让产物缺字段、标签照样全绿；它只让 ``slo_satisfaction_rate`` /
    ``slo_goodput_*`` 恒为 0，在表里与"所有配置都很差"长得一模一样。
    P1S3/P1S3B 两批共 84 份产物都带着这个缺陷。

    两个方向都要拦：低于实测下界 ⇒ 恒 0；高于上界 ⇒ 恒 1。**两种都不携带信息。**
    """

    def test_guard_accepts_calibrated_profile(self):
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3c"])["adaptive_c1"]
        self.assertEqual(gen._assert_slo_reachable(cfg), [],
                         "重标定后的 profile 不该被判为不可达")
        self.assertEqual(cfg["run"]["slo"]["tpot_ms"], 100)

    def test_guard_fires_below_floor(self):
        """反向对照：低于实测下界的阈值必须被拦下。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3c"])["adaptive_c1"]
        cfg["run"]["slo"]["tpot_ms"] = 50  # 低于实测下界 61.7
        probs = gen._assert_slo_reachable(cfg)
        self.assertTrue(probs, "50 ms 低于实测下界，守卫却没有报错")
        self.assertIn("恒不可达", " ".join(probs))

    def test_guard_fires_above_ceiling(self):
        """反向对照：高于实测上界的阈值同样不携带信息（恒 1）。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3c"])["adaptive_c1"]
        cfg["run"]["slo"]["tpot_ms"] = 9999
        probs = gen._assert_slo_reachable(cfg)
        self.assertTrue(probs, "9999 ms 高于实测上界，守卫却没有报错")
        self.assertIn("恒可达", " ".join(probs))

    def test_guard_fires_on_none(self):
        """``None`` 是"没测到"，不是"达标"——不可测的值不能当合格判据。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3c"])["adaptive_c1"]
        cfg["run"]["slo"]["tpot_ms"] = None
        self.assertTrue(gen._assert_slo_reachable(cfg))

    def test_guard_fires_on_the_executed_flawed_profiles(self):
        """守卫必须在**已执行的那两份**上真的报错，否则它是个恒真报警器。

        同时也钉住了"为什么它们需要 ``--allow-known-defects``"。
        """
        gen = _load_script("gen_phase1_step3_configs")
        for pname in ("p1s3", "p1s3b"):
            with self.subTest(profile=pname):
                cfg = gen.build(gen.PROFILES[pname])["adaptive_c1"]
                self.assertEqual(cfg["run"]["slo"]["tpot_ms"], 50,
                                 f"{pname} 复现时必须仍是 50")
                self.assertTrue(
                    gen._assert_slo_reachable(cfg),
                    f"{pname} 的 SLO 判据不可达，守卫却没报错",
                )

    def test_flawed_profiles_require_explicit_acknowledgement(self):
        """带该缺陷的 profile 默认拒绝生成，显式确认后放行。"""
        gen = _load_script("gen_phase1_step3_configs")
        self.assertNotEqual(gen.main(["--profile", "p1s3b", "--check"]), 0,
                            "带 SLO 缺陷的 profile 不该静默通过")
        self.assertEqual(
            gen.main(["--profile", "p1s3b", "--allow-known-defects", "--check"]), 0,
        )
        self.assertEqual(gen.main(["--check"]), 0,
                         "默认 profile 应当是当前要跑的那一版（p1s3d），无需逃生口")


class TestGateSeparationGuard(unittest.TestCase):
    """控制门限必须与**实测工作点**分得开（P3 后段）。

    失效形状与前面两道门不同，而且更隐蔽：门限贴着工作点时控制器**不报错、
    不停止，而是随机地动**。P1S3B 的 ``adaptive_c4`` 就是这样——放大门限 =
    ``110×0.9`` = 99.0，而 c4 的 ``tpot_p90`` 工作点是 97.8–99.2（**覆盖**门限），
    放大判据的 headroom 余量只剩 0.0112（门槛 0.1000）⇒ 三次重复的 TTFT 是
    481.4 / 481.6 / 652.1。同一份配置、同一台机器，差异全部来自一次浮点判定的
    符号。这种缺陷不会让任何 run 失败，只会让整批结论在重复之间不可复现。

    两个方向都要查：某个工作点落在门限邻域内 ⇒ 不可复现；**全部**工作点高于
    收缩门限 ⇒ 放大分支永不触发（退化成单向棘轮，与 P1S3 同类）。
    """

    def test_guard_accepts_calibrated_profile(self):
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]
        ctl = cfg["controller"]
        self.assertEqual(ctl["target_tpot_ms"], 120.0)
        self.assertEqual(ctl["hysteresis"], 0.08)
        self.assertEqual(gen._assert_gate_separation(cfg), [],
                         "重标定后的门限不该被判为贴着工作点")

    def test_thresholds_are_reported_as_designed(self):
        """门限值本身也钉住：放大 110.4 / 收缩 129.6（= 120 ∓ 0.08·120）。"""
        gen = _load_script("gen_phase1_step3_configs")
        ctl = gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]["controller"]
        t, h = ctl["target_tpot_ms"], ctl["hysteresis"]
        self.assertAlmostEqual(t * (1 - h), 110.4, places=6)
        self.assertAlmostEqual(t * (1 + h), 129.6, places=6)

    def test_guard_fires_on_the_executed_flawed_profile(self):
        """反向对照：守卫必须在**已执行的** ``p1s3b`` 上真的报错，否则是摆设。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3b"])["adaptive_c1"]
        self.assertEqual(cfg["controller"]["hysteresis"], 0.1,
                         "前提：p1s3b 沿用基线 0.1")
        probs = gen._assert_gate_separation(cfg)
        self.assertTrue(probs, "p1s3b 的放大门限落在 c4 工作点上，守卫却没报错")
        joined = " ".join(probs)
        self.assertIn("c4", joined)
        self.assertIn("不可复现", joined)

    def test_guard_fires_when_gate_sits_exactly_on_working_point(self):
        """把门限**正好**压在某个工作点上，守卫必须报错。

        这条比"在已执行的那份上报错"更硬：它证明判据是
        "门限与工作点分不分得开"，而不是"恰好那两份参数被写死了"。
        """
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]
        ctl = cfg["controller"]
        ctl["hysteresis"] = 0.08
        ctl["target_tpot_ms"] = 98.7 / 0.92  # 使 target·(1−h) == c4 的工作点 98.7
        probs = gen._assert_gate_separation(cfg)
        self.assertTrue(probs, "门限正好压在工作点上却没报错")
        self.assertIn("c4", " ".join(probs))

    def test_guard_fires_on_degenerate_ratchet(self):
        """全部工作点高于收缩门限 ⇒ 放大分支永不触发（与 P1S3 同类）。"""
        gen = _load_script("gen_phase1_step3_configs")
        cfg = gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]
        cfg["controller"]["target_tpot_ms"] = 50.0
        cfg["controller"]["hysteresis"] = 0.1
        probs = gen._assert_gate_separation(cfg)
        self.assertTrue(any("退化棘轮" in p for p in probs),
                        f"全部工作点高于收缩门限却未报棘轮：{probs}")

    def test_guard_fires_on_none(self):
        """``None`` = 没声明，不是"没问题"——不可判定的值不能当合格判据。"""
        gen = _load_script("gen_phase1_step3_configs")
        for field in ("hysteresis", "target_tpot_ms"):
            with self.subTest(field=field):
                cfg = gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]
                cfg["controller"][field] = None
                self.assertTrue(
                    gen._assert_gate_separation(cfg),
                    f"{field} 为 None 时守卫应报错",
                )

    def test_guard_reads_effective_hysteresis_not_profile_placeholder(self):
        """守卫必须读**最终产物**的 ``hysteresis``，不能读 ``Profile`` 的 ``None``。

        与 :func:`_assert_controller_movable` 的同类错误对称：那份守卫曾对
        **恰恰有缺陷的那份配置**恒真，因为它读的是 profile 里为 ``None`` 的字段。
        这里钉住同样的写法错误不会重演。
        """
        gen = _load_script("gen_phase1_step3_configs")
        profile = gen.PROFILES["p1s3c"]
        self.assertIsNone(profile.hysteresis, "前提：p1s3c 未显式覆盖 hysteresis")
        cfg = gen.build(profile)["adaptive_c1"]
        self.assertEqual(cfg["controller"]["hysteresis"], 0.1,
                         "生成的配置里它有基线值；守卫就该读这个")
        self.assertTrue(gen._assert_gate_separation(cfg))

    def test_flawed_profiles_require_explicit_acknowledgement(self):
        """带门限缺陷的 profile 默认拒绝生成，显式确认后放行。"""
        gen = _load_script("gen_phase1_step3_configs")
        self.assertNotEqual(gen.main(["--profile", "p1s3c", "--check"]), 0,
                            "门限贴着工作点的 profile 不该静默通过")
        self.assertEqual(
            gen.main(["--profile", "p1s3c", "--allow-known-defects", "--check"]), 0,
        )

    def test_current_profile_is_the_defect_free_one(self):
        """当前要跑的 profile 必须是三道门都过、且**没有声明缺陷**的那一版。"""
        gen = _load_script("gen_phase1_step3_configs")
        self.assertEqual(gen.main(["--check"]), 0, "默认 profile 应当可无条件生成")
        self.assertEqual(gen.main(["--profile", "p1s3d", "--check"]), 0)
        self.assertEqual(gen.PROFILES["p1s3d"].known_defects, (),
                         "p1s3d 若声明了已知缺陷，就不该是当前要跑的那一版")


class TestExecutedConfigsAreByteReproducible(unittest.TestCase):
    """已执行配置必须能由生成器**逐字**复现。

    这条约束不是洁癖：``results/raw/P1S3-*`` 是首批 ``MEASURED`` 证据，
    它们与"当时跑的那份配置"的对应关系一旦断了（哪怕只差一个注释行），
    追溯链就只能靠人记。生成器重构后（引入 profile、加可动性守卫）必须仍然复现，
    所以这里把它做成断言而不是靠一次手工 diff。
    """

    def test_p1s3_profile_reproduces_committed_configs(self):
        self._assert_profile_reproduces("p1s3", _CFG_DIR)

    def test_p1s3b_profile_reproduces_committed_configs(self):
        """``p1s3b`` 也是已执行证据（``P1S3B-*``/``P2-*``），同样不许漂移。

        它现在带一条**已声明的** SLO 缺陷（``run.slo.tpot_ms`` 仍为 50），
        所以复现时要带 ``--allow-known-defects``。这条测试同时钉住了"模板改了之后
        已执行配置不能跟着变"——平台模板已改为 100，若 profile 忘了显式钉死 50，
        这里会立刻报出 12 个 diff。
        """
        self._assert_profile_reproduces("p1s3b", _CFG_DIR_B)

    def test_p1s3c_profile_reproduces_committed_configs(self):
        """``p1s3c`` 的产物虽未执行，但已入库；库里那份与生成器的对应关系不许断。

        它带一条**已声明的**门限缺陷（见 :class:`TestGateSeparationGuard`），
        所以同样要带 ``--allow-known-defects``。
        """
        self._assert_profile_reproduces("p1s3c", _CFG_DIR_C)

    def test_p1s3d_profile_reproduces_committed_configs(self):
        """``p1s3d`` 是当前要跑的那一版（唯一无缺陷）。产物一旦入库就钉住。"""
        self._assert_profile_reproduces("p1s3d", _CFG_DIR_D)

    def _assert_profile_reproduces(self, profile: str, committed_dir: Path) -> None:
        import tempfile

        gen = _load_script("gen_phase1_step3_configs")
        with tempfile.TemporaryDirectory() as td:
            rc = gen.main([
                "--profile", profile, "--allow-known-defects", "--out-dir", td,
            ])
            self.assertEqual(rc, 0)
            committed = {p.name: p.read_bytes() for p in committed_dir.glob("*.yaml")}
            regenerated = {
                p.name: p.read_bytes() for p in Path(td).glob("*.yaml")
            }
            self.assertEqual(sorted(committed), sorted(regenerated),
                             "生成的文件名集合与已提交的不一致")
            for name in sorted(committed):
                with self.subTest(profile=profile, config=name):
                    self.assertEqual(
                        committed[name], regenerated[name],
                        f"{name} 无法逐字复现；已执行的配置与生成器输出必须一致，"
                        f"否则它们的追溯链就断了",
                    )

    def test_p1s3c_differs_from_p1s3b_only_in_slo_and_label(self):
        """新 profile 与已执行的 ``p1s3b`` 必须**只差** SLO 判据与标签。

        这是 P3 的单变量性断言：多一个差异，"达标率变了"就不能归因到阈值上。
        同时它也保证 P3 **不可能**顺手改到调度行为——``run.slo`` 之外无改动。
        """
        gen = _load_script("gen_phase1_step3_configs")
        a = gen.build(gen.PROFILES["p1s3b"])
        b = gen.build(gen.PROFILES["p1s3c"])
        self.assertEqual(sorted(a), sorted(b), "两版的格集合不同")
        for stem in sorted(a):
            with self.subTest(config=stem):
                diffs = gen._assert_single_variable(
                    {"a": a[stem], "b": b[stem]}, "a", "b"
                )
                self.assertEqual(
                    sorted(diffs), ["run.notes", "run.slo.tpot_ms"],
                    f"{stem} 除了 SLO 判据与标签之外还差了 "
                    f"{[d for d in diffs if d not in ('run.notes', 'run.slo.tpot_ms')]}",
                )

    def test_p1s3d_differs_from_p1s3c_only_in_gate_and_label(self):
        """``p1s3d`` 与 ``p1s3c`` 必须**只差**控制门限的两个参数与标签。

        ``target_tpot_ms`` 与 ``hysteresis`` 是**同一个变量**（门限位置）的两个
        参数，所以它们一起改不违反单变量性；除此之外多一个差异，"行为变了"就
        不能归因到门限上——尤其 ``run.slo.tpot_ms`` 必须两边都是 100，否则
        达标率的变化会同时归因到判据上。
        """
        gen = _load_script("gen_phase1_step3_configs")
        a = gen.build(gen.PROFILES["p1s3c"])
        b = gen.build(gen.PROFILES["p1s3d"])
        self.assertEqual(sorted(a), sorted(b), "两版的格集合不同")
        allowed = {
            "run.notes",
            "controller.target_tpot_ms",
            "controller.hysteresis",
        }
        for stem in sorted(a):
            with self.subTest(config=stem):
                diffs = gen._assert_single_variable(
                    {"a": a[stem], "b": b[stem]}, "a", "b"
                )
                self.assertEqual(
                    sorted(diffs), sorted(allowed),
                    f"{stem} 除了门限两参数与标签之外还差了 "
                    f"{sorted(d for d in diffs if d not in allowed)}",
                )
        self.assertEqual(
            gen.build(gen.PROFILES["p1s3d"])["adaptive_c1"]["run"]["slo"]["tpot_ms"],
            100,
            "p1s3d 的 SLO 判据必须与 p1s3c 同为 100（否则两次改动混在一起）",
        )

    def test_regenerated_configs_have_unix_line_endings(self):
        """生成物必须是 LF。Windows 上默认写 CRLF，跨机器比对会出假 diff。

        用 ``p1s3d``（**唯一**没有已知缺陷的 profile）：``p1s3b``/``p1s3c`` 都带
        一条**已声明的**缺陷，不带 ``--allow-known-defects`` 会被守卫拒掉——那条
        路径由 :class:`TestSloReachableGuard` / :class:`TestGateSeparationGuard`
        专门覆盖，不在这里顺带测。
        """
        import tempfile

        gen = _load_script("gen_phase1_step3_configs")
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(
                gen.main(["--profile", "p1s3d", "--out-dir", td]), 0
            )
            files = list(Path(td).glob("*.yaml"))
            self.assertTrue(files, "没有生成任何配置（空输入会伪装成合格结果）")
            for p in files:
                with self.subTest(config=p.name):
                    self.assertNotIn(b"\r", p.read_bytes(),
                                     f"{p.name} 含 CR；生成物必须一律 LF")


class TestMatrixRunnerInterleaves(unittest.TestCase):
    """执行顺序必须是"扫完整张表"为一轮，而不是每格背靠背跑 3 次。

    背靠背会把一次偶发停顿直接归因成配置差异（已实测 stall_p99 差 4.35×）。
    """

    def test_plan_is_interleaved(self):
        mod = _load_script("run_step3_matrix")
        cells = mod.default_cells()
        self.assertEqual(len(cells), 12)
        repeats = 3
        # 复刻 main 里的顺序生成逻辑（不实际运行远端）
        order = [(p, name) for p in range(repeats) for name in cells]
        self.assertEqual(len(order), 36)
        # 同一格的两次重复之间，必须隔着整张表
        for name in cells:
            idx = [i for i, (_, n) in enumerate(order) if n == name]
            self.assertEqual(len(idx), repeats)
            for a, b in zip(idx, idx[1:]):
                self.assertEqual(b - a, len(cells),
                                 f"{name} 的重复没有隔开整张表")

    def test_arms_and_concurrencies(self):
        mod = _load_script("run_step3_matrix")
        self.assertEqual(mod.CONCURRENCIES, (1, 2, 4, 8))
        self.assertEqual(mod.ARMS, ("static", "static_chunk", "adaptive"))


class TestRunStatusGate(unittest.TestCase):
    """矩阵的"成功"必须读**落盘产物**的计数，不能只看有没有抛异常。

    失效形状：真实后端对**单请求**的异常是"接住并记录"的——显存不足会被写进
    该请求的 outcome，进程照常收尾、``run_once`` 照常返回。于是
    **"函数返回了" ≠ "活干完了"**。实测过：注入 OOM 后产物是
    "0 成功 / 1 失败 / 7 未完成"，而矩阵报告"成功"。

    三态都要测：正常 / 明确不良 / **不可判定**。最后一种最容易漏——
    "读不到"若被当成"没问题"，后端崩在落盘前也会被计成成功。
    """

    def _mk_run(self, root: Path, *, summary=None, dropped=0, label="MEASURED"):
        run = Path(root) / "the-run"
        run.mkdir(parents=True, exist_ok=True)
        meta = {
            "data_label": label,
            "context": {"extra": {"dropped_requests": dropped}},
        }
        if summary is not None:
            meta["summary"] = summary
        (run / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        return run

    def test_healthy_run_is_ok(self):
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary={
                "n_requests": 8, "n_success": 8, "n_failed": 0,
                "n_rejected": 0, "n_unfinished": 0,
            })
            st = mod.read_run_status(run)
        self.assertTrue(st["ok"], st["reasons"])
        self.assertEqual(st["reasons"], [])
        self.assertEqual(st["counts"]["n_success"], 8)

    def test_oom_shaped_run_is_not_ok(self):
        """**用户实测的那个形状**：0 成功 / 1 失败 / 7 未完成。"""
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary={
                "n_requests": 8, "n_success": 0, "n_failed": 1,
                "n_rejected": 0, "n_unfinished": 7,
            })
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"], "0 成功的 run 竟被判成成功")
        joined = " ".join(st["reasons"])
        self.assertIn("未完成 7", joined)
        self.assertIn("失败 1", joined)
        self.assertIn("成功数 0 != 提交数 8", joined)
        # 失败信息里必须能一眼看到结局计数，否则追问题时还得回去读 JSON
        line = mod.format_run_status(st)
        self.assertIn("未完成 7", line)

    def test_partial_unfinished_is_not_ok(self):
        """少一个未完成也是"活没干完"：分位只统计成功请求，尾延迟会被低估。"""
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary={
                "n_requests": 8, "n_success": 7, "n_failed": 0,
                "n_rejected": 0, "n_unfinished": 1,
            })
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])

    def test_dropped_requests_is_not_ok(self):
        """后端自报 dropped>0 = 有请求没走到 done，产物里的计数可能对不上。"""
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, dropped=3, summary={
                "n_requests": 8, "n_success": 8, "n_failed": 0,
                "n_rejected": 0, "n_unfinished": 0,
            })
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])
        self.assertIn("dropped_requests=3", " ".join(st["reasons"]))

    def test_zero_requests_is_not_ok(self):
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary={
                "n_requests": 0, "n_success": 0, "n_failed": 0,
                "n_rejected": 0, "n_unfinished": 0,
            })
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"], "一个请求都没提交不能算成功")

    def test_missing_artifacts_is_not_ok(self):
        """**反向对照**：读不到产物时必须判失败，不能默认通过。"""
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "no-such-run"
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])
        self.assertIn("产物缺失", " ".join(st["reasons"]))
        self.assertIsNone(st["counts"]["n_success"], "不可判定不能填 0")

    def test_unparsable_meta_is_not_ok(self):
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "bad"
            run.mkdir(parents=True)
            (run / "meta.json").write_text("{not json", encoding="utf-8")
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])
        self.assertIn("不可解析", " ".join(st["reasons"]))

    def test_missing_summary_block_is_not_ok(self):
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary=None)
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])
        self.assertIn("不可判定", " ".join(st["reasons"]))

    def test_missing_counter_is_not_ok(self):
        """计数不全 ⇒ 不可判定。**不可判定不等于正常。**"""
        mod = _load_script("run_step3_matrix")
        with tempfile.TemporaryDirectory() as td:
            run = self._mk_run(td, summary={
                "n_requests": 8, "n_success": 8, "n_failed": 0, "n_rejected": 0,
            })  # 缺 n_unfinished
            st = mod.read_run_status(run)
        self.assertFalse(st["ok"])
        self.assertIn("n_unfinished", " ".join(st["reasons"]))


class TestMatrixCountsStatusFailure(unittest.TestCase):
    """矩阵必须把"跑完但没干完活"计成失败，并据此决定退出码与是否续跑。

    注入 ``runner`` / ``status_reader`` 正是为了让这条判据**可被断言**——
    它的失效形态（看异常不看产物）只在真机长跑里出现，不能只靠跑一遍试试。
    """

    def _run(self, *, statuses, keep_going):
        """``statuses`` 是逐次调用返回的状态序列；返回 (done, failures, n_calls)。"""
        mod = _load_script("run_step3_matrix")
        calls: list[str] = []
        it = iter(statuses)

        def _runner(_cfg, **kw):
            calls.append(kw["run_id"])
            return Path(kw["out_dir"]) / kw["run_id"]

        def _reader(_run_dir):
            return next(it)

        class _Cfg:
            run = type("_R", (), {"notes": "n"})()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            done, failures = mod.run_matrix(
                cells=["a", "b"], cfgs={"a": _Cfg(), "b": _Cfg()},
                run_ids={"a": "ra", "b": "rb"}, out_dir=Path("."),
                repeats=1, keep_going=keep_going,
                runner=_runner, status_reader=_reader,
            )
        return done, failures, len(calls), buf.getvalue()

    @staticmethod
    def _ok():
        return {"ok": True, "reasons": [], "counts": {
            "n_requests": 8, "n_success": 8, "n_failed": 0,
            "n_rejected": 0, "n_unfinished": 0,
        }}

    @staticmethod
    def _bad():
        return {"ok": False, "reasons": ["未完成 7 个"], "counts": {
            "n_requests": 8, "n_success": 0, "n_failed": 1,
            "n_rejected": 0, "n_unfinished": 7,
        }}

    def test_healthy_run_counts_as_done(self):
        done, failures, n, _ = self._run(statuses=[self._ok(), self._ok()], keep_going=False)
        self.assertEqual(done, 2)
        self.assertEqual(failures, [])
        self.assertEqual(n, 2)

    def test_status_failure_is_counted_and_stops(self):
        """第一格"跑完但 0 成功" ⇒ 计失败、停住、不再调用 runner。"""
        done, failures, n, out = self._run(statuses=[self._bad()], keep_going=False)
        self.assertEqual(done, 0, "0 成功的 run 被计成了成功")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "a")
        self.assertIn("运行状态不正常", failures[0][2])
        self.assertIn("未完成 7", failures[0][2])
        self.assertEqual(n, 1, "默认应当首个失败即停")
        self.assertIn("产物已保留", failures[0][2])

    def test_keep_going_continues_after_status_failure(self):
        done, failures, n, _ = self._run(
            statuses=[self._bad(), self._ok()], keep_going=True
        )
        self.assertEqual(done, 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(n, 2, "--keep-going 应当续跑")

    def test_first_cell_ok_second_bad(self):
        done, failures, n, _ = self._run(statuses=[self._ok(), self._bad()], keep_going=False)
        self.assertEqual(done, 1)
        self.assertEqual(failures[0][0], "b")
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
