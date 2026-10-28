"""数据标签与可复现性元数据的测试（规范第 8 / 10 节）。

这些测试守的是**最不能出错的一条线**：什么数据可以进结论。
规范把"绝不编造实验结果"写成绝对要求，所以这里不做"看起来合理"的模糊断言，
而是逐条钉死判定矩阵——一旦有人放宽了判定，这些测试会立刻红。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.provenance import (  # noqa: E402
    HARDWARE_REQUIRED,
    NOT_MEASURED,
    REMOTE_EXECUTION_VALUES,
    REMOTE_PLATFORM_PAIRS,
    REQUIRED_REPRO_FIELDS,
    DataLabel,
    ExecutionEnvironment,
    HardwarePlatform,
    ReproInputs,
    ReproducibilityRecord,
    build_reproducibility_record,
    env_platform,
    normalize_availability,
    parse_mthreads_gmi,
    parse_npu_smi,
    platform_evidence_conflict,
    resolve_run_label,
    summarize_labels,
    validate_platform_consistency,
    validate_platform_evidence,
)


class TestDataLabel(unittest.TestCase):
    def test_only_measured_is_eligible_for_conclusion(self):
        eligible = [label for label in DataLabel if label.eligible_for_conclusion]
        self.assertEqual(eligible, [DataLabel.MEASURED])

    def test_only_test_data_is_flagged_synthetic(self):
        synthetic = [label for label in DataLabel if label.is_synthetic]
        self.assertEqual(synthetic, [DataLabel.TEST_DATA])

    def test_label_is_json_serialisable_as_plain_string(self):
        # 继承 str 的意义：能直接写进 JSON/CSV，不需要额外转换层。
        self.assertEqual(DataLabel.MEASURED.value, "MEASURED")
        self.assertEqual(str(DataLabel.TEST_DATA.value), "TEST_DATA")


class TestResolveRunLabel(unittest.TestCase):
    """判定矩阵。每一格都是一条不能被"顺手放宽"的规则。"""

    def test_dry_run_is_test_data_regardless_of_hardware(self):
        # 关键：即使机器上真有 NPU，dry-run 的数字依然是合成的。
        for hw in (True, False, None):
            with self.subTest(hardware=hw):
                self.assertIs(
                    resolve_run_label(dry_run=True, hardware_available=hw),
                    DataLabel.TEST_DATA,
                )

    def test_real_run_without_hardware_is_hardware_required(self):
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=False),
            DataLabel.HARDWARE_REQUIRED,
        )

    def test_real_run_with_unknown_hardware_is_not_measured(self):
        # "没查过环境"不等于"测过了"。这是最容易漏的一格：
        # 若把它当 MEASURED，一份缺环境指纹的数据就能悄悄进结论。
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=None),
            DataLabel.NOT_MEASURED,
        )

    def test_real_run_with_hardware_is_measured(self):
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=True),
            DataLabel.MEASURED,
        )

    def test_simulating_backend_cannot_claim_measured(self):
        # 即使没有 --dry-run 标志，自称 sim 的后端也不许冒充实测。
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=True, backend="simulate"),
            DataLabel.TEST_DATA,
        )


class TestExecutionEnvironment(unittest.TestCase):
    """执行环境声明（规范 "Remote Execution Environment"）。

    这一组守的是两个方向相反的错误：

    * 放宽：本机跑出来的数字被当成昇腾实测（规范明文禁止）。
    * 误杀：推理其实跑在远程昇腾主机上，却因为**本机**没有卡而被判成
      HARDWARE_REQUIRED——真实测数据被聚合器拒收，且没有任何报错。
    """

    def test_default_preserves_legacy_behaviour(self):
        # 不声明执行环境时，判定必须与引入本维度之前逐格一致。
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=True),
            DataLabel.MEASURED,
        )
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=False),
            DataLabel.HARDWARE_REQUIRED,
        )
        self.assertIs(
            resolve_run_label(dry_run=False, hardware_available=None),
            DataLabel.NOT_MEASURED,
        )

    def test_declared_local_can_never_be_measured(self):
        # 最硬的一条：本机执行 -> 无论本机有没有卡，都不得成为结论证据。
        for hw in (True, False, None):
            with self.subTest(hardware=hw):
                self.assertIs(
                    resolve_run_label(
                        dry_run=False,
                        hardware_available=hw,
                        execution_environment=ExecutionEnvironment.LOCAL,
                    ),
                    DataLabel.HARDWARE_REQUIRED,
                )

    def test_remote_ascend_with_verified_hardware_is_measured(self):
        # 远程执行 + 远程指纹确认有卡 -> 这才是真正的昇腾实测。
        self.assertIs(
            resolve_run_label(
                dry_run=False,
                hardware_available=True,
                execution_environment=ExecutionEnvironment.REMOTE_ASCEND,
            ),
            DataLabel.MEASURED,
        )

    def test_remote_ascend_declaration_alone_is_not_enough(self):
        # 声明只是声明。指纹缺失 -> NOT_MEASURED；指纹说没卡 -> HARDWARE_REQUIRED。
        self.assertIs(
            resolve_run_label(
                dry_run=False,
                hardware_available=None,
                execution_environment=ExecutionEnvironment.REMOTE_ASCEND,
            ),
            DataLabel.NOT_MEASURED,
        )
        self.assertIs(
            resolve_run_label(
                dry_run=False,
                hardware_available=False,
                execution_environment=ExecutionEnvironment.REMOTE_ASCEND,
            ),
            DataLabel.HARDWARE_REQUIRED,
        )

    def test_dry_run_outranks_execution_environment(self):
        # 合成数字不会因为"声明跑在昇腾上"就变成实测。
        for env in ExecutionEnvironment:
            with self.subTest(execution_environment=env):
                self.assertIs(
                    resolve_run_label(
                        dry_run=True,
                        hardware_available=True,
                        execution_environment=env,
                    ),
                    DataLabel.TEST_DATA,
                )

    def test_parse_accepts_known_values_and_treats_blank_as_unknown(self):
        self.assertIs(ExecutionEnvironment.parse(None), ExecutionEnvironment.UNKNOWN)
        self.assertIs(ExecutionEnvironment.parse(""), ExecutionEnvironment.UNKNOWN)
        self.assertIs(
            ExecutionEnvironment.parse("local"), ExecutionEnvironment.LOCAL
        )
        # 大小写与空白不该影响判定。
        self.assertIs(
            ExecutionEnvironment.parse("  REMOTE_ASCEND "),
            ExecutionEnvironment.REMOTE_ASCEND,
        )

    def test_parse_rejects_unknown_value_instead_of_silently_downgrading(self):
        # 拼错的声明必须当场报错。静默降级成 UNKNOWN 会让人以为声明生效了，
        # 而实际判定还走老逻辑——这类"以为设了其实没设"最难发现。
        with self.assertRaises(ValueError) as ctx:
            ExecutionEnvironment.parse("remote_ascent")
        self.assertIn("remote_ascent", str(ctx.exception))

    def test_record_carries_execution_environment_from_config(self):
        record = build_reproducibility_record(
            ReproInputs(
                run_id="r1",
                config={"runtime": {"execution_environment": "remote_ascend"}},
            ),
            label=DataLabel.MEASURED,
        )
        self.assertIs(
            record.execution_environment, ExecutionEnvironment.REMOTE_ASCEND
        )
        self.assertEqual(
            record.describe()["execution_environment"], "remote_ascend"
        )

    def test_record_defaults_to_unknown(self):
        record = build_reproducibility_record(ReproInputs(run_id="r1"), label=DataLabel.NOT_MEASURED)
        self.assertIs(record.execution_environment, ExecutionEnvironment.UNKNOWN)
        self.assertEqual(record.describe()["execution_environment"], "unknown")


class TestParseNpuSmi(unittest.TestCase):
    SAMPLE = """
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.3                   Version: 23.0.3                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 92.4        39                0    / 0             |
|                           | 0000:81:00.0  | 0           0    / 0          3287 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 88.1        41                0    / 0             |
|                           | 0000:82:00.0  | 0           0    / 0          1200 / 65536         |
+===========================+===============+====================================================+
"""

    def test_extracts_model_and_count(self):
        parsed = parse_npu_smi(self.SAMPLE)
        self.assertEqual(parsed["model"], "910B3")
        self.assertEqual(parsed["count"], 2)
        self.assertTrue(parsed["raw_parsed"])

    def test_unparsable_input_reports_unknown_not_zero(self):
        # "解析不出来" 必须与 "解析出 0 张卡" 区分开。
        for text in (None, "", "npu-smi: command not found"):
            with self.subTest(text=text):
                parsed = parse_npu_smi(text)
                self.assertIsNone(parsed["count"], "未知不能被写成 0")
                self.assertEqual(parsed["model"], "")
                self.assertFalse(parsed["raw_parsed"])


class TestReproducibilityRecord(unittest.TestCase):
    def test_required_fields_cover_spec_section_8(self):
        # 规范第 8 节逐条列出的字段，一个都不能少。
        must_have = {
            "run_id", "timestamp", "git_commit", "hostname_or_machine_id",
            "device_model", "device_count", "model", "model_revision",
            "precision", "quantization", "tensor_parallel_size",
            "input_length", "output_length", "concurrency",
            "batch_configuration", "max_batched_tokens", "chunk_size",
            "prefix_cache_enabled", "graph_mode", "random_seed",
            "warmup_count", "measurement_count",
        }
        self.assertTrue(must_have.issubset(set(REQUIRED_REPRO_FIELDS)))

    def test_missing_fields_are_listed_not_silently_dropped(self):
        record = ReproducibilityRecord(run_id="r1")
        missing = record.missing_fields()
        self.assertIn("git_commit", missing)
        self.assertIn("model", missing)
        self.assertFalse(record.complete)

    def test_describe_writes_explicit_marker_instead_of_empty_string(self):
        record = ReproducibilityRecord(run_id="r1")
        described = record.describe()
        # 空字符串会被读成"这项不重要"；必须写成显式标记。
        self.assertEqual(described["model"], NOT_MEASURED)
        self.assertNotEqual(described["model"], "")
        self.assertIn("_missing_fields", described)

    def test_marker_becomes_hardware_required_when_hardware_absent(self):
        record = ReproducibilityRecord(run_id="r1", hardware_required=True)
        self.assertEqual(record.describe()["device_model"], HARDWARE_REQUIRED)

    def test_zero_is_not_treated_as_missing(self):
        # 0 是一个真实测量值（例如 warmup_count=0 表示确实不预热），
        # 不能被当成"缺失"从而替换掉。
        record = ReproducibilityRecord(run_id="r1", device_count=0, random_seed=0)
        described = record.describe()
        self.assertEqual(described["device_count"], 0)
        self.assertEqual(described["random_seed"], 0)

    def test_builder_transports_config_and_env_without_inventing(self):
        record = build_reproducibility_record(
            ReproInputs(
                run_id="r1",
                config={
                    "runtime": {"model": "Qwen2.5-7B", "precision": "bf16"},
                    "chunk": {"chunk_size": 2048, "max_num_batched_tokens": 8192},
                    "cache": {"enabled": True},
                    "scheduler": {"policy": "continuous_batching", "max_num_seqs": 32},
                },
                workload_fingerprint={
                    "prompt_len_mean": 512.0,
                    "gen_len_mean": 128.0,
                    "concurrency": 8,
                    "seed": 0,
                },
                hardware_env={"device": {"npu_smi_available": True}},
            ),
            label=DataLabel.MEASURED,
        )
        described = record.describe()
        self.assertEqual(described["model"], "Qwen2.5-7B")
        self.assertEqual(described["precision"], "bf16")
        self.assertEqual(described["chunk_size"], 2048)
        self.assertIs(described["prefix_cache_enabled"], True)
        self.assertEqual(described["concurrency"], 8)
        self.assertEqual(described["random_seed"], 0)

    def test_builder_does_not_invent_unavailable_values(self):
        record = build_reproducibility_record(
            ReproInputs(run_id="r1"), label=DataLabel.MEASURED
        )
        # 输入为空时不应该凭空产生任何值。
        self.assertEqual(record.model, "")
        self.assertIsNone(record.tensor_parallel_size)
        self.assertIsNone(record.chunk_size)


class TestSummarizeLabels(unittest.TestCase):
    def test_counts_every_label_including_zeros(self):
        counts = summarize_labels([DataLabel.MEASURED, DataLabel.MEASURED])
        self.assertEqual(counts["MEASURED"], 2)
        self.assertEqual(counts["TEST_DATA"], 0)


class TestHardwarePlatform(unittest.TestCase):
    """平台维度自身的一致性。

    这一组不测"判定对不对"，测的是**清单会不会漂移**：平台枚举、远程取值集合、
    两者的对应表是三处可以各自手写的清单，任何一处漏改都会让某个平台静默变成
    二等公民——要么无法声明，要么声明了不生效。
    """

    def test_parse_accepts_known_values_and_treats_blank_as_unknown(self):
        self.assertIs(HardwarePlatform.parse(None), HardwarePlatform.UNKNOWN)
        self.assertIs(HardwarePlatform.parse(""), HardwarePlatform.UNKNOWN)
        self.assertIs(HardwarePlatform.parse("ascend"), HardwarePlatform.ASCEND)
        self.assertIs(HardwarePlatform.parse("mthreads"), HardwarePlatform.MTHREADS)
        # 大小写与空白不该影响判定。
        self.assertIs(HardwarePlatform.parse("  MTHREADS "), HardwarePlatform.MTHREADS)

    def test_parse_rejects_misspelling_instead_of_downgrading(self):
        # 静默降级成 UNKNOWN 会让人以为声明生效了，而判定其实还走老逻辑。
        for bad in ("mthread", "moore_threads", "s4000", "ascend-npu"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError) as ctx:
                    HardwarePlatform.parse(bad)
                self.assertIn(bad, str(ctx.exception))

    def test_remote_value_set_matches_the_enum(self):
        # REMOTE_EXECUTION_VALUES 是判定真正读的那张表。它必须**恰好**等于枚举里
        # 名字带 REMOTE_ 的成员：任一侧加了取值而另一侧没加，这里立刻红。
        by_name = {
            m.value for m in ExecutionEnvironment if m.name.startswith("REMOTE_")
        }
        self.assertEqual(set(REMOTE_EXECUTION_VALUES), by_name)

    def test_pair_table_is_the_single_source_of_truth(self):
        env_values = [env for env, _ in REMOTE_PLATFORM_PAIRS]
        platform_values = [platform for _, platform in REMOTE_PLATFORM_PAIRS]
        self.assertEqual(len(env_values), len(set(env_values)), "执行环境取值重复")
        self.assertEqual(len(platform_values), len(set(platform_values)), "平台取值重复")
        # 表里列的每个远程取值都必须真的被判定当成"远程"。
        for env_value, _ in REMOTE_PLATFORM_PAIRS:
            with self.subTest(env=env_value):
                self.assertIs(
                    ExecutionEnvironment.parse(env_value).is_remote,
                    True,
                    f"{env_value} 被列在对应表里，判定却不认它是远程执行",
                )

    def test_display_name_is_defined_for_every_member(self):
        # 缺一个就会在给人看的解释串里漏出裸取值，甚至 KeyError。
        for member in HardwarePlatform:
            with self.subTest(platform=member.value):
                name = member.display_name
                self.assertTrue(name.strip())
                self.assertNotEqual(name, member.value)


class TestIsRemoteIsThreeState(unittest.TestCase):
    """``is_remote`` 必须是三态，而不是布尔。

    这是本组最容易被"顺手简化"的地方。判定原先是
    ``if env is LOCAL: return HARDWARE_REQUIRED``；换成 ``not env.is_remote``
    读起来几乎一样，却会把"未声明"（None）一起判成本机执行——于是所有没写
    execution_environment 的配置从"按指纹判定"掉成 HARDWARE_REQUIRED，
    真实测数据被静默降级。
    """

    def test_remote_values_are_true(self):
        for env in ExecutionEnvironment:
            if env.value not in REMOTE_EXECUTION_VALUES:
                continue
            with self.subTest(env=env.value):
                self.assertIs(env.is_remote, True)

    def test_local_is_false(self):
        self.assertIs(ExecutionEnvironment.LOCAL.is_remote, False)

    def test_unknown_is_none_and_deliberately_not_false(self):
        self.assertIsNone(ExecutionEnvironment.UNKNOWN.is_remote)
        # 直接钉死那条错误写法：`not is_remote` 对"未声明"取真，语义就错了。
        self.assertIsNot(ExecutionEnvironment.UNKNOWN.is_remote, False)

    def test_unset_environment_still_falls_back_to_the_fingerprint(self):
        # 端到端后果：未声明 + 指纹缺失 -> NOT_MEASURED（流程没走完），
        # 而不是 HARDWARE_REQUIRED（机器上没卡）。两者要修的东西完全不同。
        self.assertIs(
            resolve_run_label(
                dry_run=False,
                hardware_available=None,
                execution_environment=ExecutionEnvironment.UNKNOWN,
            ),
            DataLabel.NOT_MEASURED,
        )


class TestEveryRemotePlatformIsFirstClass(unittest.TestCase):
    """平台换成 S4000（MTT）时，标签判定矩阵必须逐格不变。

    判定本来就不该知道平台名。这一组把这个性质钉死：对**每一个**远程取值，
    判定结果都必须与昇腾时逐格相同——否则"以 S4000 为主"就是句空话，
    数据会卡在 HARDWARE_REQUIRED 上进不了结论，而且不报错。
    """

    def test_label_matrix_is_identical_for_every_remote_platform(self):
        cases = {
            "verified": (True, DataLabel.MEASURED),
            "absent": (False, DataLabel.HARDWARE_REQUIRED),
            "unknown": (None, DataLabel.NOT_MEASURED),
        }
        for env_value in sorted(REMOTE_EXECUTION_VALUES):
            env = ExecutionEnvironment.parse(env_value)
            for name, (hardware, expected) in cases.items():
                with self.subTest(env=env_value, hardware=name):
                    self.assertIs(
                        resolve_run_label(
                            dry_run=False,
                            hardware_available=hardware,
                            execution_environment=env,
                        ),
                        expected,
                    )

    def test_dry_run_outranks_every_remote_platform(self):
        for env_value in sorted(REMOTE_EXECUTION_VALUES):
            with self.subTest(env=env_value):
                self.assertIs(
                    resolve_run_label(
                        dry_run=True,
                        hardware_available=True,
                        execution_environment=ExecutionEnvironment.parse(env_value),
                    ),
                    DataLabel.TEST_DATA,
                )

    def test_a_non_ascend_platform_is_not_silently_disqualified(self):
        # 最直接的一格：S4000（remote_mthreads）+ 远程指纹确认有卡 -> MEASURED。
        # 若有人把判定改回"只有 remote_ascend 才算远程"，这里会红。
        self.assertIs(
            resolve_run_label(
                dry_run=False,
                hardware_available=True,
                execution_environment=ExecutionEnvironment.REMOTE_MTHREADS,
            ),
            DataLabel.MEASURED,
        )


class TestPlatformConsistency(unittest.TestCase):
    """两个声明不能自相矛盾，矛盾必须报错而不是取其一。"""

    def test_contradictory_declarations_raise(self):
        cases = [
            (ExecutionEnvironment.REMOTE_ASCEND, HardwarePlatform.MTHREADS),
            (ExecutionEnvironment.REMOTE_MTHREADS, HardwarePlatform.ASCEND),
        ]
        for env, platform in cases:
            with self.subTest(env=env.value, platform=platform.value):
                with self.assertRaises(ValueError) as ctx:
                    validate_platform_consistency(env, platform)
                message = str(ctx.exception)
                # 两个取值都要出现，否则读的人看不出是哪边写错了。
                self.assertIn(env.value, message)
                self.assertIn(platform.value, message)

    def test_error_message_offers_a_copyable_fix(self):
        # 报错必须给出**能照抄的修法**，否则还得回去翻枚举定义。
        with self.assertRaises(ValueError) as ctx:
            validate_platform_consistency(
                ExecutionEnvironment.REMOTE_ASCEND, HardwarePlatform.MTHREADS
            )
        self.assertIn("remote_mthreads", str(ctx.exception))

    def test_consistent_declarations_pass(self):
        for env_value, platform_value in REMOTE_PLATFORM_PAIRS:
            with self.subTest(env=env_value, platform=platform_value):
                validate_platform_consistency(
                    ExecutionEnvironment.parse(env_value),
                    HardwarePlatform.parse(platform_value),
                )

    def test_unknown_on_either_side_is_not_a_contradiction(self):
        # "没声明"推不出"说错了"。历史配置不写 hardware_platform，必须放行。
        cases = [
            (ExecutionEnvironment.REMOTE_ASCEND, HardwarePlatform.UNKNOWN),
            (ExecutionEnvironment.UNKNOWN, HardwarePlatform.MTHREADS),
            (ExecutionEnvironment.UNKNOWN, HardwarePlatform.UNKNOWN),
        ]
        for env, platform in cases:
            with self.subTest(env=env.value, platform=platform.value):
                validate_platform_consistency(env, platform)

    def test_local_may_declare_any_platform(self):
        # "本机就是一台整机、直接在上面跑"是真实情形。允许这么写，
        # 人才不会为了让流程走通而谎报平台。
        for member in HardwarePlatform:
            with self.subTest(platform=member.value):
                validate_platform_consistency(ExecutionEnvironment.LOCAL, member)

    def test_builder_refuses_to_record_a_contradiction(self):
        # 直接构造 ReproInputs 的调用方（脚本、测试）同样不该产出自相矛盾的记录。
        with self.assertRaises(ValueError):
            build_reproducibility_record(
                ReproInputs(
                    run_id="s4000",
                    config={
                        "runtime": {
                            "execution_environment": "remote_ascend",
                            "hardware_platform": "mthreads",
                        }
                    },
                ),
                label=DataLabel.MEASURED,
            )


class TestHardwarePlatformInRecord(unittest.TestCase):
    def test_record_carries_platform_from_config(self):
        record = build_reproducibility_record(
            ReproInputs(
                run_id="s4000",
                config={
                    "runtime": {
                        "execution_environment": "remote_mthreads",
                        "hardware_platform": "mthreads",
                    }
                },
            ),
            label=DataLabel.MEASURED,
        )
        self.assertIs(record.hardware_platform, HardwarePlatform.MTHREADS)
        self.assertEqual(record.describe()["hardware_platform"], "mthreads")

    def test_record_defaults_to_unknown_platform(self):
        record = build_reproducibility_record(
            ReproInputs(run_id="r1"), label=DataLabel.NOT_MEASURED
        )
        self.assertIs(record.hardware_platform, HardwarePlatform.UNKNOWN)
        self.assertEqual(record.describe()["hardware_platform"], "unknown")

    def test_platform_is_not_a_required_repro_field(self):
        # 陷阱：塞进 REQUIRED_REPRO_FIELDS 会让所有历史 run 一夜之间变成
        # "元数据不完整"，missing_fields() 多一项、preflight 全红。
        # 它是新增的补充维度，不是规范第 8 节点名的字段。
        self.assertNotIn("hardware_platform", REQUIRED_REPRO_FIELDS)
        self.assertNotIn(
            "hardware_platform", ReproducibilityRecord(run_id="r1").missing_fields()
        )


class TestParseMthreadsGmi(unittest.TestCase):
    """``mthreads-gmi`` 解析。与 :func:`parse_npu_smi` 同一套三态纪律。

    这一段的价值不在"能解析出一台设备"，而在**认不出时不许猜**：把认不出写成 0
    会让一台真的有卡的机器看起来没有卡，而这正是"数据进不了结论"却查不出原因的
    那类故障（现象是标签降级，原因在解析）。
    """

    SAMPLE = """\
Sat Jan  1 00:00:00 2026
---------------------------------------------------------------
mthreads-gmi: 2.7.0
---------------------------------------------------------------
ID   Name           | PCIe             | %GPU  Mem
     Device Type    | Bus-Id           |       Used,Total
---------------------------------------------------------------
0    MTT S4000      | 00000000:01:00.0 | 0%    0MiB / 49152MiB
1    MTT S4000      | 00000000:02:00.0 | 0%    0MiB / 49152MiB
---------------------------------------------------------------
"""

    def test_extracts_model_and_count(self):
        parsed = parse_mthreads_gmi(self.SAMPLE)
        self.assertEqual(parsed["model"], "MTT S4000")
        self.assertEqual(parsed["count"], 2)
        self.assertTrue(parsed["raw_parsed"])

    def test_header_line_is_not_counted_as_a_device(self):
        # 表头也带 "MTT" 一类的字眼时有风险，这里固定住"必须有行首编号"。
        # 计满 2 而不是 3，说明表头没被算进去。
        self.assertEqual(parse_mthreads_gmi(self.SAMPLE)["count"], 2)

    def test_unparsable_input_reports_unknown_not_zero(self):
        for text in (None, "", "mthreads-gmi: command not found"):
            with self.subTest(text=text):
                parsed = parse_mthreads_gmi(text)
                self.assertIsNone(parsed["count"], "未知不能被写成 0")
                self.assertEqual(parsed["model"], "")
                self.assertFalse(parsed["raw_parsed"])

    def test_a_lone_product_name_is_not_a_card(self):
        """只见产品名、数不出设备行 -> ``count=None``，不回落成 1。

        "输出里提到 MTT S4000"可能出现在版本串、警告或文档链接里。回落成 1
        会把"看到一个名字"当成"有一张卡"，于是无卡机器也可能报可用。
        """
        parsed = parse_mthreads_gmi("mthreads-gmi 2.7.0 supports MTT S4000\n")
        self.assertIsNone(parsed["count"])
        self.assertEqual(parsed["model"], "MTT S4000")
        self.assertTrue(parsed["raw_parsed"])


class TestNormalizeAvailability(unittest.TestCase):
    """多平台探测结果的合成规则。三态，取"最保守的可判定值"。

    这条规则是"平台无关"的落地方式：同一台机器上同时装了两家驱动是常态，
    合成规则必须让**任一**平台认出设备就算可用，否则换个平台就得改判定代码。
    """

    def test_no_probe_is_unknown_not_unavailable(self):
        # 一个探测都没跑 ≠ 没有硬件。
        self.assertIsNone(normalize_availability([]))

    def test_any_true_wins(self):
        self.assertIs(normalize_availability([True, False]), True)
        self.assertIs(normalize_availability([None, True]), True)
        self.assertIs(normalize_availability([False, None, True]), True)

    def test_inconclusive_probe_beats_a_clean_negative(self):
        # 一个说"没有"、一个说"认不出"：认不出的那个还没被排除，不能下否定结论。
        # 读成 False 会把"探测坏了"报成"机器没卡"，于是问题永远不会被修。
        self.assertIsNone(normalize_availability([None, False]))
        self.assertIsNone(normalize_availability([None, None]))

    def test_all_false_is_unavailable(self):
        self.assertIs(normalize_availability([False]), False)
        self.assertIs(normalize_availability([False, False]), False)

    def test_mixed_platforms_agree_on_available(self):
        """昇腾认出设备、MTT 明确没有 —— 可用。

        这是"平台维度独立"的关键点：可用性是"这台机器上有能用的加速设备"，
        不是"所有平台都认出了设备"。
        """
        self.assertIs(normalize_availability([True, False]), True)


class TestEnvPlatform(unittest.TestCase):
    """从环境指纹读**检出的**平台。只认显式字段，绝不从旧字段反推。"""

    def test_reads_explicit_platform_key(self):
        self.assertIs(
            env_platform({"platform": "ascend"}), HardwarePlatform.ASCEND
        )
        self.assertIs(
            env_platform({"platform": "mthreads"}), HardwarePlatform.MTHREADS
        )

    def test_reads_the_hardware_platform_alias(self):
        # 别名一起认，免得两个名字各写一半、读的人以为另一个是空的。
        self.assertIs(
            env_platform({"hardware_platform": "mthreads"}), HardwarePlatform.MTHREADS
        )

    def test_blank_and_missing_stay_unknown(self):
        for payload in (None, {}, {"platform": ""}, {"platform": "   "},
                        {"hostname": "somewhere"}):
            with self.subTest(payload=payload):
                self.assertIs(env_platform(payload), HardwarePlatform.UNKNOWN)

    def test_the_unknown_word_maps_to_unknown(self):
        self.assertIs(
            env_platform({"platform": "unknown"}), HardwarePlatform.UNKNOWN
        )

    def test_old_fingerprint_is_not_reverse_engineered_into_a_platform(self):
        """旧指纹只说"npu-smi 能跑"，没说这是哪台机器。

        反推成 ascend 会把一台**其实没确认过**的机器写进平台列；下游按平台筛表、
        按平台复现都不会报错，只是把人带到错误的机器上。
        """
        payload = {
            "device": {"npu_smi_available": True, "npu_smi_info": "npu-smi 23.0.3"},
            "available": True,
            "schema_version": 1,
        }
        self.assertIs(env_platform(payload), HardwarePlatform.UNKNOWN)

    def test_unregistered_platform_raises_instead_of_falling_back(self):
        """未登记平台**抛错**，不静默降级成 unknown。

        降级会把新平台的数据混进"平台未知"的桶：不报错、筛不出来，
        声明与检出的矛盾也从此永远检不出。报错消息必须给出登记方式。
        """
        with self.assertRaises(ValueError) as ctx:
            env_platform({"platform": "cambricon"})
        message = str(ctx.exception)
        self.assertIn("cambricon", message)
        self.assertIn("REMOTE_PLATFORM_PAIRS", message)


class TestPlatformEvidence(unittest.TestCase):
    """"配置**声明**的平台"与"指纹**检出**的平台"是否一致。

    与 :class:`TestPlatformConsistency` 是两条不同的线，不能互相替代：
    那个只保证两个声明不矛盾（**证明不了任何事实**，声明可以很自洽同时与真机不符）；
    这个拿"查到的"去对"说的"，才是能证明的那种。
    """

    def test_same_platform_is_no_conflict(self):
        for platform in (HardwarePlatform.ASCEND, HardwarePlatform.MTHREADS):
            with self.subTest(platform=platform):
                self.assertIsNone(
                    platform_evidence_conflict(platform, platform)
                )

    def test_differing_platforms_are_a_conflict(self):
        reason = platform_evidence_conflict(
            HardwarePlatform.MTHREADS, HardwarePlatform.ASCEND
        )
        self.assertIsNotNone(reason)
        self.assertIn("mthreads", reason)
        self.assertIn("ascend", reason)

    def test_unknown_on_either_side_is_not_a_conflict(self):
        """任一方 unknown 一律放行。

        把"没证据"当成"证据矛盾"，会让所有历史指纹与未声明平台的配置一起被拒——
        那就不是收紧，是把流水线钉死。
        """
        self.assertIsNone(
            platform_evidence_conflict(
                HardwarePlatform.UNKNOWN, HardwarePlatform.ASCEND
            )
        )
        self.assertIsNone(
            platform_evidence_conflict(
                HardwarePlatform.MTHREADS, HardwarePlatform.UNKNOWN
            )
        )

    def test_validate_blocks_the_reviewed_case(self):
        """评审复现的原例：配置声明 remote_mthreads/mthreads，指纹却是昇腾 910。

        修正前这条组合被静默接受，于是一份 MTT 的实验数据带着昇腾的指纹进入结论。
        """
        with self.assertRaises(ValueError) as ctx:
            validate_platform_evidence(
                HardwarePlatform.MTHREADS,
                HardwarePlatform.ASCEND,
                source="results/raw/_env/env.json",
            )
        message = str(ctx.exception)
        # 报错必须给出**两条**修法与各自的入口，而不是让人自己猜是哪一种。
        self.assertIn("results/raw/_env/env.json", message)
        self.assertIn("inspect_env.sh", message)
        self.assertIn("hardware_platform", message)

    def test_validate_is_silent_when_there_is_no_evidence(self):
        validate_platform_evidence(
            HardwarePlatform.MTHREADS, HardwarePlatform.UNKNOWN
        )


if __name__ == "__main__":
    unittest.main()
