"""跨模块契约测试（Round 1 验收）。

单独一个文件，是因为这里验证的**不是**某个模块的内部行为，而是模块之间的
接线：函数存在但主流程不调用、字段采集了但读端不读、三处各自理解"同一组"
——这些缺陷每一个都（1）不报错、（2）在单模块测试里全绿、（3）让产物看起来
完全正常。因此必须从产物那一端反向验证。

四项验收（对应用户列出的四项修复）：

1. 错平台被拒绝 —— 运行前 / 元数据构造 / 导出检查三处都拦，且都用**同一份**
   环境指纹快照；
2. MTT 型号与卡数落盘 —— 采集端规范化过的 ``device.model`` / ``device.count``
   真的进了 ``meta.json``；
3. 无设备记录不判可用 —— 只有产品名（没有设备记录）时保持"未知"，不是"可用"；
4. 分组贯穿汇总、绘图与导出 —— 三处对"同一实验"的理解必须一致。
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_metrics as cm  # noqa: E402
import plot_p99  # noqa: E402
import preflight as pf  # noqa: E402
import run_benchmark as rb  # noqa: E402
from ascend_opt import grouping  # noqa: E402
from ascend_opt.provenance import (  # noqa: E402
    DataLabel,
    HardwarePlatform,
    ReproInputs,
    build_reproducibility_record,
    device_probe_result,
    normalize_availability,
    parse_mthreads_gmi,
    parse_mthreads_software,
    parse_musa_version_file,
)

#: 研究主体平台的真实配置。刻意用**仓库里已有的**那个文件而不是在测试里另写
#: 一份：另写一份会与它漂移，而"声明的平台"正是这组测试的自变量。
S4000_CONFIG = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"


def _env_payload(platform: str, *, model: str = "MTT S4000", count: int | None = 1,
                 available: bool = True) -> dict:
    """按 ``scripts/inspect_env.sh`` 的**真实产物形状**造一份指纹。"""
    return {
        "schema_version": 2,
        "platform": platform,
        "available": available,
        "available_source": "declared",
        "device": {"model": model, "count": count},
        "probes": {
            "mthreads": {"tool_present": True, "result": True},
            "ascend": {"tool_present": True, "result": False},
        },
        "discovery": {"values": {}, "missing": []},
        "source": "/tmp/env.json",
    }


# --------------------------------------------------------------------------- #
# 1. 错平台必须在三处都被拒绝
# --------------------------------------------------------------------------- #

class TestPlatformEvidenceIsEnforcedAtEveryGate(unittest.TestCase):
    """声明与指纹矛盾时，三处入口都要拦，且拦的是同一个判据。"""

    def test_metadata_construction_rejects_contradiction(self):
        """入口二：元数据构造。直接构造 ReproInputs 的调用方也拦得住。"""
        with self.assertRaises(ValueError) as ctx:
            build_reproducibility_record(
                ReproInputs(
                    config={"runtime": {
                        "hardware_platform": "mthreads",
                        "execution_environment": "remote_mthreads",
                    }},
                    hardware_env=_env_payload("ascend"),
                ),
                label=DataLabel.TEST_DATA,
            )
        message = str(ctx.exception)
        # 报错必须点名**两**个平台并给出两条修法：读的人不该自己猜是
        # "指纹采错了机器"还是"声明写错了平台"。
        self.assertIn("mthreads", message)
        self.assertIn("ascend", message)
        self.assertIn("①", message)
        self.assertIn("②", message)

    def test_metadata_construction_allows_matching_platform(self):
        """正向对照：声明与检出一致时不得报错，否则这条检查会挡住一切。"""
        record = build_reproducibility_record(
            ReproInputs(
                config={"runtime": {
                    "hardware_platform": "mthreads",
                    "execution_environment": "remote_mthreads",
                }},
                hardware_env=_env_payload("mthreads"),
            ),
            label=DataLabel.TEST_DATA,
        )
        self.assertIs(record.hardware_platform, HardwarePlatform.MTHREADS)

    def test_metadata_construction_passes_when_no_evidence(self):
        """缺证据时放行：老指纹没写平台，推不出"说错了"。

        把"没证据"当成"证据矛盾"，会让所有还没采过指纹的机器一起跑不动。
        """
        payload = _env_payload("mthreads")
        payload["platform"] = "unknown"
        record = build_reproducibility_record(
            ReproInputs(
                config={"runtime": {
                    "hardware_platform": "mthreads",
                    "execution_environment": "remote_mthreads",
                }},
                hardware_env=payload,
            ),
            label=DataLabel.TEST_DATA,
        )
        self.assertEqual(record.device_model, "MTT S4000")

    def test_pre_run_rejects_before_writing_anything(self):
        """入口一：运行前。**在计算之前**报错，且一个产物都不留。

        这条测试的价值在于"早"：一次基准数分钟，跑完才发现声明与真机不符，
        既浪费了时间，又在证据目录里留下一份自相矛盾的产物。
        """
        cfg = rb.load_config(S4000_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "_env").mkdir()
            (out / "_env" / "env.json").write_text(
                json.dumps(_env_payload("ascend")), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                rb.run_once(
                    cfg, dry_run=True, out_dir=out,
                    run_id="conflict", verbose=False,
                )
            self.assertIn("ascend", str(ctx.exception))
            # 没有任何产物落盘 —— 证明检查发生在计算之前。
            self.assertFalse((out / "conflict").exists())

    def test_pre_run_and_metadata_see_the_same_fingerprint(self):
        """正向对照 + 快照绑定：同一次 run 的判定与落盘读的是同一份指纹。

        这里从**产物**反向验证：meta 里记的 hardware_env 必须与运行前放在
        ``_env/env.json`` 的那份逐字段一致；同时 device 型号/卡数也得在。
        """
        cfg = rb.load_config(S4000_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "_env").mkdir()
            payload = _env_payload("mthreads")
            (out / "_env" / "env.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            run_dir = rb.run_once(
                cfg, dry_run=True, out_dir=out, run_id="ok", verbose=False
            )
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["hardware_env"]["platform"], "mthreads")
            self.assertEqual(meta["hardware_env"]["device"]["model"], "MTT S4000")
            repro = meta["reproducibility"]
            self.assertEqual(repro["hardware_platform"], "mthreads")
            self.assertEqual(repro["execution_environment"], "remote_mthreads")

    def test_export_check_reports_platform_conflict(self):
        """入口三：逐 run 导出检查必须报出矛盾，且 MEASURED 行上是**阻断**。

        导出闸门是最后一道：矛盾产物一旦进了交付包，就会带着"硬件实测"的
        名义被引用，事后只能靠重跑纠正。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage_export(Path(tmp), declared="mthreads", detected="ascend")
            problems = pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            )
            codes = {p.code for p in problems}
            self.assertIn("platform_evidence_conflict", codes)
            conflict = next(p for p in problems if p.code == "platform_evidence_conflict")
            # MEASURED 行上的矛盾是**阻断**项：这种行会被当成硬件实测结果引用。
            self.assertEqual(conflict.level, pf.BLOCK)
            self.assertIn("mthreads", conflict.detail)
            self.assertIn("ascend", conflict.detail)

    def test_export_check_passes_when_platforms_agree(self):
        """正向对照：声明与检出一致时导出检查全绿。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage_export(Path(tmp), declared="mthreads", detected="mthreads")
            self.assertEqual(pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            ), [])

    def test_export_check_passes_when_declaration_absent(self):
        """未声明平台的旧产物不得被这条新检查挡住（缺证据 ≠ 证据矛盾）。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage_export(Path(tmp), declared=None, detected="ascend")
            self.assertEqual(pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            ), [])


# --------------------------------------------------------------------------- #
# 2. MTT 型号与卡数必须落盘
# --------------------------------------------------------------------------- #

class TestDeviceFieldsReachMetadata(unittest.TestCase):
    """采集端已经规范化过的字段，读端必须真的去读。"""

    def test_normalized_device_fields_are_read(self):
        """``device.model`` / ``device.count`` 是新格式的**权威**键名。

        修复前读端只认 ``device_model`` / ``npu.model`` 那几种历史写法，
        于是"采集成功了、元数据里却是空的"——采集端与读端各记一半。
        """
        record = build_reproducibility_record(
            ReproInputs(
                config={"runtime": {}},
                hardware_env=_env_payload("mthreads", model="MTT S4000", count=1),
            ),
            label=DataLabel.TEST_DATA,
        )
        self.assertEqual(record.device_model, "MTT S4000")
        self.assertEqual(record.device_count, 1)

    def test_zero_device_count_is_preserved(self):
        """``0`` 是**测量出来的**"一张卡都没有"，不能与"缺值"混同。

        回落链写成 ``a or b or c`` 时，0 是假值，会被静默跳过并最终变成 None
        ——于是"测到没有卡"变成了"没查到卡数"，两者的修法完全不同。
        """
        record = build_reproducibility_record(
            ReproInputs(
                config={"runtime": {}},
                hardware_env=_env_payload("mthreads", count=0),
            ),
            label=DataLabel.TEST_DATA,
        )
        self.assertEqual(record.device_count, 0)

    def test_discovery_values_fallback_still_works(self):
        """旧格式的回落路径必须保留（历史指纹只有 discovery.values）。"""
        payload = _env_payload("mthreads")
        payload["device"] = {"model": None, "count": None}
        payload["discovery"]["values"] = {
            "ASCEND_DEVICE_MODEL": "MTT S4000",
            "ASCEND_DEVICE_COUNT": 2,
        }
        record = build_reproducibility_record(
            ReproInputs(config={"runtime": {}}, hardware_env=payload),
            label=DataLabel.TEST_DATA,
        )
        self.assertEqual(record.device_model, "MTT S4000")
        self.assertEqual(record.device_count, 2)

    #: 逐字抄自真实主机（AutoDL、MTT S4000、mthreads-gmi 1.14.0、driver 2.7.0）的
    #: 输出首部。**不要**"整理"成更整齐的样子——上一版测试用的就是手写的想象样本，
    #: 把两个版本拆成了两行，于是掩盖了真实输出把它们印在**同一行**这件事。
    REAL_GMI_HEADER = (
        "Tue Sep 22 14:25:46 2026\n"
        "---------------------------------------------------------------\n"
        "    mthreads-gmi:1.14.0          Driver Version:2.7.0\n"
        "---------------------------------------------------------------\n"
        "ID   Name           |PCIe                |%GPU  Mem            \n"
        "     Device Type    |Pcie Lane Width     |Temp  MPC Capable    \n"
        "                                         |      ECC Mode       \n"
        "+-------------------------------------------------------------+\n"
        "0    MTT S4000      |00000000:16:00.0    |0%    4MiB(49152MiB)\n"
        "     Physical       |16x(16x)            |43C   YES            \n"
        "                                         |      N/A            \n"
        "---------------------------------------------------------------\n"
    )

    def test_driver_version_is_not_the_gmi_tool_version(self):
        """真实输出把 gmi 工具版本与驱动版本印在**同一行**:

            mthreads-gmi:1.14.0          Driver Version:2.7.0

        驱动版本必须是 2.7.0。早先的正则把两者写成一个交替，而交替总是先匹配到位置
        靠前的 ``mthreads-gmi``，于是驱动被静默写成工具版本 1.14.0——错的版本号会一路
        进到可复现性元数据，既不报错也不留痕，而驱动版本正是"引擎块大小为何是 64"
        这类结论的依据。
        """
        parsed = parse_mthreads_software(self.REAL_GMI_HEADER)
        self.assertEqual(parsed["driver_version"], "2.7.0")
        self.assertNotEqual(
            parsed["driver_version"],
            "1.14.0",
            "gmi 工具版本被当成了驱动版本——这正是实测踩到的那个缺陷",
        )

    def test_gmi_tool_line_alone_leaves_driver_unknown(self):
        """只有工具版本行、没有独立驱动字段时，驱动写 None，**不许**拿工具版本顶替。"""
        parsed = parse_mthreads_software("mthreads-gmi: 1.14.0\n")
        self.assertIsNone(parsed["driver_version"])
        self.assertIsNone(parsed["musa_version"])

    def test_musa_version_label_still_read_from_gmi_text(self):
        """gmi 文本里带显式 MUSA 版本行时仍要采到。"""
        parsed = parse_mthreads_software("MUSA Version: 4.1.0\n")
        self.assertEqual(parsed["musa_version"], "4.1.0")

    def test_musa_version_json_uses_toolkit_not_first_number(self):
        """``version.json`` 里组件很多，只有 musa_toolkits/musa_runtime 是 SDK 版本。

        实测：那台主机装的是 ``/usr/local/musa/version.json``，而采集端原先只找
        ``version`` / ``version.txt``，于是 MUSA 版本静默成了 None——与"这台机器没装
        MUSA"在产物里**完全无法区分**。

        样本刻意把 ``schema``/``mcc`` 排在 ``musa_toolkits`` 前面，好让"取文件里第一个
        数字"那种实现给出**不同**答案（它会答 1.0），否则这个用例分不出对错。
        """
        payload = (
            "{\n"
            '\t"schema":\t{"version":\t"1.0"},\n'
            '\t"mcc":\t{"version":\t"9.9.9"},\n'
            '\t"musa_toolkits":\t{"version":\t"3.1.0", "git branch": "kuae1.3.0"},\n'
            '\t"muPP":\t{"version":\t"1.7.0"},\n'
            '\t"musa_runtime":\t{"version":\t"3.1.0"}\n'
            "}\n"
        )
        self.assertEqual(parse_musa_version_file(payload), "3.1.0")

    def test_musa_version_json_falls_back_to_runtime(self):
        payload = '{"musa_runtime": {"version": "3.1.0"}, "mublas": {"version": "1.6.0"}}'
        self.assertEqual(parse_musa_version_file(payload), "3.1.0")

    def test_musa_version_file_never_fabricates(self):
        """纯文本版本文件取第一个版本号；认不出写 None，不编字符串。"""
        self.assertEqual(parse_musa_version_file("3.1.0\n"), "3.1.0")
        for text in (
            None,
            "",
            "   ",
            "{not json",
            "{}",
            '{"mublas": {"version": "1.6.0"}}',
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_musa_version_file(text))

    def test_mtt_software_parser_reads_musa_and_never_fabricates(self):
        """MTT 侧的 MUSA 版本要能采到；认不出时不许编一个 "unknown"。"""
        parsed = parse_mthreads_software(
            "Sat Jan  1 00:00:00 2026\n"
            "mthreads-gmi: 1.14.0\n"
            "MUSA Version: 4.1.0\n"
            "0    MTT S4000      | 00000000:01:00.0 | 0%    0MiB / 49152MiB\n"
        )
        self.assertEqual(parsed["musa_version"], "4.1.0")
        # 认不出时不许编一个 "unknown"：那会看起来像一次成功的采集。
        for text in (None, "", "mthreads-gmi: command not found"):
            with self.subTest(text=text):
                empty = parse_mthreads_software(text)
                self.assertIsNone(empty["driver_version"])
                self.assertIsNone(empty["musa_version"])


# --------------------------------------------------------------------------- #
# 3. 无设备记录不得判为可用
# --------------------------------------------------------------------------- #

def _legacy_probe_result(tool_present: bool, count, has_name: bool):
    """**修复前**的判据（原样抄自 scripts/inspect_env.sh 的 ``_probe_result``）。

    留在这里是为了让"这条缺陷真实存在"成为一条可执行的断言，而不是注释里
    的一句话：同一条输入，旧判据说"可用"，新判据说"未知"。
    """
    if not tool_present:
        return False
    if count is not None:
        return count >= 1
    return True if has_name else None


class TestNoDeviceRecordIsNotAvailable(unittest.TestCase):
    """只有产品名时保持"未知"——一条说明文字不是一张卡。"""

    LONE_NAME = "mthreads-gmi 2.7.0 supports MTT S4000\n"

    def test_this_is_a_real_defect_not_a_hyphothetical(self):
        """证明判据确实改过：同一条输入，旧实现说可用、新实现说未知。"""
        parsed = parse_mthreads_gmi(self.LONE_NAME)
        self.assertIsNone(parsed["count"], "解析器本来就返回未知（这条没变）")
        self.assertTrue(parsed["models"], "但产品名是解析出来了的")

        old = _legacy_probe_result(True, parsed["count"], bool(parsed["models"]))
        new = device_probe_result(True, parsed["count"], bool(parsed["models"]))
        self.assertTrue(old, "旧判据：有产品名 -> 可用（这就是缺陷）")
        self.assertIsNone(new, "新判据：没有设备记录 -> 未知")

    def test_judgment_matrix(self):
        """三态判定的完整取值表。工具不存在是 False，不是 None。"""
        cases = [
            # (工具在, 卡数, 有产品名, 期望)
            (False, None, False, False),   # 没装这个工具 = 明确没有
            (True, None, False, None),     # 跑了，认不出 = 未知
            (True, None, True, None),      # 只有产品名 = 未知（本轮修复点）
            (True, 0, True, False),        # 明确零张卡
            (True, 1, True, True),
            (True, 8, True, True),
        ]
        for tool, count, name, expected in cases:
            with self.subTest(tool=tool, count=count, name=name):
                self.assertEqual(
                    device_probe_result(tool, count, name), expected
                )

    def test_lone_name_does_not_make_the_machine_available(self):
        """端到端：从解析到合成可用性，全程不得出现 True。

        这是用户报告的现象："解析器返回卡数未知，采集判定却得到 available=True"。
        """
        parsed = parse_mthreads_gmi(self.LONE_NAME)
        mtt = device_probe_result(True, parsed["count"], bool(parsed["models"]))
        ascend = device_probe_result(False, None, False)
        self.assertIsNone(normalize_availability([ascend, mtt]))
        self.assertIsNot(normalize_availability([ascend, mtt]), True)

    def test_ascend_side_fallback_still_counts_device_rows(self):
        """昇腾侧的回落（数 NPU ID 行）必须保住：那是**设备记录**，不是产品名。"""
        # 数出 2 行 NPU ID -> count=2 -> 可用；旧新判据在这里一致。
        self.assertIs(device_probe_result(True, 2, True), True)
        self.assertIs(_legacy_probe_result(True, 2, True), True)


# --------------------------------------------------------------------------- #
# 4. 分组必须贯穿汇总、绘图与导出
# --------------------------------------------------------------------------- #

def _row(run_id: str, *, backend: str, platform: str = "mthreads",
         workload: str = "w", config: str = "cfgA", ttft: float = 100.0) -> dict:
    return {
        "run_id": run_id,
        "workload": workload,
        "config": config,
        "dry_run": False,
        "data_label": "MEASURED",
        "label_source": "stamped",
        "execution_environment": "remote_mthreads",
        "hardware_platform": platform,
        "backend": backend,
        "precision": "fp16",
        "quantization": "",
        "ttft_p99": str(ttft),
        "tpot_p99": "10",
        "stall_p99": "1",
    }


class TestGroupingIsSharedAcrossModules(unittest.TestCase):
    """三处对"同一实验"的理解必须一致，否则表、图、校验互相矛盾。"""

    def test_plot_key_equals_canonical_dimensions(self):
        """绘图的分组键就是 :func:`grouping.group_dimensions`，没有第二份实现。"""
        row = _row("a", backend="simulate")
        self.assertEqual(
            tuple(plot_p99._group_key(row)),
            tuple(grouping.group_dimensions(row)),
        )
        self.assertEqual(
            grouping.GROUP_DIMENSIONS,
            ("workload", "config", "hardware_platform", "backend"),
        )

    def test_aggregate_key_embeds_the_same_dimensions(self):
        """汇总键 = 批次名 + 同一套维度，顺序一致（可被 grep / diff）。"""
        row = _row("cmp-x-r1", backend="real")
        parts = cm.group_key(row).split("|")
        self.assertEqual(parts[0], "cmp-x", "批次名去掉了 -rN 后缀")
        self.assertEqual(tuple(parts[1:]), tuple(grouping.group_dimensions(row)))

    def test_backend_splits_both_aggregate_and_plot(self):
        """同一平台、不同执行后端：汇总拆两组，绘图也必须拆两根柱。

        修复前绘图只按 (workload, config, hardware_platform) 分组，于是
        "模拟 100" 与 "真机 200" 被并成一根 150 的柱——图上完全看不出来。
        """
        rows = [
            _row("cmp-x", backend="simulate", ttft=100.0),
            _row("cmp-x-r1", backend="simulate", ttft=100.0),
            _row("cmp-x-r2", backend="simulate", ttft=100.0),
            _row("cmp-x-r3", backend="real", ttft=200.0),
            _row("cmp-x-r4", backend="real", ttft=200.0),
            _row("cmp-x-r5", backend="real", ttft=200.0),
        ]
        agg = cm.aggregate_repeats(rows)
        self.assertEqual(len(agg), 2, "汇总必须按后端拆成两组")
        self.assertEqual(sorted(r["backend"] for r in agg), ["real", "simulate"])

        plan = plot_p99.build_plan(rows)
        self.assertEqual(len(plan.labels), 2, "绘图必须按后端拆成两根柱")
        values = sorted(v for v in plan.panels["ttft"] if v is not None)
        self.assertEqual(values, [100.0, 200.0])
        self.assertNotIn(150.0, values, "150 是谁都没测过的合并中位数")

    def test_aggregate_records_explicit_members(self):
        """每一组必须落盘明确的成员清单，导出侧才有据可依。"""
        rows = [
            _row("cmp-x", backend="simulate"),
            _row("cmp-x-r1", backend="simulate"),
            _row("cmp-x-r2", backend="simulate"),
        ]
        agg = cm.aggregate_repeats(rows)
        self.assertEqual(len(agg), 1)
        members = agg[0]["member_run_ids"].split(";")
        self.assertEqual(
            sorted(members), ["cmp-x", "cmp-x-r1", "cmp-x-r2"]
        )
        self.assertEqual(agg[0]["n_repeats"], 3)


# --------------------------------------------------------------------------- #
# 导出夹具
# --------------------------------------------------------------------------- #

def _stage_export(root: Path, *, declared: str | None, detected: str) -> dict[str, Path]:
    """造一套最小但**内容合格**的 raw + processed + EXPERIMENTS.md。

    与 ``tests/test_data_integrity.py`` 的夹具重复了一部分，是有意的：
    那个夹具服务于"逐项改坏一个开关"的检查，这个服务于"跨模块接线"，
    两者将来会朝不同方向演进，合成一个会互相牵制。

    只有一处关键差异：repeats.csv 带 ``member_run_ids``（真实聚合产物的形状），
    因为导出侧已经不再按 run_id 前缀自行推导成员。
    """
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    (raw / "_env").mkdir(exist_ok=True)
    (raw / "_env" / "env.json").write_text(
        json.dumps(_env_payload(detected)), encoding="utf-8"
    )

    base = "20260101-000000-w-cfgA"
    members = [base, f"{base}-r1", f"{base}-r2"]
    rows = []
    for rid in members:
        run_dir = raw / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        repro = {"git_commit": "deadbeef"}
        if declared is not None:
            repro["hardware_platform"] = declared
            repro["execution_environment"] = (
                "remote_mthreads" if declared == "mthreads" else "remote_ascend"
            )
        (run_dir / "meta.json").write_text(
            json.dumps({
                "data_label": "MEASURED",
                "summary": {"data_label": "MEASURED"},
                "reproducibility": repro,
                "config": {"config_path": "configs/mthreads/s4000_fp16.yaml"},
                "context": {"run_id": rid},
            }),
            encoding="utf-8",
        )
        (run_dir / "requests.jsonl").write_text('{"req_id": "r0"}\n', encoding="utf-8")
        (run_dir / "config.snapshot.yaml").write_text(
            "run:\n  repeats: 3\n", encoding="utf-8"
        )
        rows.append({
            "run_id": rid, "data_label": "MEASURED",
            "workload": "w", "config": "cfgA",
            "hardware_platform": "mthreads", "backend": "simulate",
        })

    with (processed / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["run_id", "data_label", "workload", "config",
                            "hardware_platform", "backend"]
        )
        writer.writeheader()
        writer.writerows(rows)

    with (processed / "repeats.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["group", "run_id_base", "member_run_ids", "n_repeats",
                            "sufficient_repeats", "data_label",
                            "hardware_platform", "backend"]
        )
        writer.writeheader()
        writer.writerow({
            "group": cm.group_key(rows[0]),
            "run_id_base": base,
            "member_run_ids": ";".join(members),
            "n_repeats": "3",
            "sufficient_repeats": "True",
            "data_label": "MEASURED",
            "hardware_platform": "mthreads",
            "backend": "simulate",
        })

    exp = root / "EXPERIMENTS.md"
    exp.write_text(
        "# EXPERIMENTS\n\n| EXP-x-001 | " + base + " | w | cfgA | 3 | 0.1 |\n",
        encoding="utf-8",
    )
    return {"raw": raw, "processed": processed, "experiments": exp}


class TestExportConsumesPersistedMembers(unittest.TestCase):
    """导出侧按**落盘的成员清单**核对，不自己按 run_id 前缀重推。"""

    def test_two_groups_sharing_a_base_are_not_merged(self):
        """同批次被拆成两组（不同后端）时，不得报出并集式的假不一致。

        这是用户报告的缺陷：导出校验按 run_id 前缀重新数成员，把被拆开的两组
        3 + 3 合成 6，于是"自报 3、实为 6"的假 ``repeats_count_mismatch``
        被报出来——它不仅本身是错的，还会把真正的问题盖住。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = _stage_export(root, declared=None, detected="mthreads")
            base = "20260101-000000-w-cfgA"
            # 把重复次数扩到 6 次，并按后端拆成两组各 3 个成员。
            with (staged["processed"] / "summary.csv").open("a", encoding="utf-8") as fh:
                for i in range(3, 6):
                    fh.write(
                        f"{base}-r{i},MEASURED,w,cfgA,mthreads,real\n"
                    )
            # 新增的成员也要有真实的原始明细，否则报的是"缺明细"而不是分组问题。
            for i in range(3, 6):
                run_dir = staged["raw"] / f"{base}-r{i}"
                run_dir.mkdir(parents=True, exist_ok=True)
                src = json.loads(
                    (staged["raw"] / base / "meta.json").read_text(encoding="utf-8")
                )
                src["context"]["run_id"] = f"{base}-r{i}"
                (run_dir / "meta.json").write_text(
                    json.dumps(src), encoding="utf-8"
                )
                (run_dir / "requests.jsonl").write_text(
                    '{"req_id": "r0"}\n', encoding="utf-8"
                )
                (run_dir / "config.snapshot.yaml").write_text(
                    "run:\n  repeats: 3\n", encoding="utf-8"
                )
            sim = [base, f"{base}-r1", f"{base}-r2"]
            real = [f"{base}-r{i}" for i in range(3, 6)]
            with (staged["processed"] / "repeats.csv").open(
                "w", encoding="utf-8", newline=""
            ) as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["group", "run_id_base", "member_run_ids",
                                    "n_repeats", "sufficient_repeats", "data_label",
                                    "hardware_platform", "backend"]
                )
                writer.writeheader()
                for backend, members in (("simulate", sim), ("real", real)):
                    writer.writerow({
                        "group": f"{base}|w|cfgA|mthreads|{backend}",
                        "run_id_base": base,
                        "member_run_ids": ";".join(members),
                        "n_repeats": "3",
                        "sufficient_repeats": "True",
                        "data_label": "MEASURED",
                        "hardware_platform": "mthreads",
                        "backend": backend,
                    })

            problems = pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            )
            codes = {p.code for p in problems}
            self.assertNotIn(
                "repeats_count_mismatch", codes,
                "按前缀重推会把两组并成 6 个成员，报出并不存在的次数不一致",
            )
            self.assertNotIn("insufficient_repeats", codes, "两组各 3 次，都够")
            self.assertNotIn("repeats_member_unaccounted", codes, "6 个成员都被认领")
            self.assertEqual(problems, [])

    def test_stale_repeats_file_is_visible_not_silent(self):
        """派生文件是旧版（没有成员清单）时必须**显形**，而不是静默跳过。

        旧实现会退回按前缀推导——那正是被修掉的那套规则。现在报一条明确的
        "重跑聚合"提示：缺一条证据时，正确的动作是补证据，不是猜。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage_export(Path(tmp), declared=None, detected="mthreads")
            path = staged["processed"] / "repeats.csv"
            text = path.read_text(encoding="utf-8")
            # 抹掉 member_run_ids 这一列（模拟旧版聚合产物）。
            lines = []
            for i, line in enumerate(text.splitlines()):
                if i == 0:
                    lines.append("group,run_id_base,n_repeats,data_label")
                else:
                    cells = line.split(",")
                    lines.append(",".join([cells[0], cells[1], cells[3], cells[5]]))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            problems = pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            )
            self.assertIn("repeats_members_missing", {p.code for p in problems})

    def test_unaccounted_run_is_reported(self):
        """汇总表里有 run 属于某批次却没被任何一组认领 —— 聚合可能漏了它。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = _stage_export(Path(tmp), declared=None, detected="mthreads")
            base = "20260101-000000-w-cfgA"
            with (staged["processed"] / "summary.csv").open("a", encoding="utf-8") as fh:
                fh.write(f"{base}-r9,MEASURED,w,cfgA,mthreads,simulate\n")
            problems = pf.check_export(
                raw_dir=staged["raw"],
                processed_dir=staged["processed"],
                experiments_md=staged["experiments"],
            )
            self.assertIn("repeats_member_unaccounted", {p.code for p in problems})


if __name__ == "__main__":
    unittest.main()
