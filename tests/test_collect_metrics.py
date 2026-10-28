"""测试聚合脚本的读写行为。

重点不在数值，而在于**不把不该进表的东西放进表里**：
- dry-run 必须默认被排除（把模拟数据混进结论是最严重的错误）
- 缺标签的旧产物不得被回推成 MEASURED（"没打标签"不等于"测过了"）
- 重复次数不足的组必须被标注出来
- raw 已不存在的派生文件必须被回收（旧列结构残留会伪装成正常数据）
- summary 的字段拍平后必须能被重新读回（列名稳定、可 diff）
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
from ascend_opt.metrics import RequestMetrics, summarize  # noqa: E402
from ascend_opt.provenance import DataLabel  # noqa: E402


def make_run(
    root: Path,
    name: str,
    *,
    config: str = "cfgA",
    workload: str = "w",
    dry_run: bool = False,
    ttft: float = 100.0,
    tpot: float = 10.0,
    n: int = 5,
    label: str | None = None,
    platform: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    reproducibility: bool = False,
) -> Path:
    """在 root 下造一个最小的合法 run 目录（meta.json + requests.jsonl）。

    ``label`` 默认按 ``dry_run`` 推一个（模拟 TRUE_TEST_DATA / 实测 MEASURED），
    显式传入则用传入值——用于构造 HARDWARE_REQUIRED 这类"不是 dry_run
    但也不是实测"的情形。

    ``platform`` / ``backend`` / ``precision`` 写成规范第 8 节的位置
    （``meta["reproducibility"]``），它们同时是分组键的成分。不传则不写该段——
    用来验证"旧产物没有这些字段"时的行为（必须是保留 unknown，不是猜一个值）。
    """
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if label is None:
        label = "TEST_DATA" if dry_run else "MEASURED"

    # 时间戳造得自洽（E2E == 完成 − 到达），这样账目自查与 TTFT 三段分解
    # 都能真的被验证，而不是靠"反正测试不看这几个字段"混过去。
    metrics = []
    for i in range(n):
        factor = 1 + 0.1 * i
        ttft_i = ttft * factor
        tpot_i = tpot * factor
        metrics.append(
            RequestMetrics(
                f"r{i}", 100, 10,
                ttft_ms=ttft_i,
                tpot_ms=tpot_i,
                tpot_p99_ms=tpot * 5,
                scheduled_arrival_ms=0.0,
                arrival_ms=0.0,
                enqueue_ms=10.0,
                first_schedule_ms=40.0,
                first_token_ms=ttft_i,
                done_ms=ttft_i + tpot_i * 9,
            )
        )
    summary = summarize(
        metrics, run_id=name, workload=workload, config=config, wall_ms=1000.0
    ).describe()
    summary["data_label"] = label

    meta = {
        "context": {
            "run_id": name,
            "workload": workload,
            "config_path": f"configs/{config}.yaml",
            "repeat_index": 0,
            "dry_run": dry_run,
        },
        "summary": summary,
        "data_label": label,
    }
    if reproducibility or platform or backend or precision:
        repro: dict[str, object] = {}
        if platform is not None:
            repro["hardware_platform"] = platform
        if backend is not None:
            repro["backend"] = backend
        if precision is not None:
            repro["precision"] = precision
        meta["reproducibility"] = repro
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for m in metrics:
            fh.write(json.dumps(m.describe(), ensure_ascii=False) + "\n")
    return run_dir


def strip_label(run_dir: Path) -> None:
    """抹掉产物里的 data_label，构造"旧版本 harness"留下的无标签产物。

    老 harness 不写标签，所以兼容路径是真实存在的一条分支——它必须被测到。
    """
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    meta.pop("data_label", None)
    (meta.get("summary") or {}).pop("data_label", None)
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


class TestCollectMetrics(unittest.TestCase):
    def _run_main(self, args):
        return cm.main(args)

    def test_dry_run_is_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True)
            make_run(raw, "20260101-000001-w-cfgA-r1", dry_run=True)

            self.assertEqual(self._run_main(["--raw", str(raw), "--out", str(out)]), 0)
            # 没有真实 run，不应该产出汇总表（空文件也不该生成）
            self.assertFalse((out / "summary.csv").exists())

    def test_include_dry_run_writes_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True)

            self.assertEqual(
                self._run_main(
                    ["--raw", str(raw), "--out", str(out), "--include-dry-run"]
                ),
                0,
            )
            rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            # 结局账目与 SLO 三口径必须出现在表里：这是"分开报告"的落点，
            # 少一列就意味着下游无法还原失败率/达标率，而不是"这一列没用"。
            for column in (
                "ttft_p99", "tpot_p99", "stall_p99", "dry_run",
                "throughput_tok_s",
                "admission_p99", "queue_p99", "prefill_p99",
                "n_failed", "n_rejected", "n_unfinished",
                "failure_rate", "rejection_rate", "unfinished_rate",
                "n_slo_satisfied", "slo_satisfaction_rate",
                "slo_goodput_req_s", "slo_goodput_tok_s",
            ):
                self.assertIn(column, rows[0])
            # goodput 这个含混的名字必须彻底消失：它既可能被读成"达标比例"，
            # 也可能被读成"达标吞吐"，而两者是不同的数。
            self.assertNotIn("goodput", rows[0])
            self.assertEqual(rows[0]["dry_run"], "True")
            # 逐请求明细也要落盘
            self.assertTrue((out / "20260101-000000-w-cfgA.csv").exists())

    def test_repeats_grouping_flags_insufficient_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")
            make_run(raw, "20260101-000000-w-cfgA-r1")

            self._run_main(["--raw", str(raw), "--out", str(out)])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_repeats"], "2")
            self.assertEqual(rows[0]["sufficient_repeats"], "False")
            # 重复次数不足时不该给出跨重复稳健 CV，避免被当成可信离散度
            self.assertEqual(rows[0]["ttft_p99_robust_cv"], "")

    def test_three_repeats_are_marked_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")
            make_run(raw, "20260101-000000-w-cfgA-r1")
            make_run(raw, "20260101-000000-w-cfgA-r2")

            self._run_main(["--raw", str(raw), "--out", str(out)])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(rows[0]["sufficient_repeats"], "True")
            self.assertNotEqual(rows[0]["ttft_p99_robust_cv"], "")

    def test_different_configs_are_not_grouped_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")
            make_run(raw, "20260101-000001-w-cfgB", config="cfgB")

            self._run_main(["--raw", str(raw), "--out", str(out)])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 2)

    def test_empty_raw_dir_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = self._run_main(
                ["--raw", str(Path(tmp) / "raw"), "--out", str(Path(tmp) / "out")]
            )
            self.assertEqual(code, 0, "盘上没有 run 是合法状态，不该让上层脚本中断")

    # ------------------------------------------------------------------ #
    # 数据标签（规范第 10 节）：标签是本脚本的过滤条件，不是装饰
    # ------------------------------------------------------------------ #

    def test_summary_carries_data_label_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")   # 显式 MEASURED

            self._run_main(["--raw", str(raw), "--out", str(out)])
            rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
            self.assertEqual(rows[0]["data_label"], "MEASURED")
            self.assertEqual(rows[0]["label_source"], "explicit")
            self.assertEqual(rows[0]["eligible_for_conclusion"], "True")

    def test_legacy_run_without_label_is_not_upgraded_to_measured(self):
        """缺标签的旧产物**不得**被回推成 MEASURED。

        早先的实现对 ``dry_run=False`` 的旧产物直接回推 ``MEASURED``，于是一批
        "只是没打标签"的数据被静默升格成可进结论的实测数据——不抛错、不告警，
        只在最后产出一张看起来完全正常的汇总表。

        回推能可靠推出的只有一件事：``dry_run=True`` 说明数字是合成的。
        反向推不出"实测"——本机无 NPU 时跑真机路径同样是 ``dry_run=False``。
        所以缺标签一律 ``NOT_MEASURED``。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            strip_label(make_run(raw, "20260101-000000-w-cfgA"))

            # 默认口径：NOT_MEASURED 不可进结论，连汇总表都不该产出
            self._run_main(["--raw", str(raw), "--out", str(out)])
            self.assertFalse(
                (out / "summary.csv").exists(),
                "缺标签的旧产物不得默认进入汇总表",
            )

            # 核对口径：必须显形为 NOT_MEASURED + legacy_inferred
            self._run_main(
                ["--raw", str(raw), "--out", str(out), "--include-non-measured"]
            )
            rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
            self.assertEqual(rows[0]["data_label"], "NOT_MEASURED")
            self.assertEqual(rows[0]["label_source"], "legacy_inferred")
            self.assertEqual(rows[0]["eligible_for_conclusion"], "False")

    def test_legacy_dry_run_is_downgraded_to_test_data(self):
        """回推只允许降级：dry_run=True 的旧产物判 TEST_DATA。

        ``dry_run=True`` 是**可靠**推断（数字必然是合成的），
        它把无标签降级为 TEST_DATA，方向上只往下走。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            strip_label(make_run(raw, "20260101-000000-w-cfgA", dry_run=True))

            self._run_main(
                ["--raw", str(raw), "--out", str(out), "--include-non-measured"]
            )
            rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
            self.assertEqual(rows[0]["data_label"], "TEST_DATA")
            self.assertEqual(rows[0]["label_source"], "legacy_inferred")

    def test_legacy_inference_never_produces_measured(self):
        """直接断言 ``effective_label``：不存在"无标签 → MEASURED"的路径。

        这是本项的不变量。逐一遍历可能的 context 形态，而不是只测一两个样例——
        漏掉 ``dry_run`` 缺失这一支，正是原来那条错误回推的藏身处。
        """
        for ctx in ({}, {"dry_run": True}, {"dry_run": False}):
            label, source = cm.effective_label({"context": ctx})
            self.assertEqual(source, "legacy_inferred")
            self.assertNotEqual(
                label, DataLabel.MEASURED,
                f"无标签产物不得被回推成 MEASURED（context={ctx}）",
            )
        # 连 context 整段缺失也不许升格
        label, _ = cm.effective_label({})
        self.assertNotEqual(label, DataLabel.MEASURED)

    def test_unknown_explicit_label_is_not_measured(self):
        """显式标签取值无法识别时不猜，按最保守的 NOT_MEASURED 处理。"""
        label, source = cm.effective_label({"data_label": "MEASURED_MAYBE"})
        self.assertEqual(label, DataLabel.NOT_MEASURED)
        self.assertEqual(source, "explicit")

    # ------------------------------------------------------------------ #
    # 派生数据卫生：raw 已不存在却留在 processed 里的旧列结构文件
    # ------------------------------------------------------------------ #

    def test_orphan_derived_csv_is_recycled(self):
        """raw 已不存在的逐 run 派生 CSV 必须被回收。

        残留文件不会报错，只会看起来像一份正常数据——本仓库实际出现过
        （列里是 ``success`` 而没有 ``outcome``，属于旧 schema 的产物）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            out.mkdir(parents=True)
            orphan = out / "20260101-000000-w-gone.csv"
            orphan.write_text("arrival_ms,success\n0,True\n", encoding="utf-8")
            make_run(raw, "20260101-000000-w-cfgA")

            self._run_main(["--raw", str(raw), "--out", str(out)])

            self.assertFalse(orphan.exists(), "孤儿派生文件必须被回收")
            # 有 raw 兜底的 run 明细不能被误删
            self.assertTrue((out / "20260101-000000-w-cfgA.csv").exists())

    def test_keep_orphans_flag_disables_recycling(self):
        """回收是可关的：不确定时应当能先留着看。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            out.mkdir(parents=True)
            orphan = out / "20260101-000000-w-gone.csv"
            orphan.write_text("arrival_ms,success\n0,True\n", encoding="utf-8")
            make_run(raw, "20260101-000000-w-cfgA")

            self._run_main(["--raw", str(raw), "--out", str(out), "--keep-orphans"])
            self.assertTrue(orphan.exists())

    def test_run_id_filter_does_not_prune_other_runs(self):
        """``--run-id`` 过滤运行时，其它 run 的明细不是孤儿，不得被删。

        判据必须是"raw 里有没有同名目录"这种结构性事实，而不是"本次有没有被选中"
        ——后者会让一次过滤运行把别人的派生数据一起清掉。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")
            make_run(raw, "20260101-000001-w-cfgB", config="cfgB")

            self._run_main(["--raw", str(raw), "--out", str(out)])
            kept = out / "20260101-000001-w-cfgB.csv"
            self.assertTrue(kept.exists())

            self._run_main(
                ["--raw", str(raw), "--out", str(out), "--run-id", "cfgA"]
            )
            self.assertTrue(kept.exists(), "过滤运行不得删掉其它 run 的明细")

    def test_group_label_takes_strictest_member(self):
        """一组里混进一个非实测，整组就不能当结论。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA")            # MEASURED
            make_run(raw, "20260101-000000-w-cfgA-r1", dry_run=True, label="TEST_DATA")

            self._run_main(
                ["--raw", str(raw), "--out", str(out), "--include-non-measured"]
            )
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data_label"], "TEST_DATA")

    def test_stale_summary_is_cleared_when_nothing_is_eligible(self):
        """没有可用 run 时必须清空旧汇总表。

        否则下游会读到**上一次**的数字却以为是本次的——派生数据被静默复用
        是这类流水线最危险的错误之一。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True)

            # 先造一份"看起来有数据"的汇总
            self._run_main(
                ["--raw", str(raw), "--out", str(out), "--include-non-measured"]
            )
            self.assertGreater((out / "summary.csv").stat().st_size, 0)

            # 再按默认口径聚合：没有 MEASURED，旧表必须被清空
            self._run_main(["--raw", str(raw), "--out", str(out)])
            self.assertEqual(
                (out / "summary.csv").stat().st_size, 0,
                "旧汇总表必须被清空，不能被下次聚合静默复用",
            )

    def test_measured_but_hardware_required_run_is_filtered(self):
        """HARDWARE_REQUIRED 的 run 不是 dry_run，但同样不能进结论。

        这正是"按 dry_run 过滤"会漏掉的情形——过滤依据必须是标签本身。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=False,
                     label="HARDWARE_REQUIRED")

            code = self._run_main(["--raw", str(raw), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertFalse(
                (out / "summary.csv").exists()
                and (out / "summary.csv").stat().st_size > 0,
                "HARDWARE_REQUIRED 不得进入汇总表",
            )


class TestPlatformReachesTheAggregate(unittest.TestCase):
    """平台/后端/精度必须真的进表、真的参与分组（评审 #2）。

    修这个问题之前，``hardware_platform`` 只是被"记录"了：``ReproducibilityRecord``
    里有这个字段、meta 里也写了，但 ``flatten_summary`` 不输出、``group_key`` 不读。
    于是两个平台的同名实验被并成一组，离散度算出来的是一个不存在的实验的离散度。

    这组测试守的是"加字段必须同时把消费者接上"——只写不读的字段等于没加。
    """

    def _run_main(self, args):
        return cm.main(args)

    def _summary(self, out: Path) -> list[dict[str, str]]:
        return list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))

    def test_platform_backend_precision_appear_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True,
                     platform="mthreads", backend="simulate", precision="fp16")

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            row = self._summary(out)[0]
            self.assertEqual(row["hardware_platform"], "mthreads")
            self.assertEqual(row["backend"], "simulate")
            self.assertEqual(row["precision"], "fp16")

    def test_two_platforms_are_never_merged_into_one_group(self):
        """评审的复现：同负载、同配置，两个平台取值 100 / 200 → 曾合成一根 150。

        合并后的中位数是一个**谁都没测过**的数，而它看起来完全正常。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True,
                     ttft=100.0, platform="ascend")
            make_run(raw, "20260101-000000-w-cfgA-r1", dry_run=True,
                     ttft=200.0, platform="mthreads")

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 2, "两个平台必须各成一组，不得合并")
            self.assertEqual(
                sorted(r["hardware_platform"] for r in rows), ["ascend", "mthreads"]
            )
            for row in rows:
                self.assertEqual(row["n_repeats"], "1")
                # 合并时这里会是 (100+200)/2 = 150 —— 必须不出现。
                self.assertNotEqual(row["ttft_p99_median"], "150.0")

    def test_repeats_of_the_same_platform_still_group_together(self):
        """正向对照：同一个平台的重复仍要合并，否则平台隔离就变成了过度拆分。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            for i, name in enumerate(["20260101-000000-w-cfgA",
                                      "20260101-000000-w-cfgA-r1",
                                      "20260101-000000-w-cfgA-r2"]):
                make_run(raw, name, dry_run=True, ttft=100.0, platform="mthreads")

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_repeats"], "3")
            self.assertEqual(rows[0]["hardware_platform"], "mthreads")

    def test_backend_is_part_of_the_group_key(self):
        """模拟与真机的数字口径本就不同，不得合起来算离散度。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True,
                     platform="mthreads", backend="simulate")
            make_run(raw, "20260101-000000-w-cfgA-r1", dry_run=True,
                     platform="mthreads", backend="real")

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(r["backend"] for r in rows), ["real", "simulate"])

    def test_missing_platform_is_unknown_not_guessed(self):
        """旧产物没有这个字段 -> ``unknown``。

        不能因为"配置里写着昇腾"就替它填上——旧 run 的执行环境本来就无从知晓，
        猜一个平台会让它混进按平台筛选的结果里，且不报错。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True)

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            row = self._summary(out)[0]
            self.assertEqual(row["hardware_platform"], "unknown")
            self.assertEqual(row["backend"], "unknown")

    def test_top_level_fallback_is_read(self):
        """字段写在 meta 顶层（而非 reproducibility 段）也要读得到。

        漏读的表现是"一列静默空掉"，不会报错——所以两种位置都固定住。
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            run_dir = make_run(raw, "20260101-000000-w-cfgA", dry_run=True)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            meta["hardware_platform"] = "ascend"
            (run_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            self.assertEqual(self._summary(out)[0]["hardware_platform"], "ascend")

    def test_group_key_names_its_dimensions(self):
        """分组键的成分必须能被 repeats.csv 自解释，不必反解那串拼接。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            make_run(raw, "20260101-000000-w-cfgA", dry_run=True,
                     platform="ascend", backend="simulate")

            self._run_main(["--raw", str(raw), "--out", str(out), "--include-dry-run"])
            rows = list(csv.DictReader((out / "repeats.csv").open(encoding="utf-8")))
            self.assertEqual(rows[0]["hardware_platform"], "ascend")
            self.assertEqual(rows[0]["backend"], "simulate")
            self.assertIn("ascend", rows[0]["group"], "分组键里必须含平台")
            self.assertIn("simulate", rows[0]["group"], "分组键里必须含后端")


if __name__ == "__main__":
    unittest.main()
