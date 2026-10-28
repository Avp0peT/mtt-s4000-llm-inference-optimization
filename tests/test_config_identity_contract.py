"""续跑时的**配置身份**判据（`config_identity_complaint`）。

★ 2026-10-03：实测到一条"**永远不复用**"的缺陷
------------------------------------------------
写端落快照用的是 `yaml.safe_dump(cfg.raw, ...)` —— 一次**重新序列化**
（注释、空行、键序、引号风格全被规范化）。而读端原先比的是

    snapshot.read_bytes() == cfg_file.read_bytes()        # 原始字节

本仓每份配置都带大段注释 ⇒ 两者**永远不可能相等** ⇒ 配置身份**恒判不一致**
⇒ 续跑**永远不复用**任何既有产物。更糟的是报错文案说"配置已变
（… 与产物里的快照不一致）"——人去找差异，**找不出**，因为差异只是注释。

⇒ 约定改为：**身份 = 参数内容的规范摘要**（`config_data_digest`）。
  注释/排版变了 ⇒ 不变；参数变了 ⇒ 变；取不到 ⇒ 拒绝。

本文件的关键用例是 `test_real_writer_snapshot_is_reusable` ——
它用**真实写端**（`run_once(dry_run=True)`）产出的快照，而不是手写的假快照。
手写的快照只能证明"判据对某些字节会怎样"，证明不了"真实写端写出来的东西
会不会被判据接受"。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import (  # noqa: E402
    CONFIG_SNAPSHOT_NAME,
    config_data_digest,
    load_config,
    run_once,
)

import run_exec_opt_matrix as M  # noqa: E402

CFG = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both.yaml"
OTHER_ARM = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both_s04_pkv.yaml"


class TestDigestProperties(unittest.TestCase):
    """`config_data_digest` 的三条性质（写读约定的基础）。"""

    def test_key_order_does_not_matter(self):
        self.assertEqual(config_data_digest({"a": 1, "b": 2}),
                         config_data_digest({"b": 2, "a": 1}))

    def test_value_type_matters(self):
        """`1` 与 `"1"` 必须不同 —— JSON 化比较是**类型化**的，YAML 文本比较不是。"""
        self.assertNotEqual(config_data_digest({"a": 1}),
                            config_data_digest({"a": "1"}))

    def test_nested_change_is_detected(self):
        self.assertNotEqual(config_data_digest({"a": {"b": [1, 2]}}),
                            config_data_digest({"a": {"b": [1, 3]}}))


class TestConfigIdentityAgainstRealWriter(unittest.TestCase):
    """★ 用**真实写端**产出的产物来验读端判据。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="cfgid_")
        cls.tmp = Path(cls._tmp.name)
        # 真实写端：dry_run 走离线模拟后端，不需要真机/真模型
        cls.run_dir = run_once(load_config(CFG), dry_run=True,
                               out_dir=cls.tmp, verbose=False)
        cls.src = CFG.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8", newline="\n")
        return p

    # ---------------- 正向：必须允许复用 ----------------

    def test_real_writer_snapshot_is_reusable(self):
        """★★ 靶心：真实写端写的快照 + 同一份配置 ⇒ **允许复用**。

        修之前这条必然失败（写端重新序列化 vs 读端比原始字节）。
        """
        why = M.config_identity_complaint(self.run_dir, CFG)
        self.assertIsNone(
            why, f"真实写端产出的快照必须能复用，实际被拒：{why}")

    def test_snapshot_bytes_really_differ_from_source(self):
        """**前提证据**：字节层面确实不同 —— 这就是旧判据必然失败的原因。

        没有这条，别人可能会以为"旧判据只是偶尔失败"。
        """
        snap = self.run_dir / CONFIG_SNAPSHOT_NAME
        self.assertNotEqual(
            snap.read_bytes(), CFG.read_bytes(),
            "写端是重新序列化 ⇒ 字节必然不同（注释被丢掉）")
        self.assertIn("#", self.src, "源配置确实带注释（这是差异的来源）")

    def test_comment_only_change_still_reuses(self):
        """只改注释 ⇒ 摘要不变 ⇒ **允许复用**（排版不影响实验）。"""
        changed = self.src.replace(
            "max_num_seqs: 32", "max_num_seqs: 32  # 只加注释，参数没变")
        self.assertNotEqual(changed, self.src, "replace 必须真的命中")
        p = self._write("comment.yaml", changed)
        self.assertIsNone(M.config_identity_complaint(self.run_dir, p))

    def test_key_reorder_still_reuses(self):
        """只调换**同一个映射内**两个键的顺序 ⇒ 同样允许复用。

        ⚠️ 必须真的只在同一层内换序。第一版我把一行挪到文件末尾，
        那改变了它的**层级**（从 `scheduler:` 下变成顶层）⇒ 参数确实变了 ⇒
        判据正确拒绝，而我的断言写反了。
        """
        block = "  policy: continuous_batching\n  max_num_seqs: 32\n"
        self.assertIn(block, self.src, "锚点必须命中，否则这条测试在测空气")
        swapped = self.src.replace(
            block, "  max_num_seqs: 32\n  policy: continuous_batching\n")
        p = self._write("reordered.yaml", swapped)
        self.assertIsNone(M.config_identity_complaint(self.run_dir, p))

    # ---------------- 反向：必须拒绝 ----------------

    def test_parameter_change_is_rejected_with_the_path(self):
        """★ 参数改变 ⇒ 拒绝，且报错要**指出改的是哪个参数**。"""
        p = self._write("param.yaml",
                        self.src.replace("max_num_seqs: 32", "max_num_seqs: 33"))
        why = M.config_identity_complaint(self.run_dir, p)
        self.assertIsNotNone(why, "参数改变必须拒绝复用")
        self.assertIn("max_num_seqs", why, "必须指出具体是哪个参数变了")

    def test_missing_snapshot_is_rejected(self):
        """快照缺失 ⇒ 摘要**测不出** ⇒ 拒绝（不得读成"没问题"）。"""
        d = self.tmp / "nosnap"
        d.mkdir(exist_ok=True)
        (d / "meta.json").write_text("{}", encoding="utf-8", newline="\n")
        why = M.config_identity_complaint(d, CFG)
        self.assertIsNotNone(why)
        self.assertIn("缺失", why)

    def test_corrupt_snapshot_is_rejected(self):
        d = self.tmp / "corrupt"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.run_dir, d)
        (d / CONFIG_SNAPSHOT_NAME).write_text("{not: valid: yaml: [",
                                              encoding="utf-8", newline="\n")
        why = M.config_identity_complaint(d, CFG)
        self.assertIsNotNone(why)
        self.assertIn("读不出", why)

    def test_non_mapping_snapshot_is_rejected(self):
        d = self.tmp / "scalar"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.run_dir, d)
        (d / CONFIG_SNAPSHOT_NAME).write_text("- just\n- a\n- list\n",
                                              encoding="utf-8", newline="\n")
        why = M.config_identity_complaint(d, CFG)
        self.assertIsNotNone(why, "快照不是映射 ⇒ 无法核验 ⇒ 拒绝")
        self.assertIn("不是映射", why)

    def test_snapshot_of_another_arm_is_rejected(self):
        """快照来自**另一臂** ⇒ 参数不同 ⇒ 拒绝（并指出差在哪些键）。"""
        d = self.tmp / "otherarm"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.run_dir, d)
        shutil.copy(OTHER_ARM, d / CONFIG_SNAPSHOT_NAME)
        why = M.config_identity_complaint(d, CFG)
        self.assertIsNotNone(why)
        self.assertIn("persistent_kv_slots", why)

    def test_conflicting_recorded_digest_is_rejected(self):
        """meta 里另记的摘要与快照打架 ⇒ 拒绝（两份记录不许挑对自己有利的那份）。"""
        d = self.tmp / "conflict"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.run_dir, d)
        mp = d / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta.setdefault("reproducibility", {})["config_digest"] = "deadbeef" * 8
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
        why = M.config_identity_complaint(d, CFG)
        self.assertIsNotNone(why)
        self.assertIn("自相矛盾", why)

    def test_missing_current_config_is_rejected(self):
        why = M.config_identity_complaint(self.run_dir, self.tmp / "nope.yaml")
        self.assertIsNotNone(why)
        self.assertIn("不存在", why)


class TestWriteReadConventionIsShared(unittest.TestCase):
    """写端与读端必须用**同一个**文件名常量（两处写字面量迟早漂移）。"""

    def test_matrix_reads_the_shared_constant(self):
        self.assertIs(M.CONFIG_SNAPSHOT_NAME, CONFIG_SNAPSHOT_NAME)

    def test_writer_uses_the_shared_constant(self):
        """写端的文件名取自常量 —— 用真实产物核验，不靠读源码。"""
        with tempfile.TemporaryDirectory() as td:
            d = run_once(load_config(CFG), dry_run=True, out_dir=Path(td),
                         verbose=False)
            self.assertTrue((d / CONFIG_SNAPSHOT_NAME).is_file())


class TestIdentityComplaintStillCoversOtherAxes(unittest.TestCase):
    """配置身份只是三根轴之一；源码/模型那两根不能被这次改动削弱。

    ⚠️ 这些用例必须让**配置身份先通过**（否则 `identity_complaint` 会在配置那一关
    就返回，根本走不到模型/源码判据 —— 第一版就是这样，断言拿到的是配置的报错）。
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="axes_")
        cls.tmp = Path(cls._tmp.name)
        real = run_once(load_config(CFG), dry_run=True, out_dir=cls.tmp,
                        verbose=False)
        #: 一份**配置身份可通过**的产物快照，供各用例复制后改自己那一轴。
        cls.snapshot = (real / CONFIG_SNAPSHOT_NAME).read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _meta_dir(self, **repro) -> Path:
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        (d / CONFIG_SNAPSHOT_NAME).write_bytes(self.snapshot)
        (d / "meta.json").write_text(
            json.dumps({"reproducibility": repro}), encoding="utf-8", newline="\n")
        return d

    def test_config_axis_passes_so_other_axes_are_reachable(self):
        """前提：不带任何 repro 字段时，配置这关是过的（否则下面三条测不到东西）。"""
        d = self._meta_dir()
        why = M.identity_complaint(d, arm="a", cfg_file=CFG,
                                   model_revision="r", source_digest="abc")
        self.assertIsNotNone(why, "没有 git_source ⇒ 必须在源码那关被拒")
        self.assertIn("源码身份", why, "★ 必须是**源码**的报错，不是配置的")

    def test_source_identity_missing_is_rejected(self):
        d = self._meta_dir()
        why = M.identity_complaint(d, arm="a", cfg_file=CFG, model_revision="r",
                                   source_digest="abc")
        self.assertIsNotNone(why)
        self.assertIn("缺失", why)

    def test_untrusted_source_is_rejected(self):
        d = self._meta_dir(git_source="guess")
        why = M.identity_complaint(d, arm="a", cfg_file=CFG, model_revision="r",
                                   source_digest="abc")
        self.assertIsNotNone(why)
        self.assertIn("不可信", why)

    def test_model_revision_mismatch_is_rejected(self):
        d = self._meta_dir(git_source="stamp", tree_digest="abc",
                           model_revision="OTHER")
        why = M.identity_complaint(d, arm="a", cfg_file=CFG, model_revision="HERE",
                                   source_digest="abc")
        self.assertIsNotNone(why)
        self.assertIn("模型", why)


if __name__ == "__main__":
    unittest.main()
