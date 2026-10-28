"""`scripts/diag_remote_reachability.py` 的用例。

这个脚本存在的唯一理由，是**不许把三种状态读成一件**：
`BANNER`（在） / `EOF`（主机在但端口没服务） / `TIMEOUT`（包被丢）。
所以用例的重点不是"能连上"，而是**三态必须分开、且 `interpret()` 的读法必须随态变化**。
全部用假 socket，**不碰网络**（离线可跑，符合"纯逻辑入口任意解释器都能跑"）。
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "diag_remote_reachability.py"


def _load():
    spec = importlib.util.spec_from_file_location("diag_remote_reachability", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diag_remote_reachability"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _FakeSock:
    """最小假 socket：只实现被测代码用到的那几个方法。"""

    def __init__(self, payload: bytes | BaseException):
        self._payload = payload
        self.closed = False

    def settimeout(self, _t):
        return None

    def recv(self, _n):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def close(self):
        self.closed = True


def _connect_returning(payload):
    def _connect(_addr, _timeout):
        return _FakeSock(payload)
    return _connect


def _connect_raising(exc):
    def _connect(_addr, _timeout):
        raise exc
    return _connect


class TestClassification(unittest.TestCase):
    """三态分类：这是整个脚本的信息量所在。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_banner(self):
        r = self.mod.probe("h", 1, "t", connect=_connect_returning(b"SSH-2.0-OpenSSH_8.9p1\r\n"))
        self.assertEqual(r["verdict"], "BANNER")
        self.assertIn("SSH-2.0-OpenSSH_8.9p1", r["detail"])
        self.assertNotIn("\r", r["detail"])  # 换行被压平，便于一行输出

    def test_eof_is_not_timeout(self):
        """★ 核心断言：EOF 与 TIMEOUT 必须是**两个不同的** verdict。"""
        eof = self.mod.probe("h", 1, "t", connect=_connect_returning(b""))
        tmo = self.mod.probe("h", 1, "t", connect=_connect_raising(socket.timeout()))
        self.assertEqual(eof["verdict"], "EOF")
        self.assertEqual(tmo["verdict"], "TIMEOUT")
        self.assertNotEqual(eof["verdict"], tmo["verdict"])
        self.assertIn("没服务", eof["detail"])
        self.assertIn("丢弃", tmo["detail"])

    def test_refused(self):
        r = self.mod.probe("h", 1, "t", connect=_connect_raising(ConnectionRefusedError()))
        self.assertEqual(r["verdict"], "REFUSED")

    def test_connected_but_silent(self):
        """连上却一个字节都不发 ⇒ 隧道假活，必须与 EOF 区分。"""
        r = self.mod.probe("h", 1, "t", connect=_connect_returning(socket.timeout()))
        self.assertEqual(r["verdict"], "TIMEOUT_AFTER_CONNECT")

    def test_socket_closed(self):
        s = _FakeSock(b"SSH-")
        self.mod.probe("h", 1, "t", connect=lambda a, t: s)
        self.assertTrue(s.closed)

    def test_error_is_caught_not_raised(self):
        r = self.mod.probe("h", 1, "t", connect=_connect_raising(OSError("boom")))
        self.assertEqual(r["verdict"], "ERROR")
        self.assertIn("boom", r["detail"])


class TestInterpret(unittest.TestCase):
    """读法必须随态变化，且"对照也不正常"时**不得**下远端结论。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def _rows(self, target_verdict, target_detail="", others=("BANNER",)):
        rows = [{"label": "目标", "target": "h:p", "verdict": target_verdict,
                 "detail": target_detail, "ms": 1.0, "note": ""}]
        rows += [{"label": f"c{i}", "target": "x", "verdict": v, "detail": "", "ms": 1.0, "note": ""}
                 for i, v in enumerate(others)]
        return rows

    def test_banner_gives_rc0(self):
        rc, lines = self.mod.interpret(self._rows("BANNER", "SSH-2.0-OpenSSH_8.9p1"))
        self.assertEqual(rc, 0)
        self.assertTrue(any("可 SSH 可达" in ln for ln in lines))
        # banner ≠ 认证通过，必须提示再跑一次 exec
        self.assertTrue(any("认证" in ln for ln in lines))

    def test_eof_points_at_port_number(self):
        rc, lines = self.mod.interpret(self._rows("EOF"))
        self.assertEqual(rc, 1)
        joined = "\n".join(lines)
        self.assertIn("端口号变了", joined)
        self.assertIn("remote.env", joined)
        # ⚠️ 不得把它说成"实例关机"
        self.assertNotIn("关机", joined)

    def test_timeout_with_healthy_control_blames_remote(self):
        rc, lines = self.mod.interpret(self._rows("TIMEOUT", others=("BANNER",)))
        self.assertEqual(rc, 1)
        joined = "\n".join(lines)
        self.assertIn("远端侧", joined)
        self.assertIn("关掉本机代理", joined)  # §5.1.2 的复测动作必须被提示

    def test_timeout_without_control_refuses_to_conclude(self):
        """★ 反向断言：对照也不正常时，必须显式说"判据没有分辨力"，不得下结论。"""
        rc, lines = self.mod.interpret(self._rows("TIMEOUT", others=("TIMEOUT", "EOF")))
        self.assertEqual(rc, 1)
        joined = "\n".join(lines)
        self.assertIn("没有分辨力", joined)
        self.assertIn("不得", joined)
        self.assertNotIn("远端侧", joined)

    def test_empty_rows_is_not_a_pass(self):
        rc, lines = self.mod.interpret([])
        self.assertEqual(rc, 2)  # 空输入不得读成"通过"

    def test_read_env_missing_file_is_empty_not_error(self):
        self.assertEqual(self.mod.read_env(Path("no/such/file.env")), {})

    def test_default_targets_include_a_negative_control(self):
        """必须自带"必然没服务"的对照，否则判据会失去分辨力。"""
        t = self.mod.default_targets("host", 1234)
        self.assertEqual(len(t), 5)
        self.assertEqual(t[0][2], 1234)
        self.assertTrue(any(port == 9 for _l, _h, port, _n in t[1:]))


if __name__ == "__main__":
    unittest.main()
