"""Windows-only IPC adjustment is explicitly gated by the job supervisor."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from combat_rl import windows_runtime as w


class WindowsRuntimeTests(unittest.TestCase):
    def test_unsupervised_or_nonwindows_startup_is_unchanged(self):
        with patch.dict(w.os.environ,{},clear=True):
            self.assertFalse(w.configure_supervised_windows_ipc())
        with patch.dict(w.os.environ,{'CS557_SUPERVISED_WINDOWS':'1'}),patch.object(w.os,'name','posix'):
            self.assertFalse(w.configure_supervised_windows_ipc())

    def test_supervised_windows_skips_only_unix_orphan_scan(self):
        class IPC:
            def remove_orphan_java_processes(self):
                raise AssertionError('Should not scan unrelated Windows process handles')
            def check_port(self):
                return 'TCP check retained'
        fake=SimpleNamespace(SocketIPC=IPC)
        with patch.dict('sys.modules',{'craftground.environment.socket_ipc':fake}), \
             patch.dict(w.os.environ,{'CS557_SUPERVISED_WINDOWS':'1'}),patch.object(w.os,'name','nt'):
            self.assertTrue(w.configure_supervised_windows_ipc())
            self.assertIsNone(IPC().remove_orphan_java_processes())
            self.assertEqual(IPC().check_port(),'TCP check retained')


if __name__=='__main__':
    unittest.main()
