"""Atomic queue status writes survive real and simulated Windows file locks."""
import ctypes
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts import queue_optimization_windows as q


class PersistenceTests(unittest.TestCase):
    def test_transient_replace_denial_retries_without_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            q.write_json(path, {'phase':'old'})
            original = Path.replace
            attempts = []
            def replace(staging, target):
                attempts.append(staging)
                self.assertEqual(q.read_json(path), {'phase':'old'})
                if len(attempts) < 3:
                    raise PermissionError('Temporary sharing violation')
                return original(staging, target)
            with patch.object(Path, 'replace', replace), patch.object(q.time, 'sleep') as sleep:
                q.write_json(path, {'phase':'running', 'remaining_seconds':None})
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(q.read_json(path)['phase'], 'running')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_permanent_denial_remains_an_error_and_preserves_previous_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            q.write_json(path, {'phase':'old'})
            with patch.object(Path, 'replace', side_effect=PermissionError('Persistent denial')), \
                 patch.object(q.time, 'sleep'):
                with self.assertRaises(PermissionError):
                    q.write_json(path, {'phase':'new'})
            self.assertEqual(q.read_json(path), {'phase':'old'})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_real_windows_reader_temporarily_prevents_atomic_replacement(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,
                                      ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            q.write_json(path, {'phase':'old'})
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            # Share read/write but deny deletion/replacement, like an active reader.
            release = threading.Timer(.3, lambda: kernel.CloseHandle(handle))
            release.start()
            try:
                q.write_json(path, {'phase':'new'})
            finally:
                release.join()
            self.assertEqual(q.read_json(path), {'phase':'new'})


if __name__ == '__main__':
    unittest.main()
