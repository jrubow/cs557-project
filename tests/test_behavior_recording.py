"""Verify optional behavior overlays and encoding without launching Minecraft."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from combat_rl import record_episode as record
from combat_rl.config import load_config
from combat_rl.endless import EndlessCombatEnv
from combat_rl.train import save_checkpoint
from test_survival_task import TelemetryBackend,environment


class RGBBackend(TelemetryBackend):
    def __init__(self):
        super().__init__()
        self.initial_env = NS(isHudHidden=True)
        self.initial_env_message = NS(hudHidden=True)
        self.ipc = NS(initial_environment=self.initial_env_message)
    def step(self,action):
        result = super().step(action)
        result[0]['rgb'] = np.full((360,640,3),80,dtype=np.uint8)
        return result


class BehaviorRecordingTests(unittest.TestCase):
    def test_encoded_frames_and_trace_agree_with_episode_ticks(self):
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest('Optional video dependency is absent')
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        config = load_config()
        env = environment(config,'C',dict(mode='batch',batch_size=2,max_seconds=.4))
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = PPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,seed=0)
                save_checkpoint(model,root/'parent',dict(config=config,condition='C',seed=0,runtime={}))
                argv = ['record','--model',str(root/'parent.zip'),'--output-dir',str(root/'video'),
                        '--mode','batch','--batch-size','2','--max-seconds','.4','--scenario-seed','53100','--track-behavior']
                with patch('sys.argv',argv),patch.object(EndlessCombatEnv,'_make_backend',side_effect=RGBBackend), \
                     patch.object(record,'runtime_metadata',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                    record.main()
                meta = json.loads((root/'video/episode.json').read_text())
                self.assertEqual(meta['behavior_summary']['counts']['ticks'],8)
                self.assertEqual(meta['annotated_frame_size'],[640,536])
                with gzip.open(root/'video/behavior-trace.json.gz','rt') as stream:
                    trace = json.load(stream)
                self.assertEqual(len(trace['ticks']),8)
                self.assertEqual(trace['episode']['counts']['ticks'],8)
                for file,expected in [('episode.mp4',9),('episode-annotated.mp4',49)]:
                    frames,_ = imageio_ffmpeg.count_frames_and_secs(str(root/'video'/file))
                    self.assertEqual(frames,expected)
                self.assertFalse((root/'video/INCOMPLETE.txt').exists())
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
