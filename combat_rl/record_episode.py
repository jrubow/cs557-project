"""Record one fresh validation episode from Minecraft's offscreen RGB frames.

Run with the train and video extras installed, and only when no other simulator
is using this runtime. RGB frames are diagnostic output, never policy input.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from combat_rl.env import CombatEnv, read_state
from combat_rl.task import NO_OP_ACTION, YAW_DEGREES, PITCH_DEGREES
from combat_rl.train import read_checkpoint, runtime_metadata


class FrameTap:
    """Observe completed raw ticks without adding actions or changing responses."""

    def __init__(self, backend):
        self.backend = backend
        self.callback = None
        self.latest = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def reset(self, **kwargs):
        result = self.backend.reset(**kwargs)
        self.latest = result[0]
        return result

    def step(self, action):
        result = self.backend.step(action)
        self.latest = result[0]
        if self.callback is not None:
            self.callback(result[0])
        return result


def action_label(action):
    move, strafe, yaw, pitch, jump, sprint, attack = map(int, action)
    return (f"Move {('still','forward','back')[move]} | Strafe {('still','left','right')[strafe]} | "
            f"Yaw {YAW_DEGREES[yaw]:+} | Pitch {PITCH_DEGREES[pitch]:+}",
            f"Attack {attack} | Jump {jump} | Sprint {sprint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario-seed', type=int)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; preserve it and choose a new directory')
    if args.width < 640 or args.height < 360 or args.width % 2 or args.height % 2:
        parser.error('Use even image dimensions of at least 640 x 360')

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from stable_baselines3 import PPO

    checkpoint, metadata = read_checkpoint(args.model)
    saved_config = metadata['config']
    config = copy.deepcopy(saved_config)
    config['environment'].update(image_width=args.width, image_height=args.height)
    seed = args.scenario_seed if args.scenario_seed is not None else config['evaluation']['validation_seed']
    if not config['evaluation']['validation_seed'] <= seed < config['evaluation']['test_seed']:
        parser.error('Choose a validation seed; held-out test scenarios are not for inspection')
    condition = metadata['condition']
    torch.set_num_threads(config['training']['torch_threads'])
    model = PPO.load(checkpoint, device='cpu')
    model.set_random_seed(seed)
    args.output_dir.mkdir(parents=True)
    font = ImageFont.load_default(size=16)
    title_font = ImageFont.load_default(size=19)
    fps = 20
    canvas_size = (args.width, args.height + 120)
    paths = [args.output_dir/'episode.mp4', args.output_dir/'episode-annotated.mp4']
    writers = []
    for path, size in zip(paths, [(args.width,args.height),canvas_size]):
        writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
            quality=8, macro_block_size=1, ffmpeg_timeout=30,
            output_params=['-preset','veryfast','-threads','1','-movflags','+faststart'])
        writer.send(None)
        writers.append(writer)
    env = CombatEnv(config, condition=condition, evaluation=True, port=args.port)
    frame_rows = []
    events = []
    frame_count = 0
    active_action = NO_OP_ACTION.copy()
    decision = 0
    last_canvas = None
    success = False
    started = time.perf_counter()
    try:
        env._raw = FrameTap(env._make_backend())
        observation, info = env.reset(seed=seed)
        baseline = env.state

        def capture(raw, initial=False):
            nonlocal frame_count, last_canvas
            rgb = raw['rgb']
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (args.height,args.width,3):
                raise RuntimeError('Expected a uint8 HWC RGB framebuffer from the pinned RAW encoder')
            rgb = np.ascontiguousarray(rgb)
            state = read_state(raw)
            tick = frame_count
            kills = max(0, state.kills-baseline.kills)
            row = dict(frame=frame_count,game_seconds=tick/fps,decision=decision,wave=env.wave,
                       health=state.health,kills=kills,x=state.x,z=state.z,yaw=state.yaw,pitch=state.pitch,
                       damage_dealt=max(0,state.damage_dealt-baseline.damage_dealt),
                       damage_taken=max(0,state.damage_taken-baseline.damage_taken))
            row.update(zip(['move','strafe','yaw_action','pitch_action','jump','sprint','attack'],
                           map(int, active_action)))
            frame_rows.append(row)
            writers[0].send(rgb)
            canvas = Image.new('RGB',canvas_size,(16,20,29))
            canvas.paste(Image.fromarray(rgb),(0,32))
            draw = ImageDraw.Draw(canvas)
            draw.text((12,7),f'PPO {condition} | seed {metadata["seed"]} | validation {seed} | 1x game time',
                      font=title_font,fill='white')
            y = args.height + 40
            draw.text((12,y),f'{tick/fps:6.2f}s | Wave {env.wave}/5 | HP {state.health:.1f}/20 | Kills {kills}',
                      font=title_font,fill=(138,225,176))
            line1,line2 = action_label(active_action)
            draw.text((12,y+27),line1,font=font,fill='white')
            draw.text((12,y+49),line2+' | Actions requested; attack is pulsed',
                      font=font,fill=(195,203,219))
            writers[1].send(np.asarray(canvas))
            last_canvas = canvas
            previous = frame_rows[-2] if len(frame_rows)>1 else row
            event = initial or kills>previous['kills'] or env.wave!=previous['wave'] or state.is_dead
            if event:
                path = args.output_dir/f'frame_{frame_count:05d}.png'
                canvas.save(path)
                events.append(dict(frame=frame_count,game_seconds=tick/fps,path=path.name,
                                   wave=env.wave,health=state.health,kills=kills))
            frame_count += 1

        capture(env._raw.latest,initial=True)
        env._raw.callback = capture
        episode_return = 0.0
        while True:
            active_action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(active_action)
            episode_return += float(reward)
            decision += 1
            if terminated or truncated:
                break
        env._raw.callback = None
        expected_frames = round(info['elapsed_seconds']*fps)+1
        if frame_count != expected_frames:
            raise RuntimeError(f'Frame/tick mismatch: {frame_count} vs {expected_frames}')
        if not np.isclose(frame_rows[-1]['kills'],info['kills']):
            raise RuntimeError('Recorded kill count differs from episode metrics')
        for key in ['damage_dealt','damage_taken']:
            if not np.isclose(frame_rows[-1][key],info[key]):
                raise RuntimeError(f'Recorded {key} differs from episode metrics')
        draw = ImageDraw.Draw(last_canvas)
        draw.rectangle((0,0,args.width,32),fill=(86,31,38))
        draw.text((12,7),f'END: {info["terminal_reason"]} | {info["waves_completed"]} waves | {info["kills"]} kills',
                  font=title_font,fill='white')
        last_canvas.save(args.output_dir/'final-frame.png')
        for _ in range(fps*2):
            writers[1].send(np.asarray(last_canvas))
        with (args.output_dir/'frames.csv').open('w',newline='') as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=list(frame_rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(frame_rows)
        report = dict(model=str(checkpoint.resolve()),model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            condition=condition,training_seed=metadata['seed'],training_steps=metadata['timesteps'],
            scenario_seed=seed,split='validation',deterministic=True,metrics=info,episode_return=episode_return,
            fps=fps,raw_frames=frame_count,annotated_frames=frame_count+fps*2,
            native_frame_size=[args.width,args.height],annotated_frame_size=list(canvas_size),
            display_overrides={'image_width':args.width,'image_height':args.height},
            model_config=saved_config,recording_config=config,runtime=runtime_metadata(),
            events=events,wall_seconds=time.perf_counter()-started,
            note='Fresh diagnostic rollout, not a replay of prior validation. No manual game input. '
                 'Only capture resolution differs; RGB/overlays never enter the structured policy input. '
                 'Raw video is 20 ticks/s; annotated video adds a 2-second end hold. '
                 'Minecraft AI is not bit-exact across runs with the same Python seed.')
        (args.output_dir/'episode.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'output_dir':str(args.output_dir),'metrics':info,'raw_frames':frame_count}),flush=True)
        success = True
    finally:
        try:
            for writer in writers:
                writer.close()
        except BaseException:
            success = False
            raise
        finally:
            try:
                env.close()
            finally:
                if not success:
                    (args.output_dir/'INCOMPLETE.txt').write_text('Recording did not finish. Inspect supervisor log before use.\n')


if __name__ == '__main__':
    main()
