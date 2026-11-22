"""Record a fresh five-wave episode or a separate endless demonstration.

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
        self.last_action = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def reset(self, **kwargs):
        result = self.backend.reset(**kwargs)
        self.latest = result[0]
        return result

    def step(self, action):
        self.last_action = dict(action)
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
    parser.add_argument('--mode', choices=['waves', 'endless', 'batch'], default='waves')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--track-crits', action='store_true', help='Add passive critical-sound cue telemetry')
    parser.add_argument('--track-behavior', action='store_true', help='Add passive aiming/movement traces and annotations')
    parser.add_argument('--spawn-seconds', type=float, default=2.0,
                        help='Endless mode only: simulated seconds between zombie spawns')
    parser.add_argument('--max-seconds', type=float, default=600,
                        help='Endless mode only: recording limit, at most 600 simulated seconds')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--hide-game-hud', action='store_true',
                        help='Keep the training display setting instead of showing the sword/crosshair/HUD')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    args.track_crits = args.track_crits or args.track_behavior
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
    canvas_size = (args.width, args.height + (176 if args.track_behavior else 146 if args.track_crits else 120))
    paths = [args.output_dir/'episode.mp4', args.output_dir/'episode-annotated.mp4']
    writers = []
    for path, size in zip(paths, [(args.width,args.height),canvas_size]):
        writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
            quality=8, macro_block_size=1, ffmpeg_timeout=30,
            output_params=['-preset','veryfast','-threads','1','-movflags','+faststart'])
        writer.send(None)
        writers.append(writer)
    if args.mode in ('endless', 'batch'):
        from combat_rl.endless import EndlessCombatEnv
        env = EndlessCombatEnv(config, condition=condition, port=args.port,
                               spawn_seconds=args.spawn_seconds, max_seconds=args.max_seconds,
                               mode='batch' if args.mode == 'batch' else 'timed', batch_size=args.batch_size)
    else:
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
        backend = env._make_backend()
        # This message has not been sent yet. Only the diagnostic display changes;
        # policy observations/actions and task settings remain the saved ones.
        backend.initial_env.isHudHidden = args.hide_game_hud
        backend.initial_env_message.hudHidden = args.hide_game_hud
        assert backend.ipc.initial_environment is backend.initial_env_message
        env._raw = FrameTap(backend)
        observation, info = env.reset(seed=seed)
        baseline = env.state
        from combat_rl.telemetry import CombatTelemetry
        tracker = CombatTelemetry() if args.track_crits else None
        if tracker is not None:
            tracker.begin(env._raw.latest['full'])
        from combat_rl.behavior import BehaviorTracker, aggregate as behavior_aggregate
        behavior_rows = []
        behavior = BehaviorTracker(config,condition,behavior_rows.append) if args.track_behavior else None
        if behavior is not None:
            behavior.begin(env._raw.latest['full'])

        def capture(raw, initial=False):
            nonlocal frame_count, last_canvas
            rgb = raw['rgb']
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (args.height,args.width,3):
                raise RuntimeError('Expected a uint8 HWC RGB framebuffer from the pinned RAW encoder')
            rgb = np.ascontiguousarray(rgb)
            state = read_state(raw)
            critical_cue = 0
            if tracker is not None and not initial:
                critical_cue = tracker.observe(raw['full'], env._raw.last_action)
            if behavior is not None and not initial:
                behavior.observe(raw['full'],env._raw.last_action,critical_cue)
            tick = frame_count
            kills = max(0, state.kills-baseline.kills)
            row = dict(frame=frame_count,game_seconds=tick/fps,decision=decision,wave=env.wave,
                       health=state.health,kills=kills,x=state.x,z=state.z,yaw=state.yaw,pitch=state.pitch,
                       damage_dealt=max(0,state.damage_dealt-baseline.damage_dealt),
                       damage_taken=max(0,state.damage_taken-baseline.damage_taken))
            if args.mode in ('endless', 'batch'):
                row.update(active_enemies=len(state.entities), spawn_commands_issued=env.spawn_commands_issued)
            if args.mode == 'batch':
                # Callback runs immediately before the task consumes this tick.
                clear_now = (not state.is_dead and state.health > 0 and not state.entities
                             and state.kills-env._batch_kills_start >= args.batch_size)
                row['batches_completed'] = env.batches_completed+int(clear_now)
            if tracker is not None:
                row['critical_sound_events'] = tracker.counts['critical_sound_events']
            row.update(zip(['move','strafe','yaw_action','pitch_action','jump','sprint','attack'],
                           map(int, active_action)))
            frame_rows.append(row)
            writers[0].send(rgb)
            canvas = Image.new('RGB',canvas_size,(16,20,29))
            canvas.paste(Image.fromarray(rgb),(0,32))
            draw = ImageDraw.Draw(canvas)
            title = (f'ENDLESS DEMO | spawn every {args.spawn_seconds:g}s | scenario {seed}'
                     if args.mode == 'endless' else
                     f'PPO {condition} | seed {metadata["seed"]} | validation {seed} | 1x game time')
            if args.mode == 'batch':
                title = f'BATCH SURVIVAL | k={args.batch_size} | scenario {seed} | 1x game time'
            draw.text((12,7),title,
                      font=title_font,fill='white')
            y = args.height + 40
            metrics_label = (f'{tick/fps:6.2f}s | HP {state.health:.1f}/20 | Kills {kills} | Alive {len(state.entities)}'
                             if args.mode in ('endless', 'batch') else
                             f'{tick/fps:6.2f}s | Wave {env.wave}/5 | HP {state.health:.1f}/20 | Kills {kills}')
            draw.text((12,y),metrics_label,
                      font=title_font,fill=(138,225,176))
            line1,line2 = action_label(active_action)
            draw.text((12,y+27),line1,font=font,fill='white')
            draw.text((12,y+49),line2+' | Actions requested; attack is pulsed',
                      font=font,fill=(195,203,219))
            if tracker is not None:
                suffix = f' | Batches cleared {row["batches_completed"]}' if args.mode == 'batch' else ''
                draw.text((12,y+73),f'Critical sound cues {tracker.counts["critical_sound_events"]}'+suffix,
                          font=font,fill=(255,210,110))
            if behavior is not None:
                retreat_hits = sum(w['group']=='retreat' and not w['partial'] and w['damage_dealt']>0
                                   and w['damage_taken']==0 for w in behavior.windows)
                turns = behavior.counts['yaw_turns']
                corrected = f'{100*behavior.counts["yaw_improving_turns"]/turns:.0f}%' if turns else 'n/a'
                draw.text((12,y+97),f'Aim turns improved {corrected} | Unanswered retreat windows {retreat_hits}',
                          font=font,fill=(171,204,255))
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
        end_label = (f'ENDLESS: {info["terminal_reason"]} | {info["kills"]} kills'
                     if args.mode in ('endless', 'batch') else
                     f'END: {info["terminal_reason"]} | {info["waves_completed"]} waves | {info["kills"]} kills')
        if args.mode == 'batch':
            end_label = f'BATCH k={args.batch_size}: {info["terminal_reason"]} | {info["kills"]} kills'
        draw.text((12,7),end_label,
                  font=title_font,fill='white')
        last_canvas.save(args.output_dir/'final-frame.png')
        for _ in range(fps*2):
            writers[1].send(np.asarray(last_canvas))
        with (args.output_dir/'frames.csv').open('w',newline='') as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=list(frame_rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(frame_rows)
        report = dict(model=str(checkpoint.resolve()),model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            mode=args.mode,policy_updated=False,
            condition=condition,training_seed=metadata['seed'],training_steps=metadata['timesteps'],
            scenario_seed=seed,split='validation',deterministic=True,metrics=info,episode_return=episode_return,
            fps=fps,raw_frames=frame_count,annotated_frames=frame_count+fps*2,
            native_frame_size=[args.width,args.height],annotated_frame_size=list(canvas_size),
            display_overrides={'image_width':args.width,'image_height':args.height,
                               'hud_hidden':args.hide_game_hud},
            model_config=saved_config,recording_config=config,runtime=runtime_metadata(),
            events=events,wall_seconds=time.perf_counter()-started,
            note='Fresh diagnostic rollout, not a replay of prior validation. No manual game input. '
                 'Only capture resolution/HUD differ; RGB/overlays never enter the structured policy input. '
                 'Raw video is 20 ticks/s; annotated video adds a 2-second end hold. '
                 'Minecraft AI is not bit-exact across runs with the same Python seed.')
        if args.mode == 'endless':
            report['note'] = ('Exploratory continuous-spawn demonstration, not a five-wave evaluation. '
                'One initial zombie, then one every fixed interval; standard arena/gear, no healing. '
                'The wave feature is fixed at 5; nearest-five/FOV observations and actions are unchanged. '
                'Recording ends on death or its time limit. No policy updates. '
                'Spawn count counts issued summon commands; raw ticks and kills are recorded. '
                'RGB/HUD are display-only; raw video is 20 ticks/s, annotation adds a two-second end hold.')
        if args.mode == 'batch':
            report['note'] = ('Exploratory cleared-batch survival, not a five-wave evaluation. '
                f'Spawn {args.batch_size} zombies initially and after every full batch clear; '
                'standard arena/gear, no between-batch healing. The wave feature is fixed at k. '
                'No policy updates or manual input. RGB/HUD/telemetry never enter policy input. '
                'Raw video is 20 ticks/s; annotation adds a two-second end hold.')
        if tracker is not None:
            report['combat_telemetry'] = tracker.summary()
            (args.output_dir/'critical-events.json').write_text(json.dumps(tracker.events)+'\n')
        if behavior is not None:
            import gzip
            details = behavior.finish()
            report['behavior_summary'] = behavior_aggregate([details])
            report['behavior_note'] = 'Geometric aiming and one-second retreat-window proxies; not proof of intent or exact hit reach.'
            with gzip.open(args.output_dir/'behavior-trace.json.gz','wt',encoding='utf-8') as stream:
                json.dump(dict(episode=details,ticks=behavior_rows),stream)
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
