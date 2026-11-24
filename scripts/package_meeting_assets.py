"""Publish portable evidence, checkpoints and full-length viewing copies for the meeting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def build(args):
    root=args.outputs.resolve(); packet=args.packet.resolve()
    packet.mkdir(parents=True,exist_ok=True)
    # Preserve exported bytes and their hashes on Windows and Linux alike.
    (packet/'.gitattributes').write_text(
        '* -text whitespace=cr-at-eol\n*.pdf binary\n*.zip binary\n*.gz binary\n*.mp4 binary\n*.jpg binary\n*.png binary\n',
        encoding='utf-8',newline='\n')
    copied=[]
    def copy(src,relative):
        src=Path(src);dst=packet/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        source_hash=digest(src)
        if digest(dst)!=source_hash:
            raise RuntimeError('Copy changed evidence bytes')
        copied.append(dict(path=Path(relative).as_posix(),source=src.relative_to(root).as_posix(),
                           bytes=dst.stat().st_size,sha256=source_hash))
        return dst

    old=read(root/'cs557-test/manifest.json')
    final=read(root/'cs557-reserved-test-20261122/manifest.json')
    behavior=read(root/'cs557-behavior-resumed-20261122/results.json')
    model_index=[];models_by_hash={}
    def model(name,src,expected,role):
        src=Path(src)
        if digest(src)!=expected:
            raise RuntimeError('Checkpoint identity changed: '+name)
        if expected in models_by_hash:
            entry=models_by_hash[expected]
            entry['aliases'].append(name);entry['roles'].append(role)
            return
        zip_path=copy(src,Path('checkpoints')/name/'final.zip')
        copy(src.with_suffix('.json'),Path('checkpoints')/name/'final.json')
        meta=read(src.with_suffix('.json'))
        entry=dict(name=name,aliases=[],roles=[role],path=zip_path.relative_to(packet).as_posix(),
            sha256=expected,condition=meta['condition'],training_seed=meta['seed'],
            actual_training_decisions=meta['timesteps'],metadata_sha256=digest(src.with_suffix('.json')))
        model_index.append(entry);models_by_hash[expected]=entry
    # Prefer the final-report names, then add original baselines and behavior checkpoints.
    for item in final['jobs']:
        model(item['name'],item['model'],item['model_sha256'],item['group'])
    for item in old['jobs']:
        if item['policy']=='ppo':
            model(item['name'],item['model'],item['model_sha256'],item['group'])
    for item in behavior['history']:
        model(item['name'],item['model'],item['model_sha256'],'behavior learning history')
    write(packet/'checkpoint-index.json',dict(models=model_index,
        five_wave_demo='checkpoints/survival10_4M_seed0/final.zip',
        exploratory_batch_demo='checkpoints/batch_plus1M_seed0/final.zip',
        note='Paths are relative to this meeting directory. Keep each final.zip beside final.json. Original metadata is unchanged; recorded local source paths are provenance.'))

    summary_files={
        'cs557-live':['manifest.json'],
        'cs557-optimization':['comparison-summary.json','confirmation.json','screening.json','final-selection.json','diagnostics.json','manifest.json'],
        'cs557-survival':['manifest.json','selection.json','status.json','reserved-test-lock.json','recovery-20261122.json'],
        'cs557-endless-calibration-recovery':['calibration.json','selection.json','critical-controls.json','critical-history.json','manifest.json','status.json'],
        'cs557-batch-finish-20261122b':['results.json','selection.json','manifest.json','status.json'],
        'cs557-behavior-resumed-20261122':['results.json','manifest.json','status.json'],
        'cs557-test':['manifest.json','results.json','status.json'],
        'cs557-reserved-test-20261122':['manifest.json','results.json','status.json'],
    }
    for folder,names in summary_files.items():
        for name in names:
            copy(root/folder/name,Path('evidence')/folder/name)
        # All episode CSVs and their evaluation metadata, but no giant event dumps.
        for src in sorted((root/folder/'evaluations').glob('*.csv')):
            copy(src,Path('evidence')/folder/'evaluations'/src.name)
            if src.with_suffix('.json').exists():
                copy(src.with_suffix('.json'),Path('evidence')/folder/'evaluations'/src.with_suffix('.json').name)
    for src in sorted((root/'cs557-live').glob('*_seed*/validation.csv')):
        copy(src,Path('evidence/cs557-live')/src.parent.name/src.name)
        if src.with_suffix('.json').exists():
            copy(src.with_suffix('.json'),Path('evidence/cs557-live')/src.parent.name/src.with_suffix('.json').name)
    # Learning logs from all trained candidates are useful for later report plots.
    for folder in ('cs557-live','cs557-optimization','cs557-survival','cs557-endless-calibration-recovery'):
        for src in sorted((root/folder).rglob('*.csv')):
            if src.name=='progress.csv' or src.name.endswith('.monitor.csv'):
                copy(src,Path('evidence/training-logs')/src.relative_to(root))
    for item in behavior['history']+behavior['references']+behavior['batch_and_transfer']:
        source=Path(item['output'])
        for name in ('episodes.json','metadata.json','ticks.csv.gz'):
            copy(source/name,Path('evidence/behavior')/item['name']/name)

    video_specs=[
        ('five-wave-clear','cs557-survival/videos/selected-51000','Five-wave clear: 15 zombies','Selected successful showcase from fixed development recordings.'),
        ('batch-42-kills','cs557-batch-finish-20261122b/videos/selected-batch-52002','Batch survival: 42 zombies','Strongest of three fixed selected-policy batch recordings (12, 24 and 42 kills).'),
        ('batch-24-kills','cs557-batch-finish-20261122b/videos/selected-batch-52000','Batch survival: 24 zombies','Another fixed development recording; eight batches cleared.'),
        ('timed-spawn-comparison','cs557-batch-finish-20261122b/videos/selected-timed2-52000','Timed arrivals: 5 zombies defeated','Optional difficulty comparison; one new zombie every two seconds. Different task from batch mode.'),
    ]
    clips=[]
    for name,folder,title,selection in video_specs:
        source=root/folder;meta=read(source/'episode.json');dst=packet/'media'/(name+'.mp4')
        copy(source/'episode.json',Path('media')/(name+'.episode.json'))
        copy(source/'frames.csv',Path('media')/(name+'.frames.csv'))
        source_sha=digest(source/'episode-annotated.mp4')
        receipt=dst.with_suffix('.export.json')
        if not (dst.exists() and receipt.exists() and read(receipt)['source_sha256']==source_sha
                and read(receipt)['sha256']==digest(dst)):
            command=[str(args.ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-n',
                '-i',str(source/'episode-annotated.mp4'),'-map','0:v:0','-an',
                '-c:v','libx264','-preset','medium','-crf','23','-pix_fmt','yuv420p',
                '-movflags','+faststart','-threads','4',str(dst)]
            print('Encoding '+name,flush=True)
            subprocess.run(command,check=True)
            write(receipt,dict(source_sha256=source_sha,sha256=digest(dst),
                codec='H.264',crf=23,full_episode=True,speed_changed=False,dimensions_changed=False,
                source_frames=meta['annotated_frames'],fps=meta['fps'],
                note='Only re-encoded for smaller download. Same frames, size and 20 fps; existing two-second end hold retained. Source has no audio.'))
        poster=packet/'media'/(name+'.jpg')
        if not poster.exists():
            subprocess.run([str(args.ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-n',
                '-sseof','-1','-i',str(dst),'-frames:v','1','-q:v','2',str(poster)],check=True)
        clips.append(dict(name=name,title=title,path=dst.relative_to(packet).as_posix(),
            poster=poster.relative_to(packet).as_posix(),bytes=dst.stat().st_size,sha256=digest(dst),
            source=folder+'/episode-annotated.mp4',source_sha256=source_sha,selection=selection,
            metrics=meta['metrics'],scenario_seed=meta['scenario_seed'],model_sha256=meta['model_sha256'],
            video_seconds=meta['annotated_frames']/meta['fps'],game_seconds=meta['metrics']['elapsed_seconds']))
        print(name+': '+str(round(dst.stat().st_size/1024/1024,2))+' MiB',flush=True)
    write(packet/'media-index.json',dict(clips=clips,
        note='These are full development demonstrations, not reserved-test episodes or typical-performance estimates. No manual play or policy updates.'))
    write(packet/'asset-provenance.json',dict(copied_files=copied,models=len(model_index),
        behavior_episodes=170,reserved_test_episodes=350,original_test_episodes=800,
        omitted='Intermediate checkpoint duplicates, full PNG frame sequences, runtime caches and raw duplicate videos remain local.'))
    print(json.dumps(dict(models=len(model_index),copied_files=len(copied),clips=len(clips),packet=str(packet))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',type=Path,required=True)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    build(parser.parse_args())
