"""Check the portable meeting packet without starting Minecraft."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',required=True,type=Path)
    p.add_argument('--write-manifest',action='store_true',help='Authoring only: create the hash list after final edits')
    args=p.parse_args();root=args.packet.resolve();manifest=root/'SHA256SUMS.txt'
    files=sorted(x for x in root.rglob('*') if x.is_file() and x!=manifest)
    if args.write_manifest:
        manifest.write_text(''.join(sha(x)+'  '+x.relative_to(root).as_posix()+'\n' for x in files),encoding='utf-8')
    expected={}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);path=(root/name).resolve()
        if not path.is_relative_to(root) or name in expected:
            raise RuntimeError('Invalid manifest entry')
        expected[name]=digest
        if not path.is_file() or sha(path)!=digest:
            raise RuntimeError('Missing or changed artifact: '+name)
    if set(expected)!={x.relative_to(root).as_posix() for x in files}:
        raise RuntimeError('Manifest does not cover the complete packet')
    for item in json.loads((root/'asset-provenance.json').read_text())['copied_files']:
        if expected[item['path']]!=item['sha256']:
            raise RuntimeError('Source-copy mismatch: '+item['path'])
    for item in json.loads((root/'checkpoint-index.json').read_text())['models']:
        if expected[item['path']]!=item['sha256']:
            raise RuntimeError('Checkpoint identity mismatch')
    print(json.dumps(dict(verified_files=len(expected),bytes=sum(x.stat().st_size for x in files),
                         models=len(json.loads((root/'checkpoint-index.json').read_text())['models']))))


if __name__=='__main__':
    main()
