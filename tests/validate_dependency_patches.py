"""Standalone deterministic patching guards; no dependency download or build."""
import copy,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from dependency_patches import apply_patches,digest

for mode in ['valid','edited_source','bad_manifest_hash','bad_result_hash','outside_target']:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'.deps/src/example';source.mkdir(parents=True)
        target=source/'source.h';target.write_bytes(b'old text\n')
        patch=dict(target='source.h',before_sha256=digest(target.read_bytes()),after_sha256=digest(b'new text\n'),old='old',new='new')
        if mode=='edited_source':target.write_bytes(b'local changes\n')
        if mode=='bad_result_hash':patch['after_sha256']='0'*64
        if mode=='outside_target':patch['target']='../outside.h'
        manifest=root/'patch.json';manifest.write_text(json.dumps(patch));dep=dict(directory='example',patches=[dict(file='patch.json',sha256=digest(manifest.read_bytes()))])
        if mode=='bad_manifest_hash':dep['patches'][0]['sha256']='0'*64
        before=target.read_bytes();failed=False
        try:apply_patches(root,dep)
        except RuntimeError:failed=True
        assert failed==(mode!='valid'),mode
        if mode=='valid':
            assert target.read_bytes()==b'new text\n'
            stamp=target.stat().st_mtime_ns;apply_patches(root,dep);assert target.stat().st_mtime_ns==stamp
        else:assert target.read_bytes()==before,mode
print('Dependency patch application/idempotence and four rejection guards passed')

for state in [0,1,2]:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'.deps/src/example';source.mkdir(parents=True)
        target=source/'source.h';texts=[b'initial\n',b'intermediate\n',b'final\n'];target.write_bytes(texts[state]);entries=[]
        for i in range(2):
            manifest=root/f'patch{i}.json'
            patch=dict(target='source.h',before_sha256=digest(texts[i]),after_sha256=digest(texts[i+1]),old=texts[i].decode(),new=texts[i+1].decode())
            manifest.write_text(json.dumps(patch));entries.append(dict(file=manifest.name,sha256=digest(manifest.read_bytes())))
        dep=dict(directory='example',patches=entries)
        apply_patches(root,dep);assert target.read_bytes()==texts[-1]
        stamp=target.stat().st_mtime_ns;apply_patches(root,dep);assert target.stat().st_mtime_ns==stamp
        # Reject a broken chain before changing any source file.
        target.write_bytes(texts[0]);patch['before_sha256']='0'*64
        manifest.write_text(json.dumps(patch));entries[-1]['sha256']=digest(manifest.read_bytes())
        try:apply_patches(root,dep)
        except RuntimeError:pass
        else:raise AssertionError('Discontinuous chain accepted')
        assert target.read_bytes()==texts[0]
print('Fresh, partial and complete patch chains plus discontinuity guards passed')
