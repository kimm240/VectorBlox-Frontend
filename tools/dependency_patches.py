"""Apply exact, hash-pinned text patches without requiring a system patch tool."""
import hashlib
import json
from pathlib import Path

def digest(data):
    return hashlib.sha256(data).hexdigest()

def apply_patches(root, dependency):
    root=Path(root).resolve()
    # LLVM archives are independently extracted beneath llvm-project.
    source=(root/'.deps/src'/dependency['directory']).resolve()
    plans=[]
    previous={}
    for entry in dependency.get('patches',[]):
        manifest_path=(root/entry['file']).resolve()
        if root not in manifest_path.parents:
            raise RuntimeError('Patch manifest outside repository')
        raw=manifest_path.read_bytes()
        if digest(raw)!=entry['sha256']:
            raise RuntimeError('Patch manifest SHA256 mismatch: '+str(manifest_path))
        patch=json.loads(raw)
        target=(source/patch['target']).resolve()
        if source not in target.parents:
            raise RuntimeError('Patch target outside dependency')
        if target in previous and patch['before_sha256']!=previous[target]:
            raise RuntimeError('Discontinuous dependency patch chain: '+str(target))
        previous[target]=patch['after_sha256']
        plans.append((entry,patch,target))
    for index,(entry,patch,target) in enumerate(plans):
        before=target.read_bytes()
        # A later patch for this exact file may already have been applied.
        # Recognize only a complete, hash-pinned downstream state; never guess
        # from matching snippets or skip validation of the manifests above.
        completed={p['after_sha256'] for _,p,t in plans[index:] if t==target}
        if digest(before) in completed:
            continue
        if digest(before)!=patch['before_sha256']:
            raise RuntimeError('Unexpected dependency source before patch: '+str(target))
        old,new=patch['old'].encode(),patch['new'].encode()
        if not old or before.count(old)!=1:
            raise RuntimeError('Patch must match exactly once: '+str(target))
        after=before.replace(old,new)
        if digest(after)!=patch['after_sha256']:
            raise RuntimeError('Patched dependency SHA256 mismatch: '+str(target))
        temporary=target.with_name(target.name+'.vbx-patch-tmp')
        temporary.write_bytes(after)
        temporary.replace(target)
        print('Applied dependency patch',entry['file'],flush=True)
