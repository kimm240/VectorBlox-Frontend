#!/usr/bin/env python3
"""Hash-pinned local dependencies. Standard library only; no system installs."""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
from dependency_patches import apply_patches
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def prepare(dep,offline):
    cache=ROOT/'.deps/downloads';cache.mkdir(parents=True,exist_ok=True)
    path=cache/dep['archive']
    if not path.exists():
        if offline:raise RuntimeError(f'Missing cached archive in offline mode: {path}')
        print('Downloading',dep['name'],dep['version'],flush=True)
        temporary=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(dep['url'],timeout=60) as response,temporary.open('wb') as out:
            shutil.copyfileobj(response,out,1024*1024)
        if sha256(temporary)!=dep['sha256']:raise RuntimeError(f'SHA256 mismatch: {temporary}')
        temporary.replace(path)
    if sha256(path)!=dep['sha256']:raise RuntimeError(f'SHA256 mismatch: {path}')
    source=ROOT/'.deps/src';source.mkdir(parents=True,exist_ok=True)
    stamp=source/(dep['name']+'.verified')
    selection_version='core-sources-v2:'+dep['sha256']
    if stamp.exists() and stamp.read_text()==selection_version and (source/dep['directory']).is_dir():return
    print('Extracting',path.name,flush=True)
    with tarfile.open(path) as tar:
        selected=[]
        for original in tar.getmembers():
            parts=Path(original.name).parts[1:]
            # Keep upstream CMake scaffolding, headers and library sources; omit
            # tests/docs/examples and target-specific CPU/GPU backend sources.
            if dep['name'] in ('llvm','mlir'):
                if parts and parts[0] in ('test','unittests','docs','examples','benchmarks'):continue
                # LLVM TargetParser still generates metadata from .td files even
                # with no code-generation targets enabled. Keep data, not backends.
                if dep['name']=='llvm' and len(parts)>3 and parts[:2]==('lib','Target') and not original.name.endswith(('.td','.def')):continue
                if dep['name']=='llvm' and len(parts)==3 and parts[:2]==('lib','Target') and original.isdir():continue
            member=copy.copy(original)
            member.name=str(Path(dep['directory']).joinpath(*parts))
            if member.islnk():member.linkname=str(Path(dep['directory']).joinpath(*Path(member.linkname).parts[1:]))
            candidate=(source/member.name).resolve()
            if source.resolve() not in candidate.parents and candidate!=source.resolve():
                raise RuntimeError('Archive path outside source directory')
            if member.isdev():raise RuntimeError('Special file in source archive')
            if member.issym() or member.islnk():
                target=(candidate.parent/member.linkname if member.issym() else source/member.linkname).resolve()
                if source.resolve() not in target.parents:raise RuntimeError('Archive link outside source directory')
            selected.append(member)
        kwargs={'filter':'data'} if 'filter' in inspect.signature(tar.extractall).parameters else {}
        tar.extractall(source,members=selected,**kwargs)
    stamp.write_text(selection_version)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-only',action='store_true')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--test',action='store_true',help='Build and run separate native regression tests (development/CI only)')
    p.add_argument('--jobs',type=int,default=min(6,os.cpu_count() or 1))
    p.add_argument('--cmake',default='cmake')
    p.add_argument('--ninja',default='ninja')
    args=p.parse_args()
    if args.jobs<1:p.error('--jobs must be positive')
    lock=json.loads((ROOT/'third_party/dependencies.lock.json').read_text())
    for dep in lock['dependencies']:
        prepare(dep,args.offline)
        apply_patches(ROOT,dep)
    if args.fetch_only:return
    ninja=shutil.which(args.ninja)
    if not ninja:raise RuntimeError('Ninja is required')
    subprocess.run([args.cmake,'-S',str(ROOT),'-B',str(ROOT/'build'),'-G','Ninja',
                    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_MAKE_PROGRAM='+ninja,
                    '-DVBX_BUILD_TESTS='+('ON' if args.test else 'OFF')],check=True)
    targets=['vbx-rewrite']+(['vbx-rewrite-tests'] if args.test else [])
    subprocess.run([args.cmake,'--build',str(ROOT/'build'),'--target',*targets,'-j',str(args.jobs)],check=True)
    if args.test:
        subprocess.run(['ctest','--test-dir',str(ROOT/'build'),'-R','^vbx-rewrite-core$','--output-on-failure'],check=True)

if __name__=='__main__':main()
