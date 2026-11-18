#!/usr/bin/env python3
"""Standalone before/after ONNX numerical correctness check (never compiles).

Inputs/outputs are matched by name, with no layout search or output permutation.
Floating outputs satisfy abs(after-before) <= atol + rtol*abs(before).
Non-floating outputs require exact equality; all nonfinite outputs fail.
"""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort

DTYPES={'tensor(float)':np.float32,'tensor(double)':np.float64,
        'tensor(float16)':np.float16,'tensor(int64)':np.int64,
        'tensor(int32)':np.int32,'tensor(int16)':np.int16,
        'tensor(int8)':np.int8,'tensor(uint8)':np.uint8,'tensor(bool)':np.bool_}


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def session(path,threads):
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    # Compare exported graphs without ORT graph fusions hiding a rewrite error.
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level=3
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def compare(reference,actual,atol,rtol):
    result={'shape':list(reference.shape),'dtype':str(reference.dtype)}
    if reference.shape!=actual.shape or reference.dtype!=actual.dtype:
        return {**result,'passed':False,'reason':'output shape/dtype mismatch',
                'actual_shape':list(actual.shape),'actual_dtype':str(actual.dtype)}
    if not np.issubdtype(reference.dtype,np.floating):
        mismatches=int(np.count_nonzero(reference!=actual))
        return {**result,'passed':mismatches==0,'mismatch_count':mismatches,'comparison':'exact'}
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(actual)):
        return {**result,'passed':False,'reason':'nonfinite output',
                'before_nonfinite':int(np.count_nonzero(~np.isfinite(reference))),
                'after_nonfinite':int(np.count_nonzero(~np.isfinite(actual)))}
    ref=reference.astype(np.float64);got=actual.astype(np.float64)
    diff=np.abs(got-ref);limit=atol+rtol*np.abs(ref)
    mismatch=diff>limit
    ratio=np.divide(diff,limit,out=np.zeros_like(diff),where=limit!=0)
    # With exact tolerances a nonzero difference has no finite error ratio.
    ratio_max=None if np.any((limit==0)&(diff!=0)) else float(ratio.max(initial=0))
    return {**result,'passed':not bool(np.any(mismatch)),
            'elements':int(diff.size),'mismatch_count':int(np.count_nonzero(mismatch)),
            'max_abs_error':float(diff.max(initial=0)),
            'mean_abs_error':float(diff.mean()) if diff.size else 0.,
            'max_relative_error_floor_1e_12':float((diff/np.maximum(np.abs(ref),1e-12)).max(initial=0)),
            'max_tolerance_ratio':ratio_max}


def run(args,report):
    if args.progress:print('Loading reference session',flush=True)
    before=session(args.before,args.threads)
    if args.progress:print('Loading rewritten session',flush=True)
    after=session(args.after,args.threads)
    if args.progress:print('Sessions ready',flush=True)
    inputs={v.name:v for v in before.get_inputs()};other={v.name:v for v in after.get_inputs()}
    if inputs.keys()!=other.keys():raise ValueError('Input name sets differ')
    outputs={v.name:v for v in before.get_outputs()};changed={v.name:v for v in after.get_outputs()}
    if outputs.keys()!=changed.keys():raise ValueError('Output name sets differ')
    for name in inputs:
        if inputs[name].type!=other[name].type:raise ValueError(f'Input dtype differs: {name}')
    names=list(outputs)
    def cases():
        if args.inputs:
            for path in args.inputs:
                with np.load(path,allow_pickle=False) as data:
                    yield str(path),{name:data[name] for name in data.files}
        else:
            for v in inputs.values():
                if v.type not in ('tensor(float)','tensor(double)','tensor(float16)'):
                    raise ValueError(f'Non-floating input {v.name}: provide domain-valid --inputs NPZ files')
                if any(not isinstance(d,int) or d<1 for d in v.shape):
                    raise ValueError(f'Dynamic/empty shape {v.name}: provide --inputs NPZ files')
            rng=np.random.default_rng(args.seed)
            for i in range(args.samples):
                feed={}
                for v in inputs.values():
                    values=rng.uniform(args.low,args.high,size=v.shape) if args.distribution=='uniform' else rng.standard_normal(v.shape)
                    feed[v.name]=values.astype(DTYPES[v.type])
                yield f'{args.distribution}:{i}',feed
            if args.edge_cases:
                for value in (0,1):
                    yield f'constant:{value}',{v.name:np.full(v.shape,value,dtype=DTYPES[v.type]) for v in inputs.values()}
    for label,feed in cases():
        if args.progress:print('Comparing',label,flush=True)
        if feed.keys()!=inputs.keys():raise ValueError(f'{label}: NPZ keys must exactly match model input names')
        for name,array in feed.items():
            for schema in (inputs[name],other[name]):
                if schema.type not in DTYPES or array.dtype!=np.dtype(DTYPES[schema.type]):
                    raise ValueError(f'{label}: incorrect/unsupported dtype for {name}')
                if array.ndim!=len(schema.shape) or any(isinstance(d,int) and d!=a for d,a in zip(schema.shape,array.shape)):
                    raise ValueError(f'{label}: shape does not match input {name}')
            if np.issubdtype(array.dtype,np.floating) and not np.all(np.isfinite(array)):
                raise ValueError(f'{label}: nonfinite input')
        ref=before.run(names,feed);got=after.run(names,feed)
        checks={n:compare(a,b,args.atol,args.rtol) for n,a,b in zip(names,ref,got)}
        report['cases'].append({'input':label,'input_sha256':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in feed.items()},
                                'outputs':checks,'passed':all(c['passed'] for c in checks.values())})
    report['passed']=bool(report['cases']) and all(c['passed'] for c in report['cases'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--report',type=Path,default=Path('artifacts/numerical-correctness.json'))
    p.add_argument('--inputs',type=Path,nargs='+',help='NPZ files: input name -> array in ONNX layout/dtype; replaces generated cases')
    p.add_argument('--samples',type=int,default=10);p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--distribution',choices=['uniform','normal'],default='uniform')
    p.add_argument('--low',type=float,default=0.);p.add_argument('--high',type=float,default=1.)
    p.add_argument('--edge-cases',action='store_true',help='Add constant zero/one inputs; use only where model domain permits')
    p.add_argument('--atol',type=float,default=1e-4);p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--progress',action='store_true')
    p.add_argument('--traceback-after',type=int,default=0,help='Optional repeated diagnostic stack interval in seconds')
    args=p.parse_args()
    if args.traceback_after<0:p.error('Nonnegative traceback interval required')
    if args.traceback_after:faulthandler.dump_traceback_later(args.traceback_after,repeat=True)
    if args.samples<1 or args.threads<1 or not all(np.isfinite(v) for v in (args.atol,args.rtol,args.low,args.high)) or min(args.atol,args.rtol)<0 or args.low>=args.high:
        p.error('Invalid sample/thread count, tolerances or input bounds')
    if args.report.resolve() in {x.resolve() for x in [args.before,args.after,*(args.inputs or [])]}:
        p.error('Report must not overwrite a model or input file')
    report={'passed':False,'before':str(args.before),'after':str(args.after),
            'runtime':f'ONNX Runtime {ort.__version__} CPUExecutionProvider, graph optimizations disabled',
            'criterion':'abs(after-before) <= atol + rtol*abs(before); non-floating exact; nonfinite fails',
            'atol':args.atol,'rtol':args.rtol,'seed':args.seed,'distribution':args.distribution,
            'input_range':[args.low,args.high] if args.distribution=='uniform' else None,
            'input_mode':'NPZ' if args.inputs else 'synthetic',
            'scope':'Sampled ONNX numerical equivalence; not a proof, INT8 accuracy or task accuracy.', 'cases':[]}
    exit_code=1
    try:
        report['model_file_sha256']={'before':file_hash(args.before),'after':file_hash(args.after)}
        run(args,report);exit_code=0 if report['passed'] else 1
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';exit_code=2
    if args.traceback_after:faulthandler.cancel_dump_traceback_later()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{"PASS" if report["passed"] else "FAIL"}: {len(report["cases"])} input cases; report: {args.report}')
    if 'error' in report:print(report['error'],file=sys.stderr)
    return exit_code

if __name__=='__main__':raise SystemExit(main())
