#!/usr/bin/env python3
"""Small numerical/regression tests. No VectorBlox SDK or ML training stack required."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
import onnxruntime as ort

def model(axis=1,opset=17,channels=3,keepdims=1,dynamic=False):
    shape=[1,'C' if dynamic else channels,7,9]
    output=[1,1,7,9] if keepdims else [1,7,9]
    kwargs={'keepdims':keepdims};inputs=['x'];initializers=[]
    if opset<18:kwargs['axes']=[axis]
    else:
        inputs+=['axes'];initializers=[numpy_helper.from_array(np.array([axis],dtype=np.int64),name='axes')]
    graph=h.make_graph([h.make_node('ReduceMean',inputs,['y'],**kwargs)],'test',
       [h.make_tensor_value_info('x',T.FLOAT,shape)],[h.make_tensor_value_info('y',T.FLOAT,output)],initializers)
    result=h.make_model(graph,opset_imports=[h.make_opsetid('',opset)]);result.ir_version=9
    return result

def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path('build/vbx-rewrite'))
    p.add_argument('--out',type=Path,default=Path('artifacts/validation'))
    args=p.parse_args();binary=args.binary.resolve();args.out.mkdir(parents=True,exist_ok=True)
    results=[];rng=np.random.default_rng(12345)
    def rewrite(source,dest,extra=()):
        run=subprocess.run([str(binary),str(source),str(dest),*extra],capture_output=True,text=True,check=True)
        return json.loads(run.stdout)
    for opset in [11,17,18,21]:
        for channels in [1,3,17]:
            for axis in [1,-3]:
                name=f'mean-o{opset}-c{channels}-a{axis}';source=args.out/(name+'.onnx');dest=args.out/(name+'.rewritten.onnx')
                onnx.save(model(axis,opset,channels),source);stats=rewrite(source,dest)
                assert stats['rewrites']==1,stats
                onnx.checker.check_model(onnx.load(dest))
                original,changed=session(source),session(dest);error=0.
                for scale in [0.001,1.,100.]:
                    x=(rng.standard_normal((1,channels,7,9))*scale).astype(np.float32)
                    before=original.run(None,{'x':x})[0];after=changed.run(None,{'x':x})[0]
                    assert np.isfinite(before).all() and np.isfinite(after).all()
                    np.testing.assert_allclose(before,after,atol=3e-5,rtol=3e-5)
                    error=max(error,float(np.max(np.abs(before-after))))
                results.append({'case':name,'max_abs_error':error,'stats':stats})
    for name,m in [('dynamic',model(dynamic=True)),('keepdims0',model(keepdims=0))]:
        source=args.out/(name+'.onnx');dest=args.out/(name+'.rewritten.onnx');onnx.save(m,source)
        stats=rewrite(source,dest);assert stats['rewrites']==0
        assert onnx.load(source).SerializeToString()==onnx.load(dest).SerializeToString()
        results.append({'case':name,'stats':stats})
    # In-place output and external data relocation must fail.
    source=args.out/'failure.onnx';onnx.save(model(),source)
    assert subprocess.run([str(binary),str(source),str(source)],capture_output=True).returncode!=0
    external=model();external.graph.initializer.append(numpy_helper.from_array(np.ones((2,2),np.float32),'unused'))
    external_path=args.out/'external.onnx'
    onnx.save_model(external,external_path,save_as_external_data=True,all_tensors_to_one_file=True,location='weights.bin',size_threshold=0)
    relocated=args.out/'relocated';relocated.mkdir(exist_ok=True)
    assert subprocess.run([str(binary),str(external_path),str(relocated/'out.onnx')],capture_output=True).returncode!=0
    rewrite(external_path,args.out/'external.rewritten.onnx')
    onnx.checker.check_model(str(args.out/'external.rewritten.onnx'))
    # Mostly non-matching graph: demonstrate root dispatch, not a runtime-speed claim.
    big=model();big.graph.ClearField('node')
    for i in range(10000):big.graph.node.append(h.make_node('Identity',['x'],[f'id{i}']))
    big.graph.node.append(h.make_node('ReduceMean',['x'],['y'],axes=[1],keepdims=1))
    source=args.out/'dispatch-10001.onnx';dest=args.out/'dispatch-10001.rewritten.onnx';onnx.save(big,source)
    timings=[]
    for _ in range(5):
        stats=rewrite(source,dest);assert stats['attempts']==1 and stats['rewrites']==1
        timings.append(stats['timing_ms'])
    results.append({'case':'dispatch_10001_nodes','median_ms':{k:statistics.median(t[k] for t in timings) for k in timings[0]},'pattern_attempts':1})
    (args.out/'results.json').write_text(json.dumps(results,indent=2))
    print(f'{len(results)} cases passed; results: {args.out / "results.json"}')

if __name__=='__main__':main()
