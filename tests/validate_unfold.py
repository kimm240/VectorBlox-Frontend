"""Standalone future-CI regression suite; never called by the compile flow."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import onnx
from onnx import helper as h, numpy_helper, TensorProto as T
import onnxruntime as ort


def fixture(k,rows,cols):
    nodes=[];weights=[]
    def constant(values):
        name=f'c{len(weights)}';weights.append(numpy_helper.from_array(np.array(values,dtype=np.int64),name));return name
    def op(kind,inputs,**attrs):
        name=f'v{len(nodes)}';nodes.append(h.make_node(kind,inputs,[name],name=name,**attrs));return name
    value='x'
    for axis,count,perm in [(2,rows,[0,1,3,2]),(3,cols,[0,1,2,4,3])]:
        parts=[]
        for j in range(count):
            part=op('Slice',[value,constant([j*k]),constant([(j+1)*k]),constant([axis])])
            part=op('Transpose',[part],perm=perm)
            parts.append(op('Unsqueeze',[part,constant([axis])]))
        value=op('Concat',parts,axis=axis)
    value=op('Reshape',[value,constant([1,1,rows,cols,k*k])],allowzero=0)
    value=op('Transpose',[value],perm=[0,1,4,2,3])
    value=op('Reshape',[value,constant([1,-1,rows,cols])],allowzero=0)
    m=h.make_model(h.make_graph(nodes,'unfold',[h.make_tensor_value_info('x',T.FLOAT,[1,1,rows*k,cols*k])],
        [h.make_tensor_value_info(value,T.FLOAT,[1,k*k,rows,cols])],weights),opset_imports=[h.make_opsetid('',17)])
    m.ir_version=9
    return onnx.shape_inference.infer_shapes(m)


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path('build/vbx-rewrite'))
    p.add_argument('--out',type=Path,default=Path('artifacts/unfold-validation'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    def rewrite(m,name):
        src=a.out/(name+'.onnx');dst=a.out/(name+'.rewritten.onnx');onnx.save(m,src)
        stats=json.loads(subprocess.check_output([str(a.binary.resolve()),str(src),str(dst)],text=True))
        onnx.checker.check_model(str(dst));return src,dst,stats
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for k,rows,cols in [(2,2,3),(3,3,2),(8,2,2)]:
        m=fixture(k,rows,cols);src,dst,stats=rewrite(m,f'k{k}')
        assert stats['rewrites']==1
        x=np.random.default_rng(k).standard_normal((1,1,rows*k,cols*k)).astype(np.float32)
        def run(path):return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
        before,after=run(src),run(dst)
        assert np.isfinite(before).all() and np.isfinite(after).all()
        np.testing.assert_array_equal(before,after)
        stable=onnx.load(dst);_,again,s=rewrite(stable,f'k{k}-again');assert s['rewrites']==0
        assert stable.SerializeToString()==onnx.load(again).SerializeToString()
        records.append({'case':f'k{k}','exact_finite_output_match':True,'stats':stats})
    for case in ['reordered_branches','shared_intermediate','wrong_slice','large_block']:
        m=fixture(9 if case=='large_block' else 2,2,3)
        if case=='reordered_branches':
            n=next(n for n in m.graph.node if n.op_type=='Concat');n.input[0],n.input[1]=n.input[1],n.input[0]
        elif case=='shared_intermediate':
            name=m.graph.node[0].output[0];v=next(v for v in m.graph.value_info if v.name==name);m.graph.output.append(v)
        elif case=='wrong_slice':
            n=m.graph.node[0];t=next(t for t in m.graph.initializer if t.name==n.input[1]);t.CopyFrom(numpy_helper.from_array(np.array([1],np.int64),t.name))
        _,dst,stats=rewrite(m,case);assert stats['rewrites']==0
        assert m.SerializeToString()==onnx.load(dst).SerializeToString();records.append({'case':case,'unchanged':True})
    (a.out/'results.json').write_text(json.dumps(records,indent=2));print(f'{len(records)} standalone unfold checks passed')

if __name__=='__main__':main()
