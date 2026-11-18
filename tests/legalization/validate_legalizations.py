"""Standalone future-CI checks for guarded backend legalization."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T


def model(nodes,ishape,oshape,weights):
    m=h.make_model(h.make_graph(nodes,'rule-check',[h.make_tensor_value_info('x',T.FLOAT,ishape)],
        [h.make_tensor_value_info('y',T.FLOAT,oshape)],weights),opset_imports=[h.make_opsetid('',17)])
    m.ir_version=9;return onnx.shape_inference.infer_shapes(m)


def cases():
    rng=np.random.default_rng(11)
    for label,dims,coefficient,ranked in [('query',[1,6,257,64],0.125,False),('negative',[1,3,129,128],-0.75,False),('small',[1,6,2,4],0.125,False),('identity',[1,6,257,64],1.,False),('ranked',[1,6,257,64],0.125,True)]:
        weights=[nh.from_array(np.array(coefficient,np.float32),'scale')]
        nodes=[h.make_node('Mul',['x','scale'],['z' if ranked else 'y'])];oshape=dims
        if ranked:
            weights.append(nh.from_array(np.array([2],np.int64),'k'));nodes.append(h.make_node('TopK',['z','k'],['y','indices'],axis=-1));oshape=dims[:-1]+[2]
        yield 'scalar-depthwise-'+label,model(nodes,dims,oshape,weights),int(label in ['query','negative'])
    for label,dims,half,shared,ranked in [
        ('full',[1,257,1536],0.5,False,False),
        ('rank4-stem',[1,40,128,128],0.5,False,False),
        ('rank4-stage2',[1,160,64,64],0.5,False,False),
        ('rank4-stage3',[1,320,32,32],0.5,False,False),
        ('rank4-stage4',[1,640,16,16],0.5,False,False),
        ('rank4-stage5',[1,1280,8,8],0.5,False,False),
        ('rank4-small',[1,3,8,8],0.5,False,False),
        ('rank4-topk',[1,40,128,128],0.5,False,True),
        ('small',[1,3,7],0.5,False,False),
        ('shared',[1,31,1536],0.5,True,False),
        ('different-scale',[1,31,1536],0.4,False,False),
        ('topk',[1,31,1536],0.5,False,True)]:
        weights=[nh.from_array(np.array(v,np.float32),n) for n,v in [('sqrt2',np.sqrt(np.float32(2))),('one',1),('half',half)]]
        nodes=[h.make_node('Div',['x','sqrt2'],['d']),h.make_node('Erf',['d'],['e']),h.make_node('Add',['e','one'],['a']),h.make_node('Mul',['x','a'],['m']),h.make_node('Mul',['m','half'],['z' if ranked else 'y'])]
        shape=dims
        if ranked:
            weights.append(nh.from_array(np.array([2],np.int64),'k'));nodes.append(h.make_node('TopK',['z','k'],['y','indices'],axis=-1));shape=dims[:-1]+[2]
        m=model(nodes,dims,shape,weights)
        if shared:m.graph.output.append(h.make_tensor_value_info('e',T.FLOAT,dims))
        yield 'gelu-tile-'+label,m,int(label in ['full','rank4-stem','rank4-stage2','rank4-stage3','rank4-stage4','rank4-stage5'])
    for k in [64,129,257]:
        nodes=[h.make_node('Transpose',['x'],['rhs'],perm=[0,1,3,2]),h.make_node('MatMul',['x','rhs'],['y'])]
        yield f'batch-matmul-reduction-{k}',model(nodes,[1,2,3,k],[1,2,3,3],[]),int(k>128)
    nodes=[h.make_node('Transpose',['x'],['rhs'],perm=[0,1,3,2]),h.make_node('MatMul',['x','rhs'],['y'])]
    yield 'batch-matmul-large-score',model(nodes,[1,2,257,64],[1,2,257,257],[]),1
    yield 'batch-matmul-cost-guard',model(nodes,[1,2,513,64],[1,2,513,513],[]),0
    for axis in [-1,3,1]:
        yield f'softmax-rows-{axis}',model([h.make_node('Softmax',['x'],['y'],axis=axis)],[1,6,17,257],[1,6,17,257],[]),int(axis!=1)
    for axis in [-1,2,1]:
        dims=[2,3,7];weights=[nh.from_array(rng.normal(size=(7,) if axis!=1 else (3,7)).astype(np.float32),'scale'),nh.from_array(rng.normal(size=(7,) if axis!=1 else (3,7)).astype(np.float32),'bias')]
        yield f'layernorm-{axis}',model([h.make_node('LayerNormalization',['x','scale','bias'],['y'],axis=axis,epsilon=1e-5)],dims,dims,weights),int(axis!=1)
    for dims,bias in [([1,257,384],True),([2,5,11],False),([2,29,384],True),([129,1,65],False)]:
        weights=[nh.from_array(rng.normal(size=dims[-1]).astype(np.float32),'scale')]
        inputs=['x','scale']
        if bias:
            weights.append(nh.from_array(rng.normal(size=dims[-1]).astype(np.float32),'bias'));inputs.append('bias')
        yield f'layernorm-layout-{dims[0]}-{dims[1]}-{dims[-1]}',model([h.make_node('LayerNormalization',inputs,['y'],axis=-1,epsilon=1e-5)],dims,dims,weights),int(dims[0]*dims[-1]<=8192)
    for label,dims,axis,index in [('rank5-inner',[2,3,4,2,2],1,1),('rank6-channel',[8,2,4,2,4,2],1,0),('rank6-negative',[8,2,4,2,4,2],-5,-1),('rank6-wide',[8,2,4,16,4,16],1,1),('rank6-last',[2,3,2,2,2,4],-1,2)]:
        actual=axis%len(dims);output=dims[:actual]+dims[actual+1:]
        weights=[nh.from_array(np.array(index,np.int64),'index')]
        yield 'gather-'+label,model([h.make_node('Gather',['x','index'],['y'],axis=axis)],dims,output,weights),1
    weights=[nh.from_array(np.array([1,5,3,2,4],np.int64),'target')]+[nh.from_array(np.array(i,np.int64),f'i{i}') for i in range(3)]
    nodes=[h.make_node('Reshape',['x','target'],['packed'],allowzero=0),h.make_node('Transpose',['packed'],['t'],perm=[2,0,3,1,4])]
    nodes += [h.make_node('Gather',['t',f'i{i}'],[f'g{i}'],axis=0) for i in range(3)]
    nodes += [h.make_node('Concat',['g0','g1','g2'],['y'],axis=0)]
    packed=model(nodes,[1,5,24],[3,2,5,4],weights)
    yield 'packed-head-gather',packed,3
    shared=onnx.ModelProto();shared.CopyFrom(packed)
    shared.graph.output.append(h.make_tensor_value_info('t',T.FLOAT,[3,1,2,5,4]))
    yield 'packed-head-shared-parent',shared,3
    for index in [0, 2, -1]:
        weights=[nh.from_array(np.array(index,np.int64),'index')]
        yield f'gather-{index}',model([h.make_node('Gather',['x','index'],['y'],axis=0)],[3,1,2,5,4],[1,2,5,4],weights),1
    for prefix in [[4],[2,3],[2,2,3]]:
        w=nh.from_array(rng.standard_normal((5,7)).astype(np.float32),'w')
        yield 'matmul-'+str(len(prefix)),model([h.make_node('MatMul',['x','w'],['y'])],prefix+[5],prefix+[7],[w]),1
    weights=[nh.from_array(rng.standard_normal((12,7)).astype(np.float32),'w'),nh.from_array(np.array([1,5,12],np.int64),'target')]
    yield 'flatten-linear',model([h.make_node('Reshape',['x','target'],['flat'],allowzero=0),h.make_node('MatMul',['flat','w'],['y'])],[1,5,3,4],[1,5,7],weights),1
    shared=model([h.make_node('Reshape',['x','target'],['flat'],allowzero=0),h.make_node('MatMul',['flat','w'],['y'])],[1,5,3,4],[1,5,7],weights)
    shared.graph.output.append(h.make_tensor_value_info('flat',T.FLOAT,[1,5,12]))
    yield 'flatten-linear-shared-guard',shared,1
    for trans in [0,1]:
        weights=[nh.from_array(rng.standard_normal((7,5) if trans else (5,7)).astype(np.float32),'w'),nh.from_array(rng.standard_normal(7).astype(np.float32),'b')]
        yield f'gemm-{trans}',model([h.make_node('Gemm',['x','w','b'],['y'],transB=trans)],[4,5],[4,7],weights),1
    weights=[nh.from_array(rng.standard_normal((2,3,4,4)).astype(np.float32),'w'),nh.from_array(rng.standard_normal(3).astype(np.float32),'b')]
    yield 'deconv',model([h.make_node('ConvTranspose',['x','w','b'],['y'],strides=[4,4],kernel_shape=[4,4])],[2,2,3,2],[2,3,12,8],weights),1
    for negative in [False,True]:
        data={'start':[-3 if negative else 0],'end':[2**63-1 if negative else 2],'axis':[-1 if negative else 5],'step':[1]}
        weights=[nh.from_array(np.array(value,np.int64),key) for key,value in data.items()]
        yield f'slice-{negative}',model([h.make_node('Slice',['x',*data],['y'])],[1,7,1,1,1,4],[1,7,1,1,1,3 if negative else 2],weights),1
    w=nh.from_array(rng.standard_normal((5,7)).astype(np.float32),'w')
    # Arithmetic upstream of discrete selection is intentionally not rewritten.
    m=model([h.make_node('MatMul',['x','w'],['z']),h.make_node('TopK',['z','k'],['y','indices'],axis=-1)],[4,5],[4,2],[w,nh.from_array(np.array([2],np.int64),'k')])
    yield 'topk-guard',m,0
    weights=[nh.from_array(rng.standard_normal((2,3,2,2)).astype(np.float32),'w')]
    yield 'deconv-stride2-unchanged',model([h.make_node('ConvTranspose',['x','w'],['y'],strides=[2,2],kernel_shape=[2,2])],[1,2,3,2],[1,3,6,4],weights),0
    weights=[nh.from_array(np.array(v,np.int64),n) for n,v in [('s',[0]),('e',[4]),('a',[5]),('t',[2])]]
    yield 'slice-step2-unchanged',model([h.make_node('Slice',['x','s','e','a','t'],['y'])],[1,7,1,1,1,4],[1,7,1,1,1,2],weights),0


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('artifacts/legalize-unit'))
    p.add_argument('--binary',type=Path,default=Path('build/vbx-rewrite'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);results=[]
    for name,m,count in cases():
        src=a.out/(name+'.onnx');dst=a.out/(name+'.rewritten.onnx');onnx.save(m,src)
        stats=json.loads(subprocess.check_output([str(a.binary.resolve()),str(src),str(dst)],text=True));assert stats['rewrites']==count,(name,stats)
        if count==0:assert m.SerializeToString()==onnx.load(dst).SerializeToString()
        subprocess.run([__import__('sys').executable,'tests/legalization/compare_onnx.py',str(src),str(dst),'--samples','3','--distribution','normal','--edge-cases','--report',str(a.out/(name+'.json'))],check=True)
        stable=a.out/(name+'.again.onnx');again=json.loads(subprocess.check_output([str(a.binary.resolve()),str(dst),str(stable)],text=True))
        assert again['rewrites']==0,(name,again)
        assert onnx.load(dst).SerializeToString()==onnx.load(stable).SerializeToString()
        results.append({'case':name,'rewrites':count,'passed':True})
    (a.out/'results.json').write_text(json.dumps(results,indent=2));print(f'{len(results)} cases passed')

if __name__=='__main__':main()
