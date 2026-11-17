"""Standalone layout-pass regressions: exact integer element order and guards."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import numpy as np

OUT=Path('artifacts/sdk-layout-tests');OUT.mkdir(parents=True,exist_ok=True)
BINARY=Path('build/vbx-rewrite').resolve()


def model(before,after):
    q={'scale':[0.125],'zero_point':[-3],'quantized_dimension':0}
    def tensor(name,shape,buffer=0,dtype='INT8'):
        return dict(name=name,shape=shape,type=dtype,buffer=buffer,quantization=copy.deepcopy(q) if dtype=='INT8' else {},is_variable=False,has_rank=True)
    return dict(version=3,operator_codes=[dict(builtin_code='RESHAPE',deprecated_builtin_code=22,version=1)],buffers=[{},dict(data=list(struct.pack('<'+'i'*len(after),*after)))],subgraphs=[dict(inputs=[0],outputs=[2],tensors=[tensor('x',before),tensor('shape',[len(after)],1,'INT32'),tensor('y',after)],operators=[dict(opcode_index=0,inputs=[0,1],outputs=[2],builtin_options_type='ReshapeOptions',builtin_options={'new_shape':after})])])


cases=[]
for i,(a,b) in enumerate([([1,1,257,64],[1,257,1,64]),([1,3,1,5],[1,3,5,1]),([1,2,1,2],[1,1,2,2])]):cases.append((f'move-{i}',model(a,b),1))
for name,a,b in [('same',[1,2,3,1],[1,2,3,1]),('nonunit-swap',[1,3,2,1],[1,2,3,1]),('batch-move',[2,1,3,1],[1,2,3,1])]:cases.append((name,model(a,b),0))
for i,(a,b) in enumerate([([1,1,6,4],[1,2,3,4]),([1,6,1,4],[1,2,3,4]),([1,6,4,1],[1,2,3,4]),([1,2,6,1],[1,2,2,3])]):
    cases.append((f'squeeze-split-{i}',model(a,b),1))
cases.append(('restore-batch',model([1,96,8,8],[8,12,8,8]),1))
cases.append(('restore-batch-nondivisible',model([1,96,8,8],[7,12,8,8]),0))
for name,a,b in [('factorization-batch',[2,1,6,4],[2,2,3,4]),('factorization-reorder',[1,1,6,4],[1,4,3,2]),('factorization-no-unit',[1,2,3,4],[1,3,2,4])]:
    cases.append((name,model(a,b),0))
m=model([1,2,6,1],[1,2,2,3]);m['subgraphs'][0]['operators'][0]['builtin_options_type']='NONE';m['subgraphs'][0]['operators'][0]['builtin_options']={}
cases.append(('factorization-no-options',m,1))
m=model([1,2,6,1],[1,2,2,3]);g=m['subgraphs'][0]
g['tensors'].append(copy.deepcopy(g['tensors'][2]));g['tensors'][3]['name']='other'
g['operators'].append(copy.deepcopy(g['operators'][0]));g['operators'][1]['outputs']=[3];g['outputs'].append(3)
cases.append(('multiple-factorizations',m,2))
for name in ['runtime-input','variable-shape','dynamic-shape','different-quantization','per-axis','float','wrong-target','produced-shape','old-transpose-version']:
    m=model([1,1,3,5],[1,3,1,5]);g=m['subgraphs'][0];ts=g['tensors']
    if name=='runtime-input':g['inputs'].append(1)
    elif name=='variable-shape':ts[1]['is_variable']=True
    elif name=='dynamic-shape':ts[0]['shape_signature']=[1,-1,3,5]
    elif name=='different-quantization':ts[2]['quantization']['scale']=[.25]
    elif name=='per-axis':
        for i in [0,2]:ts[i]['quantization']['scale']=[.125,.25]
    elif name=='float':ts[0]['type']=ts[2]['type']='FLOAT32'
    elif name=='wrong-target':m['buffers'][1]['data']=list(struct.pack('<4i',1,1,3,5))
    elif name=='produced-shape':
        m['operator_codes'].append(dict(builtin_code='CUSTOM',deprecated_builtin_code=32,custom_code='test-shape',version=1))
        g['operators'].insert(0,dict(opcode_index=1,inputs=[],outputs=[1]))
    elif name=='old-transpose-version':m['operator_codes'].append(dict(builtin_code='TRANSPOSE',deprecated_builtin_code=39,version=1))
    cases.append((name,m,int(name=='old-transpose-version')))

for name,a,b,count in [('window-single-channel',[1,4,2,2],[4,1,4],1),('window-small',[1,24,2,2],[4,6,4],1),('window-logcan',[1,1536,2,2],[16,96,4],1),('window-nonsquare',[1,24,2,3],[4,6,6],1),('window-wrong-factor',[1,24,2,2],[3,4,8],0),('window-batch-not-one',[2,24,2,2],[8,6,4],0)]:cases.append((name,model(a,b),count))
for name,a,b,count in [('pool-square',[1,4,4,96],[16,96,1,1],1),('pool-rectangular',[1,2,3,5],[6,5,1,1],1),('pool-bad-count',[1,2,3,5],[5,5,1,1],0),('pool-too-many',[1,65,65,3],[4225,3,1,1],0),('pool-batch',[2,2,3,5],[12,5,1,1],0)]:cases.append((name,model(a,b),count))
for name,source,first,second in [('shared-permutations',[1,1,3,5],[1,3,1,5],[1,3,1,5]),('dependent-permutations',[1,1,3,5],[1,3,1,5],[1,3,5,1]),('mixed-layouts',[1,6,1,4],[1,6,4,1],[1,2,3,4])]:
    m=model(source,first);g=m['subgraphs'][0];other=model(first if name=='dependent-permutations' else source,second)
    shape=len(g['tensors']);g['tensors'].append(copy.deepcopy(other['subgraphs'][0]['tensors'][1]));g['tensors'][-1]['buffer']=len(m['buffers']);m['buffers'].append(other['buffers'][1])
    output=len(g['tensors']);g['tensors'].append(copy.deepcopy(other['subgraphs'][0]['tensors'][2]));g['tensors'][-1]['name']='second'
    op=copy.deepcopy(other['subgraphs'][0]['operators'][0]);op['inputs']=[2 if name=='dependent-permutations' else 0,shape];op['outputs']=[output];g['operators'].append(op)
    g['outputs']=[output] if name=='dependent-permutations' else [2,output]
    cases.append((name,m,2))

for name in ['signature-shape-input','signature-shape-output','variable-input','variable-output','mutated-input','mutated-shape','mutated-output','external-shape-buffer','mismatched-input-signature','mismatched-output-signature','custom-options','scratch-metadata','static-signatures','empty-signatures','empty-mutation-flags','empty-custom','empty-scratch']:
    m=model([1,3,1,5],[1,3,5,1]);g=m['subgraphs'][0];ts=g['tensors'];op=g['operators'][0];count=0
    if name.startswith('signature-shape-'):
        field='inputs' if name.endswith('input') else 'outputs';m['signature_defs']=[dict(signature_key='shape',subgraph_index=0,**{field:[dict(name='shape',tensor_index=1)]})]
        if field=='outputs':count=1
    elif name=='variable-input':ts[0]['is_variable']=True
    elif name=='variable-output':ts[2]['is_variable']=True
    elif name=='mutated-input':op['mutating_variable_inputs']=[True,False]
    elif name=='mutated-shape':op['mutating_variable_inputs']=[False,True]
    elif name=='mutated-output':
        out=len(ts);ts.append(copy.deepcopy(ts[2]));ts[-1]['name']='mutator';g['operators'].append(dict(opcode_index=0,inputs=[2,1],outputs=[out],mutating_variable_inputs=[True,False]))
    elif name=='external-shape-buffer':m['buffers'][1].update(offset=128,size=16)
    elif name=='mismatched-input-signature':ts[0]['shape_signature']=[1,1,3,5]
    elif name=='mismatched-output-signature':ts[2]['shape_signature']=[1,3,1,5]
    elif name=='custom-options':op['custom_options']=[1]
    elif name=='scratch-metadata':op['intermediates']=[0]
    elif name=='static-signatures':ts[0]['shape_signature']=ts[0]['shape'][:];ts[2]['shape_signature']=ts[2]['shape'][:];count=1
    elif name=='empty-signatures':ts[0]['shape_signature']=[];ts[2]['shape_signature']=[];count=1
    elif name=='empty-mutation-flags':op['mutating_variable_inputs']=[];count=1
    elif name=='empty-custom':op['custom_options']=[];count=1
    elif name=='empty-scratch':op['intermediates']=[];count=1
    cases.append((name,m,count))

results=[]
for name,m,count in cases:
    source=OUT/f'{name}.json';target=OUT/f'{name}.out.json';again=OUT/f'{name}.again.json'
    dump=OUT/f'{name}.mlir';source.write_text(json.dumps(m));stats=json.loads(subprocess.check_output([str(BINARY),'--sdk-json',str(source),str(target),'--dump-mlir',str(dump)],text=True));assert stats['rewrites']==count,(name,stats)
    result=json.loads(target.read_text())
    if count:
        assert 'vbx_tflite.' not in dump.read_text() and 'vbx_sdk.node' in dump.read_text(),name
        g=result['subgraphs'][0];tensors=g['tensors']
        before=m['subgraphs'][0]['tensors'][0]['shape'];after=m['subgraphs'][0]['tensors'][2]['shape']
        x=np.random.default_rng(18).integers(-128,128,size=before,dtype=np.int8)
        values={0:x}
        for op in g['operators']:
            assert list(op).index('builtin_options_type')<list(op).index('builtin_options')
            code=result['operator_codes'][op['opcode_index']]
            raw=result['buffers'][tensors[op['inputs'][1]]['buffer']]['data']
            parameter=struct.unpack('<'+'i'*(len(raw)//4),bytes(raw))
            if code['builtin_code']=='TRANSPOSE':
                assert code['version']>=2
                value=values[op['inputs'][0]].transpose(parameter)
            else:
                assert code['builtin_code']=='RESHAPE'
                if op.get('builtin_options'):assert op['builtin_options_type']=='ReshapeOptions'
                value=values[op['inputs'][0]].reshape(parameter)
            values[op['outputs'][0]]=value
        for output in g['outputs']:assert np.array_equal(x.reshape(m['subgraphs'][0]['tensors'][output]['shape']),values[output]),name
        assert tensors[0]==m['subgraphs'][0]['tensors'][0] and tensors[2]==m['subgraphs'][0]['tensors'][2]
    else:assert source.read_bytes()==target.read_bytes()
    stable=json.loads(subprocess.check_output([str(BINARY),'--sdk-json',str(target),str(again)],text=True))
    assert stable['rewrites']==0 and target.read_bytes()==again.read_bytes()
    results.append(dict(case=name,rewrites=count,passed=True))
budget=OUT/'budget.out.json';budget.unlink(missing_ok=True)
r=subprocess.run([BINARY,'--sdk-json',OUT/'shared-permutations.json',budget,'--max-rewrites','1'],capture_output=True,text=True);assert r.returncode and not budget.exists() and 'limit' in r.stderr.lower(),r.stderr
(OUT/'results.json').write_text(json.dumps(results,indent=2));print(f'{len(results)} SDK layout checks passed')
