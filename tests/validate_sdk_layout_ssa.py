"""Dependent singleton-layout replacements must preserve real SSA use-def edges."""
from pathlib import Path
import json,struct,subprocess
r=Path(__file__).resolve().parents[1];f=r/'artifacts/sdk-layout-ssa-tests';f.mkdir(parents=True,exist_ok=True)
q=dict(scale=[.03125],zero_point=[-3]);shapes=[[1,1,3,5],[1,3,1,5],[1,3,5,1]]
ts=[dict(name=n,shape=s,type='INT8',buffer=0,quantization=q) for n,s in zip(['input','middle','output'],shapes)];bs=[{}];ops=[]
for i in range(2):
 shape=shapes[i+1];bs.append(dict(data=list(struct.pack('<'+'i'*len(shape),*shape))));ts.append(dict(name='shape'+str(i),shape=[len(shape)],type='INT32',buffer=i+1));ops.append(dict(opcode_index=0,inputs=[i,i+3],outputs=[i+1],builtin_options_type='ReshapeOptions',builtin_options=dict(new_shape=shape)))
m=dict(version=3,operator_codes=[dict(builtin_code='RESHAPE',deprecated_builtin_code=22,version=1)],buffers=bs,subgraphs=[dict(inputs=[0],outputs=[2],tensors=ts,operators=ops)])
source=f/'chain.json';target=f/'chain.after.json';ir=f/'chain.mlir';source.write_text(json.dumps(m));binary=r/'build/vbx-rewrite'
report=json.loads(subprocess.check_output([str(binary),'--sdk-json',str(source),str(target),'--dump-mlir',str(ir)]));assert report['rewrites']==2
text=ir.read_text();assert 'vbx_sdk.node' in text and 'tensor<' in text and 'vbx_tflite.leading_expand' not in text and 'vbx_tflite.singleton_matrix' not in text
n=json.loads(target.read_text());g=n['subgraphs'][0];assert g['inputs']==[0] and g['outputs']==[2] and g['tensors'][:5]==ts
available={0}|{i for i,t in enumerate(g['tensors']) if n['buffers'][t['buffer']].get('data')}
for op in g['operators']:
 assert all(i<0 or i in available for i in op['inputs']),op
 available.update(op['outputs'])
assert 2 in available
again=f/'chain.again.json';report=json.loads(subprocess.check_output([str(binary),'--sdk-json',str(target),str(again)]));assert report['rewrites']==0 and target.read_bytes()==again.read_bytes()
budget=f/'chain.budget.json';budget.unlink(missing_ok=True);ret=subprocess.run([str(binary),'--sdk-json',str(source),str(budget),'--max-rewrites','1'],capture_output=True);assert ret.returncode and not budget.exists()
print('Dependent singleton layout SSA, topology, idempotence and atomic budget passed')
