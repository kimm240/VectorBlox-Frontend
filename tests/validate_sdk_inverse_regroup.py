"""Independent guards for inverse-regroup SSA legalization; no compiler-time tests."""
import copy,json,struct,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];f=r/'artifacts/sdk-inverse-regroup-tests';f.mkdir(parents=True,exist_ok=True)
def fixture(a=32,b=32,h=256,w=2,c=2):
 q=dict(scale=[.0244017019867897],zero_point=[0]);ts=[dict(name=n,shape=s,type='INT8',buffer=0,quantization=copy.deepcopy(q)) for n,s in [('x',[1,b,a]),('transposed',[1,a,b]),('y',[1,h,w,c])]];ts += [dict(name='perm',shape=[3],type='INT32',buffer=1),dict(name='shape',shape=[4],type='INT32',buffer=2)];return dict(version=3,operator_codes=[dict(builtin_code='TRANSPOSE',deprecated_builtin_code=39,version=1),dict(builtin_code='RESHAPE',deprecated_builtin_code=22,version=1)],buffers=[{},dict(data=list(struct.pack('<3i',0,2,1))),dict(data=list(struct.pack('<4i',1,h,w,c)))],subgraphs=[dict(inputs=[0],outputs=[2],tensors=ts,operators=[dict(opcode_index=0,inputs=[0,3],outputs=[1]),dict(opcode_index=1,inputs=[1,4],outputs=[2])])])
rows=[]
for name in ['large-512','large-2048','large-8192','large-channel-policy','producer-v4','basic','odd','signature','root','public','shared','dynamic','variable','quant','custom','intermediate','mutation','producer-mutation','version','runtime-shape','public-shape','wrong-shape','external','single-channel','too-many-groups']:
 m=fixture(7,12,14,3,2) if name=='odd' else fixture();g=m['subgraphs'][0];ts=g['tensors'];ops=g['operators'];expected=name in ['producer-v4','basic','odd','signature']
 flags=[]
 if name.startswith('large-'):
  b=512 if name=='large-512' else 2048 if name=='large-2048' else 8192
  side={512:8,2048:16,8192:32}[b];m=fixture(32,b,256,side,side);g=m['subgraphs'][0];ts=g['tensors'];ops=g['operators'];expected=name!='large-channel-policy'
  if name=='large-8192':flags=['--max-channels','8192']
 if name=='producer-v4':m['operator_codes'][0]['version']=4
 if name=='signature':m['signature_defs']=[dict(outputs=[dict(name='output',tensor_index=2)])]
 if name=='root':g['operators']=ops[1:];g['inputs']=[1]
 if name=='public':g['outputs'].append(1)
 if name=='shared':
  t=copy.deepcopy(ts[2]);t['name']='other';ts.append(t);o=copy.deepcopy(ops[1]);o['outputs']=[5];ops.append(o);g['outputs'].append(5)
 if name=='dynamic':ts[1]['shape_signature']=[1,-1,32]
 if name=='variable':ts[1]['is_variable']=True
 if name=='quant':ts[2]['quantization']['zero_point']=[1]
 if name=='custom':ops[1]['custom_options']=[1]
 if name=='intermediate':ops[1]['intermediates']=[0]
 if name=='mutation':ops[1]['mutating_variable_inputs']=[True,False]
 if name=='producer-mutation':ops[0]['mutating_variable_inputs']=[True,False]
 if name=='version':m['operator_codes'][1]['version']=99
 if name=='runtime-shape':m['buffers'][2]={}
 if name=='public-shape':g['inputs'].append(4)
 if name=='wrong-shape':m['buffers'][2]['data'][4]=255
 if name=='external':m['buffers'][2]['offset']=256
 if name=='single-channel':
  ts[2]['shape']=[1,512,2,1];m['buffers'][2]['data']=list(struct.pack('<4i',1,512,2,1))
 if name=='too-many-groups':
  ts[0]['shape']=[1,1024,8];ts[1]['shape']=[1,8,1024];ts[2]['shape']=[1,2048,2,2];m['buffers'][2]['data']=list(struct.pack('<4i',1,2048,2,2))
 src=f/(name+'.json');dst=f/(name+'.after.json');again=f/(name+'.again.json');src.write_text(json.dumps(m));subprocess.run([r/'build/vbx-rewrite','--sdk-json',src,dst,'--dump-mlir',f/(name+'.mlir'),*flags],check=True,stdout=subprocess.DEVNULL);v=json.loads(dst.read_text());s=v['subgraphs'][0];assert any(t.get('name','').startswith('__vbx_inverse_regroup_') for t in s['tensors'])==expected,name
 assert s['tensors'][:len(ts)]==ts and s['inputs']==g['inputs'] and s['outputs']==g['outputs'] and v.get('signature_defs')==m.get('signature_defs'),name
 if expected:
  assert s['operators'][0]==ops[0] and s['operators'][-1]['outputs']==[2],name
  assert 'CONCATENATION' in (f/(name+'.mlir')).read_text()
 subprocess.run([r/'build/vbx-rewrite','--sdk-json',dst,again,*flags],check=True,stdout=subprocess.DEVNULL);assert json.loads(again.read_text())==v,name;rows.append(dict(case=name,rewritten=expected))
(f/'results.json').write_text(json.dumps(rows,indent=2));print(len(rows),'inverse-regroup guards passed')
