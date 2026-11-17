"""Guard and export checks for upstream canonicalization of inverse SDK views."""
import copy,json,struct,subprocess
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'artifacts/sdk-inverse-view-tests');args=p.parse_args()
OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
def fixture(unsigned=False):
 q=dict(scale=[.125],zero_point=[17 if unsigned else -7]);kind='UINT8' if unsigned else 'INT8'
 ts=[dict(name=n,shape=s,type=kind,buffer=0,quantization=copy.deepcopy(q)) for n,s in [('input',[1,5,1,1]),('middle',[1,5]),('restored',[1,5,1,1]),('output',[1,5,1,1])]]
 bs=[{}]
 for shape in [[1,5],[1,5,1,1]]:
  ts.append(dict(name='shape'+str(len(ts)),shape=[len(shape)],type='INT32',buffer=len(bs)));bs.append(dict(data=list(struct.pack('<'+'i'*len(shape),*shape))))
 codes=[dict(builtin_code='RESHAPE',deprecated_builtin_code=22,version=1),dict(builtin_code='QUANTIZE',deprecated_builtin_code=114,version=2)]
 ops=[dict(opcode_index=0,inputs=[0,4],outputs=[1],builtin_options_type='ReshapeOptions',builtin_options=dict(new_shape=[1,5])),dict(opcode_index=0,inputs=[1,5],outputs=[2],builtin_options_type='ReshapeOptions',builtin_options=dict(new_shape=[1,5,1,1])),dict(opcode_index=1,inputs=[2],outputs=[3],builtin_options_type='QuantizeOptions',builtin_options={})]
 return dict(version=3,operator_codes=codes,buffers=bs,subgraphs=[dict(tensors=ts,operators=ops,inputs=[0],outputs=[3])])
def main():
 OUT.mkdir(exist_ok=True);names=['basic','unsigned','signature','min-max','chain','middle-public','end-public','signature-middle','shared-middle','quantization','dtype','dynamic','variable','constant','external','parameter-public','parameter-produced','parameter-external','parameter-bytes','shape-options','custom','intermediates','mutation','version','invalid-scale','secondary-options','no-inverse'];rows=[]
 for name in names:
  m=fixture(unsigned=name=='unsigned');g=m['subgraphs'][0];ts=g['tensors'];ops=g['operators'];expected=name in names[:5]
  if name=='chain':
   for i in [1,2]:
    t=copy.deepcopy(ts[i]);t['name']='chain'+str(i);ts.append(t)
   a=copy.deepcopy(ops[0]);a['inputs'][0]=2;a['outputs']=[6]
   b=copy.deepcopy(ops[1]);b['inputs'][0]=6;b['outputs']=[7]
   ops[-1]['inputs']=[7];ops[2:2]=[a,b]
  if name=='signature':m['signature_defs']=[dict(inputs=[dict(name='x',tensor_index=0)],outputs=[dict(name='y',tensor_index=3)])]
  if name=='min-max':
   for t in ts[:4]:t['quantization'].update(min=[-1.],max=[1.])
  if name=='middle-public':g['outputs'].append(1)
  if name=='end-public':g['outputs'].append(2)
  if name=='signature-middle':m['signature_defs']=[dict(outputs=[dict(name='hidden',tensor_index=1)])]
  if name=='shared-middle':
   t=copy.deepcopy(ts[1]);t['name']='extra';ts.append(t);ops.append(dict(opcode_index=1,inputs=[1],outputs=[len(ts)-1],builtin_options_type='QuantizeOptions',builtin_options={}));g['outputs'].append(len(ts)-1)
  if name=='quantization':ts[1]['quantization']['zero_point'][0]+=1
  if name=='dtype':ts[1]['type']='UINT8'
  if name=='dynamic':ts[0]['shape_signature']=[1,-1,1,1]
  if name=='variable':ts[1]['is_variable']=True
  if name=='constant':ts[0]['buffer']=len(m['buffers']);m['buffers'].append(dict(data=[1]*5))
  if name=='external':m['buffers'][0]['offset']=1
  if name=='parameter-public':g['inputs'].append(4)
  if name=='parameter-produced':
   t=copy.deepcopy(ts[4]);t['name']='shape_source';ts.append(t);m['operator_codes'].append(dict(builtin_code='IDENTITY',deprecated_builtin_code=127,version=1));ops.insert(0,dict(opcode_index=2,inputs=[len(ts)-1],outputs=[4],builtin_options_type='NONE'))
  if name=='parameter-external':m['buffers'][1]['offset']=1
  if name=='parameter-bytes':m['buffers'][1]['data'][0]=2
  if name=='shape-options':ops[1]['builtin_options']['new_shape']=[1,1,1,5]
  if name=='custom':ops[1]['custom_options']=[1]
  if name=='intermediates':ops[1]['intermediates']=[0]
  if name=='mutation':ops[1]['mutating_variable_inputs']=[True,False]
  if name=='version':m['operator_codes'][0]['version']=2
  if name=='invalid-scale':
   for t in ts[:4]:t['quantization']['scale']=[0.]
  if name=='secondary-options':ops[1]['builtin_options_2_type']='StablehloReshapeOptions'
  if name=='no-inverse':ts[2]['shape']=[1,1,5,1];m['buffers'][2]['data']=list(struct.pack('<4i',1,1,5,1));ops[1]['builtin_options']['new_shape']=[1,1,5,1]
  paths=[OUT/(name+x) for x in ['.json','.after.json','.again.json','.report.json']];before,after,again,report=paths;before.write_text(json.dumps(m));cmd=[str(ROOT/'build/vbx-rewrite'),'--sdk-json']
  subprocess.run(cmd+[str(before),str(after),'--report',str(report)],check=True,stdout=subprocess.DEVNULL);x=json.loads(after.read_text());count=json.loads(report.read_text())['rewrites'];assert count==(2 if name=='chain' else int(expected)),(name,count)
  if expected:
   sg=x['subgraphs'][0];assert sg['tensors']==ts and x['buffers']==m['buffers'] and sg['inputs']==g['inputs'] and sg['outputs']==g['outputs'] and x.get('signature_defs')==m.get('signature_defs');assert len(sg['operators'])==1 and sg['operators'][0]['inputs']==[0]
  else:assert x==m,name
  subprocess.run(cmd+[str(after),str(again)],check=True,stdout=subprocess.DEVNULL);assert json.loads(again.read_text())==x;rows.append(dict(case=name,rewrites=count))
 before=OUT/'zero-budget.json';after=OUT/'zero-budget.after.json';before.write_text(json.dumps(fixture()));after.unlink(missing_ok=True);r=subprocess.run([str(ROOT/'build/vbx-rewrite'),'--sdk-json',str(before),str(after),'--max-rewrites','0'],capture_output=True);assert r.returncode!=0 and not after.exists()
 before=OUT/'chain.json';after=OUT/'chain-budget.after.json';after.unlink(missing_ok=True)
 r=subprocess.run([str(ROOT/'build/vbx-rewrite'),'--sdk-json',str(before),str(after),'--max-rewrites','1'],capture_output=True);assert r.returncode!=0 and not after.exists()
 (OUT/'results.json').write_text(json.dumps(rows,indent=2));print(len(rows),'inverse-view guards and 2 budget rejections passed')
if __name__=='__main__':main()
