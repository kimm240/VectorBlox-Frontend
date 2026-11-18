"""Manual frozen-pass survey; exports pretrained models, checks outputs and times CPU inference."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/model-survey'
sys.path.insert(0,str(OUT/'deps'))
torch.set_num_threads(2);torch.manual_seed(20261118)

REVISIONS = {
 'facebook/dinov2-small': 'ed25f3a31f01632728cabb09d1542f84ab7b0056',
 'nvidia/segformer-b0-finetuned-ade-512-512': '489d5cd81a0b59fab9b7ea758d3548ebe99677da',
 'timm/deit_tiny_patch16_224.fb_in1k': '80e968688553f219e4a86f940ed945a23709c16f',
 'timm/mobilevit_xxs.cvnets_in1k': '2c62d71504c6fbba9bb322f728586c4a235e5fa4',
}
REPOS = {
 'superpoint': ('https://github.com/magicleap/SuperPointPretrainedNetwork.git', '1fda796addba9b6f8e79d586a3699700a86b1cea'),
 'lightglue': ('https://github.com/cvg/LightGlue.git', '5a650c444718caffc2669dc0094cde908f42dd73'),
}

def prepare_repo(name):
 repo=OUT/'repos'/name;url,revision=REPOS[name]
 repo.parent.mkdir(parents=True,exist_ok=True)
 if not repo.exists():
  subprocess.run(['git','clone','--no-checkout',url,str(repo)],check=True)
  subprocess.run(['git','-C',str(repo),'checkout','--detach',revision],check=True)
 actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 if actual!=revision or subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
  raise RuntimeError('Expected clean pinned source: '+str(repo))
 return repo

class Outputs(torch.nn.Module):
 def __init__(self,model,field):super().__init__();self.model=model;self.field=field
 def forward(self,image):return getattr(self.model(image),self.field)

class Matcher(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,k0,d0,k1,d1):
  size=torch.tensor([[256.,256.]])
  out=self.model({'image0':{'keypoints':k0,'descriptors':d0,'image_size':size},'image1':{'keypoints':k1,'descriptors':d1,'image_size':size}})
  return out['matches0'],out['matches1'],out['matching_scores0'],out['matching_scores1']

def build(name,meta):
 if name in ['deit-tiny','mobilevit-xxs']:
  import timm
  timm.layers.set_fused_attn(False)
  key={'deit-tiny':'deit_tiny_patch16_224.fb_in1k','mobilevit-xxs':'mobilevit_xxs.cvnets_in1k'}[name]
  hf_id='timm/'+key
  model=timm.create_model(key,pretrained=True,pretrained_cfg_overlay={'hf_hub_id':hf_id+'@'+REVISIONS[hf_id]})
  size=224 if name=='deit-tiny' else 256
  meta.update(implementation='timm '+timm.__version__,pretrained_cfg=model.pretrained_cfg,scope='ImageNet classification logits')
  return model,(torch.randn(1,3,size,size),),['image']
 if name in ['dinov2-small','segformer-b0']:
  from transformers import Dinov2Model,SegformerForSemanticSegmentation
  repo={'dinov2-small':'facebook/dinov2-small','segformer-b0':'nvidia/segformer-b0-finetuned-ade-512-512'}[name]
  revision=REVISIONS[repo]
  if name=='dinov2-small':model=Outputs(Dinov2Model.from_pretrained(repo,revision=revision,attn_implementation='eager'),'last_hidden_state');size=224
  else:model=Outputs(SegformerForSemanticSegmentation.from_pretrained(repo,revision=revision),'logits');size=512
  meta.update(repository=repo,revision=revision,scope='patch features' if name=='dinov2-small' else 'ADE20K segmentation logits')
  return model,(torch.randn(1,3,size,size),),['image']
 if name=='superpoint':
  repo=prepare_repo('superpoint');sys.path.insert(0,str(repo));from demo_superpoint import SuperPointNet
  model=SuperPointNet();weight=repo/'superpoint_v1.pth';model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
  meta.update(repository='magicleap/SuperPointPretrainedNetwork',revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),scope='dense logits/descriptors; sparse NMS excluded')
  return model,(torch.rand(1,1,256,320),),['image']
 if name=='lightglue':
  repo=prepare_repo('lightglue');sys.path.insert(0,str(repo));from lightglue import LightGlue
  model=Matcher(LightGlue(features='superpoint',flash=False,depth_confidence=-1,width_confidence=-1))
  k0=torch.rand(1,128,2)*256;k1=k0+torch.randn_like(k0)*2
  d0=torch.nn.functional.normalize(torch.randn(1,128,256),dim=-1);d1=torch.nn.functional.normalize(d0+torch.randn_like(d0)*0.01,dim=-1)
  meta.update(repository='cvg/LightGlue',revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),scope='128 SuperPoint keypoints per image; adaptive stopping/pruning and flash disabled; matches and scores')
  return model,(k0,d0,k1,d1),['k0','d0','k1','d1']
 raise ValueError(name)

def session(path,opt):
 o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.graph_optimization_level=opt
 return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])

def run(name):
 p=OUT/name;p.mkdir(parents=True,exist_ok=True);meta={'name':name,'status':'failed','stage':'load','torch':torch.__version__,'onnxruntime':ort.__version__,'binary_sha256':hashlib.sha256((ROOT/'build/vbx-rewrite').read_bytes()).hexdigest()}
 try:
  model,inputs,names=build(name,meta);model.eval();meta['inputs']={n:list(x.shape) for n,x in zip(names,inputs)}
  meta['stage']='export'
  with torch.no_grad():
   expected=model(*inputs);expected=(expected,) if isinstance(expected,torch.Tensor) else expected
   torch.onnx.export(model,inputs,str(p/'exported.onnx'),input_names=names,output_names=[f'output{i}' for i in range(len(expected))],opset_version=17,do_constant_folding=True)
  raw=onnx.load(p/'exported.onnx');raw=onnx.shape_inference.infer_shapes(raw,data_prop=True);onnx.save(raw,p/'before.onnx');onnx.checker.check_model(str(p/'before.onnx'))
  feed={n:x.numpy() for n,x in zip(names,inputs)};np.savez(p/'sample0.npz',**feed)
  rng=np.random.default_rng(20261118)
  for i in [1,2]:
   extra={n:(x+rng.normal(0,0.001,x.shape).astype(np.float32)).astype(np.float32) for n,x in feed.items()}
   if name=='superpoint':extra={n:np.clip(x,0,1) for n,x in extra.items()}
   np.savez(p/f'sample{i}.npz',**extra)
  meta['stage']='export_equivalence'
  original=session(p/'before.onnx',ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
  values=original.run(None,feed)
  assert len(values)==len(expected)
  for a,b in zip(values,expected):np.testing.assert_allclose(a,b.detach().numpy(),atol=1e-4,rtol=1e-3)
  del original
  meta['stage']='rewrite';start=time.perf_counter()
  r=subprocess.run([str(ROOT/'build/vbx-rewrite'),str(p/'before.onnx'),str(p/'after.onnx'),'--report',str(p/'rewrite.json')],capture_output=True,text=True,timeout=180)
  meta['rewrite_wall_ms']=(time.perf_counter()-start)*1000
  if r.returncode:raise RuntimeError(r.stderr)
  stats=json.loads((p/'rewrite.json').read_text());meta['rewrites']=stats['rewrites'];meta['skipped']=stats['skipped']
  meta['stage']='numerical'
  r=subprocess.run([sys.executable,str(ROOT/'tests/legalization/compare_onnx.py'),str(p/'before.onnx'),str(p/'after.onnx'),'--inputs',*[str(p/f'sample{i}.npz') for i in range(3)],'--atol','1e-4','--rtol','1e-3','--report',str(p/'numerical.json')],capture_output=True,text=True,timeout=240)
  if r.returncode:raise RuntimeError(r.stdout+r.stderr)
  meta['stage']='latency';sessions=[session(p/n,ort.GraphOptimizationLevel.ORT_ENABLE_ALL) for n in ['before.onnx','after.onnx']]
  for s in sessions:
   for _ in range(5):s.run(None,feed)
  times=[[],[]]
  for i in range(30):
   for j in ([0,1] if i%2==0 else [1,0]):
    start=time.perf_counter_ns();sessions[j].run(None,feed);times[j].append((time.perf_counter_ns()-start)/1e6)
  meta['latency']={k:{'median_ms':float(np.median(t)),'p95_ms':float(np.percentile(t,95)),'samples_ms':t} for k,t in zip(['before','after'],times)}
  meta['latency_settings']={'provider':'CPUExecutionProvider','threads':2,'inter_op_threads':1,'warmup':5,'runs':30,'optimizations':'ORT_ENABLE_ALL','includes':'synchronous session.run; excludes load/export/preprocessing','alternating_order':True}
  meta.update(status='passed',stage='complete',size_bytes={k:(p/v).stat().st_size for k,v in [('before','before.onnx'),('after','after.onnx')]})
 except Exception as e:meta['error']=str(e);traceback.print_exc()
 finally:(p/'result.json').write_text(json.dumps(meta,indent=2,default=str)+'\n');print(name,meta['status'],meta['stage'],flush=True)
 return meta['status']=='passed'

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('model',choices=['superpoint','deit-tiny','dinov2-small','mobilevit-xxs','segformer-b0','lightglue']);args=parser.parse_args();sys.exit(0 if run(args.model) else 1)
