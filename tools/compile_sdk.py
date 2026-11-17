"""Compile TFLite through the pinned SDK with a C++/MLIR layout pass.

SDK source stays untouched. Numerical validation is a separate test command.
Run with the SDK's Python environment; the native rewrite binary is local.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from vbx_sdk_partitions import layout_partition_plan

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('tflite',type=Path);p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--rewrite-binary',type=Path,default=ROOT/'build/vbx-rewrite');p.add_argument('--inputs',type=Path,nargs='+',help='Optional domain-valid .npy inputs in TFLite input order');a=p.parse_args()
    source=a.tflite.resolve();output=a.output.resolve()
    if not source.is_file():p.error('Input TFLite file is missing')
    if output.exists():p.error('Output already exists; choose a fresh artifact path')
    target=json.loads((ROOT/'targets/vectorblox-3.1.json').read_text());checkout=ROOT/target['checkout']
    head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if head!=target['commit']:raise RuntimeError('SDK checkout does not match the pinned official commit')
    if subprocess.check_output(['git','-C',str(checkout),'status','--porcelain','--','python/vbx','lib'],text=True).strip():
        raise RuntimeError('SDK compiler or simulator has local changes')
    if hashlib.sha256((checkout/'lib/libvbx_cnn_sim.so').read_bytes()).hexdigest()!=target['simulator_sha256']:
        raise RuntimeError('Simulator does not match the pinned SDK artifact')
    os.environ['VBX_SDK']=os.environ['NX_SDK']=str(checkout)
    sys.path.insert(0,str(checkout/'python/vbx'))
    from vbx.generate import lite_flow as sdk
    graph,original_dir=sdk.load_graph(str(source))
    if not sdk.verify_graph(graph,'MXP','FIA',False,transformed=False):return 1
    output.parent.mkdir(parents=True,exist_ok=True)
    graph=sdk.transform_graph(graph,sdk.default_ncomp_passes,False)
    with tempfile.TemporaryDirectory() as tmp:
        before=Path(tmp)/'sdk-before.json';after=Path(tmp)/'sdk-after.json'
        before.write_text(json.dumps(graph))
        binary=a.rewrite_binary.resolve()
        output.with_suffix('.layout-provenance.json').write_text(json.dumps({
            'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()
        },indent=2)+'\n')
        subprocess.run([str(binary),'--sdk-json',str(before),str(after),'--report',str(output.with_suffix('.layout.json'))],check=True)
        graph=json.loads(after.read_text())
        from sdk_serialization import compact_tensor_slots
        graph=compact_tensor_slots(graph)
        if not sdk.verify_graph(graph,'MXP','FIA',False,transformed=True):return 1
        optimized=sdk.save_graph('legalized.tflite',original_dir,graph,copy=False)
        # Use the official partitioner/backend with our explicit layout policy.
        # TILE, singleton PACK/UNPACK and ambiguous layout chains must not be
        # fused into unsupported SDK subnodes. Numerical tests stay separate.
        splits,engines=sdk.generate_split_graphs(optimized,tmp,split_every_op=False,
            compression_vbx=target['compression'],size_config=target['hardware'],output_name=str(output.with_suffix('')))
        compiled=sdk.json_load(str(Path(tmp)/(Path(optimized).stem+'.json')))
        plan,changed=layout_partition_plan(compiled['subgraphs'][0]['operators'],compiled['operator_codes'],splits)
        output.with_suffix('.partitions.json').write_text(json.dumps({'original_splits':splits,'split_groups':changed,'partitions':plan},indent=2))
        input_paths=[str(x.resolve()) for x in a.inputs] if a.inputs else None
        if target['compression']=='ucomp':
            inputs,outputs,activations=sdk.get_tflite_io(optimized,input_paths,None,mean=0.,rgb=False,scale=1.,all_tensors=False)
        else:
            _,outputs,_=sdk.get_tflite_io(optimized,input_paths,None,mean=0.,rgb=False,scale=1.,all_tensors=False)
            inputs,_,activations=sdk.get_tflite_io(optimized,input_paths,None,mean=0.,rgb=False,scale=1.,all_tensors=True)
        inputs={name:sdk.transpose_io_to_vnnx(value) for name,value in inputs.items()}
        outputs={name:sdk.transpose_io_to_vnnx(value) for name,value in outputs.items()}
        data=sdk.generate_vnnx_from_json_subgraphs(compiled,plan,activations,target['hardware'],inputs,outputs,
            int(input_paths is not None),tmp,engines,False,compression_vbx=target['compression'])
        output.write_bytes(data)
        subprocess.run(['objcopy','-Ibinary','-Oihex',str(output),str(output.with_suffix('.hex'))],check=True)
    if not output.is_file() or output.stat().st_size==0:raise RuntimeError('SDK did not create a nonempty VNNX artifact')
    print(json.dumps({'status':'success','bytes':output.stat().st_size,'sdk_commit':head}))
    return 0


if __name__=='__main__':raise SystemExit(main())
