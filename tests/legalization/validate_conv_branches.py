"""Numerical and rejection regressions for parallel Conv branch fusion."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def model(nodes, weights, ins, outs):
    m = h.make_model(h.make_graph(nodes, 'regression', ins, outs, weights),
                     opset_imports=[h.make_opsetid('', 17)])
    m.ir_version = 9
    return onnx.shape_inference.infer_shapes(m)


def conv(stride=1, kernels=(3, 1), bias=True, different_input=False):
    rng = np.random.default_rng(7)
    nodes, weights = [], []
    ins = [h.make_tensor_value_info('x', T.FLOAT, [1, 2, 8, 10])]
    if different_input:
        ins.append(h.make_tensor_value_info('z', T.FLOAT, [1, 2, 8, 10]))
    for i, k in enumerate(kernels):
        w = f'w{i}'
        weights.append(nh.from_array(rng.normal(size=(3, 2, k, k)).astype('float32'), w))
        inputs = ['z' if i and different_input else 'x', w]
        if bias:
            weights.append(nh.from_array(rng.normal(size=3).astype('float32'), f'b{i}'))
            inputs.append(f'b{i}')
        nodes.append(h.make_node('Conv', inputs, [f'c{i}'], pads=[k//2]*4,
                                 strides=[stride]*2, name=f'branch{i}'))
    nodes.append(h.make_node('Add', ['c0', 'c1'], ['y']))
    return model(nodes, weights, ins, [h.make_tensor_value_info('y', T.FLOAT,
                          [1, 3, (8-1)//stride+1, (10-1)//stride+1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=Path('build/vbx-rewrite'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/conv-branches-validation'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    records = []

    def check(m, name, expected, exact=False, extra=()):
        src, dst = args.out/f'{name}.onnx', args.out/f'{name}.after.onnx'
        onnx.checker.check_model(m)
        onnx.save(m, src)
        stats = json.loads(subprocess.check_output([str(args.binary.resolve()), str(src), str(dst), *extra], text=True))
        assert stats['rewrites'] == expected, (name, stats)
        onnx.checker.check_model(str(dst))
        if not expected:
            assert m.SerializeToString() == onnx.load(dst).SerializeToString(), name
        else:
            rng = np.random.default_rng(11)
            feeds = {i.name: rng.normal(size=[d.dim_value for d in i.type.tensor_type.shape.dim]).astype('float32') for i in m.graph.input}
            def run(path):
                return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider']).run(None, feeds)
            for a, b in zip(run(src), run(dst)):
                assert np.isfinite(a).all() and np.isfinite(b).all()
                if exact:
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
            again = args.out/f'{name}.again.onnx'
            second = json.loads(subprocess.check_output([str(args.binary.resolve()), str(dst), str(again)], text=True))
            assert second['rewrites'] == 0, (name, second)
            assert onnx.load(dst).SerializeToString() == onnx.load(again).SerializeToString()
        records.append({'case': name, 'rewrites': expected, 'passed': True})

    for stride, kernels, bias in [(1,(3,1),True),(2,(1,3),False),(1,(3,5),True)]:
        check(conv(stride, kernels, bias), f'conv-{stride}-{kernels[0]}-{bias}', 1)
    check(conv(different_input=True), 'different-input', 0)
    for make, prefix in [(conv, 'conv')]:
        m = make()
        intermediate = m.graph.node[0].output[0]
        m.graph.output.append(next(v for v in m.graph.value_info if v.name == intermediate))
        check(m, prefix+'-shared-output', 0)
        m = make(); m.opset_import[0].version = 22
        check(m, prefix+'-future-opset', 0)
        m = make()
        w = m.graph.initializer[0]
        m.graph.input.append(h.make_tensor_value_info(w.name, w.data_type, list(w.dims)))
        check(m, prefix+'-overridable-constant', 0)
    for make, prefix in [(conv, 'conv')]:
        check(make(), prefix+'-channel-budget', 0, extra=('--max-channels', '1'))
        m = make()
        captured = m.graph.node[0].output[0]
        info = next(v for v in m.graph.value_info if v.name == captured)
        branch_output = h.make_tensor_value_info('captured', T.FLOAT,
                            [d.dim_value for d in info.type.tensor_type.shape.dim])
        branch = h.make_graph([h.make_node('Identity', [captured], ['captured'])],
                               'capture', [], [branch_output])
        m.graph.initializer.append(nh.from_array(np.array(True), 'cond'))
        m.graph.node.append(h.make_node('If', ['cond'], ['capture_result'],
                                       then_branch=branch, else_branch=branch))
        output = h.make_tensor_value_info('capture_result', T.FLOAT,
                            [d.dim_value for d in info.type.tensor_type.shape.dim])
        m.graph.output.append(output)
        check(m, prefix+'-nested-capture', 0)
    for make, prefix in [(conv, 'conv')]:
        m = make()
        m.graph.node.append(h.make_node('ArgMax', ['y'], ['index'], axis=1, keepdims=1))
        dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
        dims[1] = 1
        del m.graph.output[:]
        m.graph.output.append(h.make_tensor_value_info('index', T.INT64, dims))
        check(m, prefix+'-order-sensitive', 0)
    m = conv()
    # A different valid dilation/padding pair must not be merged.
    m.graph.node[0].attribute.extend([h.make_attribute('dilations', [2, 2])])
    next(a for a in m.graph.node[0].attribute if a.name == 'pads').ints[:] = [2]*4
    check(m, 'dilated-conv', 0)
    m = conv()
    w = m.graph.initializer[0]
    data = nh.to_array(w).copy(); data.flat[0] = np.nan
    w.CopyFrom(nh.from_array(data, w.name))
    check(m, 'nonfinite-weight', 0)
    (args.out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(records)} Conv branch regression checks passed')


if __name__ == '__main__':
    main()
