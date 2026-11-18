"""Synthetic controls for orchestration; does not stand in for full-model validation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper

ROOT = Path(__file__).resolve().parents[2]


def model(mean):
    node = h.make_node('ReduceMean', ['x'], ['y'], axes=[1], keepdims=1) if mean else h.make_node('Identity', ['x'], ['y'])
    shape = [1, 3, 2, 2]
    graph = h.make_graph([node], 'runner-control', [h.make_tensor_value_info('x', T.FLOAT, shape)],
                         [h.make_tensor_value_info('y', T.FLOAT, [1, 1, 2, 2] if mean else shape)],
                         [numpy_helper.from_array(np.ones(4, np.float32), 'unused')])
    result = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
    result.ir_version = 9
    return result


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    before = root / 'control.onnx'
    onnx.save_model(model(True), before, save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.bin', size_threshold=0)
    identity = root / 'identity.onnx'
    onnx.save(model(False), identity)
    saved = {p: p.read_bytes() for p in root.iterdir()}
    for name, second, expected in [('matched', before, True), ('unmatched', identity, False),
                                   ('missing', root / 'missing.onnx', False)]:
        out = root / name
        run = subprocess.run([sys.executable, str(Path(__file__).with_name('validate_models.py')),
                              '--xfeat', str(before), '--depth-anything-v2', str(second),
                              '--samples', '1', '--out', str(out)], capture_output=True, text=True)
        report = json.loads((out / 'results.json').read_text())
        assert (run.returncode == 0) == expected, (name, run.stdout, run.stderr)
        assert report['passed'] == expected and len(report['models']) == 2, report
        assert report['models'][0]['passed'], report
        assert not list(root.glob('*.legalized-*.onnx'))
        for path, data in saved.items():
            assert path.read_bytes() == data, path
        print(name + ': PASS')
