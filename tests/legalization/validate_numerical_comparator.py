"""Negative controls: the CI oracle must reject wrong/nonfinite ONNX outputs."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/numerical-comparator'
OUT.mkdir(parents=True, exist_ok=True)


def model(bias=0.0, output='y', shape=(2,)):
    value = np.full(shape, bias, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node('Add', ['x', 'bias'], [output])], 'comparator-control',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, list(shape))],
        [numpy_helper.from_array(value, 'bias')])
    result = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    result.ir_version = 9
    return result


before = OUT / 'before.onnx'
onnx.save(model(), before)
records = []
for name, candidate, expected in [
    ('same', model(), True),
    ('wrong-value', model(1.0), False),
    ('wrong-output-name', model(output='renamed'), False),
    ('wrong-output-shape', model(shape=(1, 2)), False),
    ('nonfinite-output', model(float('nan')), False),
]:
    after, report = OUT / (name + '.onnx'), OUT / (name + '.json')
    onnx.save(candidate, after)
    result = subprocess.run([
        sys.executable, str(ROOT / 'tests/legalization/compare_onnx.py'), str(before), str(after),
        '--samples', '2', '--edge-cases', '--report', str(report)],
        capture_output=True, text=True)
    evidence = json.loads(report.read_text())
    assert (result.returncode == 0) == expected, (name, result.stdout, result.stderr)
    assert evidence['passed'] == expected, (name, evidence)
    records.append(dict(case=name, expected_equivalent=expected, passed=True))
(OUT / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
print('5 comparator controls passed (identity accepted; wrong/nonfinite outputs rejected)')
