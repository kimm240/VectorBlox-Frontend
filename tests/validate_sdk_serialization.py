"""Check tensor compaction retains public, optional and scratch references."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sdk_serialization import compact_tensor_slots

graph = dict(buffers=[{}, {'data': [17]}], signature_defs=[dict(subgraph_index=0, inputs=[dict(name='named', tensor_index=4)])], subgraphs=[dict(tensors=[dict(name=str(i), buffer=i % 2) for i in range(7)], inputs=[0], outputs=[5], operators=[dict(inputs=[0, -1, 2], outputs=[5], intermediates=[6])])])
before = copy.deepcopy(graph)
out = compact_tensor_slots(graph)
g = out['subgraphs'][0]
assert [t['name'] for t in g['tensors']] == ['0', '2', '4', '5', '6']
assert g['inputs'] == [0] and g['outputs'] == [3]
assert g['operators'] == [dict(inputs=[0, -1, 1], outputs=[3], intermediates=[4])]
assert out['signature_defs'][0]['inputs'][0]['tensor_index'] == 2
assert out['buffers'] == before['buffers']
assert compact_tensor_slots(copy.deepcopy(out)) == out
print('Tensor serialization: IO, signatures, optional operands, intermediates, buffers and idempotence passed')
