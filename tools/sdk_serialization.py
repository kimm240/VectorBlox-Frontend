"""Compact unused tensor slots after C++ rewrites, preserving SDK metadata.

The rewrite IR deliberately retains stable tensor IDs. TFLite's all-tensors
interpreter may expose unallocated removed entries, including empty table slots.
Only serialization indices change here; buffer IDs and operator order do not.
"""
def compact_tensor_slots(graph):
    for subgraph_index, subgraph in enumerate(graph['subgraphs']):
        used = set(subgraph['inputs'] + subgraph['outputs'])
        signatures = [s for s in graph.get('signature_defs', [])
                      if s.get('subgraph_index', 0) == subgraph_index]
        for signature in signatures:
            for key in ('inputs', 'outputs'):
                used.update(e['tensor_index'] for e in signature.get(key, []) or [])
        for op in subgraph['operators']:
            for key in ('inputs', 'outputs', 'intermediates'):
                used.update(i for i in op.get(key, []) if i >= 0)
        tensors = subgraph['tensors']
        if any(i < 0 or i >= len(tensors) for i in used):
            raise ValueError('Invalid tensor reference during serialization')
        remap = {old: new for new, old in enumerate(sorted(used))}
        for key in ('inputs', 'outputs'):
            subgraph[key] = [remap[i] for i in subgraph[key]]
        for op in subgraph['operators']:
            for key in ('inputs', 'outputs', 'intermediates'):
                if key in op:
                    op[key] = [remap[i] if i >= 0 else i for i in op[key]]
        for signature in signatures:
            for key in ('inputs', 'outputs'):
                for entry in signature.get(key, []) or []:
                    entry['tensor_index'] = remap[entry['tensor_index']]
        subgraph['tensors'] = [tensors[i] for i in sorted(used)]
    return graph
