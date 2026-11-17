"""Separate ambiguous layout groups without breaking adjacent SDK LUT groups."""


def layout_partition_plan(operators, codes, splits, separate_reshapes=False):
    plan, changed = [], []
    for group in splits:
        names = [codes[operators[i]['opcode_index']]['builtin_code'] for i in group]
        isolated = set()
        # A saved plan may still treat a Reshape replaced by singleton Pack as
        # a fused subnode. Pack requires its own SDK main node. Leave adjacent
        # LUT chains intact when replaying a graph rewritten after partitioning.
        for position, index in enumerate(group):
            options = operators[index].get('builtin_options', {})
            # SDK 3.1 implements TILE only as a main node, never a subnode.
            if names[position] == 'TILE' and position > 0:
                isolated.add(position)
            if names[position] == 'PACK' and options.get('values_count') == 1 and options.get('axis') == -4:
                isolated.add(position)
            # A distributed UNPACK consumes a different concat input in each
            # branch. Saved groups must not treat those independent branches
            # as sequential subnodes of the first UNPACK.
            if names[position] == 'UNPACK' and options.get('num') == 1 and options.get('axis') in (0, -4):
                isolated.add(position)
        # SDK 3.1's fused tile mapping can derive a zero output row length when
        # a transpose precedes nearest resize. A standalone transpose keeps the
        # subsequent LUT/resize/arithmetic chain in one consistent layout.
        resizes = [i for i, name in enumerate(names) if name == 'RESIZE_NEAREST_NEIGHBOR']
        isolated.update(i for i, name in enumerate(names)
                        if name == 'TRANSPOSE' and any(j > i for j in resizes))
        if separate_reshapes:
            isolated.update(i for i, name in enumerate(names) if name == 'RESHAPE')
        else:
            for i in range(len(names) - 2):
                if names[i:i + 3] == ['RESHAPE', 'TRANSPOSE', 'RESHAPE']:
                    isolated.update(range(i, i + 3))
        if not isolated or len(group) == 1:
            plan.append(list(group))
            continue
        chunk = []
        for position, index in enumerate(group):
            if position in isolated:
                if chunk:
                    plan.append(chunk)
                    chunk = []
                plan.append([index])
            else:
                chunk.append(index)
        if chunk:
            plan.append(chunk)
        changed.append(list(group))
    assert [i for group in plan for i in group] == [i for group in splits for i in group]
    return plan, changed
