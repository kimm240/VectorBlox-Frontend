"""Standalone exact sampled comparison of integer TFLite rewrites."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

p = argparse.ArgumentParser()
p.add_argument('before', type=Path); p.add_argument('after', type=Path)
p.add_argument('--cases', type=int, default=100)
p.add_argument('--seed', type=int, default=711)
p.add_argument('--random-only', action='store_true', help='Use seeded random inputs for every case, without the initial boundary fills')
p.add_argument('--report', type=Path, required=True)
p.add_argument('--progress', action='store_true', help='Flush per-model invocation phase events')
p.add_argument('--checkpoint', action='store_true', help='Write atomic partial evidence beside the final report')
p.add_argument('--stop-on-mismatch', action='store_true', help='Stop after a failed case; partial evidence cannot pass')
p.add_argument('--builtin-only', action='store_true',
               help='Disable default delegates for an independent CPU kernel comparison')
a = p.parse_args(); assert a.cases > 0
records = []
def event(phase, case=None):
    state = {'phase': phase, 'case': case, 'requested_cases': a.cases,
             'completed_cases': len(records), 'records': records,
             'before': str(a.before), 'after': str(a.after)}
    if a.progress:
        print({k: v for k, v in state.items() if k != 'records'}, flush=True)
    if a.checkpoint:
        path = a.report.with_suffix('.progress.json')
        path.parent.mkdir(exist_ok=True, parents=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)
event('loading')
interpreter_options = {}
if a.builtin_only:
    interpreter_options['experimental_op_resolver_type'] = (
        tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
models = [tf.lite.Interpreter(model_path=str(path), num_threads=1, **interpreter_options)
          for path in [a.before, a.after]]
for m in models:
    m.allocate_tensors()
inputs = [{d['name']: d for d in m.get_input_details()} for m in models]
outputs = [{d['name']: d for d in m.get_output_details()} for m in models]
for details in outputs:
    for d in details.values():
        assert d['dtype'] in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32], (
            'This comparator requires <=32-bit integer outputs; do not cast float outputs to integers')
for pair in [inputs, outputs]:
    assert pair[0].keys() == pair[1].keys(), 'IO names differ'
    for name, x in pair[0].items():
        y = pair[1][name]
        assert np.array_equal(x['shape'], y['shape']) and x['dtype'] == y['dtype']
        for field in ['scales', 'zero_points']:
            assert np.array_equal(x['quantization_parameters'][field], y['quantization_parameters'][field])
        assert x['quantization_parameters']['quantized_dimension'] == y['quantization_parameters']['quantized_dimension']
rng = np.random.default_rng(a.seed)
for case in range(a.cases):
    for name, d in inputs[0].items():
        assert d['dtype'] in [np.int8, np.uint8]
        bounds = np.iinfo(d['dtype'])
        x = rng.integers(bounds.min, bounds.max + 1, size=d['shape'], dtype=d['dtype'])
        if case < 3 and not a.random_only:
            x.fill([bounds.min, bounds.max, d['quantization'][1]][case])
        for m, ds in zip(models, inputs):
            m.set_tensor(ds[name]['index'], x)
    timings = []
    for label, m in zip(['before', 'after'], models):
        event('invoking_' + label, case)
        started = time.perf_counter()
        m.invoke()
        timings.append(time.perf_counter() - started)
    errors = []
    for name, d in outputs[0].items():
        x = models[0].get_tensor(d['index']).astype(np.int64)
        y = models[1].get_tensor(outputs[1][name]['index']).astype(np.int64)
        errors.append(np.abs(x - y))
    records.append({'case': case, 'max_integer_error': max(int(e.max()) for e in errors),
                    'mismatches': sum(int(np.count_nonzero(e)) for e in errors),
                    'before_seconds': timings[0], 'after_seconds': timings[1]})
    event('case_complete', case)
    if a.stop_on_mismatch and records[-1]['mismatches']:
        break
complete = len(records) == a.cases
report = {'cases': len(records), 'requested_cases': a.cases,
          'completed_requested_cases': complete,
          'exact': complete and all(r['mismatches'] == 0 for r in records),
          'builtin_only': a.builtin_only, 'seed': a.seed, 'random_only': a.random_only,
          'max_integer_error': max(r['max_integer_error'] for r in records),
          'before': str(a.before), 'after': str(a.after), 'records': records}
a.report.parent.mkdir(exist_ok=True, parents=True)
a.report.write_text(json.dumps(report, indent=2))
event('complete' if complete else 'stopped_on_mismatch', len(records) - 1)
print({k: v for k, v in report.items() if k != 'records'})
raise SystemExit(0 if report['exact'] else 1)
