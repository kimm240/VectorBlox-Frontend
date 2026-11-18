"""Convert supplied verified model ONNX exports and compare outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xfeat', type=Path)
    parser.add_argument('--depth-anything-v2', type=Path)
    parser.add_argument('--repvgg-a0', type=Path)
    parser.add_argument('--repvgg-inputs', type=Path, nargs='+', help='Optional NPZ inputs')
    parser.add_argument('--xfeat-inputs', type=Path, nargs='+', help='Optional NPZ inputs')
    parser.add_argument('--depth-inputs', type=Path, nargs='+', help='Optional NPZ inputs')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/vbx-rewrite')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/legalization-models')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args()
    if not any([args.xfeat, args.depth_anything_v2, args.repvgg_a0]):
        parser.error('Supply at least one verified model ONNX path')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, source, inputs in [
        ('xfeat', args.xfeat, args.xfeat_inputs),
        ('depth-anything-v2', args.depth_anything_v2, args.depth_inputs),
        ('repvgg-a0', args.repvgg_a0, args.repvgg_inputs),
    ]:
        if source is None:
            continue
        result = {'model': name, 'source': str(source.resolve()), 'passed': False}
        temporary = None
        try:
            source = source.resolve(strict=True)
            # Keep the original parent directory so external weight paths still work.
            fd, path = tempfile.mkstemp(prefix=source.stem + '.legalized-', suffix='.onnx', dir=source.parent)
            os.close(fd)
            temporary = Path(path)
            rewrite_report = args.out / (name + '.rewrite.json')
            run = subprocess.run([str(args.binary.resolve()), str(source), str(temporary),
                                  '--report', str(rewrite_report)], capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError('Conversion failed: ' + run.stderr)
            stats = json.loads(rewrite_report.read_text())
            result['rewrites'] = stats['rewrites']
            if stats['rewrites'] < 1:
                raise RuntimeError('No legalization pattern matched this export')
            report = args.out / (name + '.numerical.json')
            command = [sys.executable, str(Path(__file__).with_name('compare_onnx.py')),
                       str(source), str(temporary), '--report', str(report),
                       '--samples', str(args.samples), '--atol', str(args.atol), '--rtol', str(args.rtol)]
            if inputs:
                command += ['--inputs', *map(str, inputs)]
            comparison = subprocess.run(command, capture_output=True, text=True)
            result['numerical_report'] = str(report)
            if comparison.returncode:
                raise RuntimeError('Numerical comparison failed: ' + comparison.stdout + comparison.stderr)
            result['passed'] = json.loads(report.read_text())['passed']
        except Exception as error:
            result['error'] = str(error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        results.append(result)
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    summary = {'passed': all(r['passed'] for r in results), 'models': results}
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Report:', args.out / 'results.json')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
