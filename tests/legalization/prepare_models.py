"""Fetch pinned official models, export ONNX, then test frontend output equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = json.loads((HERE / 'model_sources.json').read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(name, out):
    config = SOURCES[name]
    repo = out / 'repos' / name
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(['git', 'clone', '--no-checkout', config['url'], str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', config['commit']], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['commit'] or subprocess.check_output(
            ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError(f'{repo}: expected clean pinned revision {config["commit"]}')
    weight = (out / 'weights' / config['weight']) if 'weight_url' in config else repo / config['weight']
    if not weight.exists() and 'weight_url' in config:
        weight.parent.mkdir(parents=True, exist_ok=True)
        partial = weight.with_suffix('.download')
        try:
            with urllib.request.urlopen(config['weight_url'], timeout=120) as response, partial.open('wb') as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
            if digest(partial) != config['weight_sha256']:
                raise RuntimeError('Downloaded weight hash mismatch')
            partial.replace(weight)
        finally:
            partial.unlink(missing_ok=True)
    if digest(weight) != config['weight_sha256']:
        raise RuntimeError(f'{name}: weight hash mismatch')
    return repo, weight


def export(name, out):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    if torch.__version__.split('+')[0] != '2.3.0':
        raise RuntimeError('Use the pinned PyTorch 2.3.0 model-test environment')
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = {'model': name, 'passed': False, 'source': SOURCES[name],
            'route': 'official PyTorch forward -> torch.onnx.export (opset 17) -> ONNX shape inference',
            'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}
    try:
        repo, weight = prepare(name, out)
        sys.path.insert(0, str(repo))
        torch.set_num_threads(2)
        torch.manual_seed(20261112)
        if name == 'xfeat':
            from modules.model import XFeatModel
            model = XFeatModel()
            meta['scope'] = 'dense forward; sparse keypoint extraction and matching excluded'
        else:
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            meta['scope'] = 'Small relative-depth forward at 224x224; preprocessing excluded'
        model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
        model.eval()
        size = SOURCES[name]['size']
        x = torch.rand(1, 3, size, size)
        path = folder / 'model.onnx'
        with torch.no_grad():
            outputs = model(x)
            outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
            names = [f'output{i}' for i in range(len(outputs))]
            torch.onnx.export(model, (x,), str(path), input_names=['image'], output_names=names,
                              opset_version=17, do_constant_folding=True)
        # Populate intermediate tensor metadata required by guarded MLIR patterns.
        # Preserve the raw export; shape inference does not lower operators.
        raw = folder / 'exported.onnx'
        path.replace(raw)
        inferred = onnx.shape_inference.infer_shapes(onnx.load(raw), data_prop=True)
        onnx.save(inferred, path)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        actual = session.run(names, {'image': x.numpy()})
        checks = []
        for name_out, expected, got in zip(names, outputs, actual):
            ref = expected.detach().numpy()
            assert ref.shape == got.shape and ref.dtype == got.dtype
            assert np.isfinite(ref).all() and np.isfinite(got).all()
            np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-3)
            checks.append({'output': name_out, 'max_abs_error': float(np.max(np.abs(got - ref)))})
        np.savez(folder / 'sample.npz', image=x.numpy())
        meta.update(passed=True, input_shape=list(x.shape), export_checks=checks,
                    onnx_sha256=digest(path), weight_sha256=digest(weight))
    finally:
        (folder / 'export.json').write_text(json.dumps(meta, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/model-validation')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/vbx-rewrite')
    parser.add_argument('--export-model', choices=SOURCES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.export_model:
        export(args.export_model, out)
        return
    results = []
    for name in SOURCES:
        log = out / (name + '.export.log')
        with log.open('w') as stream:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(out),
                                  '--export-model', name], stdout=stream, stderr=subprocess.STDOUT)
        results.append({'model': name, 'passed': run.returncode == 0, 'log': str(log)})
        print(f"{name} export: {'PASS' if run.returncode == 0 else 'FAIL'} ({log})", flush=True)
    summary = {'exports': results, 'passed': False}
    if all(r['passed'] for r in results):
        command = [sys.executable, str(HERE / 'validate_models.py'),
                   '--xfeat', str(out / 'xfeat/model.onnx'),
                   '--depth-anything-v2', str(out / 'depth-anything-v2/model.onnx'),
                   '--out', str(out / 'comparison'), '--binary', str(args.binary.resolve())]
        run = subprocess.run(command)
        summary['passed'] = run.returncode == 0
        summary['comparison'] = str(out / 'comparison/results.json')
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
