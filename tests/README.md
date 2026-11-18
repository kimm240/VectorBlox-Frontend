# Tests

Install dependencies from the repository root:

```sh
python -m pip install -r tests/requirements.txt
```

- [Legalization tests](legalization/README.md): pattern regressions, ONNX numerical
  comparison, and XFeat / Depth Anything V2 / RepVGG-A0 full-model checks with supplied exports.
- `python tests/validate_dependency_patches.py`: dependency patch integrity and
  idempotence checks.

Tests run separately from normal ONNX conversion. GitHub Actions runs the generated
legalization fixtures and dependency checks; full pretrained model checks are opt-in.
