# Neurodeploy SDK

## Branches

- neurodeploy -- neurodeploy
- neurodeploy -- neurodeploy

## Installing dependencies

```bash
python -m pip install --upgrade build
python -m pip install --upgrade twine
```

## Uploading

Note: the version in line 7 of `pyproject.toml` must be updated for every upload.

```bash
rm -rf dist
python -m build
python -m twine upload --repository pypi dist/*
```
