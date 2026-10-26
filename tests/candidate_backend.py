"""Test-only shim: candidate inference code, untouched original assets/API."""
import importlib.util
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_paths
root = Path(os.environ['VALIDATION_PROJECT'])
model_paths.PROJECT_ROOT = root
model_paths.MODEL_ROOT = root / 'models'
spec = importlib.util.spec_from_file_location('original_backend', root / 'deploy/backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
application = backend.application
