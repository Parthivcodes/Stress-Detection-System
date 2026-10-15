"""Run a project script using dependencies installed locally in .runtime.
For the provided machine's embedded Python only; normal Python uses requirements.txt.
"""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / '.runtime'))
sys.path.insert(0, str(root / '.runtime' / 'compat'))
sys.path.insert(0, str(root))
script = sys.argv[1] if len(sys.argv) > 1 else 'server.py'
allowed = {'server.py', 'validate_submission.py', 'verify_project.py'}
if script not in allowed:
    raise SystemExit('Supported scripts: ' + ', '.join(sorted(allowed)))
runpy.run_path(str(root / script), run_name='__main__')
