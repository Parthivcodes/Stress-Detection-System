"""Friendly entry point; implementation remains in tools/train_lower_noise_simulation.py."""
from pathlib import Path
import runpy,sys
if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"tools"))
    runpy.run_path(str(root/"tools"/"train_lower_noise_simulation.py"),run_name="__main__")
