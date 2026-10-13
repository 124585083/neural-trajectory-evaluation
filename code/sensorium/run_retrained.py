"""Internal retrained_v1 stages; use run.py for the public frozen-cache replay.

Training and calibration require explicitly named stages; neither is a default.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

if __name__ == "__main__":
    from trajectory_model_eval.retrained_cli import main
    main()
