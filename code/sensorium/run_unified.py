"""Historical unified_v1 entry; its run stage fits GPFA and calibrates controls.

Use run.py for the current frozen-cache replay.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

if __name__ == "__main__":
    from trajectory_model_eval.unified_cli import main
    main()
