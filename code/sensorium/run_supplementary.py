"""Run fixed supplementary diagnostics without changing the primary analysis."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

if __name__ == "__main__":
    from trajectory_model_eval.supplementary_cli import main
    main()
