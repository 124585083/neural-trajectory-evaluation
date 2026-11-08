"""Select a validation-matched Dynamic checkpoint, export it, and score it."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from trajectory_model_eval.checkpoint_matching_cli import main
    main()
