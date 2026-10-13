"""Public frozen-cache replay; checking inputs is the default operation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from sensorium_runtime.replay import main
    main()
