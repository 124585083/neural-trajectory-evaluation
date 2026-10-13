"""Verify or copy the exact inputs required by the frozen Sensorium replay."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorium_runtime.bundle import main


if __name__ == "__main__":
    main()
