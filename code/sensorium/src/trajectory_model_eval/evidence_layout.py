"""Locate byte-preserved evidence after separating tables from figures."""
from pathlib import Path


def evidence_file(directory, relative):
    """Read original manifest members from the current project layout.

    Original manifests keep their scientific checksums and relative names.
    Only figure placement changes. Missing inputs are never synthesized.
    """
    path = Path(directory) / relative
    if path.is_file():
        return path
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index:index + 3] == ("results", "sensorium", "evidence"):
            figure = Path(*parts[:index], "results", "sensorium", "figures", *parts[index + 3:])
            if figure.is_file():
                return figure
    return path
