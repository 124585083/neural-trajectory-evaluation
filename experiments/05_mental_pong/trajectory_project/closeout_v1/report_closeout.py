"""Render the maintained closeout narrative from completed score files only."""
from pathlib import Path
import argparse
from trajectory_project.closeout_v1.closeout_setup import ROOT
from trajectory_project.publication_reports import render_closeout


def run(output_dir=None, source_root=None):
    """Render reports; an output override leaves the scientific source unchanged."""
    source = Path(source_root) if source_root is not None else ROOT
    destination = Path(output_dir) if output_dir is not None else ROOT
    return render_closeout(source, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional separate report destination.')
    parser.add_argument('--source', type=Path, help='Directory containing completed closeout score tables.')
    args = parser.parse_args()
    run(output_dir=args.output, source_root=args.source)
