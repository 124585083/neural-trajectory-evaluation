"""Reproduce the closed analysis or verify the sealed delivery."""
from pathlib import Path
import argparse
import subprocess
import sys
from datetime import datetime,timezone

PILOT=Path(__file__).resolve().parents[2]
if __name__ == "__main__": sys.path.insert(0, str(PILOT))
from trajectory_project.closeout_v1.closeout_setup import ROOT, initialize, verify_sources
from trajectory_project.step3_io import read_json, write_json, verify_records, collect_records


def tests():
    command=[sys.executable,'-B','-m','pytest',str(ROOT/'tests'),'-q','-p','no:cacheprovider']
    result=subprocess.run(command,cwd=PILOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (ROOT/'logs/tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    write_json(ROOT/'results/test_execution.json',dict(command=command,exit_code=result.returncode,
        utc=datetime.now(timezone.utc).isoformat(),output_log=str(ROOT/'logs/tests.txt')))
    print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError(result.stderr or 'Closeout tests failed')


def run_all():
    initialize();print('Unchanged sources:',verify_sources(),flush=True)
    from trajectory_project.closeout_v1.replay_a import run as replay
    replay()
    from trajectory_project.closeout_v1.null_condition import run as run_b
    run_b()
    from trajectory_project.closeout_v1.null_endpoint import prepare
    prepare()
    subprocess.run([sys.executable,'-B','-m','trajectory_project.closeout_v1.null_endpoint','--run'],cwd=PILOT,check=True)
    from trajectory_project.closeout_v1.descriptive_audit import main as descriptive
    descriptive()
    tests()
    from trajectory_project.closeout_v1.final_audit import audit,seal
    audit()
    from trajectory_project.closeout_v1.plot_closeout import plot_a,plot_nulls
    plot_a();plot_nulls()
    from trajectory_project.closeout_v1.report_closeout import run as report
    report()
    # Existing reviewed figures are byte-verified by seal; a changed rendering
    # needs an actual visual review rather than an automatically declared pass.
    seal()


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all',action='store_true',help='Run A/B/C, reuse validated complete null runs, report and seal.')
    group.add_argument('--verify',action='store_true',help='Read-only verify delivery and all original source manifests.')
    group.add_argument('--tests',action='store_true',help='Run tests and save their execution record.')
    args=parser.parse_args()
    if args.all:run_all()
    elif args.tests:tests()
    else:
        m=read_json(ROOT/'manifest.json')
        assert m['completed'] and m['status']=='CLOSED_EXPLORATORY_WITH_LIMITATIONS'
        print('Delivery records verified:',verify_records(collect_records(m)),flush=True)
        print('Historical records verified:',verify_sources(),flush=True)
        print(m['status'],flush=True)


if __name__=='__main__':main()
