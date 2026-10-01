"""Expand native compact trial labels and saved held-out outputs, without fitting."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent
if __name__ == "__main__": sys.path.insert(0, str(ROOT.parent.parent))
from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import load_trial_data
from trajectory_project.trial_endpoint_fa_gpfa_v1.decoding_trial import export_trial_predictions,export_trial_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--animal',choices=['mahler','perle'],required=True)
    p.add_argument('--representation',choices=['FA50','GPFA50'],required=True)
    p.add_argument('--round',type=int,required=True)
    p.add_argument('--trial-id',action='append')
    p.add_argument('--scores',action='store_true')
    p.add_argument('--output-gz',type=Path,required=True)
    args=p.parse_args()
    if not 0<=args.round<100:p.error('--round must be 0..99')
    prefix=f'{args.animal}_{args.representation}_r{args.round:03d}'
    if args.scores:
        result=export_trial_scores(ROOT/'readouts/trial_scores'/f'{prefix}_self_scores.npz',args.output_gz,trial_ids=args.trial_id)
    else:
        data=load_trial_data(ROOT)[args.animal]
        result=export_trial_predictions(data,ROOT/'readouts/predictions'/f'{prefix}_test_predictions.npz',args.output_gz,trial_ids=args.trial_id)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
