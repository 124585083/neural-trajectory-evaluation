"""Check publication links, language, syntax and saved-score aggregation read-only.

This command neither trains models nor reruns scientific randomizations. It is
safe to invoke from any working directory. External artifact availability is
reported by the Mental-Pong adapter; absence is distinct from hash failure.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'experiments/05_mental_pong'
TEXT_SUFFIXES = {'.md', '.py', '.js', '.json', '.jsonl', '.csv', '.yaml', '.yml',
                 '.toml', '.cff', '.txt', '.svg', '.ipynb', '.sh', '.ps1'}


def publication_files():
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT)
    return sorted({ROOT / p.decode('utf-8') for p in output.split(b'\0') if p})


def heading_ids(text):
    ids = set(re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)', text))
    seen = defaultdict(int)
    for line in text.splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        name = re.sub(r'[^\w\- ]', '', match[1].lower()).replace(' ', '-')
        count = seen[name]
        seen[name] += 1
        ids.add(name if count == 0 else f'{name}-{count}')
    return ids


def check_text_and_links(files):
    language_issues, path_issues, broken_links, syntax_issues = [], [], [], []
    text_count = links_checked = python_count = 0
    for path in files:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding='utf-8-sig')
        relative = path.relative_to(ROOT).as_posix()
        text_count += 1
        for number, line in enumerate(text.splitlines(), 1):
            if re.search('[\u3400-\u9fff\uf900-\ufaff]', line):
                language_issues.append({'file': relative, 'line': number})
            if re.search(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]', line):
                # This literal tests rejection of absolute paths, not a machine location.
                if relative in ('scripts/verify_integration.py', 'experiments/05_mental_pong/tests/test_integration.py') and 'C:/private/file' in line:
                    continue
                if relative in ('scripts/verify_integration.py', 'experiments/05_mental_pong/tests/test_integration.py') and ("'Z:/old/" in line or 'original-host' in line):
                    continue
                path_issues.append({'file': relative, 'line': number})
        if path.suffix == '.py':
            python_count += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                syntax_issues.append({'file': relative, 'line': error.lineno, 'reason': error.msg})
        if path.suffix != '.md':
            continue
        for match in re.finditer(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', text):
            target = match[1].strip().split(' "', 1)[0].strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            links_checked += 1
            address, _, fragment = unquote(target).partition('#')
            dest = (path.parent / address).resolve() if address else path
            reason = ''
            if not dest.exists():
                reason = 'missing file'
            elif fragment and dest.suffix == '.md' and fragment not in heading_ids(dest.read_text(encoding='utf-8-sig')):
                reason = 'missing heading or explicit anchor'
            if reason:
                broken_links.append({'file': relative, 'target': target, 'reason': reason})
    return {'text_files': text_count, 'python_files_parsed': python_count,
            'relative_links_checked': links_checked, 'language_issues': language_issues,
            'private_path_issues': path_issues, 'broken_links': broken_links,
            'syntax_issues': syntax_issues,
            'excluded': ['Git internals', 'ignored local configuration', 'external immutable source records',
                         'binary figures (separate visual inspection record)'],
            'required_legal_exceptions': [],
            'synthetic_path_exception': 'A synthetic Windows absolute path is used in rejection tests and in this checker.'}


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def finite_mean(values):
    values = [float(v) for v in values if v != '' and math.isfinite(float(v))]
    return statistics.mean(values) if values else float('nan')


def check_saved_scores():
    """Reaggregate saved scores; no neural predictions or labels are estimated."""
    base = MODULE / 'trajectory_project/condition_endpoint_fa_gpfa_v1'
    keys = ('animal', 'representation', 'epoch', 'coordinate')
    groups = defaultdict(list)
    for row in rows(base / 'results/round_self_metrics.csv'):
        groups[tuple(row[k] for k in keys)].append(row)
    main = rows(base / 'results/self_reconstruction_main_table.csv')
    differences = []
    for row in main:
        source = groups[tuple(row[k] for k in keys)]
        assert len(source) == 100
        for metric in ('r_obj', 'r_beh', 'RMSE_obj', 'RMSE_beh', 'Delta_r', 'Delta_RMSE'):
            differences.append(abs(finite_mean([s[metric] for s in source]) - float(row[metric])))
        differences.append(abs(float(row['Delta_r']) - (float(row['r_beh']) - float(row['r_obj']))))
        differences.append(abs(float(row['Delta_RMSE']) - (float(row['RMSE_obj']) - float(row['RMSE_beh']))))
    cross_keys = keys + ('head', 'target')
    cross_groups = defaultdict(list)
    for row in rows(base / 'results/round_cross_2x2.csv'):
        cross_groups[tuple(row[k] for k in cross_keys)].append(row)
    cross = rows(base / 'results/cross_2x2_summary.csv')
    lookup = {}
    for row in cross:
        key = tuple(row[k] for k in cross_keys)
        lookup[key] = row
        source = cross_groups[key]
        assert len(source) == 100
        for metric in ('r', 'RMSE'):
            differences.append(abs(finite_mean([s[metric] for s in source]) - float(row[metric])))
    for row in main:
        key = tuple(row[k] for k in keys)
        for head, target, suffix in [('D_obj', 'objective', 'obj'), ('D_beh', 'behavior', 'beh')]:
            cross_row = lookup[key + (head, target)]
            for metric in ('r', 'RMSE'):
                differences.append(abs(float(row[f'{metric}_{suffix}']) - float(cross_row[metric])))
    summary_text = (MODULE / 'trajectory_project/closeout_v1/FINAL_SUMMARY.md').read_text(encoding='utf-8')
    for row in main:
        if row['epoch'] == 'full' and row['coordinate'] == 'y':
            values = [row['animal'], row['representation']] + [f"{float(row[k]):.4f}" for k in
                ('r_obj', 'r_beh', 'RMSE_obj', 'RMSE_beh', 'Delta_r', 'Delta_RMSE')]
            assert '| ' + ' | '.join(values) + ' |' in summary_text
    splits = json.loads((base / 'configs/condition_splits_100.json').read_text())
    assert len(splits) == 100
    for split in splits:
        train, test = split['train_indices'], split['test_indices']
        assert len(train) == 39 and len(test) == 40
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(79))
    error = max(differences)
    assert error < 1e-10, error
    return {'status': 'passed', 'self_target_rows': len(main), 'cross_target_rows': len(cross),
            'rounds_per_group': 100, 'fixed_splits_checked': 100,
            'maximum_absolute_aggregation_difference': error,
            'english_summary_display_matches_full_precision_table': 'passed',
            'own_target_to_diagonal_identity': 'passed', 'new_model_fitting': False}


def check_evidence_index():
    records = rows(ROOT / 'results/study_evidence_index.csv')
    assert len({r['claim_id'] for r in records}) == len(records)
    allowed = {'supported_within_scope', 'not_supported', 'descriptive_observation', 'not_tested'}
    for record in records:
        assert record['status'] in allowed
        for field in ('result_file', 'protocol_file'):
            path = (ROOT / record[field]).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), record[field]
    return {'status': 'passed', 'claims': len(records), 'file_references': 2 * len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-external', action='store_true',
                        help='Check publication files without resolving external artifact roots.')
    args = parser.parse_args()
    text = check_text_and_links(publication_files())
    failures = []
    for name in ('language_issues', 'private_path_issues', 'broken_links', 'syntax_issues'):
        if text[name]:
            failures.append(name)
    try:
        scores = check_saved_scores()
    except (AssertionError, OSError, KeyError, ValueError) as error:
        scores = {'status': 'failed', 'reason': str(error)}
        failures.append('saved_scores')
    try:
        evidence = check_evidence_index()
    except (AssertionError, OSError, KeyError, ValueError) as error:
        evidence = {'status': 'failed', 'reason': str(error)}
        failures.append('evidence_index')
    external = {'status': 'skipped_by_explicit_option'}
    if not args.skip_external:
        command = [sys.executable, '-B', str(MODULE / 'run.py')]
        execution = subprocess.run(command, cwd=tempfile.gettempdir(), capture_output=True, text=True)
        try:
            external = json.loads(execution.stdout)
        except ValueError:
            external = {'status': 'failed', 'reason': execution.stderr or execution.stdout}
        if execution.returncode or external.get('status') == 'failed':
            failures.append('artifact_verification')
    result = {'status': 'passed' if not failures else 'failed', 'files_written': False,
              'scientific_analysis_rerun': False, 'text_and_links': text,
              'saved_score_checks': scores, 'evidence_index': evidence,
              'artifact_checks': external, 'failed_checks': failures}
    print(json.dumps(result, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
