"""Check an explicit publication candidate and saved-score aggregation read-only.

This command neither trains models nor reruns scientific randomizations. It is
safe to invoke from any working directory. External artifact availability is
reported by the Mental-Pong adapter; absence is distinct from hash failure.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'experiments/05_mental_pong'
MAP_RELATIVE = 'experiments/05_mental_pong/integration/revision_20261003/publication_map.csv'
# Output paths from trajectory_project/publication_reports.py. Template links
# are interpreted where the generator writes the report, not in source storage.
TEMPLATE_PROJECT = 'experiments/05_mental_pong/trajectory_project/'
TEMPLATE_OUTPUTS = {
    'closeout_report': 'closeout_v1/FINAL_REPORT.md',
    'closeout_summary': 'closeout_v1/FINAL_SUMMARY.md',
    'condition_report': 'condition_endpoint_fa_gpfa_v1/REPORT.md',
    'condition_readme': 'condition_endpoint_fa_gpfa_v1/README.md',
    'condition_protocol_diff': 'condition_endpoint_fa_gpfa_v1/protocol_diff.md',
    'mean_cancellation': 'closeout_v1/results/descriptive/mean_cancellation_interpretation.md',
    'posthoc_cases': 'closeout_v1/results/descriptive/fixed_posthoc_case_interpretation.md',
    'future_design': 'closeout_v1/future_design.md',
    'experiment_ledger': 'closeout_v1/experiment_ledger.md',
}
PRIVATE_NAME = re.compile(r'(?i)(?:^|/)(?:PRIVATE_[^/]+|REVISION_REPORT\.md|FINAL_HANDOFF[^/]*|Repository_Review_[^/]+|Codex_Repository_[^/]+)$')
PATH_PATTERNS = {
    'windows_drive': re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]'),
    'unc': re.compile(r'(?<![:\w])(?:\\\\|//)[A-Za-z0-9_.-]+[\\/][A-Za-z0-9_.-]+'),
    'personal_posix': re.compile(r'(?<![\w:])(?:/(?:home|Users)/[^/\s<>]+|/r[o]ot(?:/|\b)|~[/\\])'),
    'environment_dump': re.compile(r'["\'](?:USERPROFILE|APPDATA|LOCALAPPDATA|HOMEPATH|HOME|PATH)["\']\s*:\s*["\'][^"\']*(?:[A-Za-z]:[\\/]|/h[o]me/|/Us[e]rs/)'),
}
PUBLICATION_TEXT_PATTERNS = {
    'private_document_reference': re.compile(r'(?i)\b(?:PRIVATE[_]Codex[^\s"\']*|REVISION_REPORT\.md|Repository_Review_\d[^\s"\']*|FINAL_HANDOFF\.md)'),
    'interaction_narrative': re.compile(r'(?i)(?:awaiting.user.[r]eview|as requested by [t]he user|source user [r]equests|private [i]nterview|mock [i]nterview|assistant.s [a]ssignments|task.batch [c]ompletion)'),
    'handoff_field': re.compile(r'["\'](?:batch_status|review_crosswalk|handoff_authorization|attachment_search)["\']\s*:'),
    'download_copy_name': re.compile(r'(?i)\b[^\s/\\]+\([1-9][0-9]*\)\.(?:md|csv|pdf|docx)\b'),
    'credential_pattern': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_module(root=ROOT):
    spec = importlib.util.spec_from_file_location('publication_scope_manifest',
        Path(root) / 'experiments/05_mental_pong/revision_manifest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_membership(root):
    """Report HEAD and index membership separately; never change either."""
    def call(*args):
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
    try:
        if Path(call('rev-parse', '--show-toplevel').decode().strip()).resolve() != Path(root).resolve():
            raise ValueError('The candidate is inside a different repository.')
        decode = lambda output: {p.decode('utf-8') for p in output.split(b'\0') if p}
        return {'available': True, 'head': call('rev-parse', 'HEAD').decode().strip(),
                'head_paths': decode(call('ls-tree', '-r', '--name-only', '-z', 'HEAD')),
                'index_paths': decode(call('ls-files', '-z')),
                'untracked_paths': decode(call('ls-files', '--others', '--exclude-standard', '-z'))}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {'available': False, 'head': None, 'head_paths': set(),
                'index_paths': set(), 'untracked_paths': set()}


def candidate_inventory(root=ROOT, *, standalone=False):
    """Select only retained manifest members, then check missing and extra files.

    Worktree mode reports HEAD/index/untracked differences without calling them
    published. Standalone mode checks every file in an explicitly copied tree
    and requires no Git metadata or ignored local configuration.
    """
    root = Path(root).resolve()
    module_root = root / 'experiments/05_mental_pong'
    reader = manifest_module(root)
    legacy = rows(module_root / 'integration/migration_map.csv')
    entries = reader.read_publication_map(module_root, legacy)
    if any(not row.get('publication_disposition') for row in entries):
        raise ValueError('The candidate map needs an explicit disposition for every member.')
    kept = {row['destination'] for row in entries if row['publication_disposition'] in reader.PUBLIC_DISPOSITIONS}
    retired = {row['destination'] for row in entries if row['publication_disposition'] not in reader.PUBLIC_DISPOSITIONS}
    git = git_membership(root) if not standalone else {'available': False, 'head': None, 'head_paths': set(), 'index_paths': set(), 'untracked_paths': set()}
    if not standalone and not git['available']:
        raise ValueError('No matching Git checkout. Use --candidate-tree for a standalone candidate copy.')
    observed = ({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
                if standalone else git['head_paths'] | git['index_paths'] | git['untracked_paths'])
    issues, members = [], []
    for row in entries:
        name = row['destination']
        path = root / name
        resolved = path.resolve()
        retained = name in kept
        if retained:
            if not resolved.is_relative_to(root):
                issues.append({'file': name, 'reason': 'symlink escapes candidate root'})
            elif not path.is_file():
                issues.append({'file': name, 'reason': 'required public file missing'})
            elif row.get('change_class') != 'manifest_self' and file_sha(path) != row['publication_sha256']:
                issues.append({'file': name, 'reason': 'publication checksum mismatch'})
            if PRIVATE_NAME.search(name):
                issues.append({'file': name, 'reason': 'private document proposed for publication'})
        elif path.exists() or path.is_symlink():
            issues.append({'file': name, 'reason': 'retired file remains in the candidate worktree'})
        members.append({'path': name, 'HEAD_membership': name in git['head_paths'] if git['available'] else None,
                        'index_membership': name in git['index_paths'] if git['available'] else None,
                        'worktree_present': path.is_file(), 'publication_disposition': row['publication_disposition']})
    for name in sorted(observed - kept - retired):
        issues.append({'file': name, 'reason': 'unexpected file outside explicit public membership'})
    return [root / name for name in sorted(kept)], {
        'mode': 'standalone_candidate' if standalone else 'worktree_candidate',
        'git_identity_available': git['available'], 'HEAD': git['head'],
        'HEAD_files': len(git['head_paths']) if git['available'] else None,
        'index_files': len(git['index_paths']) if git['available'] else None,
        'untracked_nonignored_files': sorted(git['untracked_paths']),
        'retained_files': len(kept), 'intentional_exclusions': sorted(retired),
        'self_hash_exclusion': MAP_RELATIVE, 'members': members, 'issues': issues}


def publication_files():
    """Compatibility wrapper: returns the explicit worktree candidate only."""
    files, _ = candidate_inventory()
    return files


def check_private_ledger(path, files, root=ROOT, candidate_members=None):
    """Optionally check an external review ledger without printing its prose."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.is_relative_to(root):
        raise ValueError('A private review ledger must be outside the repository.')
    if path.suffix.lower() == '.csv':
        ledger = rows(path)
    else:
        ledger = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(ledger, dict):
            ledger = ledger.get('files', ledger.get('rows', []))
    dispositions = {row['path']: row['publication_disposition'] for row in (candidate_members or [])}
    by_path, issues = {}, []
    for row in ledger:
        name = row['path']
        if name in by_path:
            issues.append({'file': name, 'reason': 'duplicate ledger row'})
        by_path[name] = row
    for file in files:
        name = file.relative_to(root).as_posix()
        row = by_path.get(name)
        if not row:
            issues.append({'file': name, 'reason': 'missing private coverage row'})
            continue
        if (row.get('publication_disposition') not in {'KEEP_PUBLIC', 'EDIT_PUBLIC'} or
                (name in dispositions and row.get('publication_disposition') != dispositions[name])):
            issues.append({'file': name, 'reason': 'ledger disposition differs from candidate'})
        if row.get('candidate_hash') != file_sha(file):
            issues.append({'file': name, 'reason': 'ledger candidate hash is stale'})
        if row.get('review_coverage') not in {'full_text_review', 'full_visual_review', 'structured_data_check', 'source_read_only'}:
            issues.append({'file': name, 'reason': 'manual review is incomplete'})
    return {'status': 'failed' if issues else 'passed', 'rows_checked': len(files),
            'issues': issues, 'scope': 'Checks declared coverage and bytes; it cannot establish that a person read the text.'}


def heading_ids(text):
    parser = HTMLReferences()
    parser.feed(text)
    ids = parser.anchors.copy()
    seen = defaultdict(int)
    for line in re.sub(r'```.*?```|~~~.*?~~~', '', text, flags=re.S).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        heading = re.sub(r'<[^>]+>', '', match[1])
        heading = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading)
        name = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        count = seen[name]
        seen[name] += 1
        ids.add(name if count == 0 else f'{name}-{count}')
    return ids


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets, self.anchors = [], set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.targets.append(value)
            if key == 'id' or (tag == 'a' and key == 'name'):
                self.anchors.add(value)


def text_bytes(path):
    """Classify every file by its bytes, including extensionless text."""
    payload = Path(path).read_bytes()
    if payload.startswith((b'\xff\xfe', b'\xfe\xff')):
        return payload.decode('utf-16'), 'UTF-16'
    if b'\0' in payload:
        return None, 'binary_bytes'
    try:
        return payload.decode('utf-8-sig'), 'UTF-8'
    except UnicodeDecodeError:
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.npz', '.npy', '.pt', '.pth', '.pkl', '.zip', '.gz'}:
            return None, 'binary_format'
        raise


def _fixture_exception(relative, line):
    if relative == 'experiments/05_mental_pong/tests/test_integration.py':
        return any(token in line for token in ('C:' + '/private/file', "'Z:" + '/old/', 'original-host'))
    return relative == 'experiments/05_mental_pong/tests/test_publication_scope.py' and '# synthetic-publication-fixture' in line


def link_targets(text, suffix):
    """Read common Markdown links, reference links and HTML href/src values."""
    if suffix == '.ipynb':
        notebook = json.loads(text)
        text = '\n'.join(''.join(cell.get('source', [])) for cell in notebook.get('cells', []) if cell.get('cell_type') == 'markdown')
    text = re.sub(r'```.*?```|~~~.*?~~~', '', text, flags=re.S)
    text = re.sub(r'`[^`\n]*`', '', text)
    parser = HTMLReferences()
    parser.feed(text)
    targets = parser.targets[:]
    if suffix in ('.md', '.ipynb'):
        definitions = {m[1].strip().casefold(): m[2] for m in re.finditer(r'^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)', text, re.M)}
        targets.extend(definitions.values())
        targets.extend(m[1] for m in re.finditer(r'!?\[[^\]\n]*\]\((<[^>]+>|(?:[^()\n]|\([^()]*\))+)\)', text))
        for match in re.finditer(r'!?\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?!\()', text):
            label = (match[2] or match[1]).strip().casefold()
            if label in definitions:
                targets.append(definitions[label])
    return targets


def check_text_and_links(files, root=ROOT):
    root = Path(root).resolve()
    files = [Path(p).absolute() for p in files]
    public = {p.relative_to(root).as_posix() for p in files}
    language_issues, path_issues, broken_links, syntax_issues = [], [], [], []
    encoding_issues, publication_issues, exceptions, enumeration = [], [], [], []
    external_links, template_contexts = [], []
    text_count = links_checked = python_count = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or not path.resolve().is_relative_to(root):
            publication_issues.append({'file': relative, 'reason': 'missing or escaping retained file'})
            continue
        try:
            text, encoding = text_bytes(path)
        except UnicodeError:
            encoding_issues.append({'file': relative, 'reason': 'text could not be decoded'})
            enumeration.append({'file': relative, 'classification': 'encoding_unverified'})
            continue
        enumeration.append({'file': relative, 'classification': 'text' if text is not None else 'binary', 'encoding': encoding})
        if text is None:
            continue
        text_count += 1
        for number, line in enumerate(text.splitlines(), 1):
            if re.search('[\u3400-\u9fff\uf900-\ufaff]', line):
                language_issues.append({'file': relative, 'line': number})
            fixture = _fixture_exception(relative, line)
            if fixture:
                exceptions.append({'file': relative, 'line': number, 'reason': 'documented synthetic rejection fixture'})
                continue
            for kind, pattern in PATH_PATTERNS.items():
                if pattern.search(line):
                    path_issues.append({'file': relative, 'line': number, 'kind': kind})
            for kind, pattern in PUBLICATION_TEXT_PATTERNS.items():
                if pattern.search(line):
                    if relative == '.gitignore' and line.strip() == '/docs/' + 'REVISION_' + 'REPORT.md':
                        exceptions.append({'file': relative, 'line': number, 'reason': 'preventive ignore rule; not evidence of unpublication'})
                        continue
                    publication_issues.append({'file': relative, 'line': number, 'kind': kind})
        if path.suffix == '.py':
            python_count += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                syntax_issues.append({'file': relative, 'line': error.lineno, 'reason': error.msg})
        link_suffix = '.md' if path.name.lower().endswith('.md.in') else path.suffix.lower()
        if link_suffix not in ('.md', '.html', '.htm', '.svg', '.ipynb'):
            continue
        link_document = path
        template_name = path.name[:-6] if path.name.endswith('.md.in') else ''
        if (relative == TEMPLATE_PROJECT + 'report_templates/' + path.name and
                template_name in TEMPLATE_OUTPUTS):
            output_name = TEMPLATE_PROJECT + TEMPLATE_OUTPUTS[template_name]
            link_document = root / output_name
            template_contexts.append({'template': relative, 'generated_output': output_name,
                                      'mapping_source': TEMPLATE_PROJECT + 'publication_reports.py'})
            if output_name not in public or not link_document.is_file():
                broken_links.append({'file': relative, 'target': output_name,
                                     'reason': 'generated template output is not retained public content'})
        for raw_target in link_targets(text, link_suffix):
            target = raw_target.strip().split(' "', 1)[0].strip('<>')
            if target.casefold().startswith('file:'):
                broken_links.append({'file': relative, 'target': target, 'reason': 'local filesystem URL is not a retained public link'})
                continue
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                external_links.append({'file': relative, 'target': target, 'status': 'external_not_fetched'})
                continue
            links_checked += 1
            parts = urlsplit(unquote(target))
            address, fragment = parts.path, parts.fragment
            dest = (root / address.lstrip('/') if address.startswith('/') else link_document.parent / address).resolve() if address else link_document.resolve()
            reason = ''
            if not dest.is_relative_to(root):
                reason = 'target escapes candidate root'
            else:
                name = dest.relative_to(root).as_posix()
                member = name in public
                if dest.is_dir():
                    member = any(p.startswith(name.rstrip('/') + '/') for p in public) if name != '.' else bool(public)
                if not member:
                    reason = 'target is not retained public content'
                elif not dest.exists():
                    reason = 'missing public target'
                elif fragment and dest.suffix.lower() in ('.md', '.html', '.htm', '.svg'):
                    target_text, _ = text_bytes(dest)
                    parser = HTMLReferences(); parser.feed(target_text)
                    anchors = heading_ids(target_text) if dest.suffix.lower() == '.md' else parser.anchors
                    if fragment not in anchors:
                        reason = 'missing heading or explicit anchor'
            if reason:
                broken_links.append({'file': relative, 'target': target, 'reason': reason})
    return {'text_files': text_count, 'python_files_parsed': python_count,
            'relative_links_checked': links_checked, 'language_issues': language_issues,
            'private_path_issues': path_issues, 'broken_links': broken_links,
            'syntax_issues': syntax_issues, 'encoding_issues': encoding_issues,
            'publication_scope_issues': publication_issues, 'enumeration': enumeration,
            'external_links': external_links, 'synthetic_exceptions': exceptions,
            'template_link_contexts': template_contexts,
            'automated_scope': 'CJK, encoding, path-pattern, syntax and membership checks. These do not certify grammar, scientific wording or visual readability.',
            'manual_editorial_review': {'status': 'not_established_by_this_scanner', 'external_ledger_option': '--private-ledger'},
            'excluded': ['Files outside explicit candidate membership', 'Binary visual content (requires separate visual review)'],
            'required_legal_exceptions': [],
            'synthetic_path_exception': 'Only the named rejection-test fixtures and the preventive ignore rule listed above are exempted.'}


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


def check_evidence_index(public_files=None):
    records = rows(ROOT / 'results/study_evidence_index.csv')
    assert len({r['claim_id'] for r in records}) == len(records)
    allowed = {'supported_within_scope', 'not_supported', 'descriptive_observation', 'not_tested'}
    for record in records:
        assert record['status'] in allowed
        for field in ('result_file', 'protocol_file'):
            path = (ROOT / record[field]).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), record[field]
            if public_files is not None:
                assert path in {p.resolve() for p in public_files}, record[field]
    return {'status': 'passed', 'claims': len(records), 'file_references': 2 * len(records)}


def main():
    global ROOT, MODULE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-external', action='store_true',
                        help='Check publication files without resolving external artifact roots.')
    parser.add_argument('--candidate-tree', type=Path,
                        help='Check only an explicit copied candidate tree; Git metadata is not required.')
    parser.add_argument('--private-ledger', type=Path,
                        help='Optionally check an external per-file review ledger without publishing its contents.')
    args = parser.parse_args()
    if args.candidate_tree:
        ROOT = args.candidate_tree.resolve()
        MODULE = ROOT / 'experiments/05_mental_pong'
    failures = []
    try:
        files, candidate = candidate_inventory(ROOT, standalone=bool(args.candidate_tree))
    except (OSError, ValueError, KeyError) as error:
        files = []
        candidate = {'issues': [{'reason': str(error)}]}
    if candidate['issues']:
        failures.append('candidate_membership_or_hashes')
    text = check_text_and_links(files, ROOT)
    for name in ('language_issues', 'private_path_issues', 'broken_links', 'syntax_issues', 'encoding_issues', 'publication_scope_issues'):
        if text[name]:
            failures.append(name)
    ledger = {'status': 'not_requested', 'scope': 'No full editorial review is inferred from automated checks.'}
    if args.private_ledger:
        try:
            ledger = check_private_ledger(args.private_ledger, files, ROOT, candidate.get('members', []))
        except (OSError, ValueError, KeyError) as error:
            ledger = {'status': 'failed', 'reason': str(error)}
        if ledger['status'] == 'failed':
            failures.append('private_coverage_ledger')
    try:
        scores = check_saved_scores()
    except (AssertionError, OSError, KeyError, ValueError) as error:
        scores = {'status': 'failed', 'reason': str(error)}
        failures.append('saved_scores')
    try:
        evidence = check_evidence_index(files)
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
              'publication_candidate': candidate, 'private_ledger_check': ledger,
              'saved_score_checks': scores, 'evidence_index': evidence,
              'artifact_checks': external, 'failed_checks': failures}
    print(json.dumps(result, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
