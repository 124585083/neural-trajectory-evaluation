"""Synthetic publication-boundary fixtures; no scientific payloads are created."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = import_file('publication_scope_test_checker', REPOSITORY / 'scripts/verify_integration.py')
manifest = import_file('publication_scope_test_manifest', REPOSITORY / 'experiments/05_mental_pong/revision_manifest.py')


def synthetic_tree(root, files, excluded=()):
    module_root = root / 'experiments/05_mental_pong'
    migration = module_root / 'integration/migration_map.csv'
    migration.parent.mkdir(parents=True)
    migration.write_text('destination,destination_sha256\n', encoding='utf-8')
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    entries = []
    for name in [*files, migration.relative_to(root).as_posix()]:
        entries.append(dict(destination=name, baseline_sha256='', publication_sha256=checker.file_sha(root / name),
                            change_class='new_public_file', reason='Synthetic fixture.', publication_disposition='KEEP_PUBLIC',
                            preserved_original_sha256=''))
    for name in excluded:
        entries.append(dict(destination=name, baseline_sha256='', publication_sha256='',
                            change_class='intentional_retirement', reason='Internal workflow material excluded from distribution.',
                            publication_disposition=manifest.RETIREMENTS[name],
                            preserved_original_sha256='a' * 64))
    entries.append(dict(destination=manifest.MAP_PATH, baseline_sha256='', publication_sha256='', change_class='manifest_self',
                        reason='A recursive self-hash is excluded.', publication_disposition='KEEP_PUBLIC',
                        preserved_original_sha256=''))
    path = root / manifest.MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)
    return {row['destination'] for row in entries if row['publication_disposition'] == 'KEEP_PUBLIC'}


def test_tracked_ignored_private_document_is_rejected(tmp_path, monkeypatch):
    name = 'docs/' + 'REVISION_' + 'REPORT.md'
    members = synthetic_tree(tmp_path, {'README.md': 'Research.\n', '.gitignore': '/docs/' + 'REVISION_' + 'REPORT.md\n', name: 'Synthetic internal handoff.\n'})
    monkeypatch.setattr(checker, 'manifest_module', lambda root: manifest)
    monkeypatch.setattr(checker, 'git_membership', lambda root: dict(available=True, head='synthetic', head_paths=members,
                                                                  index_paths=members, untracked_paths=set()))
    _, result = checker.candidate_inventory(tmp_path)
    assert any(row['file'] == name and 'private document' in row['reason'] for row in result['issues'])
    row = next(row for row in result['members'] if row['path'] == name)
    assert row['HEAD_membership'] and row['index_membership']


@pytest.mark.parametrize('name', ['README.md', 'report.md.in'])
def test_local_only_targets_fail_inline_reference_and_html_links(tmp_path, name):
    public = tmp_path / name
    public.write_text('[inline](local.md)\n[reference][hidden]\n[hidden]: local.md\n<a href="local.md">HTML</a>\n', encoding='utf-8')
    (tmp_path / 'local.md').write_text('Exists locally, but is not retained public content.\n', encoding='utf-8')
    result = checker.check_text_and_links([public], tmp_path)
    assert len(result['broken_links']) >= 3
    assert all('not retained public' in row['reason'] for row in result['broken_links'])


def test_retained_headings_explicit_anchors_and_directory_links(tmp_path):
    doc = tmp_path / 'README.md'
    child = tmp_path / 'docs/note.md'; child.parent.mkdir()
    doc.write_text('[heading](docs/note.md#saved-result)\n[folder](docs/)\n<a href="docs/note.md#named">anchor</a>\n', encoding='utf-8')
    child.write_text('# Saved result\n<div id="named"></div>\n', encoding='utf-8')
    assert not checker.check_text_and_links([doc, child], tmp_path)['broken_links']


def test_template_links_use_the_verified_generated_report_directory(tmp_path):
    project = tmp_path / checker.TEMPLATE_PROJECT
    template = project / 'report_templates/closeout_report.md.in'
    output = project / 'closeout_v1/FINAL_REPORT.md'
    result_file = output.parent / 'results/synthetic.csv'
    template.parent.mkdir(parents=True)
    result_file.parent.mkdir(parents=True)
    template.write_text('[saved result](results/synthetic.csv)\n', encoding='utf-8')
    output.write_text('# Synthetic generated report\n', encoding='utf-8')
    result_file.write_text('value\n1\n', encoding='utf-8')
    result = checker.check_text_and_links([template, output, result_file], tmp_path)
    assert not result['broken_links']
    assert result['template_link_contexts'][0]['generated_output'].endswith('closeout_v1/FINAL_REPORT.md')
    result = checker.check_text_and_links([template, output], tmp_path)
    assert result['broken_links'][0]['reason'] == 'target is not retained public content'
    result_file.unlink()
    assert checker.check_text_and_links([template, output, result_file], tmp_path)['broken_links']


def test_extensionless_text_is_reviewed_and_scientific_failure_is_retained(tmp_path):
    notes = tmp_path / 'NOTICE'; notes.write_text('Scientific scope: \u795e\u7ecf\n', encoding='utf-8')
    record = tmp_path / 'failure.json'; record.write_text('{"raw_preprocessing":"fail"}', encoding='utf-8')
    result = checker.check_text_and_links([notes, record], tmp_path)
    assert result['text_files'] == 2
    assert result['language_issues'][0]['file'] == 'NOTICE'
    assert not result['publication_scope_issues']


def test_filesystem_url_is_not_treated_as_an_external_web_source(tmp_path):
    path = tmp_path / 'README.md'
    path.write_text('[local](file:///synthetic/result.md)\n', encoding='utf-8')  # synthetic-publication-fixture
    result = checker.check_text_and_links([path], tmp_path)
    assert result['broken_links'][0]['reason'].startswith('local filesystem URL')
    assert not result['external_links']


@pytest.mark.parametrize('content', [
    'as requested by the user',  # synthetic-publication-fixture
    'awaiting user review',  # synthetic-publication-fixture
    '{"batch_status":"synthetic"}',  # synthetic-publication-fixture
    'synthetic_attachment(2).md',  # synthetic-publication-fixture
    'PRIVATE_Codex_Synthetic.md',  # synthetic-publication-fixture
])
def test_publication_prose_patterns_detect_visible_synthetic_fixtures(tmp_path, content):
    path = tmp_path / 'README.md'; path.write_text(content, encoding='utf-8')
    assert checker.check_text_and_links([path], tmp_path)['publication_scope_issues']


@pytest.mark.parametrize('content', [
    'C:/Users/Synthetic/notes',  # synthetic-publication-fixture
    r'\\synthetic-server\private-share\notes',  # synthetic-publication-fixture
    '/home/synthetic/notes',  # synthetic-publication-fixture
    '/Users/synthetic/notes',  # synthetic-publication-fixture
    '{"USERPROFILE":"C:/Users/Synthetic"}',  # synthetic-publication-fixture
])
def test_private_path_patterns_are_separate_from_language_findings(tmp_path, content):
    path = tmp_path / 'public.md'; path.write_text(content, encoding='utf-8')
    result = checker.check_text_and_links([path], tmp_path)
    assert result['private_path_issues'] and not result['language_issues']


def test_intentional_retirement_is_absent_from_candidate_and_linkable_set(tmp_path, monkeypatch):
    retired = next(name for name, disposition in manifest.RETIREMENTS.items() if disposition == 'PRIVATE_ONLY')
    synthetic_tree(tmp_path, {'README.md': 'Research.\n'}, [retired])
    monkeypatch.setattr(checker, 'manifest_module', lambda root: manifest)
    files, result = checker.candidate_inventory(tmp_path, standalone=True)
    assert not result['issues']
    assert retired in result['intentional_exclusions']
    assert tmp_path / retired not in files
    unexpected = tmp_path / 'local-only.txt'; unexpected.write_text('not declared', encoding='utf-8')
    _, result = checker.candidate_inventory(tmp_path, standalone=True)
    assert any(row['file'] == 'local-only.txt' and 'unexpected' in row['reason'] for row in result['issues'])


def test_retired_file_cannot_remain_in_tree(tmp_path, monkeypatch):
    retired = next(name for name, disposition in manifest.RETIREMENTS.items() if disposition == 'PRIVATE_ONLY')
    synthetic_tree(tmp_path, {'README.md': 'Research.\n'}, [retired])
    path = tmp_path / retired; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic', encoding='utf-8')
    monkeypatch.setattr(checker, 'manifest_module', lambda root: manifest)
    _, result = checker.candidate_inventory(tmp_path, standalone=True)
    assert any('retired file remains' in row['reason'] for row in result['issues'])


def test_missing_required_file_is_not_accepted_as_retirement(tmp_path, monkeypatch):
    synthetic_tree(tmp_path, {'README.md': 'Research.\n'})
    (tmp_path / 'README.md').unlink()
    monkeypatch.setattr(checker, 'manifest_module', lambda root: manifest)
    _, result = checker.candidate_inventory(tmp_path, standalone=True)
    assert any(row['file'] == 'README.md' and 'missing' in row['reason'] for row in result['issues'])


def test_symlink_cannot_resolve_to_private_sibling(tmp_path):
    root = tmp_path / 'candidate'; root.mkdir()
    outside = tmp_path / 'outside.md'; outside.write_text('synthetic external file', encoding='utf-8')
    link = root / 'link.md'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('This host does not permit a synthetic symlink.')
    doc = root / 'README.md'; doc.write_text('[link](link.md)\n', encoding='utf-8')
    result = checker.check_text_and_links([doc, link], root)
    assert result['publication_scope_issues'] and result['broken_links']


def test_external_coverage_ledger_requires_matching_candidate_bytes(tmp_path):
    root = tmp_path / 'candidate'; root.mkdir()
    doc = root / 'README.md'; doc.write_text('Research.\n', encoding='utf-8')
    ledger = tmp_path / 'private_coverage.json'
    entry = dict(path='README.md', publication_disposition='KEEP_PUBLIC', review_coverage='full_text_review', candidate_hash=checker.file_sha(doc))
    ledger.write_text(json.dumps([entry]), encoding='utf-8')
    assert checker.check_private_ledger(ledger, [doc], root)['status'] == 'passed'
    members = [dict(path='README.md', publication_disposition='EDIT_PUBLIC')]
    assert checker.check_private_ledger(ledger, [doc], root, members)['status'] == 'failed'
    entry['review_coverage'] = 'partial'; ledger.write_text(json.dumps([entry]), encoding='utf-8')
    assert checker.check_private_ledger(ledger, [doc], root)['status'] == 'failed'
    with pytest.raises(ValueError, match='outside'):
        checker.check_private_ledger(root / 'private.json', [doc], root)
