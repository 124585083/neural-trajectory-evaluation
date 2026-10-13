"""Storage aliases for the publication adapter and retained historical modules."""
from pathlib import Path
import json
import os
import shutil
import sys

MODULE_ROOT = Path(__file__).resolve().parent.parent


def configured_root(name):
    environment = {'source_pilot': 'MENTAL_PONG_SOURCE_ROOT', 'source_data': 'MENTAL_PONG_DATA_ROOT',
                   'preservation_snapshot': 'MENTAL_PONG_SNAPSHOT_ROOT'}
    if name == 'published_module':
        return MODULE_ROOT
    if os.environ.get(environment.get(name, '')):
        return Path(os.environ[environment[name]]).expanduser().resolve()
    config = Path(os.environ.get('MENTAL_PONG_PATHS', MODULE_ROOT/'configs/paths.local.json'))
    if config.is_file():
        values = json.loads(config.read_text(encoding='utf-8-sig')).get('artifact_roots', {})
        if values.get(name):
            value = Path(values[name]).expanduser()
            return (value if value.is_absolute() else config.parent/value).resolve()
    # Pure mathematical imports remain available without external data.
    # File access fails explicitly at this missing local placeholder.
    return MODULE_ROOT/'external'/name


def resolve_path(value):
    text = str(value)
    if '://' not in text:
        return Path(value)
    alias, relative = text.split('://', 1)
    if alias not in ('source_pilot', 'source_data', 'preservation_snapshot', 'published_module'):
        raise ValueError(f'Unknown artifact alias: {alias}')
    relative = relative.replace('\\', '/')
    if Path(relative).is_absolute() or ':' in relative or '..' in Path(relative).parts:
        raise ValueError('Artifact reference must remain inside its configured root.')
    return configured_root(alias)/relative


def pdf_python():
    return Path(os.environ.get('MENTAL_PONG_PDF_PYTHON', sys.executable))


def pdftoppm():
    return Path(os.environ.get('MENTAL_PONG_PDFTOPPM', shutil.which('pdftoppm') or 'pdftoppm'))
