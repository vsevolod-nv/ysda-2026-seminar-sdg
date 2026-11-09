from pathlib import Path

import nbformat
import pytest

from seminar.data import ROOT


@pytest.mark.parametrize('filename', ['02-manual-markup.ipynb', '03-dino-markup.ipynb'])
def test_replay_only_reads_existing_objects(filename):
    notebook = nbformat.read(ROOT / filename, as_version=4)
    assert 'ALLOW_TASK_CHANGES = False' in notebook.cells[1].source
    calls = []
    env = {'ALLOW_TASK_CHANGES': False}

    def request(method, path, **kwargs):
        calls.append((method, path))
        assert method == 'GET', (method, path)
        if path.startswith('projects/'):
            return {'id': path.split('/')[-1]}
        if path.startswith('pools/'):
            return {'id': path.split('/')[-1], 'project_id': env['project_id'],
                    'reward_per_assignment': 1,
                    'defaults': {'default_overlap_for_new_task_suites': 3}}
        raise AssertionError(path)

    def forbidden_upload(*args, **kwargs):
        raise AssertionError('Task upload during replay')

    env.update(request=request, upload_tasks=forbidden_upload,
               all_items=lambda *args, **kwargs: [],
               validation=[{'type': 'MUST_PASS_MODERATION', 'active': True, 'blocker': True}])
    selected = [c for c in notebook.cells if c.cell_type == 'code' and any(
        marker in c.source for marker in ('EXISTING_PROJECT_ID =', 'pool = (',
                                         'upload_tasks(pool_id, tasks)', 'poolModeration/',
                                         'f"pools/{pool_id}/open"'))]
    assert len(selected) == 5
    for cell in selected:
        exec(cell.source, env)
    assert len(calls) == 2

    # Removing IDs must not accidentally create a project or pool in replay mode.
    source = selected[0].source
    source = source.replace(f'EXISTING_PROJECT_ID = "{env["project_id"]}"', 'EXISTING_PROJECT_ID = None')
    source = source.replace(f'EXISTING_POOL_ID = "{env["pool_id"]}"', 'EXISTING_POOL_ID = None')
    with pytest.raises(AssertionError, match='существующий проект'):
        exec(source, env)
    assert len(calls) == 2
