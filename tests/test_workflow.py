import copy
import csv
import json
from pathlib import Path

import nbformat
import pytest

from seminar.api import annotations_to_boxes, boxes_to_annotations
from seminar.data import ROOT, read_json, write_json
from seminar.results import annotation_results, check_tasks, load_answers, quality_report


def candidate(method="manual"):
    return {"candidate_id": method, "image_id": "image", "image_url": "https://example.org/image.jpg",
            "method": method, "boxes": [], "worker_id": "author-" + method,
            "assignment_id": "a-" + method, "status": "REJECTED", "seconds": 90, "reward": 1}


def vote(worker, answers=("yes", "yes", "yes"), key="manual"):
    return {"assignment_id": "review-" + worker, "task_id": "t-" + key, "worker_id": worker,
            "status": "ACCEPTED", "reward": .5,
            "input_values": {"candidate_id": key},
            "output_values": dict(zip(["complete", "no_extra", "accurate"], answers))}


def test_blind_check_excludes_authors_and_preserves_each_answer():
    tasks = check_tasks([candidate(), candidate("assisted")], "pool")
    assert len(tasks) == 2
    for task in tasks:
        assert set(task["input_values"]) == {"image", "boxes", "candidate_id"}
        assert task["unavailable_for"] == ["author-assisted", "author-manual"]
        assert task["overlap"] == 3


def test_majority_is_for_whole_result_and_needs_three_people():
    votes = [vote("r1", ("no", "yes", "yes")), vote("r2", ("yes", "no", "yes")),
             vote("r3", ("yes", "yes", "no"))]
    detail, _ = quality_report([candidate()], votes)
    assert detail.iloc[0].decision == "needs_correction"
    assert detail.iloc[0].yes_votes == 0
    detail, summary = quality_report([candidate()], [vote("r1"), vote("r2")])
    assert detail.iloc[0].decision == "pending"
    assert summary.iloc[0].checked == 0
    assert summary.iloc[0].pass_rate is None


def test_control_failure_excludes_vote_but_not_payment():
    controls = [{"id": "one", "expected": {"complete": "no", "no_extra": "yes", "accurate": "yes"}}]
    rows = []
    for i in range(3):
        worker = f"r{i}"
        rows.append(vote(worker))
        rows.append(vote(worker, ("yes" if i == 0 else "no", "yes", "yes"), "control_one"))
    detail, summary = quality_report([candidate()], rows, controls=controls)
    assert detail.iloc[0].reviewers == 2
    assert detail.iloc[0].decision == "pending"
    assert summary.iloc[0].review_paid == 1.5
    assert summary.iloc[0].annotation_paid == 0


def test_self_review_and_duplicate_worker_are_rejected():
    with pytest.raises(ValueError, match="сам размечал"):
        quality_report([candidate()], [vote("author-manual")])
    with pytest.raises(ValueError, match="Повторный голос"):
        quality_report([candidate()], [vote("same"), vote("same")])


def test_json_and_tsv_parse_same_result_and_keep_missing_time(tmp_path):
    annotations = boxes_to_annotations([[.1, .2, .5, .6]])
    answer = {"id": "a", "user_id": "w", "status": "REJECTED", "reward": 1,
              "created": "2026-01-01T00:00:00Z", "submitted": "2026-01-01T00:01:30Z",
              "tasks": [{"id": "t", "input_values": {"image_id": "image", "image": "https://example.org/a.jpg"}}],
              "solutions": [{"output_values": {"result": annotations}}]}
    write_json(tmp_path / "answers.json", [answer, answer])
    parsed = annotation_results(tmp_path / "answers.json", "manual")
    assert len(parsed) == 1 and parsed[0]["seconds"] == 90 and parsed[0]["status"] == "REJECTED"
    row = {"ASSIGNMENT:assignment_id": "a", "ASSIGNMENT:task_id": "t", "ASSIGNMENT:worker_id": "w",
           "ASSIGNMENT:status": "REJECTED", "ASSIGNMENT:reward": "1",
           "INPUT:image_id": "image", "INPUT:image": "https://example.org/a.jpg", "OUTPUT:result": json.dumps(annotations)}
    with (tmp_path / "answers.tsv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=row, delimiter="\t")
        writer.writeheader(); writer.writerow(row)
    tsv = annotation_results(tmp_path / "answers.tsv", "manual")[0]
    assert tsv["candidate_id"] == parsed[0]["candidate_id"]
    assert tsv["boxes"] == parsed[0]["boxes"]
    assert tsv["seconds"] is None
    with pytest.raises(ValueError):
        annotations_to_boxes(boxes_to_annotations([[0, 0, 0, 1]]))


@pytest.mark.skipif(not (ROOT / "data/dataset.json").exists() or not (ROOT / "data/dino_predictions.json").exists(),
                    reason="Датасет передаётся отдельно")
def test_ready_data_matches_real_dino_cache():
    data = read_json(ROOT / "data/dataset.json")
    predictions = read_json(ROOT / "data/dino_predictions.json")
    assert len(data) == len({r["image_id"] for r in data}) == 60
    assert {r["image_id"] for r in data} <= predictions["items"].keys()
    assert all((ROOT / r["image_path"]).exists() for r in data)


@pytest.mark.skipif(any(not (ROOT / name).exists() for name in
                       ("04-results.ipynb", "data/dataset.json", "data/dino_predictions.json", "data/check_controls.json")),
                    reason="Нужны локальный датасет и четвёртый ноутбук")
def test_three_notebooks_end_to_end_with_fake_api(tmp_path, monkeypatch):
    """Синтетические ответы существуют только внутри временной директории теста."""
    import seminar.api as api
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setenv("MPLBACKEND", "Agg")
    (tmp_path / "data").mkdir()
    (tmp_path / "interface").symlink_to(ROOT / "interface", target_is_directory=True)
    (tmp_path / "data/images").symlink_to(ROOT / "data/images", target_is_directory=True)
    images = read_json(ROOT / "data/dataset.json")[:2]
    write_json(tmp_path / "data/dataset.json", images)
    write_json(tmp_path / "data/dino_predictions.json", read_json(ROOT / "data/dino_predictions.json"))
    controls = read_json(ROOT / "data/check_controls.json")
    for control in controls:
        control["approved"] = True
    write_json(tmp_path / "data/check_controls.json", controls)
    monkeypatch.chdir(tmp_path)
    pools, tasks, assignments = {}, [], {}
    projects = []
    opened = set()
    moderated = set()

    def fake_request(method, path, **kwargs):
        if method == "GET" and path.endswith("/validate"):
            pool_id = path.split("/")[-2]
            return [{"type": "MUST_PASS_MODERATION", "active": pool_id not in moderated, "blocker": True}]
        if method == "PUT" and "/poolModeration/" in path:
            assert kwargs["json"] == {"status": "READY"}
            moderated.add(path.split("/")[-1])
            return {"status": "READY"}
        if method == "POST" and path == "projects":
            project = {**copy.deepcopy(kwargs["json"]), "id": f"project-{len(projects)}"}
            projects.append(project)
            return project
        if method == "POST" and path == "pools":
            assert kwargs["json"]["defaults"]["default_overlap_for_new_task_suites"] == 3
            pool = {**copy.deepcopy(kwargs["json"]), "id": f"pool-{len(pools)}"}
            pools[pool["id"]] = pool
            return pool
        if method == "POST" and path == "tasks":
            created = {}
            for i, task in enumerate(kwargs["json"]):
                created[str(i)] = {**copy.deepcopy(task), "id": f"task-{len(tasks)}"}
                tasks.append(created[str(i)])
            return {"items": created}
        if method == "GET":
            params = kwargs["params"]
            rows = ([t for t in tasks if t["pool_id"] == params["pool_id"]] if path == "tasks"
                    else assignments.get(params["pool_id"], []))
            rows = sorted([r for r in rows if r["id"] > params.get("id_gt", "")], key=lambda r: r["id"])
            return {"items": rows[:100], "has_more": len(rows) > 100}
        if method == "POST" and path.endswith("/open"):
            pool_id = path.split("/")[1]
            assert pool_id in moderated
            if pool_id in opened:
                return {"id": "operation", "status": "SUCCESS"}
            opened.add(pool_id)
            name = pools[pool_id]["private_name"]
            selected = [t for t in tasks if t["pool_id"] == pool_id]
            gold = [t for t in selected if t.get("known_solutions")]
            output = []
            for task in selected:
                if task.get("known_solutions"):
                    continue
                for worker in range(3):
                    page_tasks = [task]
                    if name == "check":
                        page_tasks.append(gold[worker % len(gold)])
                        solutions = [{"output_values": {"complete": "yes", "no_extra": "yes", "accurate": "yes"}},
                                     {"output_values": page_tasks[1]["known_solutions"][0]["output_values"]}]
                    else:
                        solutions = [{"output_values": {"result": task["input_values"]["initial_boxes"]}}]
                    output.append({"id": f"{pool_id}-a-{len(output)}", "user_id": f"{name}-worker-{worker}",
                                   "status": "ACCEPTED", "reward": 1,
                                   "created": "2026-01-01T00:00:00Z", "submitted": "2026-01-01T00:01:30Z",
                                   "tasks": page_tasks, "solutions": solutions})
            assignments[pool_id] = output
            return {"id": "operation", "status": "SUCCESS"}
        raise AssertionError((method, path))

    monkeypatch.setattr(api, "request", fake_request)
    for path in [ROOT / name for name in ("02-manual-markup.ipynb", "03-dino-markup.ipynb", "04-results.ipynb")]:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        env = {"__name__": "__main__"}
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = cell.source.replace("REWARD = None", "REWARD = 1.0")
                source = source.replace('EXISTING_PROJECT_ID = "10670"', "EXISTING_PROJECT_ID = None")
                source = source.replace('EXISTING_POOL_ID = "6756373"', "EXISTING_POOL_ID = None")
                source = source.replace('EXISTING_PROJECT_ID = "10673"', "EXISTING_PROJECT_ID = None")
                source = source.replace('EXISTING_POOL_ID = "6756505"', "EXISTING_POOL_ID = None")
                exec(compile(source, str(path), "exec"), env)
                env["display"] = lambda *args: None
        plt.close("all")
    assert len(projects) == len(pools) == 3
    passed = read_json(tmp_path / "results/passed.json")
    assert len(passed) == 12
    assert read_json(tmp_path / "results/needs_correction.json") == []
    assert read_json(tmp_path / "results/pending_checks.json") == []
    assisted_tasks = [t for t in tasks if t["pool_id"] == "pool-1"]
    assert all(len(t["unavailable_for"]) == 3 for t in assisted_tasks)
    for t in tasks:
        if t["pool_id"] == "pool-2":
            assert "method" not in t["input_values"] and "worker_id" not in t["input_values"]


def test_pool_does_not_open_while_moderation_is_pending():
    notebook = nbformat.read(ROOT / "02-manual-markup.ipynb", as_version=4)
    cell = next(c for c in notebook.cells if c.cell_type == "code" and "blockers =" in c.source)
    calls = []
    def pending(method, path, **kwargs):
        calls.append(method)
        return [{"type": "MUST_PASS_MODERATION", "active": True, "blocker": True}]
    with pytest.raises(ValueError, match="нельзя открыть"):
        exec(cell.source, {"request": pending, "validation_url": "test/validate", "pool_id": "test"})
    assert calls == ["GET"]


def test_fields_used_by_interfaces_reach_worker():
    from seminar.api import classic_view
    required = {
        "02-manual-markup.ipynb": {"image", "initial_boxes"},
        "03-dino-markup.ipynb": {"image", "initial_boxes"},
        "04-results.ipynb": {"image", "boxes"},
    }
    for filename, fields in required.items():
        if filename == "04-results.ipynb" and not (ROOT / filename).exists():
            continue
        notebook = nbformat.read(ROOT / filename, as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "project_payload = {" in c.source)
        env = {"Path": Path, "classic_view": classic_view}
        exec(cell.source, env)
        spec = env["project_payload"]["task_spec"]["input_spec"]
        # Hidden fields are removed by the platform before JS and templates run.
        visible = {key for key, value in spec.items() if not value.get("hidden", False)}
        assert fields <= visible, (filename, fields - visible)
