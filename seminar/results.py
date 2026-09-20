"""Чтение выгрузок и сводка независимой проверки."""
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd
from .api import annotations_to_boxes

FINISHED = {"SUBMITTED", "ACCEPTED", "REJECTED"}
QUESTIONS = ("complete", "no_extra", "accurate")


def load_answers(path):
    """API JSON или TSV из кабинета. Одна запись на ответ на задание."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Подложите файл результатов: {path}")
    rows = []
    if path.suffix.lower() == ".tsv":
        csv.field_size_limit(10_000_000)
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if not any(row.values()):
                    continue
                inputs = {k[6:]: v for k, v in row.items() if k.startswith("INPUT:")}
                outputs = {k[7:]: v for k, v in row.items() if k.startswith("OUTPUT:")}
                for key in ["result"]:
                    if outputs.get(key):
                        outputs[key] = json.loads(outputs[key])
                rows.append({
                    "assignment_id": row["ASSIGNMENT:assignment_id"],
                    "task_id": row["ASSIGNMENT:task_id"],
                    "worker_id": row["ASSIGNMENT:worker_id"],
                    "status": row["ASSIGNMENT:status"],
                    "created": row.get("ASSIGNMENT:started"),
                    "submitted": row.get("ASSIGNMENT:submitted"),
                    "reward": float(row["ASSIGNMENT:reward"]) if row.get("ASSIGNMENT:reward") else None,
                    "input_values": inputs, "output_values": outputs,
                })
    else:
        assignments = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(assignments, dict):
            assignments = assignments["items"]
            if isinstance(assignments, dict):
                assignments = list(assignments.values())
        for assignment in assignments:
            if assignment["status"] not in FINISHED:
                continue
            tasks, solutions = assignment["tasks"], assignment["solutions"]
            if len(tasks) != len(solutions):
                raise ValueError("Число заданий и ответов в странице не совпадает")
            for task, solution in zip(tasks, solutions):
                rows.append({
                    "assignment_id": assignment["id"], "task_id": task["id"],
                    "worker_id": assignment["user_id"], "status": assignment["status"],
                    "created": assignment.get("created"), "submitted": assignment.get("submitted"),
                    "reward": float(assignment["reward"]) if assignment.get("reward") is not None else None,
                    "input_values": task["input_values"], "output_values": solution["output_values"],
                })
    finished = {}
    for row in rows:
        if row["status"] in FINISHED:
            key = (row["assignment_id"], row["task_id"])
            if key in finished and finished[key] != row:
                raise ValueError("Две разные версии одного ответа: используйте одну актуальную выгрузку")
            finished[key] = row
    return list(finished.values())


def annotation_results(path, method):
    candidates = []
    seen_pages = set()
    for row in load_answers(path):
        values = row["input_values"]
        if "image_id" not in values or "result" not in row["output_values"]:
            raise ValueError("Нужны результаты проекта из ноутбука 01 или 02")
        if row["assignment_id"] in seen_pages:
            raise ValueError("Для измерения времени требуется одна картинка на страницу")
        seen_pages.add(row["assignment_id"])
        seconds = None
        if row["created"] and row["submitted"]:
            seconds = (pd.Timestamp(row["submitted"]) - pd.Timestamp(row["created"])).total_seconds()
            if seconds <= 0:
                raise ValueError("Некорректное время выполнения")
        key = f"{method}/{row['assignment_id']}/{row['task_id']}"
        candidates.append({
            "candidate_id": hashlib.sha256(key.encode()).hexdigest()[:24],
            "method": method, "image_id": values["image_id"], "image_url": values["image"],
            "boxes": annotations_to_boxes(row["output_values"]["result"]),
            "assignment_id": row["assignment_id"], "worker_id": row["worker_id"],
            "status": row["status"], "seconds": seconds, "reward": row["reward"],
        })
    return candidates


def workers_by_image(candidates):
    result = {}
    for row in candidates:
        result.setdefault(row["image_id"], set()).add(row["worker_id"])
    return {image: sorted(workers) for image, workers in result.items()}


def check_tasks(candidates, pool_id, overlap=3):
    excluded = workers_by_image(candidates)
    return [{
        "pool_id": pool_id, "overlap": overlap,
        "unavailable_for": excluded[row["image_id"]],
        "input_values": {"image": row["image_url"], "boxes": row["boxes"],
                         "candidate_id": row["candidate_id"]},
    } for row in candidates]


def quality_report(candidates, check_answers, overlap=3, controls=None):
    """Голос за результат — все три ответа «Да». Не усреднение ббоксов."""
    expected = {"control_" + c["id"]: c["expected"] for c in (controls or [])}
    control_pages = {}
    for row in check_answers:
        key = row["input_values"].get("candidate_id")
        if key in expected:
            correct = all(row["output_values"].get(q) == expected[key][q] for q in QUESTIONS)
            control_pages[row["assignment_id"]] = control_pages.get(row["assignment_id"], True) and correct
    candidate_map = {r["candidate_id"]: r for r in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("Повторяющиеся результаты разметки")
    excluded = workers_by_image(candidates)
    votes = {key: [] for key in candidate_map}
    review_paid = {key: [] for key in candidate_map}
    seen_workers = set()
    for row in check_answers:
        key = row["input_values"].get("candidate_id")
        if key not in candidate_map:
            if key and key.startswith("control_"):
                continue
            raise ValueError("Ответ проверки не относится к загруженным результатам")
        if row["status"] == "ACCEPTED":
            review_paid[key].append(row["reward"])
        if row["status"] not in {"SUBMITTED", "ACCEPTED"}:
            continue
        if expected and not control_pages.get(row["assignment_id"], False):
            continue
        if row["worker_id"] in excluded[candidate_map[key]["image_id"]]:
            raise ValueError("Разметчик проверяет изображение, которое сам размечал")
        pair = (key, row["worker_id"])
        if pair in seen_workers:
            raise ValueError("Повторный голос проверяющего за один результат")
        seen_workers.add(pair)
        answer = row["output_values"]
        if any(answer.get(q) not in {"yes", "no"} for q in QUESTIONS):
            raise ValueError("Проверка должна содержать три ответа yes/no")
        votes[key].append(answer)
    rows = []
    for candidate in candidates:
        key = candidate["candidate_id"]
        answers = votes[key]
        yes = sum(all(a[q] == "yes" for q in QUESTIONS) for a in answers)
        count = len(answers)
        decision = "pending"
        if count >= overlap:
            decision = "passed" if yes > count / 2 else "needs_correction" if yes < count / 2 else "disputed"
        payments = review_paid[key]
        rows.append({**candidate, "reviewers": count, "yes_votes": yes,
                     "disagreement": 0 < yes < count, "decision": decision,
                     "review_paid": None if any(p is None for p in payments) else sum(payments),
                     **{q + "_no_votes": sum(a[q] == "no" for a in answers) for q in QUESTIONS}})
    detail = pd.DataFrame(rows)
    summaries = []
    for method, group in detail.groupby("method") if not detail.empty else []:
        checked = group.decision != "pending"
        passed = group.decision == "passed"
        accepted = group.loc[group.status == "ACCEPTED", "reward"]
        summaries.append({
            "method": method, "answers": len(group), "images": group.image_id.nunique(),
            "checked": int(checked.sum()), "passed": int(passed.sum()),
            "needs_correction": int((group.decision == "needs_correction").sum()),
            "disputed": int((group.decision == "disputed").sum()),
            "pending": int((~checked).sum()),
            "pass_rate": passed.sum() / checked.sum() if checked.any() else None,
            "answers_with_time": int(group.seconds.notna().sum()),
            "median_seconds": pd.to_numeric(group.seconds).median() if group.seconds.notna().any() else None,
            "annotation_paid": None if accepted.isna().any() else accepted.sum(),
            "review_paid": None if group.review_paid.isna().any() else group.review_paid.sum(),
        })
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["total_paid"] = summary[["annotation_paid", "review_paid"]].sum(axis=1, min_count=2)
    return detail, summary
