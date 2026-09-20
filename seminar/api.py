"""Общие запросы и интерфейсы для трёх ноутбуков."""
import os
import requests
from .data import ROOT

BASE_URL = "https://tasks.yandex.ru/api/v1"


def request(method, path, **kwargs):
    token = os.getenv("TOKEN")
    if not token:
        raise ValueError("Добавьте TOKEN в .env и выполните load_dotenv()")
    url = path if path.startswith("https://") else f"{BASE_URL}/{path.lstrip('/')}"
    response = requests.request(method, url, headers={"Authorization": f"OAuth {token}"},
                                timeout=(10, 60), **kwargs)
    if not response.ok:
        print(f"HTTP {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json() if response.content else None


def all_items(path, **params):
    items = []
    params = {**params, "sort": "id", "limit": 100}
    while True:
        page = request("GET", path, params=params)
        items.extend(page["items"])
        if not page.get("has_more"):
            return items
        if not page["items"] or page["items"][-1]["id"] == params.get("id_gt"):
            raise ValueError("API не продвигает страницу результатов")
        params["id_gt"] = page["items"][-1]["id"]


def upload_tasks(pool_id, tasks, key="image_id"):
    existing = {t["input_values"][key]: t for t in all_items("tasks", pool_id=pool_id)}
    missing = []
    for task in tasks:
        old = existing.get(task["input_values"][key])
        if old:
            fields = ["input_values", "known_solutions", "unavailable_for"]
            if any(old.get(k, []) != task.get(k, []) for k in fields):
                raise ValueError("В пуле уже есть другая версия задания. Создайте новый пул.")
        else:
            missing.append(task)
    for start in range(0, len(missing), 10):
        response = request("POST", "tasks", json=missing[start:start+10],
                           params={"async_mode": "false", "open_pool": "false"})
        if response.get("validation_errors"):
            raise ValueError(response["validation_errors"])
    return {"already_uploaded": len(existing), "uploaded_now": len(missing)}


def classic_view(name):
    folder = ROOT / "interface"
    return {
        "type": "classic",
        "markup": (folder / f"{name}.html").read_text(encoding="utf-8"),
        "styles": (folder / f"{name}.css").read_text(encoding="utf-8"),
        "script": (folder / f"{name}.js").read_text(encoding="utf-8"),
        "assets": {"script_urls": ["$TOLOKA_ASSETS/js/toloka-handlebars-templates.js"] +
                   (["$TOLOKA_ASSETS/js/image-annotation.js"] if name == "task" else [])},
        "settings": {"showSkip": True, "showTimer": True, "showTitle": True,
                     "showSubmit": True, "showFullscreen": True, "showInstructions": True},
    }


def boxes_to_annotations(boxes):
    return [{"type": "rectangle", "data": {"p1": {"x": b[0], "y": b[1]},
                                              "p2": {"x": b[2], "y": b[3]}}} for b in boxes]


def annotations_to_boxes(annotations):
    boxes = []
    if not isinstance(annotations, list):
        raise ValueError("Ожидается список прямоугольников")
    for annotation in annotations:
        if annotation["type"] != "rectangle":
            raise ValueError("Ожидаются только прямоугольники")
        p1, p2 = annotation["data"]["p1"], annotation["data"]["p2"]
        box = [min(p1["x"], p2["x"]), min(p1["y"], p2["y"]),
               max(p1["x"], p2["x"]), max(p1["y"], p2["y"])]
        if not all(0 <= x <= 1 for x in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Некорректная рамка")
        boxes.append(box)
    return boxes
