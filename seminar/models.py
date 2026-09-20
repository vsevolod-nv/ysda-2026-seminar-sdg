"""Локальный inference. Входы модели не содержат эталонных рамок."""
import os
import time
from .data import ROOT

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
PROMPTS = [
    "car. van. truck. bus. trailer. bicycle. motorcycle. scooter. tram. train.",
    "stroller. shopping cart. suitcase. wheeled trash bin. delivery robot. wheelchair. forklift.",
]


def setup_cache():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_detector(device="cpu"):
    setup_cache()
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=DINO_REVISION)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID, revision=DINO_REVISION).to(device).eval()
    return processor, model


def detect(image, processor, model, prompts=None, min_score=0.10):
    import torch
    prompts = PROMPTS if prompts is None else prompts
    started = time.perf_counter()
    detections = []
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=min_score, text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        for box, score, label in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), labels):
            normalized = [box[0]/image.width, box[1]/image.height, box[2]/image.width, box[3]/image.height]
            normalized = [max(0., min(1., x)) for x in normalized]
            if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
                detections.append({"box": normalized, "score": score, "phrase": str(label)})
    return {"detections": detections, "inference_seconds": time.perf_counter() - started}
