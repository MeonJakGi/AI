from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from ultralytics import YOLO

from app.utils.product_class_map import get_product_by_class_id


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "weights" / "best.pt"

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.55
IMAGE_SIZE = 1280


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")


model = YOLO(str(MODEL_PATH))


def run_product_detection(image: Image.Image) -> List[Dict[str, Any]]:
    """
    YOLO 상품 탐지 실행.

    반환 bbox는 원본 이미지 픽셀 기준:
    x, y, width, height
    """

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            product = get_product_by_class_id(class_id)

            if product is None:
                print(f"[경고] class_id {class_id} 매핑 없음")
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "product_id": int(product["product_id"]),
                    "sku_code": product.get("sku_code"),
                    "class_name": product.get("product_name"),
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x": int(round(x1)),
                        "y": int(round(y1)),
                        "width": int(round(x2 - x1)),
                        "height": int(round(y2 - y1)),
                    },
                }
            )

    return detections