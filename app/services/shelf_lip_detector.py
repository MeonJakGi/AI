from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import math
import numpy as np
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "weights" / "shelf_lip_best.pt"

# 앞턱 모델 설정
IMGSZ = 960
CONF_THRESHOLD = 0.12
NMS_IOU_THRESHOLD = 0.50

# 윗선 추출 필터/보정값
MIN_WIDTH_RATIO = 0.06
MIN_AREA_RATIO = 0.00004
MAX_ABS_ANGLE = 22
X_BIN = 6
SMOOTH_WINDOW = 9

_model = None


def _get_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"앞턱 탐지 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    _model = YOLO(str(MODEL_PATH))
    return _model


def _polygon_to_mask(poly, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    if poly is None or len(poly) < 3:
        return mask

    pts = np.array(poly, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    cv2.fillPoly(mask, [pts], 1)

    return mask


def _moving_median(values, window: int = 9):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    if window <= 1:
        return values

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")

    smoothed = []

    for i in range(len(values)):
        smoothed.append(np.median(padded[i:i + window]))

    return np.array(smoothed, dtype=np.float32)


def _robust_fit_line(points, min_points: int = 8) -> Optional[tuple]:
    points = np.asarray(points, dtype=np.float32)

    if len(points) < min_points:
        return None

    x = points[:, 0]
    y = points[:, 1]

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None

    angle = math.degrees(math.atan(slope))

    if abs(angle) > MAX_ABS_ANGLE:
        return None

    return float(slope), float(intercept)


def _extract_upper_line_from_mask(mask: np.ndarray, conf: float) -> Optional[Dict[str, Any]]:
    """
    앞턱 mask에서 윗선을 추출한다.
    - 각 x 구간별 최상단 y를 모음
    - median smoothing
    - 직선 피팅
    """

    h, w = mask.shape[:2]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    area_ratio = len(xs) / (h * w)

    if area_ratio < MIN_AREA_RATIO:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    width = x_max - x_min
    width_ratio = width / w

    if width_ratio < MIN_WIDTH_RATIO:
        return None

    upper_points = []

    for x0 in range(x_min, x_max + 1, X_BIN):
        x1 = x0 + X_BIN
        idx = (xs >= x0) & (xs < x1)

        if idx.sum() == 0:
            continue

        x_mid = (x0 + x1) / 2

        # mask의 윗선이므로 작은 y 쪽을 사용
        y_top = np.percentile(ys[idx], 5)

        upper_points.append([x_mid, y_top])

    if len(upper_points) < 8:
        return None

    upper_points = np.array(upper_points, dtype=np.float32)
    upper_points[:, 1] = _moving_median(
        upper_points[:, 1],
        window=SMOOTH_WINDOW,
    )

    fit = _robust_fit_line(upper_points)

    if fit is None:
        return None

    slope, intercept = fit

    x1 = int(np.min(upper_points[:, 0]))
    x2 = int(np.max(upper_points[:, 0]))
    y1 = int(slope * x1 + intercept)
    y2 = int(slope * x2 + intercept)

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "y_mid": float((y1 + y2) / 2),
        "slope": float(slope),
        "intercept": float(intercept),
        "angle": float(math.degrees(math.atan(slope))),
        "conf": round(float(conf), 2),
        "width_ratio": float(abs(x2 - x1) / w),
    }


def detect_front_edge_points(image: Image.Image) -> List[Dict[str, Any]]:
    """
    shelf.front_edge_points가 null일 때 실행.

    앞턱 segmentation mask에서 윗선을 추출하고,
    row_no별 front_edge_points를 반환한다.
    """

    model = _get_model()

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    h, w = image_np.shape[:2]

    results = model.predict(
        source=image_rgb,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        retina_masks=True,
        verbose=False,
    )

    lines: List[Dict[str, Any]] = []

    for result in results:
        # segmentation 모델 기준
        if result.masks is not None and result.masks.xy is not None:
            polys = result.masks.xy

            if result.boxes is not None and result.boxes.conf is not None:
                confs = result.boxes.conf.detach().cpu().numpy().tolist()
            else:
                confs = [1.0] * len(polys)

            for poly, conf in zip(polys, confs):
                mask = _polygon_to_mask(poly, h, w)
                line = _extract_upper_line_from_mask(mask, conf)

                if line is not None:
                    lines.append(line)

        # 혹시 bbox 모델로 들어온 경우 fallback
        elif result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                line = {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y1),
                    "y_mid": float(y1),
                    "angle": 0.0,
                    "conf": round(conf, 2),
                    "width_ratio": float((x2 - x1) / w),
                }

                lines.append(line)

    if not lines:
        raise ValueError("앞턱 윗선 추출 결과가 없습니다.")

    # 위쪽 선반부터 row_no 부여
    lines = sorted(lines, key=lambda item: item["y_mid"])

    front_edge_points: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines, start=1):
        front_band_px = max(60, int(h * 0.04))

        front_edge_points.append(
            {
                "row_no": idx,
                "points_xy": [
                    [line["x1"], line["y1"]],
                    [line["x2"], line["y2"]],
                ],
                "front_y": None,
                "polygon": None,
                "front_band_px": front_band_px,
                "back_band_px": int(front_band_px * 2),
                "conf": line["conf"],
                "angle": round(line["angle"], 2),
                "width_ratio": round(line["width_ratio"], 4),
            }
        )

    return front_edge_points