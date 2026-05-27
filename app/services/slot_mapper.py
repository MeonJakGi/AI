from typing import Any, Dict, List, Optional


LOW_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_FRONT_BAND_PX = 80

SLOT_PADDING_X = 30
SLOT_PADDING_Y = 10

def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _find_slot_for_bbox(
    bbox: Dict[str, int],
    slots: List[Any],
) -> Optional[Dict[str, Any]]:
    """
    bbox의 bottom center 기준으로 slot 매핑.
    """

    bottom_center_x = bbox["x"] + bbox["width"] / 2
    bottom_center_y = bbox["y"] + bbox["height"]

    for slot in slots:
        slot_dict = _to_dict(slot)

        # x1 = slot_dict["x"]
        # y1 = slot_dict["y"]
        # x2 = x1 + slot_dict["width"]
        # y2 = y1 + slot_dict["height"]

        x1 = slot_dict["x"] - SLOT_PADDING_X
        y1 = slot_dict["y"] - SLOT_PADDING_Y
        x2 = slot_dict["x"] + slot_dict["width"] + SLOT_PADDING_X
        y2 = slot_dict["y"] + slot_dict["height"] + SLOT_PADDING_Y

        if x1 <= bottom_center_x <= x2 and y1 <= bottom_center_y <= y2:
            return slot_dict

    return None


def _get_line_y_at_x(points_xy: List[List[float]], x: float) -> Optional[float]:
    """
    points_xy 두 점으로 이루어진 앞턱 윗선에서
    특정 x 위치의 y값을 계산한다.
    """

    if not points_xy or len(points_xy) < 2:
        return None

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    if x2 == x1:
        return float((y1 + y2) / 2)

    ratio = (x - x1) / (x2 - x1)
    return float(y1 + ratio * (y2 - y1))


def _get_front_edge_for_row(
    row_no: int,
    front_edge_points: List[Any],
) -> Optional[Dict[str, Any]]:
    for edge in front_edge_points:
        edge_dict = _to_dict(edge)

        if edge_dict.get("row_no") == row_no:
            return edge_dict

    return None


def _get_front_y_for_bbox(
    bbox: Dict[str, int],
    row_no: int,
    front_edge_points: List[Any],
) -> Optional[float]:
    edge = _get_front_edge_for_row(
        row_no=row_no,
        front_edge_points=front_edge_points,
    )

    if edge is None:
        return None

    center_x = bbox["x"] + bbox["width"] / 2

    if edge.get("points_xy"):
        return _get_line_y_at_x(edge["points_xy"], center_x)

    if edge.get("front_y") is not None:
        return float(edge["front_y"])

    if edge.get("polygon"):
        # 윗선 기준이므로 polygon의 가장 위쪽 y 사용
        ys = [point[1] for point in edge["polygon"]]
        return float(min(ys))

    return None


def _get_front_band_px(
    row_no: int,
    front_edge_points: List[Any],
) -> int:
    edge = _get_front_edge_for_row(
        row_no=row_no,
        front_edge_points=front_edge_points,
    )

    if edge is None:
        return DEFAULT_FRONT_BAND_PX

    return int(edge.get("front_band_px") or DEFAULT_FRONT_BAND_PX)


def _decide_depth_position(
    bbox: Dict[str, int],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> str:
    """
    FRONT/BACK 판단.

    기준:
    - 상품 bbox의 bottom_y가 앞턱 윗선 y - front_band_px 이상이면 FRONT
    - 그보다 위에 있으면 BACK
    """

    if row_no is None:
        return "UNKNOWN"

    front_y = _get_front_y_for_bbox(
        bbox=bbox,
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if front_y is None:
        return "UNKNOWN"

    front_band_px = _get_front_band_px(
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    bottom_y = bbox["y"] + bbox["height"]

    if bottom_y >= front_y - front_band_px:
        return "FRONT"

    return "BACK"


def map_detections_to_slots(
    detections: List[Dict[str, Any]],
    slots: List[Any],
    front_edge_points: List[Any],
) -> List[Dict[str, Any]]:
    """
    YOLO 탐지 결과를 slot에 매핑하고 FRONT/BACK을 계산한다.
    """

    mapped: List[Dict[str, Any]] = []

    for det in detections:
        bbox = det["bbox"]
        slot = _find_slot_for_bbox(bbox, slots)

        if slot is None:
            mapped.append(
                {
                    "slot_id": None,
                    "product_id": det.get("product_id"),
                    "class_id": det["class_id"],
                    "class_name": det.get("class_name"),
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                    "confidence": det["confidence"],
                    "depth_position": "UNKNOWN",
                    "is_misplaced": False,
                    "is_low_confidence": det["confidence"] < LOW_CONFIDENCE_THRESHOLD,
                }
            )
            continue

        expected_class_id = int(slot["class_id"])
        detected_class_id = int(det["class_id"])

        is_misplaced = detected_class_id != expected_class_id

        depth_position = _decide_depth_position(
            bbox=bbox,
            row_no=slot.get("row_no"),
            front_edge_points=front_edge_points,
        )

        mapped.append(
            {
                "slot_id": int(slot["slot_id"]),
                "product_id": det.get("product_id"),
                "class_id": det["class_id"],
                "class_name": det.get("class_name"),
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
                "confidence": det["confidence"],
                "depth_position": depth_position,
                "is_misplaced": is_misplaced,
                "is_low_confidence": det["confidence"] < LOW_CONFIDENCE_THRESHOLD,
            }
        )

    return mapped