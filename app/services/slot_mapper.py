from typing import Any, Dict, List, Optional


LOW_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_FRONT_THRESHOLD = 80


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

        x1 = slot_dict["x"]
        y1 = slot_dict["y"]
        x2 = x1 + slot_dict["width"]
        y2 = y1 + slot_dict["height"]

        if x1 <= bottom_center_x <= x2 and y1 <= bottom_center_y <= y2:
            return slot_dict

    return None


def _get_line_y_at_x(points_xy: List[List[float]], x: float) -> Optional[float]:
    """
    두 점으로 이루어진 앞턱 기준선에서 x 위치의 y값 계산.
    """

    if not points_xy or len(points_xy) < 2:
        return None

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    if x2 == x1:
        return float((y1 + y2) / 2)

    ratio = (x - x1) / (x2 - x1)
    return float(y1 + ratio * (y2 - y1))


def _get_front_y_for_row(
    row_no: int,
    center_x: float,
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    row_no에 맞는 front_y를 찾는다.
    """

    for edge in front_edge_points:
        edge_dict = _to_dict(edge)

        if edge_dict.get("row_no") != row_no:
            continue

        if edge_dict.get("points_xy"):
            return _get_line_y_at_x(edge_dict["points_xy"], center_x)

        if edge_dict.get("front_y") is not None:
            return float(edge_dict["front_y"])

        if edge_dict.get("polygon"):
            ys = [point[1] for point in edge_dict["polygon"]]
            return float(max(ys))

    return None


def _decide_depth_position(
    bbox: Dict[str, int],
    row_no: int,
    front_edge_points: List[Any],
) -> str:
    """
    bbox와 앞턱 좌표를 비교해서 FRONT / BACK / UNKNOWN 판단.

    기준:
    - bbox bottom_y가 front_y 근처면 FRONT
    - front_y보다 위쪽이면 BACK
    """

    center_x = bbox["x"] + bbox["width"] / 2
    bottom_y = bbox["y"] + bbox["height"]

    front_y = _get_front_y_for_row(
        row_no=row_no,
        center_x=center_x,
        front_edge_points=front_edge_points,
    )

    if front_y is None:
        return "UNKNOWN"

    if bottom_y >= front_y - DEFAULT_FRONT_THRESHOLD:
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
                    "product_id": det["product_id"],
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

        expected_product_id = int(slot["product_id"])
        detected_product_id = int(det["product_id"])

        is_misplaced = detected_product_id != expected_product_id

        depth_position = _decide_depth_position(
            bbox=bbox,
            row_no=int(slot["row_no"]),
            front_edge_points=front_edge_points,
        )

        mapped.append(
            {
                "slot_id": slot["slot_id"],
                "product_id": det["product_id"],
                "sku_code": det.get("sku_code"),
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