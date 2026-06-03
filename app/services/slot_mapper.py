from typing import Any, Dict, List, Optional
import math


LOW_CONFIDENCE_THRESHOLD = 0.4

SLOT_PADDING_X = 30
SLOT_PADDING_Y = 10

# FRONT/BACK 기본 판단 기준
FRONT_DEPTH_SCORE_THRESHOLD = 0.65
FRONT_ABSOLUTE_MIN_SCORE = 0.65
FRONT_GROUP_MARGIN_SCORE = 0.08
FRONT_DISTANCE_GUARD_PX = 55

# 앞턱선이 slot bottom보다 아래로 내려간 경우 보정
FRONT_EDGE_SLOT_BOTTOM_OVERSHOOT_PX = 10
FRONT_EDGE_OVERSHOOT_GROUP_CUT_SCORE = 0.70

# 정면/측면 구분 기준
FRONT_VIEW_EDGE_ANGLE_MAX_DEG = 4.0

# 정면 전용 row cluster 보정
FRONT_VIEW_ROW_CLUSTER_GAP_PX = 25
FRONT_VIEW_MAX_ABOVE_FRONT_EDGE_PX = 75

# stack 보정 기준: 측면에서만 사용
TALL_ITEM_ASPECT_RATIO = 1.3

STACK_MIN_COUNT = 2
STACK_X_CENTER_TOLERANCE_RATIO = 0.45
STACK_X_CENTER_TOLERANCE_MAX_PX = 45
STACK_MAX_VERTICAL_GAP_PX = 55
STACK_MIN_Y_SEPARATION_RATIO = 0.25
STACK_MIN_X_OVERLAP_RATIO = 0.45

STACK_PROMOTE_MIN_BACK_SCORE = 0.75
STACK_PROMOTE_MAX_SCORE_GAP = 0.25

STACK_PROMOTABLE_MIN_HEIGHT_PX = 45
STACK_PROMOTABLE_MAX_HEIGHT_PX = 95
STACK_PROMOTABLE_MIN_ASPECT_RATIO = 0.8
STACK_PROMOTABLE_MAX_ASPECT_RATIO = 3.5


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    raise TypeError(
        f"_to_dict() expected dict or Pydantic model, got {type(obj).__name__}: {obj}"
    )


def _get_depth_point_ratio_for_bbox(bbox: Dict[str, int]) -> float:
    """
    FRONT/BACK 판단 기준점.

    실험 버전:
    - 상품 모양별 기준점 분리 X
    - 모든 상품을 bbox bottom center 기준으로 통일

    즉, depth 판단 기준점:
    px = bbox center x
    py = bbox bottom y
    """
    return 1.0


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

        x1 = slot_dict["x"] - SLOT_PADDING_X
        y1 = slot_dict["y"] - SLOT_PADDING_Y
        x2 = slot_dict["x"] + slot_dict["width"] + SLOT_PADDING_X
        y2 = slot_dict["y"] + slot_dict["height"] + SLOT_PADDING_Y

        if x1 <= bottom_center_x <= x2 and y1 <= bottom_center_y <= y2:
            return slot_dict

    return None


def _get_line_y_at_x(points_xy: List[List[float]], x: float) -> Optional[float]:
    """
    points_xy 두 점으로 이루어진 앞턱선에서
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

        if int(edge_dict.get("row_no")) == int(row_no):
            return edge_dict

    return None


def _get_front_edge_line_for_row(
    row_no: int,
    front_edge_points: List[Any],
) -> Optional[List[List[float]]]:
    """
    row_no에 해당하는 앞턱선 points_xy 반환.
    """

    edge = _get_front_edge_for_row(
        row_no=row_no,
        front_edge_points=front_edge_points,
    )

    if edge is None:
        return None

    points_xy = edge.get("points_xy")

    if not points_xy or len(points_xy) < 2:
        return None

    return points_xy


def _get_edge_angle_deg_from_points(points_xy: List[List[float]]) -> Optional[float]:
    """
    앞턱선 points_xy의 기울기 각도 계산.

    정면: 거의 0도
    측면: 기울어진 각도
    """

    if not points_xy or len(points_xy) < 2:
        return None

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    if abs(dx) < 1:
        return None

    return math.degrees(math.atan2(dy, dx))


def _is_front_view_by_front_edges(front_edge_points: List[Any]) -> bool:
    """
    정면 이미지인지 판단한다.

    기준:
    - 앞턱선들이 거의 수평이면 정면
    - 측면은 앞턱선이 사선이므로 False
    """

    if not front_edge_points:
        return False

    angles = []

    for edge in front_edge_points:
        edge_dict = _to_dict(edge)
        points_xy = edge_dict.get("points_xy")

        angle = _get_edge_angle_deg_from_points(points_xy)

        if angle is None:
            continue

        angles.append(abs(angle))

    if not angles:
        return False

    return max(angles) <= FRONT_VIEW_EDGE_ANGLE_MAX_DEG


def _is_front_edge_below_slot_bottom(
    bbox: Dict[str, int],
    slot_dict: Dict[str, Any],
    front_edge_points: List[Any],
) -> bool:
    """
    앞턱선이 해당 slot의 bottom보다 너무 아래에 잡혔는지 확인한다.

    이런 경우 depth_score 계산이 이상해질 수 있으므로,
    앞턱선 대신 slot bottom 기준으로 score를 계산한다.
    """

    row_no = slot_dict.get("row_no")

    if row_no is None:
        return False

    points_xy = _get_front_edge_line_for_row(
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if points_xy is None:
        return False

    px = bbox["x"] + bbox["width"] / 2
    front_line_y = _get_line_y_at_x(points_xy, px)

    if front_line_y is None:
        return False

    slot_bottom_y = float(slot_dict["y"] + slot_dict["height"])

    return front_line_y > slot_bottom_y + FRONT_EDGE_SLOT_BOTTOM_OVERSHOOT_PX


def _get_signed_distance_to_front_edge(
    bbox: Dict[str, int],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    상품 bbox 기준점과 앞턱선 사이의 signed distance 계산.

    실험 버전 기준점:
    - bbox bottom center
    """

    if row_no is None:
        return None

    points_xy = _get_front_edge_line_for_row(
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if points_xy is None:
        return None

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    px = bbox["x"] + bbox["width"] / 2

    depth_point_ratio = _get_depth_point_ratio_for_bbox(bbox)
    py = bbox["y"] + bbox["height"] * depth_point_ratio

    dx = x2 - x1
    dy = y2 - y1

    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        return None

    signed_distance = (dx * (py - y1) - dy * (px - x1)) / length

    return round(float(signed_distance), 2)


def _get_depth_score_in_slot(
    bbox: Dict[str, int],
    slot_dict: Dict[str, Any],
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    slot 내부에서 상품이 앞쪽에 얼마나 가까운지 score로 계산한다.

    score 해석:
    - 0.0 근처: slot 뒤쪽
    - 1.0 근처: 앞턱선 쪽
    - 1.0 초과: 앞턱선보다 더 앞쪽/아래쪽
    """

    row_no = slot_dict.get("row_no")

    if row_no is None:
        return None

    px = bbox["x"] + bbox["width"] / 2
    back_py = float(slot_dict["y"])

    # 앞턱선이 slot bottom보다 아래에 있으면 slot bottom 기준으로 계산
    if _is_front_edge_below_slot_bottom(
        bbox=bbox,
        slot_dict=slot_dict,
        front_edge_points=front_edge_points,
    ):
        slot_bottom_y = float(slot_dict["y"] + slot_dict["height"])

        depth_point_ratio = _get_depth_point_ratio_for_bbox(bbox)
        obj_py = bbox["y"] + bbox["height"] * depth_point_ratio

        denom = slot_bottom_y - back_py

        if denom <= 1:
            return None

        score = (obj_py - back_py) / denom
        return round(float(score), 3)

    obj_distance = _get_signed_distance_to_front_edge(
        bbox=bbox,
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if obj_distance is None:
        return None

    points_xy = _get_front_edge_line_for_row(
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if points_xy is None:
        return None

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    dx = x2 - x1
    dy = y2 - y1

    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        return None

    back_distance = (dx * (back_py - y1) - dy * (px - x1)) / length

    if back_distance >= -1:
        return None

    score = (obj_distance - back_distance) / (0 - back_distance)

    return round(float(score), 3)


def _get_depth_distance(
    bbox: Dict[str, int],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    디버깅용 distance.
    """

    return _get_signed_distance_to_front_edge(
        bbox=bbox,
        row_no=row_no,
        front_edge_points=front_edge_points,
    )


def _decide_depth_by_front_edge_distance(
    bbox: Dict[str, int],
    slot_dict: Optional[Dict[str, Any]],
    front_edge_points: List[Any],
) -> str:
    """
    1차 FRONT/BACK 판단.
    """

    if slot_dict is None:
        return "UNKNOWN"

    depth_score = _get_depth_score_in_slot(
        bbox=bbox,
        slot_dict=slot_dict,
        front_edge_points=front_edge_points,
    )

    if depth_score is None:
        return "UNKNOWN"

    if depth_score >= FRONT_DEPTH_SCORE_THRESHOLD:
        return "FRONT"

    return "BACK"


def _refine_depth_by_slot_and_class(
    mapped: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    같은 slot + class 안에서 FRONT/BACK을 다시 정한다.

    - 가장 앞쪽 score 그룹만 FRONT 후보
    - 그룹 전체가 앞쪽 기준을 못 넘으면 전부 BACK
    """

    groups: Dict[tuple, List[Dict[str, Any]]] = {}

    for det in mapped:
        slot_id = det.get("slot_id")
        class_id = det.get("class_id")
        depth_score = det.get("depth_score")

        if slot_id is None or class_id is None or depth_score is None:
            continue

        key = (int(slot_id), int(class_id))
        groups.setdefault(key, []).append(det)

    for _, dets in groups.items():
        scores = [
            float(det["depth_score"])
            for det in dets
            if det.get("depth_score") is not None
        ]

        if not scores:
            continue

        max_score = max(scores)

        if max_score < FRONT_ABSOLUTE_MIN_SCORE:
            for det in dets:
                det["depth_position"] = "BACK"
                det["depth_front_cut"] = FRONT_ABSOLUTE_MIN_SCORE
            continue

        has_front_edge_overshoot = any(
            det.get("front_edge_below_slot_bottom")
            for det in dets
        )

        if has_front_edge_overshoot:
            front_cut = FRONT_EDGE_OVERSHOOT_GROUP_CUT_SCORE
        else:
            front_cut = max(
                FRONT_ABSOLUTE_MIN_SCORE,
                max_score - FRONT_GROUP_MARGIN_SCORE,
            )

        for det in dets:
            score = det.get("depth_score")

            if score is None:
                det["depth_position"] = "UNKNOWN"
                continue

            distance = det.get("depth_distance")
            is_overshoot_slot = bool(det.get("front_edge_below_slot_bottom"))

            passes_distance_guard = True

            if not is_overshoot_slot and distance is not None:
                passes_distance_guard = float(distance) >= -FRONT_DISTANCE_GUARD_PX

            if float(score) >= front_cut and passes_distance_guard:
                det["depth_position"] = "FRONT"
            else:
                det["depth_position"] = "BACK"

            det["depth_front_cut"] = round(float(front_cut), 3)
            det["front_distance_guard_passed"] = passes_distance_guard

    return mapped


def _cluster_front_view_rows_by_bottom_y(
    dets: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    정면 이미지에서 같은 slot/class 상품들을 bottom_y 기준으로 row cluster 생성.
    """

    valid_dets = [
        det for det in dets
        if det.get("y") is not None and det.get("height") is not None
    ]

    if not valid_dets:
        return []

    sorted_dets = sorted(
        valid_dets,
        key=lambda det: float(det["y"] + det["height"]),
        reverse=True,
    )

    clusters: List[List[Dict[str, Any]]] = []

    for det in sorted_dets:
        bottom_y = float(det["y"] + det["height"])

        placed = False

        for cluster in clusters:
            cluster_bottoms = [
                float(item["y"] + item["height"])
                for item in cluster
            ]
            cluster_mean_bottom = sum(cluster_bottoms) / len(cluster_bottoms)

            if abs(bottom_y - cluster_mean_bottom) <= FRONT_VIEW_ROW_CLUSTER_GAP_PX:
                cluster.append(det)
                placed = True
                break

        if not placed:
            clusters.append([det])

    return clusters


def _is_close_to_front_edge_in_front_view(
    det: Dict[str, Any],
    front_edge_points: List[Any],
) -> bool:
    """
    정면 이미지에서 front candidate row가 실제 앞턱선 근처인지 확인한다.
    """

    row_no = det.get("row_no")

    if row_no is None:
        return False

    points_xy = _get_front_edge_line_for_row(
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if points_xy is None:
        return False

    bbox_center_x = float(det["x"] + det["width"] / 2)
    bbox_bottom_y = float(det["y"] + det["height"])

    front_line_y = _get_line_y_at_x(points_xy, bbox_center_x)

    if front_line_y is None:
        return False

    distance_above_front_edge = float(front_line_y - bbox_bottom_y)

    return distance_above_front_edge <= FRONT_VIEW_MAX_ABOVE_FRONT_EDGE_PX


def _refine_front_view_depth_by_bottom_row(
    mapped: List[Dict[str, Any]],
    front_edge_points: List[Any],
) -> List[Dict[str, Any]]:
    """
    정면 이미지 전용 보정.

    - 같은 slot/class 안에서 bottom_y row cluster 생성
    - 가장 아래쪽 cluster를 front candidate로 선택
    - front candidate가 앞턱선 근처에 있을 때만 FRONT
    - 앞줄이 없고 뒷줄만 남은 경우는 BACK
    """

    if not _is_front_view_by_front_edges(front_edge_points):
        return mapped

    groups: Dict[tuple, List[Dict[str, Any]]] = {}

    for det in mapped:
        slot_id = det.get("slot_id")
        class_id = det.get("class_id")

        if slot_id is None or class_id is None:
            continue

        key = (int(slot_id), int(class_id))
        groups.setdefault(key, []).append(det)

    for _, dets in groups.items():
        clusters = _cluster_front_view_rows_by_bottom_y(dets)

        if not clusters:
            continue

        front_candidate_cluster = clusters[0]

        cluster_is_real_front = any(
            _is_close_to_front_edge_in_front_view(
                det=det,
                front_edge_points=front_edge_points,
            )
            for det in front_candidate_cluster
        )

        front_ids = {
            id(det)
            for det in front_candidate_cluster
        }

        for det in dets:
            if cluster_is_real_front and id(det) in front_ids:
                det["depth_position"] = "FRONT"
                det["front_view_row_cluster"] = "FRONT_ROW"
                det["front_view_front_edge_guard_passed"] = True
            else:
                det["depth_position"] = "BACK"
                det["front_view_row_cluster"] = "BACK_ROW"
                det["front_view_front_edge_guard_passed"] = False

            det["front_view_row_cluster_refined"] = True

    return mapped


def _x_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["x"])
    ax2 = float(a["x"] + a["width"])
    bx1 = float(b["x"])
    bx2 = float(b["x"] + b["width"])

    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    base = max(1.0, min(float(a["width"]), float(b["width"])))

    return overlap / base


def _center_x(det: Dict[str, Any]) -> float:
    return float(det["x"] + det["width"] / 2)


def _center_y(det: Dict[str, Any]) -> float:
    return float(det["y"] + det["height"] / 2)


def _is_stack_promotable_compact_item(det: Dict[str, Any]) -> bool:
    """
    측면 stack 보정 대상인지 판단.
    """

    width = max(float(det["width"]), 1.0)
    height = max(float(det["height"]), 1.0)
    aspect_ratio = height / width

    if height < STACK_PROMOTABLE_MIN_HEIGHT_PX:
        return False

    if height > STACK_PROMOTABLE_MAX_HEIGHT_PX:
        return False

    if aspect_ratio < STACK_PROMOTABLE_MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > STACK_PROMOTABLE_MAX_ASPECT_RATIO:
        return False

    return True


def _is_vertical_stack_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> bool:
    """
    두 상품 bbox가 같은 세로 stack인지 판단.
    """

    ax = _center_x(a)
    bx = _center_x(b)

    ay = _center_y(a)
    by = _center_y(b)

    min_width = max(1.0, min(float(a["width"]), float(b["width"])))
    min_height = max(1.0, min(float(a["height"]), float(b["height"])))

    x_tolerance = min(
        max(12.0, min_width * STACK_X_CENTER_TOLERANCE_RATIO),
        STACK_X_CENTER_TOLERANCE_MAX_PX,
    )

    if abs(ax - bx) > x_tolerance:
        return False

    overlap_ratio = _x_overlap_ratio(a, b)

    if overlap_ratio < STACK_MIN_X_OVERLAP_RATIO:
        return False

    if abs(ay - by) < min_height * STACK_MIN_Y_SEPARATION_RATIO:
        return False

    upper = a if ay < by else b
    lower = b if ay < by else a

    upper_bottom = float(upper["y"] + upper["height"])
    lower_top = float(lower["y"])

    vertical_gap = lower_top - upper_bottom

    if vertical_gap < -min_height * 0.65:
        return False

    if vertical_gap > STACK_MAX_VERTICAL_GAP_PX:
        return False

    return True


def _promote_stacked_front_items(
    mapped: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    측면 이미지용 stack 보정.

    정면에서는 실행하지 않는다.
    """

    groups: Dict[tuple, List[Dict[str, Any]]] = {}

    for det in mapped:
        slot_id = det.get("slot_id")
        class_id = det.get("class_id")

        if slot_id is None or class_id is None:
            continue

        if not _is_stack_promotable_compact_item(det):
            continue

        key = (int(slot_id), int(class_id))
        groups.setdefault(key, []).append(det)

    for _, dets in groups.items():
        if len(dets) < STACK_MIN_COUNT:
            continue

        visited = set()

        for i, base in enumerate(dets):
            if i in visited:
                continue

            stack_items = [base]
            stack_indices = {i}

            for j, other in enumerate(dets):
                if i == j:
                    continue

                if _is_vertical_stack_pair(base, other):
                    stack_items.append(other)
                    stack_indices.add(j)

            if len(stack_items) < STACK_MIN_COUNT:
                continue

            visited.update(stack_indices)

            has_front = any(
                item.get("depth_position") == "FRONT"
                for item in stack_items
            )

            if not has_front:
                continue

            front_scores = [
                float(item["depth_score"])
                for item in stack_items
                if item.get("depth_position") == "FRONT"
                and item.get("depth_score") is not None
            ]

            if not front_scores:
                continue

            max_front_score = max(front_scores)

            for item in stack_items:
                if item.get("depth_position") != "BACK":
                    continue

                score = item.get("depth_score")

                if score is None:
                    continue

                score = float(score)

                if score < STACK_PROMOTE_MIN_BACK_SCORE:
                    continue

                if max_front_score - score > STACK_PROMOTE_MAX_SCORE_GAP:
                    continue

                item["depth_position"] = "FRONT"
                item["stack_promoted_front"] = True

    return mapped


def map_detections_to_slots(
    detections: List[Dict[str, Any]],
    slots: List[Any],
    front_edge_points: List[Any],
) -> List[Dict[str, Any]]:
    """
    YOLO 탐지 결과를 slot에 매핑하고 FRONT/BACK을 계산한다.

    공통:
    - bbox bottom center로 slot 매핑
    - bbox bottom center로 depth score 계산

    정면:
    - row cluster + front edge guard

    측면:
    - depth score + stack 보정
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
                    "depth_distance": None,
                    "is_misplaced": False,
                    "is_low_confidence": det["confidence"] < LOW_CONFIDENCE_THRESHOLD,
                }
            )
            continue

        expected_class_id = int(slot["class_id"])
        detected_class_id = int(det["class_id"])

        is_misplaced = detected_class_id != expected_class_id

        depth_position = _decide_depth_by_front_edge_distance(
            bbox=bbox,
            slot_dict=slot,
            front_edge_points=front_edge_points,
        )

        depth_score = _get_depth_score_in_slot(
            bbox=bbox,
            slot_dict=slot,
            front_edge_points=front_edge_points,
        )

        depth_distance = _get_depth_distance(
            bbox=bbox,
            row_no=slot.get("row_no"),
            front_edge_points=front_edge_points,
        )

        mapped.append(
            {
                "slot_id": int(slot["slot_id"]),
                "slot_code": slot.get("slot_code"),
                "row_no": int(slot["row_no"]),
                "col_no": int(slot["col_no"]),
                "product_id": det.get("product_id"),
                "class_id": det["class_id"],
                "class_name": det.get("class_name"),
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
                "confidence": det["confidence"],
                "depth_position": depth_position,
                "depth_distance": depth_distance,
                "is_misplaced": is_misplaced,
                "is_low_confidence": det["confidence"] < LOW_CONFIDENCE_THRESHOLD,
                "depth_score": depth_score,
                "front_edge_below_slot_bottom": _is_front_edge_below_slot_bottom(
                    bbox=bbox,
                    slot_dict=slot,
                    front_edge_points=front_edge_points,
                ),
            }
        )

    mapped = _refine_depth_by_slot_and_class(mapped)

    is_front_view = _is_front_view_by_front_edges(front_edge_points)

    if is_front_view:
        mapped = _refine_front_view_depth_by_bottom_row(
            mapped=mapped,
            front_edge_points=front_edge_points,
        )
    else:
        mapped = _promote_stacked_front_items(mapped)
 

    return mapped