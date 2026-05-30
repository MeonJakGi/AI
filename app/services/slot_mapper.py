from typing import Any, Dict, List, Optional


LOW_CONFIDENCE_THRESHOLD = 0.4

SLOT_PADDING_X = 30
SLOT_PADDING_Y = 10

# 앞턱선에 대한 수직거리 허용값
# signed_distance >= -이 값이면 FRONT
FRONT_PERPENDICULAR_TOLERANCE_PX = 35
FRONT_PERPENDICULAR_TOLERANCE_RATIO = 0.28
FRONT_PERPENDICULAR_TOLERANCE_MAX_PX = 60
FRONT_ADAPTIVE_TOLERANCE_RATIO = 0.45
FRONT_ADAPTIVE_TOLERANCE_MAX_PX = 80
FRONT_ADAPTIVE_ANGLE_CUTOFF_DEG = 5.0
FRONT_DEPTH_SCORE_THRESHOLD = 0.50
FRONT_ABSOLUTE_MIN_SCORE = 0.50
FRONT_GROUP_MARGIN_SCORE = 0.08


# bbox 모양 기준
TALL_ITEM_ASPECT_RATIO = 1.3
FLAT_ITEM_ASPECT_RATIO = 1.05

# 위아래 stack 보정 기
STACK_MIN_COUNT = 2
STACK_X_CENTER_TOLERANCE_RATIO = 0.35
STACK_X_CENTER_TOLERANCE_MAX_PX = 28
STACK_MAX_VERTICAL_GAP_PX = 45
STACK_MIN_Y_SEPARATION_RATIO = 0.35
STACK_MIN_X_OVERLAP_RATIO = 0.70


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
    앞턱선 수직거리 계산에 사용할 bbox 내부 기준점.

    상품 모양별 기준점:
    - 세로로 긴 상품: 0.85
      너무 아래인 1.0은 뒤쪽 상품도 FRONT가 될 수 있고,
      너무 위인 0.65는 앞줄 프링글스/포키까지 BACK이 될 수 있음.

    - 낮고 넓은 상품: 1.0
      왕뚜껑/컵라면/햇반류는 bottom 기준을 유지해야 앞줄이 살아남음.

    - 일반 상품: 0.93
      칸쵸/포카칩 같은 일반 상품은 bottom보다 살짝 위를 보되,
      기존 0.85처럼 너무 위로 올리지는 않음.
    """

    width = max(float(bbox["width"]), 1.0)
    height = max(float(bbox["height"]), 1.0)

    aspect_ratio = height / width

    # 프링글스/포키처럼 세로로 긴 상품
    if aspect_ratio >= TALL_ITEM_ASPECT_RATIO:
        return 0.85

    # 왕뚜껑/햇반/컵라면처럼 낮고 넓은 상품
    if aspect_ratio <= FLAT_ITEM_ASPECT_RATIO:
        return 1.0

    # 칸쵸/포카칩/일반 봉지류
    return 0.93


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

def _get_front_edge_angle_for_row(
    row_no: int,
    front_edge_points: List[Any],
) -> float:
    """
    row_no에 해당하는 앞턱선 angle을 가져온다.
    angle이 없으면 0도로 본다.
    """

    edge = _get_front_edge_for_row(
        row_no=row_no,
        front_edge_points=front_edge_points,
    )

    if edge is None:
        return 0.0

    angle = edge.get("angle")

    if angle is None:
        return 0.0

    return float(angle)




def _get_front_tolerance_px(
    slot_dict: Optional[Dict[str, Any]],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> float:
    """
    FRONT/BACK 판단용 tolerance.

    핵심:
    - 앞턱선이 거의 수평이면 정면 이미지일 가능성이 높으므로 tolerance를 넓힌다.
    - 앞턱선이 많이 기울어져 있으면 측면/사선 이미지일 가능성이 높으므로 기본 tolerance만 사용한다.
    - 정면/측면 로직을 따로 나누는 게 아니라, angle에 따라 tolerance만 연속적으로 조절한다.
    """

    base_tolerance = float(FRONT_PERPENDICULAR_TOLERANCE_PX)

    if slot_dict is None or row_no is None:
        return base_tolerance

    slot_height = float(slot_dict.get("height") or 0)

    if slot_height <= 0:
        return base_tolerance

    angle = abs(
        _get_front_edge_angle_for_row(
            row_no=int(row_no),
            front_edge_points=front_edge_points,
        )
    )

    # angle이 0에 가까우면 1.0, cutoff 이상이면 0.0
    angle_weight = max(
        0.0,
        min(
            1.0,
            1.0 - angle / FRONT_ADAPTIVE_ANGLE_CUTOFF_DEG,
        ),
    )

    adaptive_tolerance = min(
        FRONT_ADAPTIVE_TOLERANCE_MAX_PX,
        slot_height * FRONT_ADAPTIVE_TOLERANCE_RATIO,
    )

    # 수평에 가까울수록 adaptive tolerance를 더 많이 반영
    final_tolerance = base_tolerance + (
        adaptive_tolerance - base_tolerance
    ) * angle_weight

    return float(max(base_tolerance, final_tolerance))


def _get_front_edge_line_for_row(
    row_no: int,
    front_edge_points: List[Any],
) -> Optional[List[List[float]]]:
    """
    row_no에 해당하는 앞턱선 points_xy를 반환한다.
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


def _get_signed_distance_to_front_edge(
    bbox: Dict[str, int],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    상품 bbox 기준점과 앞턱선 사이의 부호 있는 수직 거리.

    기준점:
    - bbox 내부의 depth point
    - 상품 모양에 따라 bbox의 65~75% 지점을 사용

    반환값:
    - 양수: 앞턱선 아래쪽/앞쪽
    - 0 근처: 앞턱선 근처
    - 음수: 앞턱선 위쪽/뒤쪽
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
    - 1.0 근처: 앞턱선 근처
    - 1.0 초과: 앞턱선보다 더 앞쪽/아래쪽

    계산 방식:
    - 앞턱선을 기준선으로 사용
    - 같은 x 위치에서 slot top을 뒤쪽 기준점으로 사용
    - 상품 bbox bottom center가 그 사이 어디에 있는지 정규화
    """

    row_no = slot_dict.get("row_no")

    if row_no is None:
        return None

    px = bbox["x"] + bbox["width"] / 2

    # 상품 기준점: bbox bottom center
    # obj_py = bbox["y"] + bbox["height"]

    # 뒤쪽 기준점: 같은 x 위치에서 slot의 위쪽 y
    back_py = float(slot_dict["y"])

    obj_distance = _get_signed_distance_to_front_edge(
        bbox={
            "x": bbox["x"],
            "y": bbox["y"],
            "width": bbox["width"],
            "height": bbox["height"],
        },
        row_no=int(row_no),
        front_edge_points=front_edge_points,
    )

    if obj_distance is None:
        return None

    # slot top 위치의 signed distance를 직접 계산
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

    # back_distance는 보통 음수여야 함
    if back_distance >= -1:
        return None

    # 앞턱선 distance는 0으로 보고 정규화
    score = (obj_distance - back_distance) / (0 - back_distance)

    return round(float(score), 3)


def _decide_depth_by_front_edge_distance(
    bbox: Dict[str, int],
    slot_dict: Optional[Dict[str, Any]],
    front_edge_points: List[Any],
) -> str:
    """
    slot 내부 depth_score로 FRONT/BACK 판단.

    depth_score:
    - 0에 가까움: 뒤쪽
    - 1에 가까움: 앞턱선 쪽
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
    같은 slot + 같은 class 안에서 FRONT/BACK을 다시 정한다.

    핵심:
    - 전역 threshold 하나로 판단하지 않음
    - 같은 slot + 같은 class 안에서 가장 앞쪽 상품군만 FRONT
    - 단, 그 그룹 전체가 앞쪽 기준에 못 오면 전부 BACK
      → 뒷줄만 있는 슬롯이 FRONT로 잡히는 문제 방지
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

        # 뒷줄만 있는 경우:
        # 제일 앞쪽인 상품조차 절대 기준을 못 넘으면 전부 BACK
        if max_score < FRONT_ABSOLUTE_MIN_SCORE:
            for det in dets:
                det["depth_position"] = "BACK"
                det["depth_front_cut"] = FRONT_ABSOLUTE_MIN_SCORE
            continue

        # 앞줄 후보가 있을 때만, 가장 앞쪽 상품군 근처를 FRONT로 인정
        front_cut = max(
            FRONT_ABSOLUTE_MIN_SCORE,
            max_score - FRONT_GROUP_MARGIN_SCORE,
        )

        for det in dets:
            score = det.get("depth_score")

            if score is None:
                det["depth_position"] = "UNKNOWN"
                continue

            if float(score) >= front_cut:
                det["depth_position"] = "FRONT"
            else:
                det["depth_position"] = "BACK"

            det["depth_front_cut"] = round(float(front_cut), 3)

    return mapped

def _get_depth_distance(
    bbox: Dict[str, int],
    row_no: Optional[int],
    front_edge_points: List[Any],
) -> Optional[float]:
    """
    디버깅용 거리값.
    앞턱선 기준 signed distance를 반환한다.
    """

    return _get_signed_distance_to_front_edge(
        bbox=bbox,
        row_no=row_no,
        front_edge_points=front_edge_points,
    )


def _x_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["x"])
    ax2 = float(a["x"] + a["width"])
    bx1 = float(b["x"])
    bx2 = float(b["x"] + b["width"])

    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    base = max(1.0, min(float(a["width"]), float(b["width"])))

    return overlap / base


def _is_tall_item(det: Dict[str, Any]) -> bool:
    """
    육포/봉지처럼 세로로 긴 상품인지 판단.
    세로로 긴 상품은 stack 승격 대상에서 제외한다.
    """

    width = max(float(det["width"]), 1.0)
    height = max(float(det["height"]), 1.0)

    return height / width >= TALL_ITEM_ASPECT_RATIO


def _center_x(det: Dict[str, Any]) -> float:
    return float(det["x"] + det["width"] / 2)


def _center_y(det: Dict[str, Any]) -> float:
    return float(det["y"] + det["height"] / 2)


def _is_vertical_stack_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> bool:
    """
    두 상품 bbox가 같은 세로 stack인지 판단한다.

    조건:
    - x 중심이 가까워야 함
    - x축 overlap도 충분해야 함
    - y 중심은 충분히 달라야 함
    - 위 상품 bottom과 아래 상품 top 사이 간격이 너무 크면 안 됨
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

    if vertical_gap < -min_height * 0.35:
        return False

    if vertical_gap > STACK_MAX_VERTICAL_GAP_PX:
        return False

    return True


def _promote_stacked_front_items(
    mapped: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    위아래로 쌓인 상품만 FRONT로 승격한다.

    특정 class_id 사용 X.
    특정 상품명 사용 X.

    조건:
    - 같은 slot + 같은 class
    - 2개 이상이 같은 세로열에 있음
    - 그 세로 stack 안에 FRONT가 하나라도 있으면,
      같은 stack의 BACK 상품을 FRONT로 승격

    세로로 긴 상품은 제외한다.
    육포/봉지류처럼 길게 서 있는 상품이 stack으로 오인되는 것을 막기 위함.
    """

    groups: Dict[tuple, List[Dict[str, Any]]] = {}

    for det in mapped:
        slot_id = det.get("slot_id")
        class_id = det.get("class_id")

        if slot_id is None or class_id is None:
            continue

        if _is_tall_item(det):
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

            for item in stack_items:
                if item.get("depth_position") == "BACK":
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
                "depth_score": depth_score
            }
        )

    mapped = _refine_depth_by_slot_and_class(mapped)
    mapped = _promote_stacked_front_items(mapped)

    return mapped