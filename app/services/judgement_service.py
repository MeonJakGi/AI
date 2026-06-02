from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    raise TypeError(
        f"_to_dict() expected dict or Pydantic model, got {type(obj).__name__}: {obj}"
    )


def _get_min_front_quantity(slot_dict: Dict[str, Any]) -> int:
    min_front_quantity = slot_dict.get("min_front_quantity")

    if min_front_quantity is not None:
        return int(min_front_quantity)

    expected_quantity = int(slot_dict.get("expected_quantity", 0))
    return max(expected_quantity // 2, 1)

def _make_slot_bbox(slot_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    slot 좌표를 crop용 bbox로 변환.
    NEED_REFILL처럼 탐지 bbox가 없는 경우 사용.
    """
    return {
        "x": int(round(float(slot_dict["x"]))),
        "y": int(round(float(slot_dict["y"]))),
        "width": int(round(float(slot_dict["width"]))),
        "height": int(round(float(slot_dict["height"]))),
    }


def _make_detection_bbox(det: Dict[str, Any]) -> Dict[str, int]:
    """
    detection 좌표를 crop용 bbox로 변환.
    현재 slot_mapper.py에서 x, y, width, height를 이미 넘기고 있으므로 이 값을 사용.
    """
    return {
        "x": int(round(float(det["x"]))),
        "y": int(round(float(det["y"]))),
        "width": int(round(float(det["width"]))),
        "height": int(round(float(det["height"]))),
    }

BACK_COVERED_X_OVERLAP_RATIO = 0.35
BACK_COVERED_CENTER_TOLERANCE_RATIO = 0.65
BACK_COVERED_CENTER_TOLERANCE_MAX_PX = 70


def _center_x(det: Dict[str, Any]) -> float:
    return float(det["x"] + det["width"] / 2)


def _x_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["x"])
    ax2 = float(a["x"] + a["width"])
    bx1 = float(b["x"])
    bx2 = float(b["x"] + b["width"])

    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    base = max(1.0, min(float(a["width"]), float(b["width"])))

    return overlap / base


def _is_back_covered_by_front(
    back_det: Dict[str, Any],
    front_det: Dict[str, Any],
) -> bool:
    """
    BACK 상품이 FRONT 상품과 같은 열에 있으면,
    FRONT 상품 뒤에 가려진 것으로 보고
    매대 추정 수량 계산에서 BACK 상품을 별도로 더하지 않는다.
    """

    overlap_ratio = _x_overlap_ratio(back_det, front_det)

    if overlap_ratio >= BACK_COVERED_X_OVERLAP_RATIO:
        return True

    center_distance = abs(_center_x(back_det) - _center_x(front_det))

    min_width = max(
        1.0,
        min(float(back_det["width"]), float(front_det["width"])),
    )

    center_tolerance = min(
        min_width * BACK_COVERED_CENTER_TOLERANCE_RATIO,
        BACK_COVERED_CENTER_TOLERANCE_MAX_PX,
    )

    return center_distance <= center_tolerance


def _count_uncovered_back_quantity(
    expected_product_detections: List[Dict[str, Any]],
) -> int:
    """
    BACK 상품 중 앞에 같은 열 FRONT 상품이 없는 것만 센다.

    - FRONT 상품 1개는 뒤에 가려진 상품까지 포함해 2개로 추정
    - BACK 상품은 앞에 같은 열 FRONT 상품이 없을 때만 1개로 추정
    """

    front_detections = [
        det for det in expected_product_detections
        if det.get("depth_position") == "FRONT"
    ]

    back_detections = [
        det for det in expected_product_detections
        if det.get("depth_position") == "BACK"
    ]

    uncovered_back_count = 0

    for back_det in back_detections:
        covered_by_front = any(
            _is_back_covered_by_front(
                back_det=back_det,
                front_det=front_det,
            )
            for front_det in front_detections
        )

        if not covered_by_front:
            uncovered_back_count += 1

    return uncovered_back_count


def _select_issue_bbox_for_need_check(slot_detections: List[Dict[str, Any]]):
    for det in slot_detections:
        if det.get("is_misplaced", False):
            return (
                _make_detection_bbox(det),
                "MISPLACED_DETECTION",
                round(float(det["confidence"]), 2),
            )

    return None, None, None

def decide_stock_status(
    front_quantity: int,
    min_front_quantity: int,
    warehouse_quantity: int,
    has_misplaced: bool,
) -> Tuple[str, str]:
    """
    Stock.status 판단 로직.

    기준:
    1. 오진열
       → NEED_CHECK

    2. 앞열이 충분함
       front_quantity >= min_front_quantity
       → ENOUGH

    3. 앞열이 부족하지만 창고에 채워넣을 재고가 있음
       warehouse_quantity > 0
       → NEED_REFILL

    4. 앞열이 부족하지만 창고에 채워넣을 재고가 없음
       warehouse_quantity <= 0
       → ORDER_NEEDED

    주의:
    - 발주 필요 리스트는 여기서 판단하지 않는다.
    """

    if has_misplaced:
        return "NEED_CHECK", "MISPLACED_PRODUCT"

    if front_quantity >= min_front_quantity:
        return "ENOUGH", "FRONT_QUANTITY_ENOUGH"

    if warehouse_quantity > 0:
        return "NEED_REFILL", "FRONT_UNDER_AND_WAREHOUSE_STOCK_EXISTS"

    return "ORDER_NEEDED", "FRONT_UNDER_AND_WAREHOUSE_STOCK_EMPTY"


def analyze_stock_results(
    mapped_detections: List[Dict[str, Any]],
    slots: List[Any],
) -> List[Dict[str, Any]]:
    """
    slot별 Stock 결과 생성.

    Inventory = 전체 재고 상태
    Stock = 매대 진열 상태

    창고 재고 계산:
    estimated_shelf_quantity = front_quantity * 2 + uncovered_back_quantity

    - FRONT 상품은 뒤쪽 상품을 가린다고 보고 2개로 추정
    - BACK 상품은 앞에 같은 열 FRONT 상품이 없을 때만 1개로 추정

    warehouse_quantity
    = Inventory.total_quantity - estimated_shelf_quantity
    """

    detections_by_slot = defaultdict(list)

    for det in mapped_detections:
        if det.get("slot_id") is None:
            continue

        detections_by_slot[int(det["slot_id"])].append(det)

    # # class_id별 현재 매대 탐지 수량 합계
    # # slot_id가 null이어도 이미지 안에서 탐지된 상품이면 포함
    # shelf_detected_quantity_by_class = defaultdict(int)

    # for det in mapped_detections:
    #     class_id = det.get("class_id")

    #     if class_id is None:
    #         continue

    #     shelf_detected_quantity_by_class[int(class_id)] += 1

    stock_results: List[Dict[str, Any]] = []

    for slot in slots:
        slot_dict = _to_dict(slot)

        slot_id = int(slot_dict["slot_id"])

        # Product.product_id = 상품 PK, 결과 반환/DB 연결용
        raw_product_id = slot_dict.get("product_id")
        expected_product_id = int(raw_product_id) if raw_product_id is not None else None

        # Product.class_id = YOLO class_id, 탐지 결과 비교/판단용
        expected_class_id = int(slot_dict["class_id"])

        slot_detections = detections_by_slot.get(slot_id, [])

        # 해당 slot에서 기대 상품 class_id와 같은 탐지만 정상 수량으로 계산
        expected_product_detections = [
            det
            for det in slot_detections
            if det.get("class_id") is not None
            and int(det["class_id"]) == expected_class_id
        ]

        front_quantity = sum(
            1
            for det in expected_product_detections
            if det["depth_position"] == "FRONT"
        )

        back_quantity = sum(
            1
            for det in expected_product_detections
            if det["depth_position"] == "BACK"
        )

        detected_quantity = len(expected_product_detections)

        has_misplaced = any(
            det.get("is_misplaced", False)
            for det in slot_detections
        )

        # has_low_confidence = any(
        #     det.get("is_low_confidence", False)
        #     for det in slot_detections
        # )

        # has_unknown_depth = any(
        #     det.get("depth_position") == "UNKNOWN"
        #     for det in slot_detections
        # )

        if slot_detections:
            confidence = round(
                sum(float(det["confidence"]) for det in slot_detections)
                / len(slot_detections),
                2,
            )
        else:
            confidence = 0.00

        total_quantity = int(slot_dict["total_quantity"])
        min_front_quantity = _get_min_front_quantity(slot_dict)

        # BACK 상품 중 앞에 같은 열 FRONT 상품이 없는 것만 계산
        uncovered_back_quantity = _count_uncovered_back_quantity(
            expected_product_detections
        )

        # 매대 추정 수량
        # FRONT 상품은 뒤쪽 상품을 가린다고 보고 2개로 계산
        # BACK 상품은 앞에 같은 열 FRONT 상품이 없을 때만 1개로 계산
        estimated_shelf_quantity = front_quantity * 2 + uncovered_back_quantity

        # 창고 재고 = 전체 재고 - 매대 추정 수량
        warehouse_quantity = max(
            total_quantity - estimated_shelf_quantity,
            0,
        )

        status, status_reason = decide_stock_status(
            front_quantity=front_quantity,
            min_front_quantity=min_front_quantity,
            warehouse_quantity=warehouse_quantity,
            has_misplaced=has_misplaced,
            # has_low_confidence=has_low_confidence,
            # has_unknown_depth=has_unknown_depth,
        )

        slot_bbox = _make_slot_bbox(slot_dict)

        issue_bbox = None
        bbox_source = None
        issue_confidence = None

        if status == "NEED_CHECK":
            issue_bbox, bbox_source, issue_confidence = _select_issue_bbox_for_need_check(
                slot_detections
            )

            if issue_bbox is None:
                issue_bbox = slot_bbox
                bbox_source = "SLOT"
                issue_confidence = None

        elif status == "NEED_REFILL":
            issue_bbox = slot_bbox
            bbox_source = "SLOT"
            issue_confidence = None

        elif status == "ORDER_NEEDED":
            issue_bbox = slot_bbox
            bbox_source = "SLOT"
            issue_confidence = None

        stock_results.append(
            {
                "slot_id": slot_id,
                "product_id": expected_product_id,
                "class_id": expected_class_id,
                "status": status,
                "is_misplaced": has_misplaced,
                "front_quantity": front_quantity,
                "back_quantity": back_quantity,
                "detected_quantity": detected_quantity,
                "confidence": confidence,
                "status_reason": status_reason,
                "issue_bbox": issue_bbox,
                "bbox_source": bbox_source,
                "issue_confidence": issue_confidence,

            }
        )

    return stock_results