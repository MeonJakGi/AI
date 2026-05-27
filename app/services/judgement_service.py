from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


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


def _select_issue_bbox_for_need_check(slot_detections: List[Dict[str, Any]]):
    for det in slot_detections:
        if det.get("is_misplaced", False):
            return (
                _make_detection_bbox(det),
                "MISPLACED_DETECTION",
                round(float(det["confidence"]), 2),
            )

    for det in slot_detections:
        if det.get("is_low_confidence", False):
            return (
                _make_detection_bbox(det),
                "LOW_CONFIDENCE_DETECTION",
                round(float(det["confidence"]), 2),
            )

    for det in slot_detections:
        if det.get("depth_position") == "UNKNOWN":
            return (
                _make_detection_bbox(det),
                "UNKNOWN_DEPTH_DETECTION",
                round(float(det["confidence"]), 2),
            )

    for det in slot_detections:
        return (
            _make_detection_bbox(det),
            "DETECTION",
            round(float(det["confidence"]), 2),
        )

    return None, None, None

def decide_stock_status(
    front_quantity: int,
    min_front_quantity: int,
    warehouse_quantity: int,
    has_misplaced: bool,
    has_low_confidence: bool,
    has_unknown_depth: bool,
) -> Tuple[str, str]:
    """
    Stock.status 판단 로직.

    기준:
    1. 오진열 / 저신뢰 / FRONT-BACK 판단 불가
       → NEED_CHECK

    2. 앞열이 충분한지 1차 판단
       front_quantity >= min_front_quantity
       → ENOUGH

    3. 앞열이 충분하지 않으면 2차 판단
       창고 재고 있음
       → NEED_REFILL

       창고 재고 없음
       → ORDER_NEEDED

    주의:
    - ORDER_NEEDED는 보충 필요 리스트에 띄우는 대상이 아님.
    - 보충 필요 리스트는 NEED_REFILL, NEED_CHECK만 표시.
    - 발주 필요 리스트는 Inventory.total_quantity <= reorder_point 기준으로
      백엔드가 별도로 조회.
    """

    if has_misplaced:
        return "NEED_CHECK", "MISPLACED_PRODUCT"

    if has_low_confidence:
        return "NEED_CHECK", "LOW_CONFIDENCE_DETECTION"

    if has_unknown_depth:
        return "NEED_CHECK", "DEPTH_POSITION_UNKNOWN"

    # 1차: 앞열 충분 여부 판단
    if front_quantity >= min_front_quantity:
        return "ENOUGH", "FRONT_QUANTITY_ENOUGH"

    # 2차: 충분하지 않으면 창고 재고 기준으로 판단
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
    warehouse_quantity
    = Inventory.total_quantity - 현재 이미지에서 탐지된 해당 상품의 매대 수량

    단, 여기서 뒤에 숨은 재고를 추정하지 않음.
    AI가 실제로 탐지한 수량만 사용함.
    """

    detections_by_slot = defaultdict(list)

    for det in mapped_detections:
        if det.get("slot_id") is None:
            continue

        detections_by_slot[int(det["slot_id"])].append(det)

    # class_id별 현재 매대 탐지 수량 합계
    # slot_id가 null이어도 이미지 안에서 탐지된 상품이면 포함
    shelf_detected_quantity_by_class = defaultdict(int)

    for det in mapped_detections:
        class_id = det.get("class_id")

        if class_id is None:
            continue

        shelf_detected_quantity_by_class[int(class_id)] += 1

    stock_results: List[Dict[str, Any]] = []

    for slot in slots:
        slot_dict = _to_dict(slot)

        slot_id = int(slot_dict["slot_id"])

        # Product.product_id = 상품 PK, 결과 반환/DB 연결용
        expected_product_id = int(slot_dict["product_id"])

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

        has_low_confidence = any(
            det.get("is_low_confidence", False)
            for det in slot_detections
        )

        has_unknown_depth = any(
            det.get("depth_position") == "UNKNOWN"
            for det in slot_detections
        )

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

        # 해당 상품이 현재 매대 전체에서 탐지된 수량
        shelf_detected_quantity = shelf_detected_quantity_by_class.get(
            expected_class_id,
            0,
        )

        # 창고 재고 = 전체 재고 - 매대에서 실제 탐지된 재고
        warehouse_quantity = max(
            total_quantity - shelf_detected_quantity,
            0,
        )

        status, status_reason = decide_stock_status(
            front_quantity=front_quantity,
            min_front_quantity=min_front_quantity,
            warehouse_quantity=warehouse_quantity,
            has_misplaced=has_misplaced,
            has_low_confidence=has_low_confidence,
            has_unknown_depth=has_unknown_depth,
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