from collections import defaultdict
from typing import Any, Dict, List


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def decide_stock_status(
    front_quantity: int,
    min_front_quantity: int,
    inventory_quantity: int,
    has_misplaced: bool,
    has_low_confidence: bool,
    has_unknown_depth: bool,
) -> str:
    """
    stock.status 판단.

    ENOUGH:
    - 앞줄 수량 충분

    NEED_REFILL:
    - 앞줄 부족
    - 창고 재고 있음

    ORDER_NEEDED:
    - 앞줄 부족
    - 창고 재고 없음

    NEED_CHECK:
    - 오진열
    - 저신뢰
    - depth UNKNOWN
    """

    if has_misplaced or has_low_confidence or has_unknown_depth:
        return "NEED_CHECK"

    if front_quantity >= min_front_quantity:
        return "ENOUGH"

    if inventory_quantity > 0:
        return "NEED_REFILL"

    return "ORDER_NEEDED"


def _build_status_reason(status: str) -> str:
    if status == "NEED_CHECK":
        return "MISPLACED_OR_LOW_CONFIDENCE_OR_DEPTH_UNKNOWN"

    if status == "ENOUGH":
        return "FRONT_QUANTITY_ENOUGH"

    if status == "NEED_REFILL":
        return "FRONT_QUANTITY_UNDER_MIN_FRONT_QUANTITY_AND_INVENTORY_EXISTS"

    if status == "ORDER_NEEDED":
        return "FRONT_QUANTITY_UNDER_MIN_FRONT_QUANTITY_AND_INVENTORY_EMPTY"

    return "UNKNOWN_REASON"


def analyze_stock_results(
    mapped_detections: List[Dict[str, Any]],
    slots: List[Any],
) -> List[Dict[str, Any]]:
    """
    slot별 stock 결과 생성.
    """

    detections_by_slot = defaultdict(list)

    for det in mapped_detections:
        if det["slot_id"] is None:
            continue

        detections_by_slot[det["slot_id"]].append(det)

    stock_results: List[Dict[str, Any]] = []

    for slot in slots:
        slot_dict = _to_dict(slot)

        slot_id = int(slot_dict["slot_id"])
        expected_product_id = int(slot_dict["product_id"])

        slot_detections = detections_by_slot.get(slot_id, [])

        expected_product_detections = [
            det
            for det in slot_detections
            if int(det["product_id"]) == expected_product_id
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

        has_misplaced = any(det["is_misplaced"] for det in slot_detections)
        has_low_confidence = any(det["is_low_confidence"] for det in slot_detections)
        has_unknown_depth = any(
            det["depth_position"] == "UNKNOWN"
            for det in slot_detections
        )

        if slot_detections:
            confidence = round(
                sum(det["confidence"] for det in slot_detections) / len(slot_detections),
                4,
            )
        else:
            confidence = 0.0

        inventory_quantity = int(slot_dict["inventory_quantity"])
        reorder_point = int(slot_dict["reorder_point"])
        min_front_quantity = int(slot_dict["min_front_quantity"])

        status = decide_stock_status(
            front_quantity=front_quantity,
            min_front_quantity=min_front_quantity,
            inventory_quantity=inventory_quantity,
            has_misplaced=has_misplaced,
            has_low_confidence=has_low_confidence,
            has_unknown_depth=has_unknown_depth,
        )

        total_quantity = inventory_quantity + detected_quantity
        order_list_needed = total_quantity <= reorder_point

        stock_results.append(
            {
                "slot_id": slot_id,
                "product_id": expected_product_id,
                "sku_code": slot_dict.get("sku_code"),
                "status": status,
                "front_quantity": front_quantity,
                "back_quantity": back_quantity,
                "detected_quantity": detected_quantity,
                "confidence": confidence,
                "status_reason": _build_status_reason(status),
                "order_list_needed": order_list_needed,
            }
        )

    return stock_results