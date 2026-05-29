from typing import Any, Dict, List, Optional


SLOT_PADDING_X = 25
SLOT_PADDING_Y = 15

ROW_Y_THRESHOLD_RATIO = 0.45
CLUSTER_X_GAP_RATIO = 1.0

AUTO_SLOT_MIN_CONFIDENCE = 0.30
AUTO_SLOT_NMS_IOU_THRESHOLD = 0.50

AUTO_ROW_X_MARGIN_PX = 80
AUTO_ROW_MAX_DISTANCE_RATIO = 1.15
AUTO_ROW_MAX_DISTANCE_PX = 180


def _bbox_center_x(bbox: Dict[str, int]) -> float:
    return float(bbox["x"] + bbox["width"] / 2)


def _bbox_center_y(bbox: Dict[str, int]) -> float:
    return float(bbox["y"] + bbox["height"] / 2)


def _bbox_bottom_y(bbox: Dict[str, int]) -> float:
    return float(bbox["y"] + bbox["height"])


def _merge_bboxes(bboxes: List[Dict[str, int]]) -> Dict[str, int]:
    x1 = min(b["x"] for b in bboxes)
    y1 = min(b["y"] for b in bboxes)
    x2 = max(b["x"] + b["width"] for b in bboxes)
    y2 = max(b["y"] + b["height"] for b in bboxes)

    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(x2 - x1),
        "height": int(y2 - y1),
    }

def _bbox_iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax1 = float(a["x"])
    ay1 = float(a["y"])
    ax2 = float(a["x"] + a["width"])
    ay2 = float(a["y"] + a["height"])

    bx1 = float(b["x"])
    by1 = float(b["y"])
    bx2 = float(b["x"] + b["width"])
    by2 = float(b["y"] + b["height"])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union

def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _get_line_y_at_x(points_xy: List[List[float]], x: float) -> float:
    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    if x2 == x1:
        return float((y1 + y2) / 2)

    ratio = (x - x1) / (x2 - x1)
    return float(y1 + ratio * (y2 - y1))

def _assign_row_by_front_edge(
    bbox: Dict[str, int],
    front_edge_points: List[Any],
) -> Optional[int]:
    """
    bbox bottom center가 어느 앞턱선 row에 가까운지로 shelf row를 배정한다.

    단, 아무 detection이나 row에 배정하면
    배경 매대/사람/장바구니 상품까지 auto slot으로 만들어진다.

    그래서 아래 조건을 만족하는 detection만 row에 배정한다.
    - bbox center x가 앞턱선 x 범위 근처에 있어야 함
    - bbox bottom이 앞턱선과 너무 멀면 제외
    """

    px = bbox["x"] + bbox["width"] / 2
    py = bbox["y"] + bbox["height"]
    bbox_h = max(float(bbox["height"]), 1.0)

    best_row_no = None
    best_dist = float("inf")

    for edge in front_edge_points:
        edge_dict = _to_dict(edge)

        row_no = edge_dict.get("row_no")
        points_xy = edge_dict.get("points_xy")

        if row_no is None or not points_xy or len(points_xy) < 2:
            continue

        x1, _ = points_xy[0]
        x2, _ = points_xy[1]

        min_x = min(x1, x2) - AUTO_ROW_X_MARGIN_PX
        max_x = max(x1, x2) + AUTO_ROW_X_MARGIN_PX

        # 앞턱선 x 범위 밖 detection은 제외
        if px < min_x or px > max_x:
            continue

        line_y = _get_line_y_at_x(points_xy, px)
        dist = abs(py - line_y)

        if dist < best_dist:
            best_dist = dist
            best_row_no = int(row_no)

    if best_row_no is None:
        return None

    max_allowed_dist = min(
        AUTO_ROW_MAX_DISTANCE_PX,
        bbox_h * AUTO_ROW_MAX_DISTANCE_RATIO,
    )

    # 앞턱선과 너무 멀면 우리 선반 상품이 아니라고 보고 제외
    if best_dist > max_allowed_dist:
        return None

    return best_row_no

def _group_by_class(detections: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}

    for det in detections:
        class_id = int(det["class_id"])
        groups.setdefault(class_id, []).append(det)

    return groups



def _split_x_clusters(row_dets: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    같은 row 안에서도 x 간격이 너무 멀면 다른 slot으로 분리한다.
    """

    if not row_dets:
        return []

    sorted_dets = sorted(
        row_dets,
        key=lambda d: d["bbox"]["x"],
    )

    clusters: List[List[Dict[str, Any]]] = [[sorted_dets[0]]]

    for det in sorted_dets[1:]:
        bbox = det["bbox"]
        prev_bbox = clusters[-1][-1]["bbox"]

        prev_right = prev_bbox["x"] + prev_bbox["width"]
        gap = bbox["x"] - prev_right

        avg_width = (
            float(prev_bbox["width"]) + float(bbox["width"])
        ) / 2

        if gap <= avg_width * CLUSTER_X_GAP_RATIO:
            clusters[-1].append(det)
        else:
            clusters.append([det])

    return clusters

def filter_detections_for_slot_generation(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    auto slot 생성 전에 detection을 정리한다.

    목적:
    - confidence가 너무 낮은 bbox 제거
    - 같은 class에서 많이 겹치는 중복 bbox 제거
    """

    # 1. confidence 낮은 detection 제거
    filtered = [
        det for det in detections
        if float(det.get("confidence", 0.0)) >= AUTO_SLOT_MIN_CONFIDENCE
    ]

    # 2. class별 NMS
    class_groups: Dict[int, List[Dict[str, Any]]] = {}

    for det in filtered:
        class_id = int(det["class_id"])
        class_groups.setdefault(class_id, []).append(det)

    final_detections: List[Dict[str, Any]] = []

    for _, dets in class_groups.items():
        dets = sorted(
            dets,
            key=lambda d: float(d.get("confidence", 0.0)),
            reverse=True,
        )

        kept: List[Dict[str, Any]] = []

        for det in dets:
            bbox = det["bbox"]

            is_duplicate = False

            for kept_det in kept:
                kept_bbox = kept_det["bbox"]

                iou = _bbox_iou(bbox, kept_bbox)

                if iou >= AUTO_SLOT_NMS_IOU_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(det)

        final_detections.extend(kept)

    return final_detections

def generate_slots_from_detections(
    detections: List[Dict[str, Any]],
    front_edge_points: List[Any],
) -> List[Dict[str, Any]]:
    """
    상품 detection 결과를 기반으로 slot 후보를 자동 생성한다.

    핵심:
    - 정면/측면 모두 앞턱선 기준으로 shelf row를 먼저 배정한다.
    - 같은 row 안에서 class_id별로 묶는다.
    - 같은 row + class 안에서 x축으로 가까운 bbox를 하나의 slot으로 묶는다.
    """

    # detections = filter_detections_for_slot_generation(detections)

    slots: List[Dict[str, Any]] = []
    slot_id = 1

    # 1. 앞턱선 기준으로 row 먼저 배정
    row_groups: Dict[int, List[Dict[str, Any]]] = {}

    for det in detections:
        bbox = det["bbox"]

        row_no = _assign_row_by_front_edge(
            bbox=bbox,
            front_edge_points=front_edge_points,
        )

        # 앞턱선 기준으로 row 배정이 안 되면
        # 우리 대상 선반 상품이 아니라고 보고 auto slot 생성에서 제외
        if row_no is None:
            det["auto_row_no"] = None
            continue

        det["auto_row_no"] = row_no
        row_groups.setdefault(row_no, []).append(det)

    # 2. row 안에서 class별로 묶고 x cluster 생성
    for row_no in sorted(row_groups.keys()):
        row_dets = row_groups[row_no]

        class_groups: Dict[int, List[Dict[str, Any]]] = {}

        for det in row_dets:
            class_id = int(det["class_id"])
            class_groups.setdefault(class_id, []).append(det)

        row_slots: List[Dict[str, Any]] = []

        for class_id, class_dets in class_groups.items():
            x_clusters = _split_x_clusters(class_dets)

            for cluster in x_clusters:
                bboxes = [det["bbox"] for det in cluster]
                merged = _merge_bboxes(bboxes)

                x = max(0, merged["x"] - SLOT_PADDING_X)
                y = max(0, merged["y"] - SLOT_PADDING_Y)
                width = merged["width"] + SLOT_PADDING_X * 2
                height = merged["height"] + SLOT_PADDING_Y * 2

                representative = cluster[0]

                row_slots.append(
                    {
                        "slot_id": slot_id,
                        "slot_code": f"auto-slot-{slot_id}",
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                        "row_no": int(row_no),
                        "col_no": 0,  # 아래에서 x순으로 다시 부여
                        "product_id": representative.get("product_id"),
                        "class_id": int(class_id),
                        "product_name": representative.get("class_name"),
                        "expected_quantity": len(cluster),
                        "min_front_quantity": 1,
                        "min_display_quantity": 1,
                        "total_quantity": len(cluster),
                        "reorder_point": 1,
                    }
                )

                slot_id += 1

        # 3. 같은 row 안에서 x좌표 순서대로 col_no 부여
        row_slots = sorted(row_slots, key=lambda s: s["x"])

        for col_idx, slot in enumerate(row_slots, start=1):
            slot["col_no"] = col_idx
            slots.append(slot)

    return slots