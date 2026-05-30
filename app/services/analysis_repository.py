import json
import math
from typing import Any, Dict, List, Optional

from app.db import get_connection

def _parse_json(value: Any):
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        # JSON 문자열이 이중으로 들어온 경우까지 방어
        for _ in range(2):
            parsed = json.loads(value)

            if isinstance(parsed, (dict, list)):
                return parsed

            if isinstance(parsed, str):
                value = parsed.strip()
                continue

            return parsed

    return value

def _calc_angle(points_xy: List[List[float]]) -> float:
    if not points_xy or len(points_xy) < 2:
        return 0.0

    x1, y1 = points_xy[0]
    x2, y2 = points_xy[1]

    return round(math.degrees(math.atan2(y2 - y1, x2 - x1)), 2)


def _normalize_front_edge_points(value: Any) -> List[Dict[str, Any]]:
    """
    DB의 shelf.front_edge_points 형식을
    slot_mapper.py / visualizer.py가 기대하는 형식으로 변환한다.

    DB 현재 형식:
    {
        "view": "FRONT",
        "row_points": [[...], ...],
        "shelf_lip_points": [[x,y], [x,y], [x,y], [x,y], ...]
    }

    AI 서버 기대 형식:
    [
        {
            "row_no": 1,
            "points_xy": [[x1, y1], [x2, y2]],
            "angle": ...
        },
        ...
    ]
    """
    parsed = _parse_json(value)

    if parsed is None:
        return []

    # 이미 기대 형식인 경우
    if isinstance(parsed, list):
        if all(isinstance(item, dict) and "points_xy" in item for item in parsed):
            return parsed

        # 혹시 [[x1,y1], [x2,y2], ...] 같은 점 리스트만 들어온 경우
        return []

    if not isinstance(parsed, dict):
        return []

    # 1순위: shelf_lip_points 사용
    # 현재 DB 값은 4개 점이 하나의 앞턱 polygon이고,
    # 그중 앞의 2개 점이 앞턱 윗선으로 보임.
    shelf_lip_points = parsed.get("shelf_lip_points")

    if isinstance(shelf_lip_points, list) and len(shelf_lip_points) >= 4:
        edges = []

        for idx in range(0, len(shelf_lip_points), 4):
            polygon = shelf_lip_points[idx:idx + 4]

            if len(polygon) < 2:
                continue

            points_xy = [
                [float(polygon[0][0]), float(polygon[0][1])],
                [float(polygon[1][0]), float(polygon[1][1])],
            ]

            edges.append({
                "row_no": len(edges) + 1,
                "points_xy": points_xy,
                "angle": _calc_angle(points_xy),
            })

        return edges

    # 2순위: row_points 사용
    # row_points가 5개씩 한 row라면 각 row의 첫 점과 마지막 점을 앞턱선으로 사용
    row_points = parsed.get("row_points")

    if isinstance(row_points, list) and len(row_points) >= 2:
        edges = []

        points_per_row = 5

        for idx in range(0, len(row_points), points_per_row):
            row = row_points[idx:idx + points_per_row]

            if len(row) < 2:
                continue

            points_xy = [
                [float(row[0][0]), float(row[0][1])],
                [float(row[-1][0]), float(row[-1][1])],
            ]

            edges.append({
                "row_no": len(edges) + 1,
                "points_xy": points_xy,
                "angle": _calc_angle(points_xy),
            })

        return edges

    return []


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class AnalysisRepository:
    def mark_analysis_processing(self, shelf_image_id: int) -> None:
        """
        AI 서버가 POST 요청을 받자마자 shelf_image 상태를 PROCESSING으로 변경.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE shelf_image
                    SET analysis_status = 'PROCESSING',
                        analysis_requested_at = COALESCE(analysis_requested_at, NOW()),
                        last_checked_at = NOW(),
                        analysis_message = NULL
                    WHERE shelf_image_id = %s
                    """,
                    (shelf_image_id,),
                )

                if cursor.rowcount == 0:
                    raise ValueError(f"shelf_image_id={shelf_image_id}가 존재하지 않습니다.")

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def mark_analysis_completed(
        self,
        shelf_image_id: int,
        image_width: Optional[int],
        image_height: Optional[int],
        message: str = "AI analysis completed",
    ) -> None:
        """
        detection_result / stock / alarm 저장까지 끝나면 COMPLETE로 변경.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE shelf_image
                    SET analysis_status = 'COMPLETE',
                        image_width = COALESCE(%s, image_width),
                        image_height = COALESCE(%s, image_height),
                        analysis_completed_at = NOW(),
                        last_checked_at = NOW(),
                        analysis_message = %s
                    WHERE shelf_image_id = %s
                    """,
                    (image_width, image_height, message, shelf_image_id),
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def mark_analysis_failed(self, shelf_image_id: int, message: str) -> None:
        """
        분석 중 에러가 나면 FAILED로 변경하고 retry_count 증가.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE shelf_image
                    SET analysis_status = 'FAILED',
                        retry_count = retry_count + 1,
                        last_checked_at = NOW(),
                        analysis_message = %s
                    WHERE shelf_image_id = %s
                    """,
                    (message[:1000], shelf_image_id),
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def get_analysis_context(self, shelf_image_id: int, shelf_id: int) -> Dict[str, Any]:
        """
        request로 받은 shelf_id를 기준으로 분석에 필요한 정보를 조회한다.

        이제 shelf_image_id로 camera -> shelf를 찾지 않는다.
        shelf_id는 request에서 직접 받는다.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # 1. shelf_image 존재 여부 확인
                cursor.execute(
                    """
                    SELECT
                        shelf_image_id,
                        camera_id,
                        image_width,
                        image_height
                    FROM shelf_image
                    WHERE shelf_image_id = %s
                    """,
                    (shelf_image_id,),
                )

                image_row = cursor.fetchone()

                if image_row is None:
                    raise ValueError(
                        f"shelf_image_id={shelf_image_id}가 존재하지 않습니다."
                    )

                # 2. request로 받은 shelf_id 기준으로 shelf 조회
                cursor.execute(
                    """
                    SELECT
                        shelf_id,
                        store_id,
                        front_edge_points
                    FROM shelf
                    WHERE shelf_id = %s
                    """,
                    (shelf_id,),
                )

                shelf_row = cursor.fetchone()

                if shelf_row is None:
                    raise ValueError(
                        f"shelf_id={shelf_id}가 존재하지 않습니다."
                    )

                # 3. shelf_id 기준으로 slot / planogram / product / inventory 조회
                cursor.execute(
                    """
                    SELECT
                        sl.slot_id,
                        sl.slot_code,
                        sl.x,
                        sl.y,
                        sl.width,
                        sl.height,
                        sl.row_no,
                        sl.col_no,
                        p.product_id,
                        p.class_id,
                        p.product_name,
                        pg.expected_quantity,
                        pg.min_front_quantity,
                        pg.min_display_quantity,
                        COALESCE(i.total_quantity, 0) AS total_quantity,
                        COALESCE(i.reorder_point, 0) AS reorder_point
                    FROM slot sl
                    JOIN planogram pg
                        ON pg.slot_id = sl.slot_id
                    AND pg.is_active = TRUE
                    JOIN product p
                        ON p.product_id = pg.product_id
                    LEFT JOIN inventory i
                        ON i.store_id = %s
                    AND i.product_id = p.product_id
                    WHERE sl.shelf_id = %s
                    ORDER BY sl.row_no, sl.col_no, sl.slot_id
                    """,
                    (shelf_row["store_id"], shelf_id),
                )

                slots = cursor.fetchall()

            return {
                "shelf_image_id": int(image_row["shelf_image_id"]),
                "camera_id": image_row.get("camera_id"),
                "shelf_id": int(shelf_row["shelf_id"]),
                "store_id": int(shelf_row["store_id"]),
                "front_edge_points": _normalize_front_edge_points(
                    shelf_row.get("front_edge_points")
                ),
                "slots": slots,
            }

        finally:
            conn.close()

    def save_shelf_front_edge_points(
        self,
        shelf_id: int,
        front_edge_points: List[Dict[str, Any]],
    ) -> None:
        """
        shelf.front_edge_points가 비어 있어서 AI가 앞턱선을 탐지한 경우,
        다음 분석부터 재사용할 수 있도록 shelf 테이블에 저장.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE shelf
                    SET front_edge_points = %s
                    WHERE shelf_id = %s
                    """,
                    (_json_dumps(front_edge_points), shelf_id),
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def get_product_id_by_class_id(self) -> Dict[int, int]:
        """
        YOLO class_id를 DB product_id로 변환하기 위한 매핑 조회.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT class_id, product_id
                    FROM product
                    """
                )
                rows = cursor.fetchall()

            return {
                int(row["class_id"]): int(row["product_id"])
                for row in rows
            }

        finally:
            conn.close()

    def save_analysis_results(
        self,
        shelf_image_id: int,
        mapped_detections: List[Dict[str, Any]],
        stock_results: List[Dict[str, Any]],
    ) -> None:
        """
        분석 결과 저장.

        저장 순서:
        1. 같은 shelf_image_id 기존 detection_result 삭제
        2. 같은 shelf_image_id 기존 stock 삭제
        3. detection_result 저장
        4. 기존 current stock 비활성화
        5. stock 저장
        6. stock 상태가 ENOUGH가 아니면 alarm 저장
        """
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # 같은 이미지 재분석 시 중복 저장 방지
                cursor.execute(
                    "DELETE FROM detection_result WHERE shelf_image_id = %s",
                    (shelf_image_id,),
                )

                cursor.execute(
                    "DELETE FROM stock WHERE shelf_image_id = %s",
                    (shelf_image_id,),
                )

                for det in mapped_detections:
                    cursor.execute(
                        """
                        INSERT INTO detection_result (
                            shelf_image_id,
                            slot_id,
                            product_id,
                            x,
                            y,
                            width,
                            height,
                            confidence,
                            depth_position,
                            is_misplaced,
                            is_low_confidence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            shelf_image_id,
                            det.get("slot_id"),
                            det.get("product_id"),
                            int(det["x"]),
                            int(det["y"]),
                            int(det["width"]),
                            int(det["height"]),
                            round(float(det["confidence"]), 2),
                            det.get("depth_position"),
                            bool(det.get("is_misplaced", False)),
                            bool(det.get("is_low_confidence", False)),
                        ),
                    )

                slot_ids = sorted({
                    int(item["slot_id"])
                    for item in stock_results
                })

                if slot_ids:
                    placeholders = ", ".join(["%s"] * len(slot_ids))
                    cursor.execute(
                        f"""
                        UPDATE stock
                        SET is_current = FALSE
                        WHERE is_current = TRUE
                          AND slot_id IN ({placeholders})
                        """,
                        tuple(slot_ids),
                    )

                for item in stock_results:
                    product_id = item.get("product_id")

                    if product_id is None:
                        raise ValueError(
                            f"slot_id={item.get('slot_id')}의 product_id가 없어 stock 저장이 불가능합니다."
                        )

                    issue_bbox = item.get("issue_bbox") or {}

                    cursor.execute(
                        """
                        INSERT INTO stock (
                            shelf_image_id,
                            slot_id,
                            product_id,
                            status,
                            is_misplaced,
                            front_quantity,
                            back_quantity,
                            detected_quantity,
                            confidence,
                            status_reason,
                            issue_x,
                            issue_y,
                            issue_width,
                            issue_height,
                            bbox_source,
                            is_current,
                            changed_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, NOW())
                        """,
                        (
                            shelf_image_id,
                            int(item["slot_id"]),
                            int(product_id),
                            item["status"],
                            bool(item.get("is_misplaced", False)),
                            int(item.get("front_quantity", 0)),
                            int(item.get("back_quantity", 0)),
                            int(item.get("detected_quantity", 0)),
                            round(float(item.get("confidence", 0.0)), 2),
                            item.get("status_reason"),
                            issue_bbox.get("x"),
                            issue_bbox.get("y"),
                            issue_bbox.get("width"),
                            issue_bbox.get("height"),
                            item.get("bbox_source"),
                        ),
                    )

                    stock_id = cursor.lastrowid
                    status = item["status"]

                    # SKU 완전 부재 또는 오진열인 경우에만 alarm 생성
                    if _should_create_alarm(item):
                        alarm_type = _to_alarm_type(item)
                        message = _make_alarm_message(item)

                        cursor.execute(
                            """
                            INSERT INTO alarm (
                                stock_id,
                                alarm_type,
                                message,
                                is_read
                            ) VALUES (%s, %s, %s, FALSE)
                            """,
                            (stock_id, alarm_type, message),
                        )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()


def _should_create_alarm(item: Dict[str, Any]) -> bool:
    """
    alarm 생성 기준:
    1. SKU가 매대에서 완전히 비어 있음
    2. 오진열 상태임

    단순 보충 필요 / 발주 필요은 alarm 생성하지 않음.
    """
    detected_quantity = int(item.get("detected_quantity", 0))
    is_misplaced = bool(item.get("is_misplaced", False))

    # 오진열
    if is_misplaced:
        return True

    # SKU 완전 부재
    if detected_quantity == 0:
        return True

    return False


def _to_alarm_type(item: Dict[str, Any]) -> str:
    """
    DB alarm_type CHECK 조건에 맞게 변환.

    SKU 완전 부재 → SHELF_EMPTY
    오진열 → NEED_CHECK
    """
    detected_quantity = int(item.get("detected_quantity", 0))
    is_misplaced = bool(item.get("is_misplaced", False))

    if is_misplaced:
        return "NEED_CHECK"

    if detected_quantity == 0:
        return "SHELF_EMPTY"

    raise ValueError(
        f"alarm 생성 대상이 아닌 item입니다. "
        f"slot_id={item.get('slot_id')}, "
        f"status={item.get('status')}, "
        f"detected_quantity={detected_quantity}, "
        f"is_misplaced={is_misplaced}"
    )


def _make_alarm_message(item: Dict[str, Any]) -> str:
    slot_id = item.get("slot_id")
    product_id = item.get("product_id")
    detected_quantity = int(item.get("detected_quantity", 0))
    is_misplaced = bool(item.get("is_misplaced", False))

    if is_misplaced:
        return (
            f"slot_id={slot_id}, product_id={product_id} 상품이 "
            f"기준 위치와 다른 곳에 진열되어 있습니다."
        )

    if detected_quantity == 0:
        return (
            f"slot_id={slot_id}, product_id={product_id} 상품이 "
            f"매대에서 완전히 비어 있습니다."
        )

    return (
        f"slot_id={slot_id}, product_id={product_id} 상품의 "
        f"진열 상태 확인이 필요합니다."
    )