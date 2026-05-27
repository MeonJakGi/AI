from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[2]
DEBUG_DIR = BASE_DIR / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _get_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_analysis_result(
    image: Image.Image,
    front_edge_points: List[Any],
    slots: List[Any],
    detections: List[Dict[str, Any]],
    shelf_image_id: int,
) -> str:
    """
    분석 결과를 이미지 위에 시각화해서 저장한다.

    표시 내용:
    - slot 영역
    - 앞턱 윗선
    - 상품 bbox
    - slot_id / product_id / FRONT/BACK
    """

    debug_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_image)
    font = _get_font(18)

    # 1. slot 영역 그리기
    for slot in slots:
        slot_dict = _to_dict(slot)

        x1 = int(slot_dict["x"])
        y1 = int(slot_dict["y"])
        x2 = int(slot_dict["x"] + slot_dict["width"])
        y2 = int(slot_dict["y"] + slot_dict["height"])

        slot_id = slot_dict["slot_id"]
        slot_code = slot_dict.get("slot_code", "")

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="yellow",
            width=3,
        )

        draw.text(
            (x1 + 5, y1 + 5),
            f"SLOT {slot_id} {slot_code}",
            fill="yellow",
            font=font,
        )

    # 2. 앞턱 윗선 그리기
    for edge in front_edge_points:
        edge_dict = _to_dict(edge)

        points_xy = edge_dict.get("points_xy")
        row_no = edge_dict.get("row_no")

        if points_xy and len(points_xy) >= 2:
            p1 = tuple(map(int, points_xy[0]))
            p2 = tuple(map(int, points_xy[1]))

            draw.line(
                [p1, p2],
                fill="cyan",
                width=5,
            )

            draw.text(
                (p1[0], p1[1] - 25),
                f"FRONT EDGE row {row_no}",
                fill="cyan",
                font=font,
            )

    # 3. 상품 탐지 bbox 그리기
    for det in detections:
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = x1 + int(det["width"])
        y2 = y1 + int(det["height"])

        slot_id = det.get("slot_id")
        product_id = det.get("product_id")
        class_name = det.get("class_name", "")
        confidence = det.get("confidence")
        depth_position = det.get("depth_position")

        if depth_position == "FRONT":
            color = "lime"
        elif depth_position == "BACK":
            color = "orange"
        else:
            color = "red"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3,
        )

        label = f"{depth_position} | slot:{slot_id} | p:{product_id} | {confidence}"

        draw.rectangle(
            [x1, max(0, y1 - 25), x1 + 420, y1],
            fill=color,
        )

        draw.text(
            (x1 + 3, max(0, y1 - 22)),
            label,
            fill="black",
            font=font,
        )

    output_path = DEBUG_DIR / f"analyze_{shelf_image_id}.png"
    debug_image.save(output_path)

    return str(output_path)

def draw_front_edge_only(
    image: Image.Image,
    front_edge_points: List[Any],
    shelf_image_id: int,
) -> str:
    """
    앞턱 탐지 결과만 이미지 위에 시각화해서 저장한다.

    표시 내용:
    - 앞턱 윗선
    - row_no
    - 양 끝점
    - front_y / angle / width_ratio 정보
    """

    debug_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_image)
    font = _get_font(22)

    for edge in front_edge_points:
        edge_dict = _to_dict(edge)

        points_xy = edge_dict.get("points_xy")
        row_no = edge_dict.get("row_no")
        front_y = edge_dict.get("front_y")
        angle = edge_dict.get("angle")
        width_ratio = edge_dict.get("width_ratio")
        conf = edge_dict.get("conf")

        if not points_xy or len(points_xy) < 2:
            continue

        p1 = tuple(map(int, points_xy[0]))
        p2 = tuple(map(int, points_xy[1]))

        # 앞턱 선
        draw.line(
            [p1, p2],
            fill="cyan",
            width=8,
        )

        # 양 끝점
        r = 8
        draw.ellipse(
            [p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r],
            fill="red",
        )
        draw.ellipse(
            [p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r],
            fill="red",
        )

        # 라벨
        label = (
            f"row {row_no} | y={front_y} | "
            f"angle={angle} | width={width_ratio} | conf={conf}"
        )

        text_x = max(5, min(p1[0], p2[0]))
        text_y = max(5, min(p1[1], p2[1]) - 35)

        # 글씨 배경
        text_box_width = 620
        text_box_height = 30

        draw.rectangle(
            [text_x, text_y, text_x + text_box_width, text_y + text_box_height],
            fill="black",
        )

        draw.text(
            (text_x + 5, text_y + 4),
            label,
            fill="cyan",
            font=font,
        )

    output_path = DEBUG_DIR / f"front_edge_only_{shelf_image_id}.png"
    debug_image.save(output_path)

    return str(output_path)