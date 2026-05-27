"""
shelf_lip_detector.py

IPYNB의 선반 앞턱 윗선 추출 코드를 서버용으로 옮긴 버전입니다.
- slot 정보는 사용하지 않습니다.
- 업로드된 ipynb/텍스트 코드의 파라미터와 후처리 흐름을 최대한 그대로 반영했습니다.
- FastAPI main.py에서는 detect_front_edge_points(image)를 그대로 호출하면 됩니다.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import os
import math

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# =========================
# 모델 경로
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("SHELF_LIP_MODEL_PATH", str(BASE_DIR / "weights" / "shelf_lip_best.pt")))


# =========================
# 예측 설정 - IPYNB 셀 5 기준
# =========================
IMGSZ = 960
CONF_THRES = 0.12
IOU_THRES = 0.50

# =========================
# 윗선 후보 필터링 설정 - IPYNB 셀 5 기준
# =========================
MIN_CANDIDATE_WIDTH_RATIO = 0.06
MIN_FINAL_WIDTH_RATIO = 0.12
MIN_AREA_RATIO = 0.00004

# 실제 사진/측면 이미지에서는 사선이 있을 수 있어서 넉넉하게 둠
MAX_ABS_ANGLE = 22

X_BIN = 6
SMOOTH_WINDOW = 9

# 같은 선반 라인의 조각 병합 기준
MERGE_Y_THRESH = 25
MERGE_SLOPE_THRESH = 0.25


# =========================
# target shelf group 선택 설정 - IPYNB 셀 5 기준
# =========================
USE_TARGET_GROUP_SELECTION = True
TARGET_GROUP_ANGLE_THRESH = 7.0
TARGET_GROUP_MIN_WIDTH_RATIO = 0.18
TARGET_GROUP_MIN_LINES = 2
TARGET_GROUP_CENTER_WEIGHT = 0.8
TARGET_GROUP_COUNT_WEIGHT = 3.0
TARGET_GROUP_WIDTH_WEIGHT = 2.0
TARGET_GROUP_XCOVER_WEIGHT = 2.0
TARGET_GROUP_YSPAN_WEIGHT = 1.0

# =========================
# 최종 중복선 제거 설정 - IPYNB 셀 5 기준
# =========================
USE_CLOSE_SUPPRESSION = False
MIN_VERTICAL_GAP = 80
UPPER_EDGE_WIDTH_KEEP_RATIO = 0.50
MAX_SHELF_LINES = 10

# =========================
# 선 연장 설정 - IPYNB 셀 5 기준
# =========================
EXTEND_LINE_TO_FULL_WIDTH = False
EXTEND_MARGIN_RATIO = 0.08

# 서버 로그용
DEBUG_FRONT_EDGE = True

_model: Optional[YOLO] = None


def _debug(message: str) -> None:
    if DEBUG_FRONT_EDGE:
        print(f"[front_edge] {message}")


def _get_model() -> YOLO:
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"앞턱 탐지 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    _model = YOLO(str(MODEL_PATH))
    return _model


# =========================
# 셀 6. 마스크에서 윗선 추출 함수
# =========================
def polygon_to_mask(poly, h: int, w: int) -> np.ndarray:
    """YOLO segmentation polygon을 binary mask로 변환"""
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return mask

    pts = np.array(poly, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def moving_median(values, window: int = 9):
    """y값 smoothing용 median filter"""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = []
    for i in range(len(values)):
        smoothed.append(np.median(padded[i:i + window]))
    return np.array(smoothed, dtype=np.float32)


def robust_fit_line(points, max_iter: int = 5, min_points: int = 8):
    """point들로 y = ax + b 직선을 robust fitting"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < min_points:
        return None

    fit_points = points.copy()

    for _ in range(max_iter):
        if len(fit_points) < min_points:
            break

        x = fit_points[:, 0]
        y = fit_points[:, 1]

        try:
            a, b = np.polyfit(x, y, 1)
        except Exception:
            return None

        pred = a * x + b
        residual = np.abs(y - pred)
        threshold = max(6, np.percentile(residual, 80))
        keep = residual <= threshold

        if keep.sum() < min_points:
            break
        if keep.sum() == len(fit_points):
            break

        fit_points = fit_points[keep]

    if len(fit_points) < min_points:
        return None

    x = fit_points[:, 0]
    y = fit_points[:, 1]

    try:
        a, b = np.polyfit(x, y, 1)
    except Exception:
        return None

    return float(a), float(b), fit_points


def extract_upper_line_from_mask(
    mask,
    conf=1.0,
    min_width_ratio=0.06,
    min_area_ratio=0.00004,
    max_abs_angle=22,
    x_bin=6,
    smooth_window=9,
):
    """
    mask 전체 중심선이 아니라 mask의 윗변을 따라가는 선 추출.
    IPYNB에서 최종 활성화된 함수 기준으로 box 하단 band 로직은 주석 처리된 상태라 반영하지 않습니다.
    """
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    area = len(xs)
    area_ratio = area / (h * w)
    if area_ratio < min_area_ratio:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    width = x_max - x_min
    width_ratio = width / w

    if width_ratio < min_width_ratio:
        return None

    upper_points = []

    for x0 in range(x_min, x_max + 1, x_bin):
        x1 = x0 + x_bin
        idx = (xs >= x0) & (xs < x1)
        if idx.sum() == 0:
            continue

        x_mid = (x0 + x1) / 2
        y_top = np.percentile(ys[idx], 5)
        upper_points.append([x_mid, y_top])

    if len(upper_points) < 8:
        return None

    upper_points = np.array(upper_points, dtype=np.float32)
    upper_points[:, 1] = moving_median(upper_points[:, 1], window=smooth_window)

    fit_result = robust_fit_line(upper_points)
    if fit_result is None:
        return None

    a, b, fit_points = fit_result
    angle = math.degrees(math.atan(a))

    if abs(angle) > max_abs_angle:
        return None

    x1 = int(np.min(fit_points[:, 0]))
    x2 = int(np.max(fit_points[:, 0]))
    y1 = int(a * x1 + b)
    y2 = int(a * x2 + b)

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    line_width_ratio = abs(x2 - x1) / w

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "y_mid": float((y1 + y2) / 2),
        "slope": float(a),
        "intercept": float(b),
        "angle": float(angle),
        "conf": float(conf),
        "max_conf": float(conf),
        "line_length": float(line_length),
        "width_ratio": float(line_width_ratio),
        "source_count": 1,
        "_points": fit_points.tolist(),
    }


# =========================
# 셀 7. 후처리 함수
# =========================
def line_y_at(line, x):
    return line["slope"] * x + line["intercept"]


def get_line_x_center(line):
    x1 = line.get("detected_x1", line["x1"])
    x2 = line.get("detected_x2", line["x2"])
    return (x1 + x2) / 2


def get_line_x_span(line):
    x1 = line.get("detected_x1", line["x1"])
    x2 = line.get("detected_x2", line["x2"])
    return min(x1, x2), max(x1, x2)


def get_union_x_coverage(lines, image_w):
    """
    여러 선의 x구간 union coverage 계산.
    배경의 짧은 선반보다, 대상 매대처럼 넓게 퍼진 그룹을 선호하기 위함.
    """
    intervals = []

    for line in lines:
        x1, x2 = get_line_x_span(line)
        x1 = max(0, min(int(x1), image_w - 1))
        x2 = max(0, min(int(x2), image_w - 1))

        if x2 > x1:
            intervals.append((x1, x2))

    if len(intervals) == 0:
        return 0.0

    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))

    total = sum(e - s for s, e in merged)

    return total / image_w


def extend_line_to_valid_image_range(line, image_w, image_h):
    """
    선을 이미지 전체 폭까지 연장하되,
    y가 이미지 밖으로 나가면 이미지 안에 들어오는 x 범위까지만 사용.
    EXTEND_LINE_TO_FULL_WIDTH=True일 때만 사용.
    """
    new_line = line.copy()

    a = float(new_line["slope"])
    b = float(new_line["intercept"])

    new_line["detected_x1"] = int(new_line.get("detected_x1", new_line["x1"]))
    new_line["detected_y1"] = int(new_line.get("detected_y1", new_line["y1"]))
    new_line["detected_x2"] = int(new_line.get("detected_x2", new_line["x2"]))
    new_line["detected_y2"] = int(new_line.get("detected_y2", new_line["y2"]))

    x_left = 0
    x_right = image_w - 1

    if abs(a) > 1e-6:
        x_at_y0 = (0 - b) / a
        x_at_yh = ((image_h - 1) - b) / a

        valid_x_min = min(x_at_y0, x_at_yh)
        valid_x_max = max(x_at_y0, x_at_yh)

        x_left = max(0, int(math.ceil(valid_x_min)))
        x_right = min(image_w - 1, int(math.floor(valid_x_max)))

        if x_left >= x_right:
            x_left = int(new_line["detected_x1"])
            x_right = int(new_line["detected_x2"])

    y_left = int(a * x_left + b)
    y_right = int(a * x_right + b)

    y_left = int(np.clip(y_left, 0, image_h - 1))
    y_right = int(np.clip(y_right, 0, image_h - 1))

    new_line["x1"] = int(x_left)
    new_line["y1"] = int(y_left)
    new_line["x2"] = int(x_right)
    new_line["y2"] = int(y_right)
    new_line["y_mid"] = float((y_left + y_right) / 2)
    new_line["line_length"] = float(
        math.sqrt((x_right - x_left) ** 2 + (y_right - y_left) ** 2)
    )
    new_line["width_ratio"] = float(abs(x_right - x_left) / image_w)

    return new_line


def extend_line_with_detected_margin(line, image_w, image_h, margin_ratio=0.08):
    """
    이미지 전체 폭까지 연장하지 않고,
    원래 탐지된 구간 기준으로 좌우 margin만 확장한다.
    """
    new_line = line.copy()

    a = float(new_line["slope"])
    b = float(new_line["intercept"])

    detected_x1 = int(new_line.get("detected_x1", new_line["x1"]))
    detected_x2 = int(new_line.get("detected_x2", new_line["x2"]))

    new_line["detected_x1"] = int(new_line.get("detected_x1", new_line["x1"]))
    new_line["detected_y1"] = int(new_line.get("detected_y1", new_line["y1"]))
    new_line["detected_x2"] = int(new_line.get("detected_x2", new_line["x2"]))
    new_line["detected_y2"] = int(new_line.get("detected_y2", new_line["y2"]))

    if detected_x1 > detected_x2:
        detected_x1, detected_x2 = detected_x2, detected_x1

    margin = int(image_w * margin_ratio)

    x1 = max(0, detected_x1 - margin)
    x2 = min(image_w - 1, detected_x2 + margin)

    y1 = int(a * x1 + b)
    y2 = int(a * x2 + b)

    y1 = int(np.clip(y1, 0, image_h - 1))
    y2 = int(np.clip(y2, 0, image_h - 1))

    new_line["x1"] = int(x1)
    new_line["y1"] = int(y1)
    new_line["x2"] = int(x2)
    new_line["y2"] = int(y2)
    new_line["y_mid"] = float((y1 + y2) / 2)
    new_line["line_length"] = float(
        math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    )
    new_line["width_ratio"] = float(abs(x2 - x1) / image_w)

    return new_line


def refit_line_from_points(points, image_w, image_h):
    """
    여러 조각 line의 point를 합쳐 다시 하나의 직선으로 fitting
    """
    fit_result = robust_fit_line(points)

    if fit_result is None:
        return None

    a, b, fit_points = fit_result

    x1 = int(np.min(fit_points[:, 0]))
    x2 = int(np.max(fit_points[:, 0]))

    y1 = int(a * x1 + b)
    y2 = int(a * x2 + b)

    x1 = int(np.clip(x1, 0, image_w - 1))
    x2 = int(np.clip(x2, 0, image_w - 1))
    y1 = int(np.clip(y1, 0, image_h - 1))
    y2 = int(np.clip(y2, 0, image_h - 1))

    angle = math.degrees(math.atan(a))
    line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    width_ratio = abs(x2 - x1) / image_w

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "y_mid": float((y1 + y2) / 2),
        "slope": float(a),
        "intercept": float(b),
        "angle": float(angle),
        "line_length": float(line_length),
        "width_ratio": float(width_ratio),
        "_points": fit_points.tolist()
    }


def merge_duplicate_lines(lines, image_w, image_h, y_thresh=25, slope_thresh=0.25):
    """
    같은 선반 row가 여러 조각으로 잡힌 경우 하나로 병합.
    단, y 차이가 큰 서로 다른 선반 라인은 병합하지 않음.
    """
    if len(lines) == 0:
        return []

    lines = sorted(lines, key=lambda x: x["y_mid"])
    groups = []

    for line in lines:
        placed = False

        for group in groups:
            ref = group[0]
            x_ref = image_w / 2

            y_line = line_y_at(line, x_ref)
            y_ref = line_y_at(ref, x_ref)

            y_diff = abs(y_line - y_ref)
            slope_diff = abs(line["slope"] - ref["slope"])

            if y_diff <= y_thresh and slope_diff <= slope_thresh:
                group.append(line)
                placed = True
                break

        if not placed:
            groups.append([line])

    merged = []

    for group in groups:
        all_points = []
        confs = []

        for line in group:
            all_points.extend(line["_points"])
            confs.append(line.get("conf", 1.0))

        all_points = np.asarray(all_points, dtype=np.float32)

        merged_line = refit_line_from_points(
            all_points,
            image_w=image_w,
            image_h=image_h
        )

        if merged_line is None:
            merged_line = sorted(
                group,
                key=lambda x: x.get("line_length", 0),
                reverse=True
            )[0]

        merged_line["conf"] = float(np.mean(confs))
        merged_line["max_conf"] = float(np.max(confs))
        merged_line["source_count"] = int(len(group))

        # detected 좌표가 없으면 현재 좌표를 detected로 저장
        merged_line["detected_x1"] = int(merged_line.get("detected_x1", merged_line["x1"]))
        merged_line["detected_y1"] = int(merged_line.get("detected_y1", merged_line["y1"]))
        merged_line["detected_x2"] = int(merged_line.get("detected_x2", merged_line["x2"]))
        merged_line["detected_y2"] = int(merged_line.get("detected_y2", merged_line["y2"]))

        merged.append(merged_line)

    return merged


def build_angle_groups(lines, angle_thresh=7.0):
    """
    merged_lines를 angle 기준으로 그룹화.
    비슷한 기울기의 선반선끼리 같은 후보 그룹으로 묶는다.
    """
    if len(lines) == 0:
        return []

    lines_sorted = sorted(lines, key=lambda x: x["angle"])

    groups = []

    for line in lines_sorted:
        placed = False

        for group in groups:
            group_angles = [g["angle"] for g in group]
            group_angle_median = float(np.median(group_angles))

            if abs(line["angle"] - group_angle_median) <= angle_thresh:
                group.append(line)
                placed = True
                break

        if not placed:
            groups.append([line])

    return groups


def split_group_by_x_position(group, image_w, x_center_thresh_ratio=0.18):
    """
    angle이 비슷한 그룹 안에서, x 위치가 비슷한 선들끼리 다시 sub-grouping.
    """
    if len(group) == 0:
        return []

    x_center_thresh = image_w * x_center_thresh_ratio
    group_sorted = sorted(group, key=lambda line: get_line_x_center(line))

    subgroups = []

    for line in group_sorted:
        placed = False
        line_center = get_line_x_center(line)

        for subgroup in subgroups:
            centers = [get_line_x_center(g) for g in subgroup]
            subgroup_center = float(np.median(centers))

            if abs(line_center - subgroup_center) <= x_center_thresh:
                subgroup.append(line)
                placed = True
                break

        if not placed:
            subgroups.append([line])

    return subgroups


def score_target_group(group, image_w, image_h):
    """
    target shelf group 후보 점수 계산.
    """
    if len(group) == 0:
        return -1e9

    widths = np.array([line.get("width_ratio", 0.0) for line in group], dtype=float)
    y_mids = np.array([line.get("y_mid", 0.0) for line in group], dtype=float)
    x_centers = np.array([get_line_x_center(line) / image_w for line in group], dtype=float)

    n_lines = len(group)
    median_width = float(np.median(widths)) if len(widths) > 0 else 0.0
    x_coverage = float(get_union_x_coverage(group, image_w))
    y_span = float((np.max(y_mids) - np.min(y_mids)) / image_h) if len(y_mids) > 1 else 0.0
    center_penalty = float(abs(np.mean(x_centers) - 0.5))

    # 너무 짧은 선이 많은 그룹은 감점
    short_ratio = float(np.mean(widths < TARGET_GROUP_MIN_WIDTH_RATIO)) if len(widths) > 0 else 1.0

    score = (
        TARGET_GROUP_COUNT_WEIGHT * n_lines
        + TARGET_GROUP_WIDTH_WEIGHT * median_width
        + TARGET_GROUP_XCOVER_WEIGHT * x_coverage
        + TARGET_GROUP_YSPAN_WEIGHT * y_span
        - TARGET_GROUP_CENTER_WEIGHT * center_penalty
        - 1.5 * short_ratio
    )

    return float(score)


def select_target_shelf_group(merged_lines, image_w, image_h, verbose=False):
    """
    merged_lines 중에서 대상 매대에 해당하는 선반 그룹만 선택.
    IPYNB 개선 버전 그대로: angle 그룹화 → x 위치 sub-grouping → 최고 점수 subgroup 선택.
    """
    if not USE_TARGET_GROUP_SELECTION:
        return merged_lines

    if len(merged_lines) == 0:
        return []

    angle_groups = build_angle_groups(
        merged_lines,
        angle_thresh=TARGET_GROUP_ANGLE_THRESH
    )

    group_infos = []
    global_group_idx = 0

    for angle_group_idx, angle_group in enumerate(angle_groups):
        subgroups = split_group_by_x_position(
            angle_group,
            image_w=image_w,
            x_center_thresh_ratio=0.18
        )

        for subgroup_idx, subgroup in enumerate(subgroups):
            score = score_target_group(subgroup, image_w, image_h)

            angles = [line["angle"] for line in subgroup]
            y_mids = [line["y_mid"] for line in subgroup]
            widths = [line.get("width_ratio", 0.0) for line in subgroup]
            x_centers = [get_line_x_center(line) / image_w for line in subgroup]

            group_infos.append({
                "group_idx": global_group_idx,
                "angle_group_idx": angle_group_idx,
                "subgroup_idx": subgroup_idx,
                "group": subgroup,
                "score": score,
                "n_lines": len(subgroup),
                "angle_median": float(np.median(angles)),
                "y_min": float(np.min(y_mids)),
                "y_max": float(np.max(y_mids)),
                "median_width": float(np.median(widths)),
                "x_coverage": float(get_union_x_coverage(subgroup, image_w)),
                "x_center_mean": float(np.mean(x_centers)),
            })

            global_group_idx += 1

    if len(group_infos) == 0:
        return merged_lines

    # 최소 선 개수 조건 만족 그룹 우선
    valid_groups = [
        info for info in group_infos
        if info["n_lines"] >= TARGET_GROUP_MIN_LINES
    ]

    if len(valid_groups) == 0:
        valid_groups = group_infos

    best_info = max(valid_groups, key=lambda x: x["score"])

    selected = best_info["group"]
    selected = sorted(selected, key=lambda x: x["y_mid"])

    if verbose:
        print("\n[target_shelf_group 디버그]")
        for info in sorted(group_infos, key=lambda x: x["score"], reverse=True):
            print(
                f"group={info['group_idx']} | "
                f"angle_group={info['angle_group_idx']} | "
                f"subgroup={info['subgroup_idx']} | "
                f"score={info['score']:.3f} | "
                f"n={info['n_lines']} | "
                f"angle={info['angle_median']:.2f} | "
                f"x_center={info['x_center_mean']:.2f} | "
                f"y=({info['y_min']:.1f}, {info['y_max']:.1f}) | "
                f"median_width={info['median_width']:.3f} | "
                f"x_cov={info['x_coverage']:.3f}"
            )

        print("선택된 group:", best_info["group_idx"])

    return selected


def suppress_abnormally_close_lines(lines, image_w):
    """
    가까운 중복선 제거.
    지금은 USE_CLOSE_SUPPRESSION=False면 사용되지 않음.
    """
    if len(lines) <= 1:
        return lines

    x_ref = image_w / 2

    for line in lines:
        line["_y_ref"] = float(line_y_at(line, x_ref))

    lines = sorted(lines, key=lambda x: x["_y_ref"])

    keep = [True] * len(lines)

    for i in range(len(lines) - 1):
        if not keep[i]:
            continue

        y_gap = lines[i + 1]["_y_ref"] - lines[i]["_y_ref"]
        slope_diff = abs(lines[i + 1]["slope"] - lines[i]["slope"])

        if y_gap <= MIN_VERTICAL_GAP and slope_diff <= MERGE_SLOPE_THRESH:
            upper = lines[i]
            lower = lines[i + 1]

            upper_width = upper.get("width_ratio", 0)
            lower_width = lower.get("width_ratio", 0)
            best_width = max(upper_width, lower_width)

            if upper_width >= best_width * UPPER_EDGE_WIDTH_KEEP_RATIO:
                keep[i + 1] = False
            else:
                keep[i] = False

    selected = [line for line, k in zip(lines, keep) if k]

    for line in selected:
        line.pop("_y_ref", None)

    return selected


def select_final_shelf_lines(lines, image_w, image_h):
    """
    최종 선반 기준선 선택
    - 짧은 선 제거
    - score 계산
    - 필요 시 가까운 중복선 제거
    - 선 연장 방식 선택
    - 위에서 아래 순서로 shelf_index 부여
    """
    final_lines = []

    for line in lines:
        width_ratio = line.get("width_ratio", 0)

        if width_ratio < MIN_FINAL_WIDTH_RATIO:
            continue

        length_score = min(width_ratio, 1.0)
        conf_score = line.get("max_conf", line.get("conf", 1.0))

        line = dict(line)
        line["score"] = float(0.75 * length_score + 0.25 * conf_score)

        final_lines.append(line)

    if USE_CLOSE_SUPPRESSION:
        final_lines = suppress_abnormally_close_lines(
            final_lines,
            image_w=image_w
        )

    final_lines = sorted(
        final_lines,
        key=lambda x: x["score"],
        reverse=True
    )

    final_lines = final_lines[:MAX_SHELF_LINES]
    final_lines = sorted(final_lines, key=lambda x: x["y_mid"])

    if EXTEND_LINE_TO_FULL_WIDTH:
        final_lines = [
            extend_line_to_valid_image_range(
                line,
                image_w=image_w,
                image_h=image_h
            )
            for line in final_lines
        ]
    else:
        final_lines = [
            extend_line_with_detected_margin(
                line,
                image_w=image_w,
                image_h=image_h,
                margin_ratio=EXTEND_MARGIN_RATIO
            )
            for line in final_lines
        ]

    final_lines = sorted(final_lines, key=lambda x: x["y_mid"])

    for idx, line in enumerate(final_lines, start=1):
        line["shelf_index"] = idx

    return final_lines


def line_to_jsonable(line):
    """json 저장 가능한 형태로 변환"""
    return {
        "shelf_index": int(line.get("shelf_index", -1)),
        "x1": int(line["x1"]),
        "y1": int(line["y1"]),
        "x2": int(line["x2"]),
        "y2": int(line["y2"]),
        "y_mid": float(line["y_mid"]),
        "slope": float(line["slope"]),
        "intercept": float(line["intercept"]),
        "angle": float(line["angle"]),
        "conf": float(line.get("conf", 0.0)),
        "max_conf": float(line.get("max_conf", line.get("conf", 0.0))),
        "line_length": float(line.get("line_length", 0.0)),
        "width_ratio": float(line.get("width_ratio", 0.0)),
        "score": float(line.get("score", 0.0)),
        "source_count": int(line.get("source_count", 1)),
        "detected_x1": int(line.get("detected_x1", line["x1"])),
        "detected_y1": int(line.get("detected_y1", line["y1"])),
        "detected_x2": int(line.get("detected_x2", line["x2"])),
        "detected_y2": int(line.get("detected_y2", line["y2"])),
        "target_group_selected": bool(line.get("target_group_selected", False)),
    }


# =========================
# 셀 8. 이미지 예측 함수 + 추가 후처리
# =========================
def filter_front_large_shelf_candidates(lines, image_w, image_h):
    """
    왼쪽/오른쪽 배경 선반, 이미지 상단 배경선을 제외하고
    앞쪽 큰 매대에 해당할 가능성이 높은 line만 남긴다.

    디버그 출력 추가:
    - merged_lines 중 어떤 선이 왜 탈락했는지 확인하기 위함.
    """
    filtered = []

    for idx, line in enumerate(lines):
        x1 = line["x1"]
        x2 = line["x2"]
        y_mid = line["y_mid"]

        x_min = min(x1, x2)
        x_max = max(x1, x2)

        width_ratio = (x_max - x_min) / image_w
        x_center_ratio = ((x_min + x_max) / 2) / image_w
        y_ratio = y_mid / image_h
        angle = line.get("angle", 0.0)
        conf = line.get("conf", 0.0)

        reason = None

        # 기존 ipynb 기준 조건
        if width_ratio < 0.25:
            reason = f"width_ratio too small: {width_ratio:.3f}"
        elif x_center_ratio < 0.25:
            reason = f"x_center too left: {x_center_ratio:.3f}"
        elif x_center_ratio > 0.90:
            reason = f"x_center too right: {x_center_ratio:.3f}"
        elif y_ratio < 0.18:
            reason = f"y_ratio too top: {y_ratio:.3f}"

        print(
            f"[front_edge][filter] idx={idx} "
            f"y_mid={y_mid:.1f} "
            f"angle={angle:.2f} "
            f"conf={conf:.2f} "
            f"width={width_ratio:.3f} "
            f"x_center={x_center_ratio:.3f} "
            f"y_ratio={y_ratio:.3f} "
            f"=> {'DROP: ' + reason if reason else 'KEEP'}"
        )

        if reason is not None:
            continue

        filtered.append(line)

    return filtered


def recover_lines_by_gap_anomaly(
    final_lines,
    candidate_lines,
    image_height,
    gap_ratio_thresh=1.45,
    min_gap_height_ratio=0.18,
    y_margin_ratio=0.18,
    duplicate_y_thresh=45
):
    """
    선반 개수를 고정하지 않고,
    final_lines 사이의 비정상적으로 큰 y gap 안에서만 누락 후보 line을 복구하는 함수.
    """
    if len(final_lines) < 2:
        return final_lines

    final_lines = sorted(final_lines, key=lambda x: x["y_mid"])

    gaps = []
    for i in range(len(final_lines) - 1):
        gap = final_lines[i + 1]["y_mid"] - final_lines[i]["y_mid"]
        gaps.append(gap)

    if len(gaps) == 0:
        return final_lines

    median_gap = np.median(gaps)
    recovered_lines = list(final_lines)

    for i, gap in enumerate(gaps):
        upper = final_lines[i]
        lower = final_lines[i + 1]

        is_large_gap = gap > median_gap * gap_ratio_thresh
        is_physically_large = gap > image_height * min_gap_height_ratio

        if not (is_large_gap and is_physically_large):
            continue

        y_min = upper["y_mid"]
        y_max = lower["y_mid"]

        expected_y = (y_min + y_max) / 2
        y_margin = gap * y_margin_ratio

        near_candidates = []
        for cand in candidate_lines:
            cy = cand["y_mid"]

            if any(abs(cy - line["y_mid"]) < duplicate_y_thresh for line in recovered_lines):
                continue

            if y_min + duplicate_y_thresh < cy < y_max - duplicate_y_thresh:
                if abs(cy - expected_y) <= max(y_margin, 70):
                    near_candidates.append(cand)

        if len(near_candidates) == 0:
            continue

        best = max(
            near_candidates,
            key=lambda x: (
                x.get("width_ratio", 0) * 0.6
                + x.get("conf", 0) * 0.3
                - abs(x["y_mid"] - expected_y) / image_height * 0.1
            )
        )

        best = dict(best)
        best["recovered"] = True
        best["recovery_reason"] = "large_gap_candidate"
        recovered_lines.append(best)

    recovered_lines = sorted(recovered_lines, key=lambda x: x["y_mid"])

    for idx, line in enumerate(recovered_lines, start=1):
        line["shelf_index"] = idx

    return recovered_lines


def synthesize_missing_lines_by_gap(
    final_lines,
    image_w,
    image_h,
    gap_ratio_thresh=1.35,
    min_gap_height_ratio=0.10,
):
    """
    YOLO mask 후보 자체가 없는 큰 y gap에 대해 위/아래 선을 기준으로 synthetic line을 생성.
    IPYNB에서 쓰던 min positive gap 기준 보간 방식 그대로 반영.
    """
    if len(final_lines) < 2:
        return final_lines

    lines = sorted([dict(line) for line in final_lines], key=lambda x: x["y_mid"])
    gaps = [lines[i + 1]["y_mid"] - lines[i]["y_mid"] for i in range(len(lines) - 1)]
    positive_gaps = [g for g in gaps if g > 0]

    if not positive_gaps:
        return lines

    reference_gap = min(positive_gaps)
    min_large_gap = max(reference_gap * gap_ratio_thresh, image_h * min_gap_height_ratio)
    synthesized = []
    x_ref = image_w / 2

    for i, gap in enumerate(gaps):
        upper = lines[i]
        lower = lines[i + 1]
        synthesized.append(upper)

        if gap < min_large_gap:
            continue

        y_ref = (upper["y_mid"] + lower["y_mid"]) / 2
        slope = float((upper["slope"] + lower["slope"]) / 2)
        intercept = float(y_ref - slope * x_ref)

        ux1 = upper.get("detected_x1", upper["x1"])
        ux2 = upper.get("detected_x2", upper["x2"])
        lx1 = lower.get("detected_x1", lower["x1"])
        lx2 = lower.get("detected_x2", lower["x2"])

        neighbor_xs = [ux1, ux2, lx1, lx2]
        x_margin = int(image_w * 0.015)
        x1 = int(np.clip(min(neighbor_xs) - x_margin, 0, image_w - 1))
        x2 = int(np.clip(max(neighbor_xs) + x_margin, 0, image_w - 1))
        if x1 > x2:
            x1, x2 = x2, x1

        y1 = int(np.clip(slope * x1 + intercept, 0, image_h - 1))
        y2 = int(np.clip(slope * x2 + intercept, 0, image_h - 1))
        angle = math.degrees(math.atan(slope))
        line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        synth_line = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "y_mid": float((y1 + y2) / 2),
            "slope": float(slope),
            "intercept": float(intercept),
            "angle": float(angle),
            "conf": 0.0,
            "max_conf": 0.0,
            "line_length": float(line_length),
            "width_ratio": float(abs(x2 - x1) / image_w),
            "score": 0.0,
            "source_count": 0,
            "detected_x1": x1,
            "detected_y1": y1,
            "detected_x2": x2,
            "detected_y2": y2,
            "target_group_selected": False,
            "synthesized": True,
            "synthesis_reason": "large_gap_interpolation",
        }
        synthesized.append(synth_line)

    synthesized.append(lines[-1])
    synthesized = sorted(synthesized, key=lambda x: x["y_mid"])

    for idx, line in enumerate(synthesized, start=1):
        line["shelf_index"] = idx

    return synthesized


def snap_synthesized_lines_to_strong_edge(lines, img_bgr, image_w, image_h):
    """
    synthesized line을 주변 강한 가로 edge 위치로 보정.
    IPYNB의 grad_y + whole-line offset 방식 그대로 반영.
    """
    if len(lines) == 0:
        return lines

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.abs(grad_y)

    snapped_lines = []

    for line in lines:
        if not line.get("synthesized", False):
            snapped_lines.append(line)
            continue

        a = float(line["slope"])
        b = float(line["intercept"])
        x1 = int(min(line["x1"], line["x2"]))
        x2 = int(max(line["x1"], line["x2"]))

        if x2 - x1 < image_w * 0.15:
            snapped_lines.append(line)
            continue

        step = max(10, int(image_w / 120))
        xs = np.arange(x1, x2 + 1, step, dtype=np.int32)
        if len(xs) < 12:
            snapped_lines.append(line)
            continue

        offsets = np.arange(-70, 151, 3, dtype=np.int32)
        best_score = -1e9
        best_offset = 0

        for off in offsets:
            ys = np.round(a * xs + b + off).astype(np.int32)
            valid = (ys >= 2) & (ys < image_h - 2)
            if valid.sum() < len(xs) * 0.65:
                continue

            vxs = xs[valid]
            vys = ys[valid]
            samples = []
            coverage_hits = 0

            for x, y in zip(vxs, vys):
                xl = max(0, int(x) - 18)
                xr = min(image_w - 1, int(x) + 18)
                row_score = float(edge_strength[int(y), xl:xr + 1].mean())
                samples.append(row_score)
                if row_score > 18:
                    coverage_hits += 1

            if not samples:
                continue

            samples = np.asarray(samples, dtype=np.float32)
            mean_edge = float(np.mean(samples))
            p70_edge = float(np.percentile(samples, 70))
            continuity = coverage_hits / max(1, len(samples))

            distance_penalty = abs(float(off)) / 150.0
            lower_bonus = max(0.0, float(off)) / 150.0 * 4.0

            score = 0.55 * mean_edge + 0.35 * p70_edge + 18.0 * continuity - 7.0 * distance_penalty + lower_bonus

            if score > best_score:
                best_score = score
                best_offset = int(off)

        if best_score < 18:
            snapped_lines.append(line)
            continue

        new_b = b + best_offset
        new_line = dict(line)
        new_y1 = int(np.clip(a * x1 + new_b, 0, image_h - 1))
        new_y2 = int(np.clip(a * x2 + new_b, 0, image_h - 1))
        angle = math.degrees(math.atan(a))

        new_line.update({
            "x1": x1,
            "y1": new_y1,
            "x2": x2,
            "y2": new_y2,
            "detected_x1": x1,
            "detected_y1": new_y1,
            "detected_x2": x2,
            "detected_y2": new_y2,
            "y_mid": float((new_y1 + new_y2) / 2),
            "slope": float(a),
            "intercept": float(new_b),
            "angle": float(angle),
            "line_length": float(math.sqrt((x2 - x1) ** 2 + (new_y2 - new_y1) ** 2)),
            "width_ratio": float(abs(x2 - x1) / image_w),
            "snapped_to_edge": True,
            "snap_mode": "whole_line_offset",
            "snap_offset": int(best_offset),
            "snap_score": float(best_score),
        })
        snapped_lines.append(new_line)

    snapped_lines = sorted(snapped_lines, key=lambda x: x["y_mid"])
    for idx, line in enumerate(snapped_lines, start=1):
        line["shelf_index"] = idx

    return snapped_lines


def suppress_close_keep_upper_line(lines, image_w, image_h, y_thresh_ratio=0.035, slope_thresh=0.08):
    """
    최종 결과에서 너무 가까운 중복선 제거. 가까우면 위쪽 선 유지.
    IPYNB v10 safe 방식 반영.
    """
    if len(lines) <= 1:
        return lines

    y_thresh = max(14, image_h * y_thresh_ratio)
    x_ref = image_w / 2
    items = []

    for line in lines:
        item = dict(line)
        item["_y_ref"] = float(line_y_at(item, x_ref))
        items.append(item)

    items = sorted(items, key=lambda x: x["_y_ref"])
    kept = []

    for line in items:
        duplicate_idx = None

        for idx, kept_line in enumerate(kept):
            y_gap = abs(line["_y_ref"] - kept_line["_y_ref"])
            slope_gap = abs(line["slope"] - kept_line["slope"])

            l1, r1 = sorted([line["x1"], line["x2"]])
            l2, r2 = sorted([kept_line["x1"], kept_line["x2"]])
            overlap = max(0, min(r1, r2) - max(l1, l2))
            min_width = max(1, min(r1 - l1, r2 - l2))
            overlap_ratio = overlap / min_width

            if y_gap <= y_thresh and slope_gap <= slope_thresh and overlap_ratio >= 0.45:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            kept.append(line)
        else:
            # 가까운 중복이면 y가 더 위인 선 유지
            if line["_y_ref"] < kept[duplicate_idx]["_y_ref"]:
                kept[duplicate_idx] = line

    kept = sorted(kept, key=lambda x: x["_y_ref"])
    for line in kept:
        line.pop("_y_ref", None)

    for idx, line in enumerate(kept, start=1):
        line["shelf_index"] = idx

    return kept


def read_image_bgr(image_path) -> np.ndarray:
    """
    IPYNB의 read_image_bgr와 동일한 이미지 로더.
    한글 경로/일반 경로 모두 PIL로 읽고 BGR(OpenCV)로 변환한다.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}\n원인: {e}")


def _predict_upper_lines_from_image(
    image: Image.Image,
    model: YOLO,
    imgsz=IMGSZ,
    conf_thres=CONF_THRES,
    iou_thres=IOU_THRES,
    min_width_ratio=MIN_CANDIDATE_WIDTH_RATIO,
    min_area_ratio=MIN_AREA_RATIO,
    verbose_group=False,
):
    """
    IPYNB의 predict_upper_lines_for_image()와 최대한 동일하게 실행한다.

    중요:
    - 노트북은 model.predict(source=str(image_path)) 로 실행했다.
    - 이전 서버 코드는 PIL/np array를 바로 넣어서, 같은 모델이어도 mask 좌표/후처리 결과가 달라질 수 있었다.
    - 그래서 여기서는 입력 PIL 이미지를 임시 jpg 파일로 저장한 뒤, 노트북처럼 path 기반으로 예측한다.
    """
    image_rgb = image.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        image_rgb.save(tmp_path, format="JPEG", quality=95)

        # IPYNB read_image_bgr(image_path)와 동일한 방식
        img = read_image_bgr(tmp_path)
        h, w = img.shape[:2]

        # IPYNB와 동일하게 source=str(image_path)
        result = model.predict(
            source=str(tmp_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            retina_masks=True,
            verbose=False,
        )[0]

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    raw_lines = []

    if result.masks is None:
        return [], result

    polys = result.masks.xy

    if result.boxes is not None and result.boxes.conf is not None:
        confs = result.boxes.conf.detach().cpu().numpy().tolist()
    else:
        confs = [1.0] * len(polys)

    for poly, conf in zip(polys, confs):
        mask = polygon_to_mask(poly, h, w)

        line = extract_upper_line_from_mask(
            mask,
            conf=conf,
            min_width_ratio=min_width_ratio,
            min_area_ratio=min_area_ratio,
            max_abs_angle=MAX_ABS_ANGLE,
            x_bin=X_BIN,
            smooth_window=SMOOTH_WINDOW,
        )

        if line is not None:
            raw_lines.append(line)

    _debug(f"raw_lines={len(raw_lines)}")

    merged_lines = merge_duplicate_lines(
        raw_lines,
        image_w=w,
        image_h=h,
        y_thresh=MERGE_Y_THRESH,
        slope_thresh=MERGE_SLOPE_THRESH
    )

    _debug(f"merged_lines={len(merged_lines)}")

    front_candidate_lines = filter_front_large_shelf_candidates(
        merged_lines,
        image_w=w,
        image_h=h
    )

    _debug(f"front_candidate_lines={len(front_candidate_lines)}")

    group_input_lines = (
        front_candidate_lines
        if len(front_candidate_lines) >= TARGET_GROUP_MIN_LINES
        else merged_lines
    )

    if verbose_group or DEBUG_FRONT_EDGE:
        print("front_candidate_lines count:", len(front_candidate_lines))
        print("target group input line count:", len(group_input_lines))

    # front_candidate_lines까지 통과했다면 이미 앞쪽 큰 매대 후보로 필터링된 상태다.
    # 여기서 다시 select_target_shelf_group()을 적용하면
    # angle이 다른 실제 앞턱 후보가 버려질 수 있으므로,
    # 디버그 단계에서는 group_input_lines를 그대로 살린다.
    if len(front_candidate_lines) >= TARGET_GROUP_MIN_LINES:
        target_group_lines = sorted(group_input_lines, key=lambda x: x["y_mid"])

        if verbose_group:
            print("[front_edge] skip select_target_shelf_group")
            print("[front_edge] use all front_candidate_lines as target_group_lines")

    else:
        target_group_lines = select_target_shelf_group(
            group_input_lines,
            image_w=w,
            image_h=h,
            verbose=verbose_group
        )

    _debug(f"target_group_lines={len(target_group_lines)}")

    for line in target_group_lines:
        line["target_group_selected"] = True

    final_lines = select_final_shelf_lines(
        target_group_lines,
        image_w=w,
        image_h=h
    )

    _debug(f"final_lines after select_final={len(final_lines)}")

    recovery_candidates = front_candidate_lines if len(front_candidate_lines) > 0 else merged_lines
    final_lines = recover_lines_by_gap_anomaly(
        final_lines=final_lines,
        candidate_lines=recovery_candidates,
        image_height=img.shape[0],
        gap_ratio_thresh=1.18,
        min_gap_height_ratio=0.08,
        y_margin_ratio=0.38,
        duplicate_y_thresh=45
    )

    _debug(f"final_lines after recover={len(final_lines)}")

    final_lines = [
        extend_line_with_detected_margin(
            line,
            image_w=w,
            image_h=h,
            margin_ratio=EXTEND_MARGIN_RATIO
        )
        if line.get("recovered", False) and not EXTEND_LINE_TO_FULL_WIDTH
        else line
        for line in final_lines
    ]

    final_lines = sorted(final_lines, key=lambda x: x["y_mid"])
    for idx, line in enumerate(final_lines, start=1):
        line["shelf_index"] = idx

    final_lines = synthesize_missing_lines_by_gap(
        final_lines,
        image_w=w,
        image_h=h,
        gap_ratio_thresh=1.28,
        min_gap_height_ratio=0.08
    )

    _debug(f"final_lines after synthesize={len(final_lines)}")

    final_lines = snap_synthesized_lines_to_strong_edge(
        final_lines,
        img_bgr=img,
        image_w=w,
        image_h=h
    )

    final_lines = suppress_close_keep_upper_line(
        final_lines,
        image_w=w,
        image_h=h,
        y_thresh_ratio=0.035,
        slope_thresh=0.08
    )

    final_lines = refine_line_endpoints_by_edge_support(
    final_lines,
    img_bgr=img,
    image_w=w,
    image_h=h,
    max_extend_ratio=0.22,
    step_px=8,
    patch_half_width=18,
    min_abs_edge_score=10.0,
    support_ratio=0.45,
    max_miss_count=2,
    )

    final_lines = extend_short_right_edges(
    final_lines,
    image_w=w,
    image_h=h,
    min_width_ratio=0.35,
    right_percentile=85,
    margin_ratio=0.01,
    )

    front_left_x = estimate_front_shelf_left_x(
    final_lines,
    image_w=w,
    image_h=h,
    )

    if front_left_x is not None:
        trimmed_lines = []

        for line in final_lines:
            trimmed = trim_background_part_from_line(
                line,
                image_w=w,
                image_h=h,
                front_left_x=front_left_x,
            )

            if trimmed is not None:
                trimmed_lines.append(trimmed)

        final_lines = sorted(trimmed_lines, key=lambda x: x["y_mid"])

        for idx, line in enumerate(final_lines, start=1):
            line["shelf_index"] = idx

    _debug(f"final_lines after close suppression={len(final_lines)}")

    final_lines = sorted(final_lines, key=lambda x: x["y_mid"])
    for idx, line in enumerate(final_lines, start=1):
        line["shelf_index"] = idx

    _debug(f"final_lines final={len(final_lines)}")

    return final_lines, result

def detect_front_edge_points(image: Image.Image) -> List[Dict[str, Any]]:
    """
    shelf.front_edge_points가 null일 때 실행.

    YOLO segmentation mask에서 앞턱 윗선을 추출하고,
    IPYNB 후처리 흐름을 거쳐 row_no별 front_edge_points를 반환한다.

    이 함수는 slot 정보를 사용하지 않는다.
    """
    model = _get_model()

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    h, _ = image_np.shape[:2]

    lines, _ = _predict_upper_lines_from_image(
        image=image,
        model=model,
        imgsz=IMGSZ,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        min_width_ratio=MIN_CANDIDATE_WIDTH_RATIO,
        min_area_ratio=MIN_AREA_RATIO,
        verbose_group=False,
    )

    if not lines:
        raise ValueError("앞턱 윗선 추출 결과가 없습니다.")

    front_edge_points: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines, start=1):
        front_band_px = max(60, int(h * 0.04))

        front_edge_points.append(
            {
                "row_no": idx,
                "front_y": int(round(float(line["y_mid"]))),
                "points_xy": [
                    [int(line["x1"]), int(line["y1"])],
                    [int(line["x2"]), int(line["y2"])],
                ],
                "polygon": None,
                "front_band_px": int(front_band_px),
                "back_band_px": int(front_band_px * 2),
                "conf": round(float(line.get("conf", 0.0)), 2),
                "angle": round(float(line.get("angle", 0.0)), 2),
                "width_ratio": round(float(line.get("width_ratio", 0.0)), 4),
            }
        )

    return front_edge_points


# 디버깅용: line 전체 정보가 필요할 때만 사용

def detect_front_edge_lines_debug(image: Image.Image) -> List[Dict[str, Any]]:
    model = _get_model()
    lines, _ = _predict_upper_lines_from_image(image=image, model=model, verbose_group=True)
    return [line_to_jsonable(line) for line in lines]

def trim_background_part_from_line(line, image_w, image_h, front_left_x):
    """
    한 line이 왼쪽 배경 매대까지 이어진 경우,
    front_left_x 기준으로 왼쪽 배경 부분을 잘라낸다.
    """

    new_line = dict(line)

    x1 = int(new_line["x1"])
    x2 = int(new_line["x2"])

    left_x = min(x1, x2)
    right_x = max(x1, x2)

    # 이미 front_left_x 오른쪽에서 시작하면 그대로 둔다.
    if left_x >= front_left_x:
        return new_line

    # 자르고 나서 너무 짧아지면 버린다.
    if right_x - front_left_x < image_w * 0.12:
        return None

    a = float(new_line["slope"])
    b = float(new_line["intercept"])

    new_left_x = int(front_left_x)
    new_left_y = int(np.clip(a * new_left_x + b, 0, image_h - 1))

    # x1, x2 순서 보존
    if x1 <= x2:
        new_line["x1"] = new_left_x
        new_line["y1"] = new_left_y
    else:
        new_line["x2"] = new_left_x
        new_line["y2"] = new_left_y

    new_line["y_mid"] = float((new_line["y1"] + new_line["y2"]) / 2)
    new_line["line_length"] = float(
        math.sqrt(
            (new_line["x2"] - new_line["x1"]) ** 2
            + (new_line["y2"] - new_line["y1"]) ** 2
        )
    )
    new_line["width_ratio"] = float(abs(new_line["x2"] - new_line["x1"]) / image_w)
    new_line["background_trimmed"] = True
    new_line["front_left_x"] = int(front_left_x)

    return new_line

def estimate_front_shelf_left_x(lines, image_w, image_h):
    """
    slot 없이 대상 매대의 왼쪽 시작 x를 추정한다.

    아이디어:
    - x=0 근처에서 시작하는 선은 배경 매대까지 섞였을 가능성이 큼
    - 정상적인 앞쪽 매대 선들은 어느 정도 오른쪽에서 시작함
    - 그 정상 후보들의 left x 중 가장 왼쪽 값을 target shelf left로 본다
    """

    left_candidates = []

    for line in lines:
        x1 = int(line["x1"])
        x2 = int(line["x2"])
        left_x = min(x1, x2)

        y_ratio = float(line["y_mid"]) / image_h
        width_ratio = float(line.get("width_ratio", 0.0))

        # 너무 위쪽 배경선 제외
        if y_ratio < 0.15:
            continue

        # 너무 짧은 선 제외
        if width_ratio < 0.20:
            continue

        # x=0 근처에서 시작하는 건 배경까지 섞인 선일 가능성이 크므로 제외
        if left_x < image_w * 0.10:
            continue

        left_candidates.append(left_x)

    if len(left_candidates) == 0:
        return None

    # 너무 많이 오른쪽으로 자르지 않기 위해 가장 왼쪽 정상 시작점 사용
    front_left_x = int(min(left_candidates) - image_w * 0.015)
    front_left_x = int(np.clip(front_left_x, 0, image_w - 1))

    return front_left_x

def extend_short_right_edges(
    lines,
    image_w,
    image_h,
    min_width_ratio=0.35,
    right_percentile=85,
    margin_ratio=0.01,
):
    """
    일부 앞턱선이 오른쪽 끝까지 가지 못하고 짧게 끝나는 문제 보정.

    - y 위치와 angle은 그대로 유지
    - 오른쪽 끝점만 다른 정상 line들의 오른쪽 끝 기준으로 확장
    - slot 정보 사용 안 함
    """

    if len(lines) == 0:
        return lines

    right_candidates = []

    for line in lines:
        x1 = int(line["x1"])
        x2 = int(line["x2"])

        left = min(x1, x2)
        right = max(x1, x2)

        width_ratio = (right - left) / image_w

        if width_ratio >= min_width_ratio:
            right_candidates.append(right)

    if len(right_candidates) == 0:
        return lines

    target_right = int(np.percentile(right_candidates, right_percentile))
    target_right = int(np.clip(target_right + image_w * margin_ratio, 0, image_w - 1))

    extended = []

    for line in lines:
        new_line = dict(line)

        x1 = int(new_line["x1"])
        x2 = int(new_line["x2"])

        left = min(x1, x2)
        right = max(x1, x2)
        width_ratio = (right - left) / image_w

        # 이미 충분히 오른쪽까지 가면 그대로 둔다
        if right >= target_right * 0.97:
            extended.append(new_line)
            continue

        # 짧게 끝난 선만 오른쪽으로 확장
        if width_ratio < min_width_ratio or right < target_right:
            a = float(new_line["slope"])
            b = float(new_line["intercept"])

            new_right_x = target_right
            new_right_y = int(np.clip(a * new_right_x + b, 0, image_h - 1))

            if x1 <= x2:
                new_line["x2"] = int(new_right_x)
                new_line["y2"] = int(new_right_y)
            else:
                new_line["x1"] = int(new_right_x)
                new_line["y1"] = int(new_right_y)

            new_line["y_mid"] = float((new_line["y1"] + new_line["y2"]) / 2)
            new_line["line_length"] = float(
                math.sqrt(
                    (new_line["x2"] - new_line["x1"]) ** 2
                    + (new_line["y2"] - new_line["y1"]) ** 2
                )
            )
            new_line["width_ratio"] = float(abs(new_line["x2"] - new_line["x1"]) / image_w)
            new_line["right_edge_extended"] = True
            new_line["target_right_x"] = int(target_right)

        extended.append(new_line)

    extended = sorted(extended, key=lambda x: x["y_mid"])

    for idx, line in enumerate(extended, start=1):
        line["shelf_index"] = idx

    return extended

def refine_line_endpoints_by_edge_support(
    lines,
    img_bgr,
    image_w,
    image_h,
    max_extend_ratio=0.22,
    step_px=8,
    patch_half_width=18,
    min_abs_edge_score=10.0,
    support_ratio=0.45,
    max_miss_count=2,
):
    """
    정면/측면 분기 없이, 실제 이미지 edge가 이어지는 만큼만
    앞턱선의 좌우 끝점을 보정한다.

    핵심:
    - line의 slope/intercept는 유지한다.
    - 왼쪽/오른쪽으로 조금씩 이동하면서 해당 위치 주변의 edge 강도를 확인한다.
    - edge가 충분히 강하면 endpoint를 확장한다.
    - edge가 연속으로 약하면 확장을 멈춘다.

    장점:
    - 정면 매대: 수평 앞턱 edge가 이어지므로 왼쪽/오른쪽으로 자연스럽게 확장됨
    - 측면 매대: 해당 기울기 방향으로 실제 edge가 없으면 확장되지 않음
    """

    if len(lines) == 0:
        return lines

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 선반 앞턱은 주로 수평/사선 경계라 y방향 gradient가 잘 잡힘
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.abs(grad_y)

    def edge_score_at(line, x):
        a = float(line["slope"])
        b = float(line["intercept"])

        y = int(round(a * x + b))

        if x < 0 or x >= image_w or y < 2 or y >= image_h - 2:
            return None

        x1 = max(0, int(x) - patch_half_width)
        x2 = min(image_w - 1, int(x) + patch_half_width)

        # 해당 y 주변의 edge 평균
        patch = edge_strength[y - 1:y + 2, x1:x2 + 1]

        if patch.size == 0:
            return None

        return float(np.mean(patch))

    refined = []

    for line in lines:
        new_line = dict(line)

        x1 = int(new_line["x1"])
        x2 = int(new_line["x2"])

        left = min(x1, x2)
        right = max(x1, x2)

        if right <= left:
            refined.append(new_line)
            continue

        # 현재 line 내부 edge score를 기준으로 adaptive threshold 생성
        sample_xs = np.linspace(left, right, num=30)
        inner_scores = []

        for sx in sample_xs:
            score = edge_score_at(new_line, int(round(sx)))
            if score is not None:
                inner_scores.append(score)

        if len(inner_scores) < 5:
            refined.append(new_line)
            continue

        base_score = float(np.percentile(inner_scores, 60))
        threshold = max(min_abs_edge_score, base_score * support_ratio)

        max_extend_px = int(image_w * max_extend_ratio)

        # -------------------------
        # 왼쪽으로 확장
        # -------------------------
        best_left = left
        miss_count = 0

        x = left - step_px
        min_x = max(0, left - max_extend_px)

        while x >= min_x:
            score = edge_score_at(new_line, x)

            if score is not None and score >= threshold:
                best_left = x
                miss_count = 0
            else:
                miss_count += 1

            if miss_count >= max_miss_count:
                break

            x -= step_px

        # -------------------------
        # 오른쪽으로 확장
        # -------------------------
        best_right = right
        miss_count = 0

        x = right + step_px
        max_x = min(image_w - 1, right + max_extend_px)

        while x <= max_x:
            score = edge_score_at(new_line, x)

            if score is not None and score >= threshold:
                best_right = x
                miss_count = 0
            else:
                miss_count += 1

            if miss_count >= max_miss_count:
                break

            x += step_px

        # 너무 과하게 짧아지거나 이상하면 기존 유지
        if best_right - best_left < image_w * 0.20:
            refined.append(new_line)
            continue

        a = float(new_line["slope"])
        b = float(new_line["intercept"])

        new_left_y = int(np.clip(a * best_left + b, 0, image_h - 1))
        new_right_y = int(np.clip(a * best_right + b, 0, image_h - 1))

        # x1, x2 순서 보존
        if x1 <= x2:
            new_line["x1"] = int(best_left)
            new_line["y1"] = int(new_left_y)
            new_line["x2"] = int(best_right)
            new_line["y2"] = int(new_right_y)
        else:
            new_line["x1"] = int(best_right)
            new_line["y1"] = int(new_right_y)
            new_line["x2"] = int(best_left)
            new_line["y2"] = int(new_left_y)

        new_line["y_mid"] = float((new_line["y1"] + new_line["y2"]) / 2)
        new_line["line_length"] = float(
            math.sqrt(
                (new_line["x2"] - new_line["x1"]) ** 2
                + (new_line["y2"] - new_line["y1"]) ** 2
            )
        )
        new_line["width_ratio"] = float(abs(new_line["x2"] - new_line["x1"]) / image_w)
        new_line["endpoint_refined_by_edge"] = True
        new_line["edge_threshold"] = float(threshold)

        refined.append(new_line)

    refined = sorted(refined, key=lambda x: x["y_mid"])

    for idx, line in enumerate(refined, start=1):
        line["shelf_index"] = idx

    return refined