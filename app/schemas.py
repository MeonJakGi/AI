from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FrontEdgePoint(BaseModel):
    row_no: int
    front_y: Optional[float] = None
    points_xy: Optional[List[List[float]]] = None
    polygon: Optional[List[List[float]]] = None
    front_band_px: Optional[int] = 80
    back_band_px: Optional[int] = 160


class SlotInput(BaseModel):
    slot_id: int
    slot_code: str

    x: float
    y: float
    width: float
    height: float

    row_no: int
    col_no: Optional[int] = None

    product_id: int
    product_name: Optional[str] = None

    expected_quantity: int
    min_front_quantity: int
    min_display_quantity: int

    inventory_quantity: int
    reorder_point: int


class AnalyzeShelfRequest(BaseModel):
    shelf_image_id: int
    shelf_id: int
    store_id: int

    image_url: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    front_edge_points: List[FrontEdgePoint]
    slots: List[SlotInput]


class DetectionResponse(BaseModel):
    slot_id: Optional[int]
    product_id: int
    class_name: Optional[str] = None

    x: int
    y: int
    width: int
    height: int
    confidence: float

    depth_position: str
    is_misplaced: bool
    is_low_confidence: bool


class StockResultResponse(BaseModel):
    slot_id: int
    product_id: int

    status: str
    front_quantity: int
    back_quantity: int
    detected_quantity: int
    confidence: float
    status_reason: str

    order_list_needed: bool = False


class AnalyzeShelfResponse(BaseModel):
    shelf_image_id: int
    detections: List[DetectionResponse]
    stock_results: List[StockResultResponse]
    summary: Dict[str, Any]