from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FrontEdgePoint(BaseModel):
    row_no: int
    front_y: Optional[float] = None
    points_xy: Optional[List[List[float]]] = None
    polygon: Optional[List[List[float]]] = None
    front_band_px: Optional[int] = 80
    back_band_px: Optional[int] = 160

    # 앞턱 모델 추출 품질 참고용. DB 저장 필수값은 아님.
    conf: Optional[float] = None
    angle: Optional[float] = None
    width_ratio: Optional[float] = None


class SlotInput(BaseModel):
    slot_id: int
    slot_code: str

    x: float
    y: float
    width: float
    height: float

    row_no: Optional[int] = None
    col_no: Optional[int] = None

    # 백엔드가 말한 리스트 번호 기준이라면 product_id = YOLO class_id
    product_id: int
    product_name: Optional[str] = None

    expected_quantity: int
    min_front_quantity: int
    min_display_quantity: int

    # Entity 기준 Inventory.totalQuantity
    total_quantity: int
    reorder_point: int


class AnalyzeShelfRequest(BaseModel):
    shelf_image_id: int
    shelf_id: int
    store_id: int

    image_url: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    # null이면 AI 서버가 shelf_lip_best.pt로 앞턱 윗선 추출
    front_edge_points: Optional[List[FrontEdgePoint]] = None

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
    is_misplaced: bool

    front_quantity: int
    back_quantity: int
    detected_quantity: int

    confidence: float
    status_reason: str


class AnalyzeShelfResponse(BaseModel):
    shelf_image_id: int

    # Spring이 shelf.front_edge_points가 null일 때 저장할 수 있도록 반환
    front_edge_points: Optional[List[FrontEdgePoint]] = None

    detections: List[DetectionResponse]
    stock_results: List[StockResultResponse]
    summary: Dict[str, Any]