from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.schemas import AnalyzeShelfRequest, AnalyzeShelfResponse
from app.services.image_loader import load_image
from app.services.product_detector import run_product_detection
from app.services.shelf_lip_detector import detect_front_edge_points
from app.services.slot_mapper import map_detections_to_slots
from app.services.judgement_service import analyze_stock_results
from app.services.visualizer import draw_analysis_result, draw_front_edge_only

app = FastAPI(title="Be:show AI Server")


@app.post("/api/beshow/analysis", response_model=AnalyzeShelfResponse)
def analyze_shelf(request: AnalyzeShelfRequest):
    image = load_image(request.image_url)

    front_edge_points = request.front_edge_points
    front_edge_source = "REQUEST"

    # shelf.front_edge_points가 null이면 AI 서버가 앞턱 모델로 탐지
    if not front_edge_points:
        try:
            front_edge_points = detect_front_edge_points(image)
            front_edge_source = "AI_DETECTED"

        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "front_edge_points가 null입니다. "
                    "AI 서버에서 앞턱을 탐지하려면 "
                    "weights/shelf_lip_best.pt 파일이 필요합니다. "
                    f"{str(e)}"
                ),
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"앞턱 윗선 추출 중 오류가 발생했습니다: {str(e)}",
            )

    detections = run_product_detection(image)

    mapped_detections = map_detections_to_slots(
        detections=detections,
        slots=request.slots,
        front_edge_points=front_edge_points,
    )

    # debug_image_path = draw_analysis_result(
    # image=image,
    # front_edge_points=front_edge_points,
    # slots=request.slots,
    # detections=mapped_detections,
    # shelf_image_id=request.shelf_image_id,
    # )

    debug_image_path = draw_front_edge_only(
    image=image,
    front_edge_points=front_edge_points,
    shelf_image_id=request.shelf_image_id,
    )

    stock_results = analyze_stock_results(
        mapped_detections=mapped_detections,
        slots=request.slots,
    )

    summary = {
        "detected_count": len(mapped_detections),
        "stock_result_count": len(stock_results),
        "need_refill_count": sum(
            1 for item in stock_results
            if item["status"] == "NEED_REFILL"
        ),
        "order_needed_count": sum(
            1 for item in stock_results
            if item["status"] == "ORDER_NEEDED"
        ),
        "need_check_count": sum(
            1 for item in stock_results
            if item["status"] == "NEED_CHECK"
        ),
        "front_edge_source": front_edge_source,
        "front_edge_count": len(front_edge_points),
        "debug_image_path": debug_image_path
    }

    return AnalyzeShelfResponse(
        shelf_image_id=request.shelf_image_id,
        front_edge_points=front_edge_points,
        detections=mapped_detections,
        stock_results=stock_results,
        summary=summary,
    )

@app.get("/ai/debug-image/{shelf_image_id}")
def get_debug_image(shelf_image_id: int):
    image_path = Path("debug_outputs") / f"analyze_{shelf_image_id}.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="디버그 이미지를 찾을 수 없습니다."
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"analyze_{shelf_image_id}.png",
    )

@app.get("/ai/debug-front-edge/{shelf_image_id}")
def get_debug_front_edge_image(shelf_image_id: int):
    image_path = Path("debug_outputs") / f"front_edge_only_{shelf_image_id}.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="앞턱 디버그 이미지를 찾을 수 없습니다."
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"front_edge_only_{shelf_image_id}.png",
    )