from fastapi import FastAPI

from app.schemas import AnalyzeShelfRequest, AnalyzeShelfResponse
from app.services.image_loader import load_image
from app.services.product_detector import run_product_detection
from app.services.slot_mapper import map_detections_to_slots
from app.services.judgement_service import analyze_stock_results


app = FastAPI(title="Be:show AI Server")


@app.post("/ai/analyze-shelf", response_model=AnalyzeShelfResponse)
def analyze_shelf(request: AnalyzeShelfRequest):
    image = load_image(request.image_url)

    detections = run_product_detection(image)

    mapped_detections = map_detections_to_slots(
        detections=detections,
        slots=request.slots,
        front_edge_points=request.front_edge_points,
    )

    stock_results = analyze_stock_results(
        mapped_detections=mapped_detections,
        slots=request.slots,
    )

    summary = {
        "detected_count": len(mapped_detections),
        "stock_result_count": len(stock_results),
        "need_refill_count": sum(1 for item in stock_results if item["status"] == "NEED_REFILL"),
        "order_needed_count": sum(1 for item in stock_results if item.get("order_list_needed") is True),
        "need_check_count": sum(1 for item in stock_results if item["status"] == "NEED_CHECK"),
    }

    return AnalyzeShelfResponse(
        shelf_image_id=request.shelf_image_id,
        detections=mapped_detections,
        stock_results=stock_results,
        summary=summary,
    )