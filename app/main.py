from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AnalyzeShelfRequest, AnalyzeShelfResponse
from app.services.analysis_repository import AnalysisRepository
from app.services.image_loader import load_image
from app.services.judgement_service import analyze_stock_results
from app.services.product_detector import run_product_detection
from app.services.shelf_lip_detector import detect_front_edge_points
from app.services.slot_mapper import map_detections_to_slots
from app.services.visualizer import draw_analysis_result
from app.services.backend_client import post_analysis_completed

app = FastAPI(title="Be:show AI Server")
# app.mount(
#     "/test_images",
#     StaticFiles(directory="test_images"),
#     name="test_images",
# )


def _safe_mark_failed(repo: AnalysisRepository, shelf_image_id: int, message: str) -> None:
    try:
        repo.mark_analysis_failed(shelf_image_id=shelf_image_id, message=message)
    except Exception:
        # 실패 상태 저장 중 발생한 DB 예외가 원래 분석 예외를 덮어쓰지 않도록 한다.
        pass


def _attach_product_ids_to_detections(detections, product_id_by_class_id):
    for det in detections:
        class_id = int(det["class_id"])
        det["product_id"] = product_id_by_class_id.get(class_id)
    return detections


@app.post("/api/beshow/analysis", response_model=AnalyzeShelfResponse)
def analyze_shelf(request: AnalyzeShelfRequest):
    repo = AnalysisRepository()

    try:
        # 1. 요청을 받자마자 shelf_image 상태를 분석중으로 변경
        repo.mark_analysis_processing(request.shelf_image_id)

        # 2. DB에서 shelf/store/front_edge/slot/planogram/inventory 조회
        context = repo.get_analysis_context(
            shelf_image_id=request.shelf_image_id,
            shelf_id=request.shelf_id,
        )
        slots = context["slots"]

        if not slots:
            raise HTTPException(
                status_code=400,
                detail="해당 shelf에 활성 planogram/slot 정보가 없어 분석할 수 없습니다.",
            )

        # 3. image_url로 이미지 로드
        image = load_image(request.image_url)
        image_width, image_height = image.size

        # 4. shelf.front_edge_points가 있으면 DB 값을 사용, 없으면 기존 앞턱 모델로 탐지
        front_edge_points = context.get("front_edge_points") or []

        # if isinstance(front_edge_points, str):
        #     front_edge_points = json.loads(front_edge_points)
        front_edge_source = "DB"

        if not front_edge_points:
            try:
                front_edge_points = detect_front_edge_points(image)
                front_edge_source = "AI_DETECTED"
                repo.save_shelf_front_edge_points(
                    shelf_id=context["shelf_id"],
                    front_edge_points=front_edge_points,
                )
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "shelf.front_edge_points가 비어 있습니다. "
                        "AI 서버에서 앞턱을 탐지하려면 weights/shelf_lip_best.pt 파일이 필요합니다. "
                        f"{str(e)}"
                    ),
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"앞턱 윗선 추출 중 오류가 발생했습니다: {str(e)}",
                )

        # 5. 상품 탐지 실행 후 DB product_id 매핑
        detections = run_product_detection(image)
        product_id_by_class_id = repo.get_product_id_by_class_id()
        detections = _attach_product_ids_to_detections(
            detections=detections,
            product_id_by_class_id=product_id_by_class_id,
        )

        # 6. DB slot 기준으로 detection을 slot에 매핑하고 FRONT/BACK 계산
        mapped_detections = map_detections_to_slots(
            detections=detections,
            slots=slots,
            front_edge_points=front_edge_points,
        )

        # 7. slot별 최종 stock 상태 생성
        stock_results = analyze_stock_results(
            mapped_detections=mapped_detections,
            slots=slots,
        )

        # 8. 디버그 이미지 저장
        debug_image_path = draw_analysis_result(
            image=image,
            front_edge_points=front_edge_points,
            slots=slots,
            detections=mapped_detections,
            shelf_image_id=request.shelf_image_id,
        )

        # 9. detection_result / stock / alarm 저장
        repo.save_analysis_results(
            shelf_image_id=request.shelf_image_id,
            mapped_detections=mapped_detections,
            stock_results=stock_results,
        )

        summary = {
            "shelf_id": request.shelf_id,
            "shelf_image_id": request.shelf_image_id,
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
            "debug_image_path": debug_image_path,
            "slot_source": "DB",
            "slot_count": len(slots),
        }

        # 10. 모든 저장이 끝난 후 shelf_image 상태를 완료로 변경
        repo.mark_analysis_completed(
            shelf_image_id=request.shelf_image_id,
            image_width=image_width,
            image_height=image_height,
            message="AI analysis completed",
        )

        # 분석 완료 후 백엔드에 완료 알림 POST
        try:
            post_analysis_completed(
                shelf_id=request.shelf_id,
                shelf_image_id=request.shelf_image_id,
                status="COMPLETE",
                summary=summary,
                message="AI analysis completed",
            )
        except Exception as callback_error:
            # callback 실패 때문에 분석 자체를 실패 처리하지는 않음
            print(f"[WARN] 백엔드 callback 실패: {callback_error}")

        return AnalyzeShelfResponse(
            shelf_image_id=request.shelf_image_id,
            front_edge_points=front_edge_points,
            detections=mapped_detections,
            stock_results=stock_results,
            summary=summary,
        )

    except HTTPException as e:
        _safe_mark_failed(repo, request.shelf_image_id, str(e.detail))
        raise e
    except ValueError as e:
        _safe_mark_failed(repo, request.shelf_image_id, str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _safe_mark_failed(repo, request.shelf_image_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"AI 분석 처리 중 오류가 발생했습니다: {str(e)}",
        )


@app.get("/ai/debug-image/{shelf_image_id}")
def get_debug_image(shelf_image_id: int):
    image_path = Path("debug_outputs") / f"analyze_{shelf_image_id}.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="디버그 이미지를 찾을 수 없습니다.",
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"analyze_{shelf_image_id}.png",
    )
