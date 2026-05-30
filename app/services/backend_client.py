import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


def post_analysis_completed(
    shelf_id: int,
    shelf_image_id: int,
    status: str,
    summary: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    AI 분석 완료 후 백엔드에 결과 완료 알림 POST.
    DB 저장은 이미 AI 서버에서 끝낸 상태라고 가정.
    """
    base_url = os.getenv("BACKEND_BASE_URL")
    callback_path = os.getenv(
        "BACKEND_ANALYSIS_COMPLETED_PATH",
        "/api/beshow/analysis/completed",
    )

    if not base_url:
        raise ValueError("BACKEND_BASE_URL 환경변수가 설정되지 않았습니다.")

    url = base_url.rstrip("/") + "/" + callback_path.lstrip("/")

    payload = {
        "shelf_id": shelf_id,
        "shelf_image_id": shelf_image_id,
        "status": status,
        "summary": summary or {},
        "message": message,
    }

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    try:
        return response.json()
    except Exception:
        return {"text": response.text}