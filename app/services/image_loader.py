from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def load_image(image_url: str) -> Image.Image:
    """
    image_url로 이미지를 읽는다.

    지원:
    - S3 presigned URL
    - 일반 http/https URL
    - 로컬 이미지 경로
    """

    if image_url.startswith("http://") or image_url.startswith("https://"):
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    image_path = Path(image_url)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_url}")

    return Image.open(image_path).convert("RGB")