import os
import re
from fastapi import APIRouter, HTTPException, UploadFile, status
from paddleocr import PaddleOCR
import requests
from models.OCRModel import *
from models.RestfulModel import *
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "id")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)


def group_data(txts):
    data = {
        "products": [],
        "discount": None,
        "total": None
    }

    # Regex untuk mendeteksi nama produk, harga, diskon, dan total
    product_name_pattern = re.compile(r'^[A-Za-z0-9\s.,\-]+$')
    price_pattern = re.compile(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)')
    discount_pattern = re.compile(r'DISKON:?\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)')
    total_pattern = re.compile(r'TOTAL:?\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)')

    i = 0
    while i < len(txts):
        if i + 2 < len(txts):
            product_name = txts[i].strip()
            quantity_str = txts[i + 1].strip()
            price_str = txts[i + 2].strip()

            # Validasi nama produk dengan regex
            if not product_name_pattern.match(product_name):
                i += 1
                continue

            try:
                quantity = int(quantity_str)
            except ValueError:
                quantity = None

            match = price_pattern.search(price_str)
            if match:
                price = match.group(0)
                if price is not None and quantity is not None:
                    data["products"].append({
                        "name": product_name,
                        "quantity": quantity,
                        "price": price
                    })
            i += 3  # Lanjutkan ke produk berikutnya
        else:
            break

    discount_match = discount_pattern.search(' '.join(txts))
    if discount_match:
        data["discount"] = discount_match.group(1)

    total_match = total_pattern.search(' '.join(txts))
    if total_match:
        data["total"] = total_match.group(1)

    return data




@router.get('/predict-by-path', response_model=RestfulModel, summary="Path")
def predict_by_path(image_path: str):
    result = ocr.ocr(image_path, cls=True)
    result = result[0]
    txts = [line[1][0] for line in result]
    final = group_data(txts)

    restfulModel = RestfulModel(resultcode=200, message="Success", data=final, cls=OCRModel)
    return restfulModel

@router.post('/predict-by-base64', response_model=RestfulModel, summary="Base64")
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    result = ocr.ocr(img=img, cls=True)
    txts = [line[1][0] for line in result]
    final = group_data(txts)

    restfulModel = RestfulModel(resultcode=200, message="Success", data=final, cls=OCRModel)
    return restfulModel

@router.post('/predict-by-file', response_model=RestfulModel, summary="File")
async def predict_by_file(file: UploadFile):
    restfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr.ocr(img=img, cls=True)
        result = result[0]
        txts = [line[1][0] for line in result]
        final = group_data(txts)

        restfulModel = RestfulModel(resultcode=200, message="Success", data=final, cls=OCRModel)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=".jpg or .png files are required.")
    
    return restfulModel

@router.get('/predict-by-url', response_model=RestfulModel, summary="URL")
async def predict_by_url(imageUrl: str):
    restfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = ocr.ocr(img=img, cls=True)
        result = result[0]
        txts = [line[1][0] for line in result]
        final = group_data(txts)

        restfulModel = RestfulModel(resultcode=200, message="Success", data=final, cls=OCRModel)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=".jpg or .png image URLs only.")
    
    return restfulModel
