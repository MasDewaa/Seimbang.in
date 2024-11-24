# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, UploadFile, status
from models.OCRModel import *
from models.RestfulModel import *
from paddleocr import PaddleOCR
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "id")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)


@router.get('/predict-by-path', response_model=RestfulModel, summary="Path")
def predict_by_path(image_path: str):
    result = ocr.ocr(image_path, cls=True)
    result = result[0]
    txts = [line[1][0]for line in result]
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=txts, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="Base64 ")
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    result = ocr.ocr(img=img, cls=True)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel, summary="")
async def predict_by_file(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):  
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr.ocr(img=img, cls=True)
        result = result[0]
        txts = [line[1][0]for line in result]
        restfulModel = RestfulModel(
            resultcode=200, message="Success", data=txts, cls=OCRModel)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=".jpg .png"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="URL")
async def predict_by_url(imageUrl: str):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"): 
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = ocr.ocr(img=img, cls=True)
        result = result[0]
        txts = [line[1][0]for line in result]
        restfulModel = RestfulModel(
            resultcode=200, message="Success", data=txts, cls=OCRModel)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=".jpg .png "
        )
    return restfulModel
