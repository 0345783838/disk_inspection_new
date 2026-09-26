import io
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from src.dtos.meta import Params, UvParams, UvBox
from src.service.check_disk_service_yolo import DiskCheckingService
from src.service.control_plc_service import PlcControllingService

inspection_router = APIRouter()
communication_router = APIRouter()
disk_checking_service = DiskCheckingService()
plc_controlling_service = PlcControllingService()
logger = logging.getLogger(__name__)


def decode_image(image):
    if image.file is None:
        raise HTTPException(status_code=400, detail="Missing image")
    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    try:
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as ex:
        raise HTTPException(status_code=400, detail="Invalid image") from ex
    if decoded is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return decoded


def parse_form(model, value):
    try:
        return model.model_validate_json(value)
    except (ValidationError, ValueError, TypeError) as ex:
        raise HTTPException(status_code=400, detail="Invalid inspection parameters") from ex


def inspect(operation, *args):
    started = time.perf_counter()
    try:
        return operation(*args)
    except (ValueError, SyntaxError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    finally:
        logger.info("%s completed in %.1f ms", operation.__name__, (time.perf_counter() - started) * 1000)


@inspection_router.post('/check_disk_white')
def check_disk_white(image: UploadFile = File(...)):
    return inspect(disk_checking_service.check_disk_white, decode_image(image))


@inspection_router.post('/check_disk_uv')
def check_disk_uv(image: UploadFile = File(...), uv_box: str = Form(...)):
    geometry = parse_form(UvBox, uv_box)
    return inspect(disk_checking_service.check_disk_uv, decode_image(image),
                   geometry.crop_box, geometry.uv_box_1, geometry.uv_box_2,
                   geometry.mid_1, geometry.mid_2)


@inspection_router.post('/check_disk_swagger')
def check_disk_swagger(image: UploadFile = File(...)):
    rendered = inspect(disk_checking_service.check_disk_swagger, decode_image(image))
    success, encoded = cv2.imencode('.jpg', rendered)
    if not success:
        raise HTTPException(status_code=500, detail="Cannot encode inspection image")
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


@inspection_router.post('/check_disk_debug')
def check_disk_debug(image: UploadFile = File(...), params_json: str = Form(...)):
    return inspect(disk_checking_service.check_disk_debug, decode_image(image), parse_form(Params, params_json))


@inspection_router.post('/check_disk_uv_debug')
def check_disk_uv_debug(image: UploadFile = File(...), params_json: str = Form(...)):
    return inspect(disk_checking_service.check_disk_uv_debug, decode_image(image), parse_form(UvParams, params_json))


@inspection_router.get('/check_service_status')
def check_service_status():
    return {"status": "running"}


@communication_router.get(path='/connect_plc')
def connect_plc(ip: str, port: int):
    return {"Success": plc_controlling_service.connect_plc(ip, port)}


@communication_router.get(path='/disconnect_plc')
def disconnect_plc():
    return {"Success": plc_controlling_service.disconnect_plc()}


@communication_router.get(path='/control_uv_light')
def control_uv(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_uv() if status else plc_controlling_service.turn_off_uv()}


@communication_router.get(path='/control_white_light')
def control_led_1(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_led() if status else plc_controlling_service.turn_off_led()}


@communication_router.get(path='/check_connection')
def check_connection():
    return {"Success": plc_controlling_service.check_connection()}


@communication_router.get(path='/read_trigger')
def read_trigger():
    success, status = plc_controlling_service.read_trigger()
    return {"Success": success, "Status": status}


@communication_router.get(path='/reset_trigger')
def reset_trigger():
    return {"Success": plc_controlling_service.reset_trigger()}


@communication_router.get(path='/on_error_abnormal')
def on_error_abnormal():
    return {"Success": plc_controlling_service.on_error_abnormal()}


@communication_router.get(path='/on_error_mixing')
def on_error_mixing():
    return {"Success": plc_controlling_service.on_error_mixing()}


@communication_router.get(path='/on_ok_signal')
def on_ok_signal():
    return {"Success": plc_controlling_service.on_ok_signal()}


@communication_router.get(path='/off_ok_signal')
def off_ok_signal():
    return {"Success": plc_controlling_service.off_ok_signal()}

