import time
import cv2
import numpy as np
from src.dtos.meta import DataResponse, ErrorCode, DataDebugResponse
from src.service.base_service import BaseService
import base64


class PlcControllingService(BaseService):
    def __init__(self):
        super().__init__()
        pass

    def connect_plc(self, ip, port):
        return self.plc_controller.connect(ip, port)

    def disconnect_plc(self):
        return self.plc_controller.disconnect()

    def turn_on_uv(self):
        return self.plc_controller.on_uv()

    def turn_off_uv(self):
        return self.plc_controller.off_uv()

    def turn_on_led(self):
        return self.plc_controller.on_led()

    def turn_off_led(self):
        return self.plc_controller.off_led()

    def check_connection(self):
        return self.plc_controller.check_connection()

    def read_trigger(self):
        return self.plc_controller.read_trigger()

    def reset_trigger(self):
        return self.plc_controller.reset_trigger()

    def on_error_abnormal(self):
        return self.plc_controller.on_error_abnormal()

    def on_error_mixing(self):
        return self.plc_controller.on_error_mixing()

    def on_ok_signal(self):
        return self.plc_controller.on_ok_signal()

    def off_ok_signal(self):
        return self.plc_controller.off_ok_signal()