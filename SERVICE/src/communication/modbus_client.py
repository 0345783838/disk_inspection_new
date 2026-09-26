from pymodbus.client import ModbusTcpClient
import traceback
import time
from threading import RLock
from functools import wraps


def synchronized(operation):
    @wraps(operation)
    def call(self, *args, **kwargs):
        with self._lock:
            return operation(self, *args, **kwargs)
    return call


class MbClient:
    # ip="192.168.0.211", port=8000,
    def __init__(self, id=1):
        self.HMI_IP = "192.168.2.10"
        self.PORT = 8000
        self.UNIT_ID = id
        self.client = None
        self._lock = RLock()

    @synchronized
    def connect(self, ip, port):
        try:
            self.disconnect()
            self.client = ModbusTcpClient(ip, port=port, timeout=1)
            if self.client.connect():
                self.HMI_IP = ip
                self.PORT = port
                return True
            else:
                return False

        except Exception:
            print("[Mb Client] Exception", traceback.format_exc())
            return False

    @synchronized
    def disconnect(self):
        if self.client:
            try:
                self.client.close()
                self.client = None
                return True
            except Exception:
                return False
        return True

    # ================= CHECK CONNECTION =================
    @synchronized
    def check_connection(self) -> bool:
        """
        Kiểm tra kết nối Modbus còn sống hay không
        """
        if not self.client:
            return False

        try:
            rr = self.client.read_coils(30, count=1)
            if rr is None or rr.isError():
                return False

            return True

        except Exception as ex:
            return False

    # ====================================================

    def on_uv(self):
        return self.__write_bit(20, True)

    def off_uv(self):
        return self.__write_bit(20, False)

    def on_led(self):
        return self.__write_bit(21, True)

    def off_led(self):
        return self.__write_bit(21, False)

    def on_ok_signal(self):
        return self.__write_bit(30, True)

    def off_ok_signal(self):
        return self.__write_bit(30, False)

    def on_error_abnormal(self):
        return self.__write_bit(32, True)

    def off_error_abnormal(self):
        return self.__write_bit(32, False)

    def on_error_mixing(self):
        return self.__write_bit(33, True)

    def off_error_mixing(self):
        return self.__write_bit(33, False)

    def read_trigger(self):
        return self.__read_bit(0)

    def reset_trigger(self):
        return self.__write_bit(0, False)

    @synchronized
    def __write_bit(self, addr: int, value: bool = False):
        try:
            # if not self.check_connection():
            #     return False

            rr = self.client.write_coil(addr, value)
            if rr is None or rr.isError():
                return False

            return True

        except Exception:
            return False

    @synchronized
    def __read_bit(self, addr: int):
        try:
            # if not self.check_connection():
            #     return -1

            rr = self.client.read_coils(addr, count=1)
            if rr is None or rr.isError():
                return -1, False
            if not rr.bits:
                return -1, False
            return 1, rr.bits[0]

        except Exception:
            print(traceback.format_exc())
            return -1, False


if __name__ == "__main__":
    a = MbClient(1)
    a.connect("192.168.2.10",8000)
    while True:
        time.sleep(1)
        a.read_trigger()