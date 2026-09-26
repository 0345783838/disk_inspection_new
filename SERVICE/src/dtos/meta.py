from typing import List, Optional
from pydantic import BaseModel, Field


class DataResponse(BaseModel):
    Result: int = Field(default=1, strict=True, ge=0, le=2)  # 0/1/2 - ok/ng/warning
    ErrorCode: Optional[str] = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: Optional[str] = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    ResImg: Optional[str] = None  # base64 result encoded image
    MaxDiskDistance: float = 0
    MinDiskDistance: float = 0
    CropBox: Optional[str] = None  # x1,x2,y1,y2
    UvBox1: Optional[str] = None  # x1,x2,y1,y2
    UvBox2: Optional[str] = None  # x1,x2,y1,y2
    Mid1: Optional[str] = None
    Mid2: Optional[str] = None


class DataResponseUv(BaseModel):
    Result: bool = False  # True/False
    ErrorCode: Optional[str] = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: Optional[str] = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    CountUvDisk: int = 0
    ResImg: Optional[str] = None  # base64 result encoded image


class DataDebugResponse(BaseModel):
    Result: int = Field(default=1, strict=True, ge=0, le=2)  # 0/1/2 - ok/ng/warning
    ErrorCode: Optional[str] = None
    ErrorDesc: Optional[str] = None
    DetectImg: Optional[str] = None  # base64 result encoded image
    SegmentImg: Optional[str] = None  # base64 result encoded image
    FinalImg: Optional[str] = None  # base64 result encoded image
    CropBox: Optional[str] = None  # x1,x2,y1,y2
    UvBox1: Optional[str] = None  # x1,x2,y1,y2
    UvBox2: Optional[str] = None  # x1,x2,y1,y2
    Mid1: Optional[str] = None
    Mid2: Optional[str] = None


class DataDebugUVResponse(BaseModel):
    Result: bool = False  # True/False
    CountUvDisk: int = 0
    ThresholdImg: Optional[str] = None  # base64 result encoded image
    FinalImg: Optional[str] = None  # base64 result encoded image


class ErrorCode:
    PASS = ("PASS", "Khay đĩa đạt chất lượng")
    ABNORMAL = ("ERROR_001", "Khay đĩa có bất thường!")
    WARNING_NUM_DISK = ("WARNING_001", "Số lượng đĩa trong khay đang thiếu!")
    ERR_NUM_DISK = ("ERROR_002", "Số lượng khe đĩa trong khay bất thường!")
    ERR_MIXING_DISK = ("ERROR_003", "Phát sinh mixing đĩa!")


class ClassifyResult:
    OK = 'ok'
    NG = 'ng'
    NO_DISK = 'no_disk'


class InspectionState:
    OK = 0
    NG = 1
    WARNING = 2


class Params(BaseModel):
    segment_threshold: float = Field(ge=0, le=1)
    segment_iou: float = Field(ge=0, le=1)
    detect_threshold: float = Field(ge=0, le=1)
    detect_iou: float = Field(ge=0, le=1)
    caliper_min_edge_distance: float = Field(gt=0)
    caliper_max_edge_distance: float = Field(gt=0)
    caliper_length_rate: float = Field(gt=0, le=1)
    caliper_thickness_list: List[int] = Field(min_length=1)
    disk_num: int = Field(gt=0)
    disk_max_distance: float = Field(ge=0)
    disk_min_distance: float = Field(ge=0)
    disk_min_area: float = Field(ge=0)


class UvBox(BaseModel):
    crop_box: str = Field(min_length=1)
    uv_box_1: str = Field(min_length=1)
    uv_box_2: str = Field(min_length=1)
    mid_1: str = Field(min_length=1)
    mid_2: str = Field(min_length=1)


class UvParams(UvBox):
    uv_disk_lower_threshold: List[int] = Field(min_length=3, max_length=3)
    uv_disk_upper_threshold: List[int] = Field(min_length=3, max_length=3)
    uv_disk_min_area: float = Field(ge=0)
