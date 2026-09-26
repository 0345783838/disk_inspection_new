# Import the Model Achitecture here!!!
from src.communication.modbus_client import MbClient
from src.inference.segment.unet_onnx_segmentor import OnnxSegmentor as OnnxSegmentorUnet
from src.inference.yolo_segment.onnx_segmentor import OnnxSegmentor
from src.inference.yolo_classify.yolo_classifier import YoloClassifier
from src.inference.yolo_detect.onnx_detector import OnnxDetector
import decouple

from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper

# Load config here!!!
config = decouple.config
Csv = decouple.Csv

if isinstance(config, decouple.AutoConfig):
    from pathlib import Path
    config = decouple.Config(decouple.RepositoryEnv(str(Path(__file__).resolve().parents[2] / 'config' / 'config.env')))

# Check if there is any config


# Get the config from the config file

# Segmentation model
DISK_SEGMENT_MODEL_PATH = config('DISK_SEGMENT_MODEL_PATH')
DISK_SEGMENT_MODEL_LABELS = config('DISK_SEGMENT_MODEL_LABELS',
                                        cast=lambda v: [s.strip() for s in v.split(',')])
DISK_SEGMENT_CONF_THRESH = config('DISK_SEGMENT_CONF_THRESH', cast=float)
DISK_SEGMENT_IOU_THRESH = config('DISK_SEGMENT_IOU_THRESH', cast=float)
DISK_SEGMENT_USING_GPU = config('DISK_SEGMENT_USING_GPU', cast=int)
DISK_SEGMENT_NUM_THREADS = config('DISK_SEGMENT_NUM_THREADS', cast=int)

# Detection model
DISK_POINT_DETECT_MODEL_PATH = config('DISK_POINT_DETECT_MODEL_PATH')
DISK_POINT_DETECT_MODEL_LABELS = config('DISK_POINT_DETECT_MODEL_LABELS',
                                        cast=lambda v: [s.strip() for s in v.split(',')])
DISK_POINT_DETECT_CONF_THRESH = config('DISK_POINT_DETECT_CONF_THRESH', cast=float)
DISK_POINT_DETECT_IOU_THRESH = config('DISK_POINT_DETECT_IOU_THRESH', cast=float)

# Classification model
POINT_CLASSIFY_MODEL_PATH = config('POINT_CLASSIFY_MODEL_PATH')
POINT_CLASSIFY_LABELS = config('POINT_CLASSIFY_LABELS', cast=lambda v: [s.strip() for s in v.split(',')])
POINT_CLASSIFY_CONF_THRESH = config('POINT_CLASSIFY_CONF_THRESH', cast=float)
POINT_CLASSIFY_IMAGE_SIZE = config('POINT_CLASSIFY_IMAGE_SIZE', cast=Csv(cast=int))
POINT_CLASSIFY_BATCH_SIZE = config('POINT_CLASSIFY_BATCH_SIZE', cast=int, default=1)

CALIPER_MIN_EDGE_DISTANCE = config('CALIPER_MIN_EDGE_DISTANCE', cast=float)
CALIPER_SUBPIXEL = config('CALIPER_SUB_PIXEL', cast=bool)
CALIPER_MAX_PAIRS = config('CALIPER_MAX_PAIRS', cast=int)
CALIPER_MAX_EDGE_DISTANCE = config('CALIPER_MAX_EDGE_DISTANCE', cast=float)
CALIPER_THICKNESS_LIST = config('CALIPER_THICKNESS_LIST', cast=lambda v: [int(s.strip()) for s in v.split(',')])
CALIPER_LENGTH_RATE = config('CALIPER_LENGTH_RATE', cast=float)
CALIPER_POLARITY = config('CALIPER_POLARITY')
CALIPER_ANGLE = config('CALIPER_ANGLE', cast=float)
CALIPER_RETURN_PROFILE = config('CALIPER_RETURN_PROFILE', cast=bool)

NUM_DISK = config('NUM_DISK', cast=int)
MAX_DISK_DISTANCE = config('MAX_DISK_DISTANCE', cast=float)
MIN_DISK_DISTANCE = config('MIN_DISK_DISTANCE', cast=float)
MIN_DISK_AREA = config('MIN_DISK_AREA', cast=float)

UV_DISK_LOWER_THRESHOLD = config('UV_DISK_LOWER_THRESHOLD', cast=lambda v: tuple(int(s.strip()) for s in v.split(',')))
UV_DISK_UPPER_THRESHOLD = config('UV_DISK_UPPER_THRESHOLD', cast=lambda v: tuple(int(s.strip()) for s in v.split(',')))
UV_MIN_DISK_AREA = config('UV_MIN_DISK_AREA', cast=float)


# Initialize the models
# disk_segmentor = OnnxSegmentorUnet(DISK_SEGMENT_MODEL_PATH,
#                                    DISK_SEGMENT_CONF_THRESH,
#                                    gpu_mode=DISK_SEGMENT_USING_GPU,
#                                    num_threads=DISK_SEGMENT_NUM_THREADS)

disk_point_detect_model = OnnxDetector(path=DISK_POINT_DETECT_MODEL_PATH,
                                       label=DISK_POINT_DETECT_MODEL_LABELS,
                                       conf_thres=DISK_POINT_DETECT_CONF_THRESH,
                                       iou_thres=DISK_POINT_DETECT_IOU_THRESH)

point_classification_model = YoloClassifier(model_path=POINT_CLASSIFY_MODEL_PATH,
                                            labels=POINT_CLASSIFY_LABELS,
                                            input_size=POINT_CLASSIFY_IMAGE_SIZE,
                                            conf_thres=POINT_CLASSIFY_CONF_THRESH,
                                            batch_size=POINT_CLASSIFY_BATCH_SIZE)

caliper = AdvancedMultiEdgeCaliper(min_edge_distance=CALIPER_MIN_EDGE_DISTANCE,
                                   subpixel=CALIPER_SUBPIXEL,
                                   max_pairs=CALIPER_MAX_PAIRS,
                                   pair_max_gap=CALIPER_MAX_EDGE_DISTANCE,
                                   thickness_list=CALIPER_THICKNESS_LIST,
                                   length_rate=CALIPER_LENGTH_RATE,
                                   polarity=CALIPER_POLARITY,
                                   angle_deg=CALIPER_ANGLE,
                                   return_profiles=CALIPER_RETURN_PROFILE)

disk_segmentor_yolo = OnnxSegmentor(path=DISK_SEGMENT_MODEL_PATH,
                                    label=DISK_SEGMENT_MODEL_LABELS,
                                    conf_thres=DISK_SEGMENT_CONF_THRESH,
                                    iou_thres=DISK_SEGMENT_IOU_THRESH,)

num_disk = NUM_DISK
max_disk_distance = MAX_DISK_DISTANCE
min_disk_distance = MIN_DISK_DISTANCE
min_disk_area = MIN_DISK_AREA

uv_disk_lower_threshold = UV_DISK_LOWER_THRESHOLD
uv_disk_upper_threshold = UV_DISK_UPPER_THRESHOLD
uv_min_disk_area = UV_MIN_DISK_AREA

plc_controller = MbClient()


class BaseService:
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # self.disk_segmentor = disk_segmentor
            self.disk_point_detect_model = disk_point_detect_model
            self.point_classification_model = point_classification_model
            self.caliper = caliper

            self.num_disk = num_disk
            self.max_disk_distance = max_disk_distance
            self.min_disk_distance = min_disk_distance
            self.min_disk_area = min_disk_area

            self.uv_disk_lower_threshold = uv_disk_lower_threshold
            self.uv_disk_upper_threshold = uv_disk_upper_threshold
            self.uv_min_disk_area = uv_min_disk_area

            self.plc_controller = plc_controller

            self.disk_segmentor_yolo = disk_segmentor_yolo
        else:
            pass
