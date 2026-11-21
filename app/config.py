from http import HTTPStatus
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHECKPOINT_PATH = ROOT / "dl_model" / "checkpoints" / "mnist_classifier.pt"

# ---------- model / preprocessing ----------
IMAGE_SIZE = 28
INPUT_RESOLUTION = (IMAGE_SIZE, IMAGE_SIZE)
BOX = 20        # 数字缩放后的最大边长
THRESHOLD = 50  # 二值化阈值（像素值高于此值视为笔画）

# ---------- api ----------
API_PREFIX = "/api"
ROUTE_HEALTH = "/health"
ROUTE_PREDICT = "/predict"
STATIC_MOUNT_NAME = "static"

# ---------- response ----------
STATUS_OK = HTTPStatus.OK
STATUS_BAD_REQUEST = HTTPStatus.BAD_REQUEST
STATUS_UNPROCESSABLE_ENTITY = HTTPStatus.UNPROCESSABLE_ENTITY

TOP_K = 3
DETAIL_INVALID_IMAGE = "invalid image"
DETAIL_EMPTY_DRAWING = "empty drawing"
