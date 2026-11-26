from pathlib import Path

# torchvision extracts MNIST into <root>/MNIST/raw/
DATA_ROOT = Path(__file__).resolve().parent / "data"
MNIST_RAW_DIR = DATA_ROOT / "MNIST" / "raw"

TRAIN_IMAGES = MNIST_RAW_DIR / "train-images-idx3-ubyte"
TRAIN_LABELS = MNIST_RAW_DIR / "train-labels-idx1-ubyte"
TEST_IMAGES = MNIST_RAW_DIR / "t10k-images-idx3-ubyte"
TEST_LABELS = MNIST_RAW_DIR / "t10k-labels-idx1-ubyte"
