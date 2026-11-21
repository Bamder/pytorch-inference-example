from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
PREVIEW_DIR = HERE / "preview"

transform = transforms.ToTensor()

train_set = datasets.MNIST(root=DATA_ROOT,
                           train=True,
                           download=True,
                           transform=transform)

test_set = datasets.MNIST(root=DATA_ROOT,
                          train=False,
                          download=True,
                          transform=transform)

def export_preview(number:int, dataset):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(number):
        Image.fromarray(dataset.data[i].numpy()).save(
            PREVIEW_DIR / f"sample_{i}_{dataset.targets[i]}.png"
        )

if __name__ == "__main__":
    export_preview(10, train_set)