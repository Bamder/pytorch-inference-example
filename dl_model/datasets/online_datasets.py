from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms

from dl_model.datasets.paths import DATA_ROOT

PREVIEW_DIR = Path(__file__).resolve().parent / "preview"

transform = transforms.ToTensor()


def get_train_set():
    return datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )


def get_test_set():
    return datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )


def export_preview(number: int, dataset):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(number):
        Image.fromarray(dataset.data[i].numpy()).save(
            PREVIEW_DIR / f"sample_{i}_{dataset.targets[i]}.png"
        )


if __name__ == "__main__":
    export_preview(10, get_train_set())
    get_test_set()  # download the test split as well
