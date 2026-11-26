import torch
import argparse

from torch import nn
from pathlib import Path
from tqdm import tqdm

from dl_model.model.scripts import data_loader
from dl_model.model.structure.classifier import Classifier
from dl_model.datasets.paths import (
    TEST_IMAGES,
    TEST_LABELS,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "mnist_classifier.pt"


class Trainer:

    def __init__(
        self, 
        model : nn.Module, 
        optimizer : torch.optim.Optimizer | None = None,
        device: str | torch.device | None = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = self._resolve_device(device)
    

    def _resolve_device(self, device : str | torch.device | None = None):
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device


    def set_optimizer(self, optimizer : torch.optim.Optimizer):
        self.optimizer = optimizer


    def set_device(self, device: str | torch.device | None = None):
        self.device = self._resolve_device(device)


    def load_checkpoint(
        self,
        checkpoint_path : Path,
        only_model : bool = False
    ):
        
        self.model.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])      

        if only_model:
            return
        
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been initiallized yet. Please build optimizer with the same algorithm as checkpoint.")
 
        self.optimizer.load_state_dict(checkpoint["optimizer"])


    def save(self, path : Path):

        if self.optimizer is None:
            raise RuntimeError("Can not save optimizer as it is not initiallized. Please set_optimizer() or run train() first.")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_type": type(self.optimizer).__name__,
            "optimizer_hparams": {
                key: value
                for key, value in self.optimizer.param_groups[0].items()
                if key != "params"          
            },
            "epoch": "unsaved",
            "loss": "unsaved"
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(path)


    def train(
        self,
        sample_path : Path,
        label_path : Path,
        batch_size : int = 16,
        epochs : int = 30,
        learning_rate : float = 1e-3
    ):

        self.model.to(self.device) # migrate the tensors to particular device

        images, labels = data_loader.load_set( 
            sample_path=sample_path,
            label_path=label_path
        )
        images = torch.from_numpy(images).unsqueeze(1).float().div_(255.0)
        labels= torch.from_numpy(labels).long()

        loss_func = nn.CrossEntropyLoss(reduction="mean")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # set training flag
        self.model.train()

        for epoch in range(epochs):
            print(f"* epoch {epoch} / {epochs}")

            batch_generator = data_loader.BatchGenerator(
                samples=images, 
                labels=labels, 
                batch_size=batch_size
            )

            for batch in tqdm(batch_generator, total=len(batch_generator), desc=f"epoch {epoch}"):
                
                batch_images, batch_labels = batch

                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # clear the accumulated gradient
                self.optimizer.zero_grad()

                logits = self.model(batch_images)

                # average loss of this batch
                loss = loss_func(logits, batch_labels)

                loss.backward()

                # optimize the params
                self.optimizer.step()

        self.model.train(False)


    def resume(
        self,
        checkpoint_path : Path,
        sample_path : Path,
        label_path : Path,
        batch_size : int = 16,
        epochs : int = 30,
        learning_rate : float = 1e-3
    ):

        self.load_checkpoint(checkpoint_path=checkpoint_path)
        
        self.train(
            sample_path=sample_path,
            label_path=label_path,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )


    def test(
        self, 
        sample_path : Path,
        label_path : Path,
        batch_size : int = 256
    ):

        self.model.to(self.device)
        self.model.eval()
        
        images, labels = data_loader.load_set( 
            sample_path=sample_path,
            label_path=label_path
        )

        images = torch.from_numpy(images).unsqueeze(1).float().div_(255.0)
        labels= torch.from_numpy(labels).long()

        loss_func = nn.CrossEntropyLoss(reduction="sum")

        total_loss = 0.0
        total_correction = 0
        total_samples = 0

        batch_generator = data_loader.BatchGenerator(
            samples=images,
            labels=labels,
            batch_size=batch_size,
            shuffle=False
        )

        with torch.no_grad():

            for batch in tqdm(batch_generator, total=len(batch_generator), desc="Running test..."):
                batch_images, batch_labels = batch

                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)


                logits = self.model(batch_images)

                total_loss += loss_func(logits, batch_labels).item()
                total_correction += (logits.argmax(dim=1) == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        if total_samples == 0:
            raise RuntimeError("no sample to evaluate, check dataset paths.")

        result = {
            "loss": total_loss / total_samples,
            "accuracy": total_correction / total_samples,
            "num_samples": total_samples,
        }

        print(
            f"* test loss is {result['loss']:.4f} | "
            f"accuracy {result['accuracy']:.2%} ({total_correction}/{total_samples})"
        )

        return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser  = argparse.ArgumentParser(
        prog="dl_model.model.scripts.train",
        description="Trainer for classifier: test-only or full train-test run."
    )

    parser.add_argument(
        "--mode",
        choices=("testonly", "full"),
        default="full",
        help="testonly: only load checkpoint and run test; full: run train before test (default)."
    )

    return parser

def run_with_argument(args: argparse.Namespace) -> dict:
    model = Classifier((28, 28))
    
    trainer = Trainer(model)

    if args.mode == "full":
        trainer.train(
            sample_path=TRAIN_IMAGES,
            label_path=TRAIN_LABELS,
            batch_size=64,
            epochs=30,
            learning_rate=1e-3
        )
        trainer.save(CHECKPOINT)

        print(f"* checkpoint saved to {CHECKPOINT}")

        trainer = Trainer(model)

    trainer.load_checkpoint(
        CHECKPOINT,
        only_model=True
    )

    result = trainer.test(
        sample_path=TEST_IMAGES,
        label_path=TEST_LABELS
    )

    return result



if __name__ == "__main__":
    run_with_argument(build_argument_parser().parse_args())

