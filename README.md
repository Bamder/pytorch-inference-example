# PyTorch Inference Service Example

A small, layered example of serving a PyTorch image classifier behind FastAPI.

Architecture: **Model–View–Presenter** (Passive View variant).

Reference task: MNIST handwritten digit recognition — draw a digit, get the prediction.

> This is an **example**, not a framework: there is no plugin system. Copy it and adapt the pieces listed under [Reusing this](#reusing-this).

## Architecture

```
HTTP
 └─ app/api.py                                           route + request schema
     └─ app/presenter.py                                 orchestrates decode → infer → render (no HTTP knowledge)
         ├─ app/model.py                                 preprocessing + inference
         │   └─ dl_model/model/structure/classifier.py   nn.Module
         └─ app/view.py                                  passive view: shapes the JSON response
```

Dependencies point inward. The presenter never touches HTTP or PyTorch; the view never decides anything, it only renders what the presenter tells it.

## Quick start

Requires Python 3.12+ and the following packages:

```bash
pip install fastapi uvicorn torch torchvision pillow numpy tqdm
```

The trained checkpoint is included at `dl_model/checkpoints/mnist_classifier.pt`.

```bash
# Windows
start.bat

# or manually run:
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> and draw a digit.

## HTTP API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| POST | `/api/predict` | Body: `{"image_base64": "data:image/png;base64,..."}` |

Successful response:

```json
{
  "digit": 7,
  "confidence": 0.99,
  "top3": [
    { "digit": 7, "score": 0.99 },
    { "digit": 1, "score": 0.004 },
    { "digit": 2, "score": 0.002 }
  ]
}
```

`400` invalid image, `422` empty canvas.

## Train / evaluate

Dataset is not committed. Download and extract MNIST first (also generates previews):

```bash
python -m dl_model.datasets.online_datasets
```

Then, from the repository root:

```bash
python -m dl_model.model.scripts.train --mode full       # train 30 epochs, then test
python -m dl_model.model.scripts.train --mode testonly   # evaluate the existing checkpoint
```

Training writes `dl_model/checkpoints/mnist_classifier.pt`.

## Layout

```
app/
  main.py        composition root: model, presenter, routes, static files
  api.py         FastAPI routes and request schema
  presenter.py   use-case orchestration
  model.py       preprocessing + inference
  view.py        passive view and JSON response
  static/        drawing canvas frontend
dl_model/
  datasets/      MNIST download and preview export
  model/structure/classifier.py   small CNN
  model/scripts/ trainer, evaluator, IDX loader, batching
  checkpoints/   trained weights
start.bat        Windows launcher
```

## Reusing this

| Want to change | Touch |
|---|---|
| Dataset / file format | `dl_model/model/scripts/data_loader.py` |
| Model architecture | `dl_model/model/structure/classifier.py`, then `app/main.py` |
| Input preprocessing | `to_tensor()` in `app/model.py` |
| Output format | `JsonResponseView` in `app/view.py` |
| UI | `app/static/` |

## Limitations

- Hard-coded for 1×28×28 grayscale input and 10 classes.
- Two data paths: a hand-written IDX loader for training, torchvision for download/preview.
- No tests in the repository yet.
