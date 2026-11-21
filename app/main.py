"""composition root：创建应用、加载 Model、挂载 View。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dl_model.model.structure.classifier import Classifier
from app.api import router
from app.model import RecognizerModel
from app.presenter import PredictionPresenter

STATIC = Path(__file__).parent / "static"
CHECKPOINT = Path(__file__).parents[1] / "dl_model" / "checkpoints" / "mnist_classifier.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = RecognizerModel(Classifier((28, 28)))
    model.load_checkpoint(CHECKPOINT)

    app.state.presenter = PredictionPresenter(model=model)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
