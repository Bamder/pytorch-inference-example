"""composition root：创建应用、加载 Model、挂载 View。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dl_model.model.structure.classifier import Classifier
from app.api import router
from app.config import (
    API_PREFIX,
    CHECKPOINT_PATH,
    INPUT_RESOLUTION,
    STATIC_DIR,
    STATIC_MOUNT_NAME,
)
from app.model import RecognizerModel
from app.presenter import PredictionPresenter


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = RecognizerModel(Classifier(INPUT_RESOLUTION))
    model.load_checkpoint(CHECKPOINT_PATH)

    app.state.presenter = PredictionPresenter(model=model)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router, prefix=API_PREFIX)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name=STATIC_MOUNT_NAME)
