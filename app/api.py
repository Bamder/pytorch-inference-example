from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.view import JsonResponseView

router = APIRouter()


class PredictRequest(BaseModel):
    image_base64: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
def predict_digit(body: PredictRequest, request: Request):

    view = JsonResponseView()
    request.app.state.presenter.predict(body.image_base64, view)

    return view.render()
