from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.config import ROUTE_HEALTH, ROUTE_PREDICT
from app.view import JsonResponseView

router = APIRouter()


class PredictRequest(BaseModel):
    image_base64: str


@router.get(ROUTE_HEALTH)
def health():
    return {"status": "ok"}


@router.post(ROUTE_PREDICT)
def predict_digit(body: PredictRequest, request: Request):

    view = JsonResponseView()
    request.app.state.presenter.predict(body.image_base64, view)

    return view.render()
