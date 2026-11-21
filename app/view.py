import base64
from typing import Any

from fastapi.responses import JSONResponse


class JsonResponseView:

    def __init__(self):
        self._status_code = 200
        self._body: dict[str, Any] = {}

    def show_result(self, digit: int, probabilities: list[float]) -> None:

        ranked = sorted(
            enumerate(probabilities), 
            key=lambda item: item[1],
            reverse=True
        )

        self._status_code = 200
        self._body = {
            "digit": digit,
            "confidence": ranked[0][1],
            "top3": [{"digit": index, "score": score} for index, score in ranked[:3]],
        }

    def show_empty_canvas(self) -> None:
        self._status_code = 422
        self._body = {"detail": "empty drawing"}

    def show_invalid_alarm(self) -> None:
        self._status_code = 400
        self._body = {"detail": "invalid image"}

    # make data to json response
    def render(self) -> JSONResponse:
        return JSONResponse(self._body, status_code=self._status_code)


def decode_data_url(value: str) -> bytes:

    # format of dataURL: "data:[<mime>][;base64],<payload>"
    payload = value.split(",", 1)[-1]
    try:
        return base64.b64decode(payload, validate=True) # raw bytes stream
    except ValueError as exc:
        raise ValueError("invalid image") from exc



