"""Presenter：编排「解码 → 推理 → 渲染」，不关心 HTTP 细节。"""

from app.model import RecognizerModel
from app.view import JsonResponseView, decode_data_url


class PredictionPresenter:

    def __init__(self, model: RecognizerModel, persistent_view: None = None):

        self.model = model
        self.view = persistent_view # placeholder

    def predict(self, image_data_url: str, temp_view: JsonResponseView) -> None:

        try:
            image = decode_data_url(image_data_url)
            result = self.model.recognize(image)

        except ValueError:
            temp_view.show_invalid_alarm()
            return

        if result is None:
            temp_view.show_empty_canvas()
            return

        temp_view.show_result(*result)
