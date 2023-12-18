from typing import Any, Union
from piq import FID
import torch

class FIDMetric:
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, first_dl, second_dl) -> Any:
        metric = FID()
        return metric(first_dl, second_dl)