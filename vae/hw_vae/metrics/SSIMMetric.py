from typing import Any, Union
from piq import SSIMLoss
import torch

class SSIMMetric(SSIMLoss):
    def __init__(self,
                 kernel_size: int = 11, 
                 kernel_sigma: float = 1.5, 
                 k1: float = 0.01, 
                 k2: float = 0.03, 
                 downsample: bool = True, 
                 reduction: str = 'mean', 
                 data_range: int | float = 1) -> None:
        super().__init__(kernel_size, kernel_sigma, k1, k2, downsample, reduction, data_range)
    
    @torch.inference_mode()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)