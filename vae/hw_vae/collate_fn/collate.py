import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> dict:
    images = torch.stack([items["image"] for items in dataset_items], dim=0)
    return {
        "image": images
    }