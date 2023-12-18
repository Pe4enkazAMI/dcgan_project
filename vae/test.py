import argparse
import json
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from hw_vae.trainer import Trainer
from hw_vae.utils import prepare_device
from hw_vae.utils.object_loading import get_dataloaders
import numpy as np
import torchaudio
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

def dp(test_data_dir):
    data = list(filter(lambda x: x != ".DS_Store", os.listdir(test_data_dir)))
    data_paths = list(map(lambda x: test_data_dir + "/" + x, data))
    return data_paths


@hydra.main(version_base=None, config_path="", config_name="config_hydra_test")
def main(cfg):
    OmegaConf.resolve(cfg)

    print(f'{OmegaConf.to_yaml(cfg)}')

    logger = logging.getLogger("train")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...")
    checkpoint = torch.load(cfg.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if cfg["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    print(device)
    model = model.to(device)
    model.eval()
    noise = torch.randn(128, 128, 1, 1)
    fakes = model.generate(noise)
    
    save_image(fakes, "grid.png")



if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    print("start training")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()