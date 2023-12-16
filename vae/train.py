import warnings
import numpy as np
import torch
import sys 
import os 
import logging
from pathlib import Path
from datetime import datetime

from hw_vae.trainer import Trainer
from hw_vae.utils import prepare_device
from hw_vae.utils.object_loading import get_dataloaders

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

@hydra.main(version_base=None, config_path="", config_name="config_hydra")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    print(f'{OmegaConf.to_yaml(cfg)}')

    logger = logging.getLogger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(cfg)
    print("DATALOADERS SETUP COMPLETED...")
    # build model architecture, then print to console
    model = instantiate(cfg["arch"])
    logger.info(model)
    print("MODEL SETUP COMPLETED...")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    metric = instantiate(cfg["Metric"])
    loss = instantiate(cfg["Loss"]).to(device)
    print("LOSS SETUP COMPLETED...")
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    params_g = model.generator.parameters()
    g_optimizer = instantiate(cfg["g_optimizer"], params=params_g)
    params_d = model.discriminator.parameters()
    d_optimizer = instantiate(cfg["d_optimizer"], params=params_d)
    print("OPT AND LR SETUP COMPLETED...")



    save_dir = Path(cfg["trainer"]["save_dir"])
    exper_name = cfg["name"]
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")
    _save_dir = str(save_dir / "models" / exper_name / run_id)
    _log_dir = str(save_dir / "log" / exper_name / run_id)

    # make directory for saving checkpoints and log.

    Path(_save_dir).mkdir(parents=True, exist_ok=True)
    Path(_log_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        criterion=loss,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=None,
        metric=metric,
        len_epoch=cfg["trainer"].get("len_epoch", None),
        ckpt_dir=_save_dir
    )
    trainer.train()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    print("start training")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()