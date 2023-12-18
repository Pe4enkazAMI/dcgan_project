import torch
from hw_vae.base import BaseTrainer

from hw_vae.utils import MetricTracker, inf_loop
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import PIL
import torchvision.transforms as T
import matplotlib.pyplot as plt
from hw_vae.metrics import SSIMMetric
from hw_vae.metrics import FIDMetric



class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            g_optimizer,
            d_optimizer,
            config,
            device,
            dataloaders,
            ckpt_dir,
            metric,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        print(device)
        super().__init__(model=model, 
                         criterion=criterion,
                         g_optimizer=g_optimizer,
                         d_optimizer=d_optimizer,
                         lr_shceduler=lr_scheduler,
                         config=config,
                         device=device,
                         ckpt_dir=ckpt_dir)
        
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.device = device
        self.metric = metric
        self.denorm = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler

        self.log_step = self.config["trainer"].get("log_step", 50)
        self.batch_accum_steps = self.config["trainer"].get("batch_accum_steps", 1)

        self.loss_keys = ["GANLoss"]
        self.fixed_noise = torch.randn(64, 128, 1, 1, device=device)
        self.train_metrics = MetricTracker(
            "GLoss", "DLoss", "SSIM", "FID", "grad_norm", writer=self.writer
        )
        self.evaluation_metrics = MetricTracker("GANLoss", "SSIMMetric", writer=self.writer)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        names = ["image"]
        for tensor_for_gpu in names:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        progress_bar = tqdm(range(self.len_epoch), desc='train')
        preds = {"image": []}
        for batch_idx, batch in enumerate(self.train_dataloader):
            stop = False
            progress_bar.update(1)
            try:
                batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                        batch_idx=batch_idx
                )

            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    f"Train Epoch: {epoch} {self._progress(batch_idx)} \
                        GLoss: {batch['GLoss'].item()}, DLoss: {batch['DLoss']}"
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.len_epoch:
                stop = True
                break
            if stop:
                break
        log = last_train_metrics

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(batch["VLBLoss"].item())

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})


        with torch.no_grad():
            self.model.eval()
            fake = self.model.generate(self.fixed_noise).detach().cpu()
            self.writer.add_image("real_example_images", fake)
            self.writer.add_image("train_loop_example", batch["image_fake"])
        
            fid = self.fid_metric(self.denorm(batch["image"][:16,...]).reshape(16, -1),
                                                         self.denorm(batch["image_fake"][:16, ...]).reshape(16, -1)).item()
            print("FID", fid)
            self.writer.add_scalar("FID", fid)
            ssim = self.ssim_metric(self.denorm(batch["image"][:16, ...]),
                                                           self.denorm(batch["image_fake"][:16, ...])).item()
            print("SSIM", ssim)
            self.writer.add_scalar("SSIM", ssim)
        return log
    
    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.model.discriminator.zero_grad()
            real_cpu = batch["image"]
            b_size = real_cpu.shape[0]
            label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
            output = self.model.discriminate(real_cpu).view(-1)
            errD_real = self.criterion(output, label)
            errD_real.backward()

            D_x = output.mean().item()
            noise = torch.randn(b_size, 128, 1, 1, device=self.device)
            fake = self.model.generate(noise)
            batch["image_fake"] = fake
            label.fill_(0)
            output = self.model.discriminate(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.d_optimizer.step()


            self.model.generator.zero_grad()
            label.fill_(1)  
            
            output = self.model.discriminate(fake).view(-1)
            
            errG = self.criterion(output, label)
            
            errG.backward()
            D_G_z2 = output.mean().item()
            
            self.g_optimizer.step()

            metrics.update("GLoss", errG.item())
            metrics.update("DLoss", errD.item())
            batch["GLoss"] = errG
            batch["DLoss"] = errD

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, audio, sr, name):
        self.writer.add_audio(f"Audio_{name}", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))